#include "messages.h"
#include "server.h"

#include <fmt/core.h>

#include <cpptrace/cpptrace.hpp>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

extern "C" void LogAndExitHandler(int sig);
extern "C" void FatalSignalHandler(int sig);

namespace
{

// Expand %h (hostname) and %p (pid) in a server-log path so each node's per-node
// singleton writes a distinct file. run_job points $NATIVE_CKPT_SERVER_LOG at the
// shared ci-results dir with a %h token, and that dir is uploaded as a CI artifact,
// so the crash backtrace this daemon logs survives the job and is collectible
// per-node even on a multinode run where every node runs its own server.
std::string ExpandLogPath(std::string path)
{
    char        hostbuf[256] = {};    // NOLINT — zero-init guarantees NUL-termination even on truncation
    std::string const host   = (::gethostname(hostbuf, sizeof(hostbuf) - 1) == 0) ? std::string(hostbuf) : std::string("unknown-host");
    auto              subst   = [](std::string& s, char const* tok, std::string const& val) {
        std::string const token(tok);
        for (std::size_t pos = s.find(token); pos != std::string::npos; pos = s.find(token, pos + val.size()))
        {
            s.replace(pos, token.size(), val);
        }
    };
    subst(path, "%h", host);
    subst(path, "%p", std::to_string(::getpid()));
    return path;
}

// Detach inherited stdin/stderr so the daemon never blocks on a pipe owned by the
// now-departed client. Stdout is intentionally left pointing at the launcher pipe
// so RunServer's "READY\n" handshake (server.cpp) still reaches the launcher; that
// fd is single-write-only and SIGPIPE is ignored, so it can't hang us later.
// Stderr log path honors $NATIVE_CKPT_SERVER_LOG; falls back to
// /tmp/native_ckpt_server.<pid>.log.
void RedirectStdio()
{
    char const* envLog = ::getenv("NATIVE_CKPT_SERVER_LOG");
    std::string logPath = (envLog != nullptr && *envLog != '\0') ? ExpandLogPath(std::string(envLog))
                                                                  : fmt::format("/tmp/native_ckpt_server.{}.log", ::getpid());

    static constexpr mode_t LogFileMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int                     logFd       = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, LogFileMode);    // NOLINT
    if (logFd < 0) { logFd = ::open("/dev/null", O_WRONLY | O_CLOEXEC); }                                                     // NOLINT
    int nullFd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);                                                                   // NOLINT

    if (nullFd >= 0)
    {
        ::dup2(nullFd, STDIN_FILENO);
        if (nullFd > STDERR_FILENO) { ::close(nullFd); }
    }
    if (logFd >= 0)
    {
        ::dup2(logFd, STDERR_FILENO);
        if (logFd > STDERR_FILENO) { ::close(logFd); }
    }
}

// Logs a breadcrumb then re-raises with the default disposition so we exit with the standard
// signal status. Safe because RedirectStdio has pointed STDERR_FILENO at a regular file: write()
// on a regular file cannot block on a dead pipe and cannot raise SIGPIPE.
extern "C" void LogAndExitHandler(int sig)
{
    char const* name = [](int signal) -> char const* {
        switch (signal)
        {
        case SIGINT: return "SIGINT (terminating)";
        case SIGTERM: return "SIGTERM (terminating)";
        case SIGQUIT: return "SIGQUIT (terminating)";
        default: return "unknown signal";
        }
    }(sig);

    // Only async-signal-safe calls in here.
    static constexpr char Prefix[] = "[native_ckpt:svr][SIGNAL] received ";    // NOLINT
    static constexpr char Suffix[] = "\n";                                     // NOLINT
    (void)::write(STDERR_FILENO, Prefix, sizeof(Prefix) - 1);                  // NOLINT
    (void)::write(STDERR_FILENO, name, std::strlen(name));
    (void)::write(STDERR_FILENO, Suffix, sizeof(Suffix) - 1);    // NOLINT

    ::signal(sig, SIG_DFL);    // NOLINT(concurrency-mt-unsafe)
    ::raise(sig);
}

// Write a demangled, file:line-resolved backtrace to STDERR_FILENO (RedirectStdio points it at
// the log file) via cpptrace. Shared by the fatal-signal and std::terminate handlers.
//
// Not strictly async-signal-safe: generate_trace() unwinds and resolves symbols (allocates,
// reads the binary). The fatal-signal handler accepts this — it re-raises to the default
// disposition afterward, so the kernel still drops a core even if a trace from deep heap
// corruption is truncated, and a demangled trace is far more useful for diagnosing the crash
// than a mangled-but-signal-safe one. absorb_trace_exceptions (set in main) keeps it from
// throwing; resolving one trace at startup warms the resolver so the first crash is cheap.
void DumpBacktraceToLog()
{
    fmt::print(stderr, "[native_ckpt:svr][CRASH] backtrace (most recent call first):\n");
    cpptrace::generate_trace().print(std::cerr);
}

// Fatal-signal handler: a crash (segfault, abort/assert, bus error, ...) otherwise dies with
// no log line, so the server log the heartbeat-failure diagnostic dumps is empty. Log the
// signal + a backtrace, then restore the default disposition and re-raise so the process still
// exits with the signal's status and drops a core.
extern "C" void FatalSignalHandler(int sig)
{
    char const* name = [](int signal) -> char const* {
        switch (signal)
        {
        case SIGSEGV: return "SIGSEGV (segfault)";
        case SIGABRT: return "SIGABRT (abort/assert)";
        case SIGBUS: return "SIGBUS (bad memory access)";
        case SIGFPE: return "SIGFPE (arithmetic)";
        case SIGILL: return "SIGILL (illegal instruction)";
        default: return "fatal signal";
        }
    }(sig);
    static constexpr char Prefix[] = "[native_ckpt:svr][CRASH] crashed on ";    // NOLINT
    static constexpr char Suffix[] = "\n";                                      // NOLINT
    (void)::write(STDERR_FILENO, Prefix, sizeof(Prefix) - 1);
    (void)::write(STDERR_FILENO, name, std::strlen(name));
    (void)::write(STDERR_FILENO, Suffix, sizeof(Suffix) - 1);
    DumpBacktraceToLog();
    ::signal(sig, SIG_DFL);    // NOLINT(concurrency-mt-unsafe)
    ::raise(sig);
}

// std::terminate handler: an uncaught C++ exception calls terminate (not a signal), so we run
// in a normal context where full symbolization is safe — recover the what() and print a
// demangled, file:line-resolved trace via cpptrace. On the Itanium ABI the stack is not unwound
// before terminate, so the trace still includes the throw site. [[noreturn]]: always abort()s.
[[noreturn]] void TerminateHandler()
{
    static constexpr char Hdr[] = "[native_ckpt:svr][CRASH] std::terminate — uncaught exception\n";    // NOLINT
    (void)::write(STDERR_FILENO, Hdr, sizeof(Hdr) - 1);
    if (std::exception_ptr const ex = std::current_exception())
    {
        try
        {
            std::rethrow_exception(ex);
        } catch (std::exception const& e)
        {
            fmt::print(stderr, "[native_ckpt:svr][CRASH] what(): {}\n", e.what());
        } catch (...)
        {
            fmt::print(stderr, "[native_ckpt:svr][CRASH] non-std exception\n");
        }
    }
    DumpBacktraceToLog();
    std::abort();
}

void InstallSignalHandlers()
{
    // SIGPIPE: a dead reader on a socket or pipe is expected for a long-lived daemon; let write()
    // return EPIPE at the call site instead of trapping. The previous handler wrote to stderr,
    // which could itself trigger SIGPIPE and recurse / hang.
    struct sigaction ign{};
    ign.sa_handler = SIG_IGN;
    sigemptyset(&ign.sa_mask);
    ign.sa_flags = 0;
    ::sigaction(SIGPIPE, &ign, nullptr);

    // SIGHUP: after setsid() we have no controlling terminal, so the kernel shouldn't deliver it.
    // Ignore defensively (default disposition would terminate the daemon when we want it to survive
    // client exit).
    ::sigaction(SIGHUP, &ign, nullptr);

    // SIGINT/SIGTERM/SIGQUIT: log a breadcrumb then re-raise. sa_mask blocks all four signals
    // (plus SIGPIPE) while the handler runs so it can't be re-entered.
    struct sigaction exitSa{};
    exitSa.sa_handler = &LogAndExitHandler;
    sigemptyset(&exitSa.sa_mask);
    sigaddset(&exitSa.sa_mask, SIGINT);
    sigaddset(&exitSa.sa_mask, SIGTERM);
    sigaddset(&exitSa.sa_mask, SIGQUIT);
    sigaddset(&exitSa.sa_mask, SIGPIPE);
    exitSa.sa_flags = SA_RESTART;
    ::sigaction(SIGINT, &exitSa, nullptr);
    ::sigaction(SIGTERM, &exitSa, nullptr);
    ::sigaction(SIGQUIT, &exitSa, nullptr);

    // Crash signals: log a marker + backtrace then re-raise (FatalSignalHandler). Run on an
    // alternate stack (SA_ONSTACK) so a stack-overflow SIGSEGV can still be reported; SA_NODEFER
    // lets the final re-raise redeliver. SIGABRT also covers abort()/assert() and the
    // terminate-handler's abort.
    static char altStackStorage[64 * 1024];    // NOLINT — fixed size; SIGSTKSZ is not constant on modern glibc
    stack_t    altStack{};
    altStack.ss_sp    = static_cast<void*>(altStackStorage);
    altStack.ss_size  = sizeof(altStackStorage);
    altStack.ss_flags = 0;
    ::sigaltstack(&altStack, nullptr);

    struct sigaction crashSa{};
    crashSa.sa_handler = &FatalSignalHandler;
    sigemptyset(&crashSa.sa_mask);
    crashSa.sa_flags = SA_ONSTACK | SA_NODEFER;
    for (int const sig : {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL}) { ::sigaction(sig, &crashSa, nullptr); }

    // Uncaught C++ exceptions call std::terminate (not a signal): recover what() + backtrace.
    std::set_terminate(&TerminateHandler);
}
}    // namespace

int main(int /* argc */, char const* const /* argv */[])
try
{
    ::setsid();
    RedirectStdio();
    // Prime cpptrace before installing the handlers: absorb_trace_exceptions keeps trace
    // generation from ever throwing out of a handler, and resolving one trace now warms the
    // unwinder + libdwarf reader so the first crash backtrace is cheap.
    cpptrace::absorb_trace_exceptions(true);
    (void)cpptrace::generate_trace();
    InstallSignalHandlers();
    fmt::print(stderr, "[native_ckpt:svr][INFO] Daemon started pid={} sid={} ppid={}\n", ::getpid(), ::getsid(0), ::getppid());
    RunServer();
    fmt::print(stderr, "[native_ckpt:svr][INFO] Daemon exiting normally pid={}\n", ::getpid());
    return 0;
} catch (checkpoint::ipc::ErrorAlreadyRunning const& e)
{
    // Another instance is already running; the launcher will connect to it.
    fmt::print(stderr, "[native_ckpt:svr][INFO] {}\n", e.what());
    return 0;
} catch (std::exception const& e)
{
    fmt::print(stderr, "[native_ckpt:svr][ERROR] Fatal error: {}\n", e.what());
    return 1;
}
