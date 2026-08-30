#include "config.h"
#include "logging.h"
#include "messages.h"
#include "server.h"
#include "service_client.h"

#include <cpptrace/cpptrace.hpp>
#include <fmt/core.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

// TODO: avoid backtrace()
#include <execinfo.h>

extern "C" void LogAndExitHandler(int sig);
extern "C" void FatalSignalHandler(int sig);

static std::string GetLogPath()
{ return checkpoint::config::GetConfigValue<checkpoint::config::ServerLog>().string(); }

// Detach stdin/stderr so the daemon never blocks on a pipe owned by the departed client.
// Stdout stays on the launcher pipe for RunServer's "READY\n" handshake (single-write, SIGPIPE ignored).
// Stderr goes to the log file; if it can't be opened the daemon refuses to start.
static void RedirectStdio()
{
    auto                    logPath     = GetLogPath();
    static constexpr mode_t LogFileMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int                     logFd
        = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, LogFileMode);    // NOLINT(cppcoreguidelines-pro-type-vararg)
    if (logFd < 0)
    {
        // Refuse to start without a log: stderr still points at the launcher pipe here (not yet
        // redirected), so this throw reaches main's catch and the launcher sees the real reason.
        int const err = errno;
        throw std::runtime_error(fmt::format("cannot open server log {}: {}", logPath, std::strerror(err)));
    }
    int nullFd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);    // NOLINT(cppcoreguidelines-pro-type-vararg)

    if (nullFd >= 0)
    {
        ::dup2(nullFd, STDIN_FILENO);
        if (nullFd > STDERR_FILENO) { ::close(nullFd); }
    }
    ::dup2(logFd, STDERR_FILENO);
    if (logFd > STDERR_FILENO) { ::close(logFd); }
}

// Log a breadcrumb then re-raise with the default disposition to exit with the standard signal
// status. Safe: RedirectStdio pointed STDERR at a regular file, so write() can't block or SIGPIPE.
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
    static constexpr std::string_view Prefix = "[native_ckpt:svr][SIGNAL] received ";
    static constexpr std::string_view Suffix = "\n";
    (void)::write(STDERR_FILENO, std::data(Prefix), Prefix.size());
    (void)::write(STDERR_FILENO, name, std::strlen(name));
    (void)::write(STDERR_FILENO, std::data(Suffix), Suffix.size());

    ::signal(sig, SIG_DFL);    // NOLINT(concurrency-mt-unsafe)
    ::raise(sig);
}

// Full demangled, file:line backtrace via cpptrace. Used by the std::terminate handler, which
// runs in a normal context where allocation + symbolization are safe.
static void DumpBacktraceToLog()
{
    fmt::print(stderr, "[native_ckpt:svr][CRASH] backtrace (most recent call first):\n");
    cpptrace::generate_trace().print(std::cerr);
}

// Async-signal-safe backtrace via glibc execinfo: cpptrace allocates (and can_signal_safe_unwind()
// is false on this build's unwind backend), so it re-aborts on a corrupt heap — the common SIGABRT
// case. backtrace_symbols_fd() writes "<object>(<symbol>+0x<off>) [0x<addr>]" to the fd without
// malloc; resolve offline with addr2line. backtrace()'s lazy unwinder load is warmed in main().
static void DumpSafeBacktrace(int fd)
{
    static constexpr std::string_view Hdr = "[native_ckpt:svr][CRASH] backtrace (most recent call first):\n";
    (void)::write(fd, std::data(Hdr), Hdr.size());
    static constexpr int                MaxFrames = 128;
    static std::array<void*, MaxFrames> Frames{};    // static: keep the buffer off the small sigaltstack
    int const                           n = ::backtrace(Frames.data(), MaxFrames);
    if (n <= 0)
    {
        static constexpr std::string_view Msg = "[native_ckpt:svr][CRASH]   (no frames captured)\n";
        (void)::write(fd, std::data(Msg), Msg.size());
        return;
    }
    ::backtrace_symbols_fd(Frames.data(), n, fd);    // async-signal-safe: writes to fd, never calls malloc
}

// Fatal-signal handler: a crash otherwise dies with no log line. Log the signal + a backtrace,
// then restore the default disposition and re-raise to exit with the signal's status + a core.
extern "C" void FatalSignalHandler(int sig)
{
    // Re-entry/cross-thread guard: SA_NODEFER leaves the signal unblocked, so a re-fault here would
    // re-enter and spin forever. Only the first fatal signal runs the backtrace; any concurrent or
    // re-entrant signal restores the default disposition and re-raises so the kernel finishes up.
    static sig_atomic_t volatile InHandler = 0;
    if (__atomic_exchange_n(&InHandler, 1, __ATOMIC_SEQ_CST) != 0)
    {
        ::signal(sig, SIG_DFL);    // NOLINT(concurrency-mt-unsafe)
        ::raise(sig);
        return;
    }

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
    (void)::write(STDERR_FILENO, std::data(Prefix), sizeof(Prefix) - 1);
    (void)::write(STDERR_FILENO, name, std::strlen(name));
    (void)::write(STDERR_FILENO, std::data(Suffix), sizeof(Suffix) - 1);
    // Signal-safe (no allocation) so it still produces a trace when the heap is corrupt — the
    // common SIGABRT cause. DumpBacktraceToLog's symbolizing trace would re-abort here.
    DumpSafeBacktrace(STDERR_FILENO);
    ::signal(sig, SIG_DFL);    // NOLINT(concurrency-mt-unsafe)
    ::raise(sig);
}

// std::terminate handler: an uncaught C++ exception calls terminate (not a signal), so we run
// in a normal context where full symbolization is safe — recover the what() and print a
// demangled, file:line-resolved trace via cpptrace.
[[noreturn]] static void TerminateHandler()
{
    static constexpr std::string_view Hdr = "[native_ckpt:svr][CRASH] std::terminate — uncaught exception\n";
    (void)::write(STDERR_FILENO, std::data(Hdr), Hdr.size());
    if (std::exception_ptr const ex = std::current_exception())
    {
        try
        {
            std::rethrow_exception(ex);
        } catch (std::exception const& e) { fmt::print(stderr, "[native_ckpt:svr][CRASH] what(): {}\n", e.what()); } catch (...)
        {
            fmt::print(stderr, "[native_ckpt:svr][CRASH] non-std exception\n");
        }
    }
    DumpBacktraceToLog();
    std::abort();
}

static void InstallSignalHandlers()
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
    static std::array<char, 64ul * 1024ul> AltStackStorage{};
    stack_t                                altStack{};
    altStack.ss_sp    = static_cast<void*>(AltStackStorage.data());
    altStack.ss_size  = AltStackStorage.size();
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
    // Warm glibc's unwinder too: backtrace()'s first call may lazily dlopen libgcc_s (which
    // allocates), so prime it now — the fatal handler's DumpSafeBacktrace must not allocate.
    {
        std::array<void*, 1> warm{};
        (void)::backtrace(warm.data(), 1);
    }
    InstallSignalHandlers();
    fmt::print(stderr, "[native_ckpt:svr][INFO] Daemon started pid={} sid={} ppid={}\n", ::getpid(), ::getsid(0), ::getppid());
    RunServer();
    fmt::print(stderr, "[native_ckpt:svr][INFO] Daemon exiting normally pid={}\n", ::getpid());
    checkpoint::logging::Shutdown();
    return 0;
} catch (checkpoint::ipc::ErrorAlreadyRunning const& e)
{
    // Another candidate won the instance election. Do not let this candidate's
    // launcher observe EOF until the elected daemon is actually answering pings;
    // otherwise several ranks can race ahead while its queues are still booting.
    fmt::print(stderr, "[native_ckpt:svr][INFO] {}\n", e.what());
    if (!TryConnectToCheckpointService())
    {
        fmt::print(stderr, "[native_ckpt:svr][ERROR] Elected checkpoint server did not become ready\n");
        return 1;
    }
    fmt::print(stdout, "READY\n");
    std::error_code ec;
    std::filesystem::remove(GetLogPath(), ec);
    checkpoint::logging::Shutdown();

    return 0;
} catch (std::exception const& e)
{
    fmt::print(stderr, "[native_ckpt:svr][ERROR] Fatal error: {}\n", e.what());
    checkpoint::logging::Shutdown();
    return 1;
}
