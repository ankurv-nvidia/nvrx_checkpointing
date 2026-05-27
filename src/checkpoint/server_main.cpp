#include "messages.h"
#include "server.h"

#include <fmt/core.h>

#include <csignal>
#include <cstdlib>
#include <cstring>

extern "C" void LogAndExitHandler(int sig);

namespace
{

// Detach inherited stdin/stderr so the daemon never blocks on a pipe owned by the
// now-departed client. Stdout is intentionally left pointing at the launcher pipe
// so RunServer's "READY\n" handshake (server.cpp) still reaches the launcher; that
// fd is single-write-only and SIGPIPE is ignored, so it can't hang us later.
// Stderr log path honors $NATIVE_CKPT_SERVER_LOG; falls back to
// /tmp/native_ckpt_server.<pid>.log.
void RedirectStdio()
{
    char const* envLog = ::getenv("NATIVE_CKPT_SERVER_LOG");
    std::string logPath
        = (envLog != nullptr && *envLog != '\0') ? std::string(envLog) : fmt::format("/tmp/native_ckpt_server.{}.log", ::getpid());

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
}
}    // namespace

int main(int /* argc */, char const* const /* argv */[])
try
{
    ::setsid();
    RedirectStdio();
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
