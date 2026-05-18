#include "messages.h"
#include "server.h"

#include <fmt/core.h>

#include <csignal>
#include <cstring>

extern "C" void LogAndContinueHandler(int sig);
extern "C" void LogAndExitHandler(int sig);

namespace
{

// Handler that logs and lets the daemon continue running (for SIGHUP, SIGPIPE).
extern "C" void LogAndContinueHandler(int sig)
{
    char const* name = [](int signal) -> char const* {
        switch (signal)
        {
        case SIGHUP: return "SIGHUP (ignored, daemon continues)";
        case SIGINT: return "SIGINT (terminating)";
        case SIGTERM: return "SIGTERM (terminating)";
        case SIGPIPE: return "SIGPIPE (ignored, daemon continues)";
        case SIGQUIT: return "SIGQUIT (terminating)";
        default: return "unknown signal";
        }
    }(sig);

    // WriteSignalLog runs inside a signal handler, and only async-signal-safe functions are legal there.
    // fmt::print / printf / std::cout are not.
    static constexpr char Prefix[] = "[native_ckpt:svr][SIGNAL] received ";    // NOLINT
    static constexpr char Suffix[] = "\n";                                     // NOLINT
    (void)::write(STDERR_FILENO, Prefix, sizeof(Prefix) - 1);                  // NOLINT
    (void)::write(STDERR_FILENO, name, std::strlen(name));
    (void)::write(STDERR_FILENO, Suffix, sizeof(Suffix) - 1);    // NOLINT
}

// Handler that logs and then re-raises with the default disposition so we exit
// with the standard signal status (and any wrapper sees the real cause).
extern "C" void LogAndExitHandler(int sig)
{
    LogAndContinueHandler(sig);
    ::signal(sig, SIG_DFL);    // NOLINT(concurrency-mt-unsafe)
    ::raise(sig);
}

void InstallSignalHandlers()
{
    // Survive parent (client) death and broken stdout pipe.
    ::signal(SIGHUP, &LogAndContinueHandler);     // NOLINT(concurrency-mt-unsafe)
    ::signal(SIGPIPE, &LogAndContinueHandler);    // NOLINT(concurrency-mt-unsafe)

    // Log before terminating on explicit kill signals so we have a breadcrumb.
    ::signal(SIGINT, &LogAndExitHandler);     // NOLINT(concurrency-mt-unsafe)
    ::signal(SIGTERM, &LogAndExitHandler);    // NOLINT(concurrency-mt-unsafe)
    ::signal(SIGQUIT, &LogAndExitHandler);    // NOLINT(concurrency-mt-unsafe)
}
}    // namespace

int main(int /* argc */, char const* const /* argv */[])
try
{
    ::setsid();
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
