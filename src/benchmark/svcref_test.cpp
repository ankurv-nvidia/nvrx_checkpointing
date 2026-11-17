#include "svcref.h"
#include "server.h"
#include "service_client.h"
#include "telemetry.h"

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/process/v2/pid.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

namespace
{
using namespace std::chrono_literals;
namespace http = boost::beast::http;
using Tcp      = boost::asio::ip::tcp;

uint64_t Pid()
{ return static_cast<uint64_t>(boost::process::v2::current_pid()); }

class Peer
{
    public:
    Peer() : _acceptor(_ioc, Tcp::endpoint{boost::asio::ip::address_v4::loopback(), 0})
    {
        _acceptor.non_blocking(true);
        _thread = std::jthread([this](std::stop_token stop) {
            while (!stop.stop_requested())
            {
                boost::system::error_code error;
                Tcp::socket               socket(_ioc);
                _acceptor.accept(socket, error);
                if (error)
                {
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                try
                {
                    boost::beast::tcp_stream         stream(std::move(socket));
                    boost::beast::flat_buffer        buffer;
                    http::request<http::string_body> request;
                    http::read(stream, buffer, request);
                    http::response<http::string_body> response{available ? http::status::ok : http::status::service_unavailable, 11};
                    auto const                        target = request.target();
                    if (target == "/api/peer/acquireref")
                    {
                        acquiredToken = std::stoull(std::string(request["X-Checkpoint-Token"]));
                        ++acquisitions;
                        response.body() = std::to_string(acquiredToken.load());
                    }
                    else if (target.starts_with("/api/peer/releaseref/"))
                    {
                        ++releaseAttempts;
                        bool const rejected = rejectReleases.load();
                        response.result(rejected ? http::status::service_unavailable : http::status::ok);
                        if (!rejected)
                        {
                            releasedToken = std::stoull(std::string(target.substr(std::string_view("/api/peer/releaseref/").size())));
                            ++releases;
                        }
                    }
                    else
                    {
                        ++probes;
                        response.body() = std::to_string(generation.load());
                    }
                    response.prepare_payload();
                    http::write(stream, response);
                } catch (boost::system::system_error const&) {}
            }
        });
    }

    [[nodiscard]] std::string Address() const { return "127.0.0.1:" + std::to_string(_acceptor.local_endpoint().port()); }

    std::atomic<bool>     available{true};
    std::atomic<uint64_t> generation{7};
    std::atomic<unsigned> probes{};
    std::atomic<unsigned> acquisitions{};
    std::atomic<unsigned> releaseAttempts{};
    std::atomic<unsigned> releases{};
    std::atomic<uint64_t> acquiredToken{};
    std::atomic<uint64_t> releasedToken{};
    std::atomic<bool>     rejectReleases{false};

    private:
    boost::asio::io_context _ioc;
    Tcp::acceptor           _acceptor;
    std::jthread            _thread;
};

class Shutdown
{
    public:
    explicit Shutdown(uint16_t index, SvcRef::Busy busy = {}, std::function<bool()> drained = {}) :
        _finished(std::async(std::launch::async, [this, index, pending = std::move(busy), flush = std::move(drained)] {
                      SvcRef::WaitForShutdown(index, _stop, pending, flush);
                  }).share())
    {}
    ~Shutdown()
    {
        _stop = true;
        _finished.wait();
    }

    bool FinishedWithin(std::chrono::milliseconds timeout) const
    {
        if (_finished.wait_for(timeout) != std::future_status::ready) { return false; }
        _finished.get();
        return true;
    }

    private:
    std::atomic<bool>        _stop{false};
    std::shared_future<void> _finished;
};

template <typename TPredicate> bool Eventually(TPredicate&& predicate)
{
    auto const deadline = std::chrono::steady_clock::now() + 3s;
    while (!predicate())
    {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::sleep_for(10ms);
    }
    return true;
}
}    // namespace

TEST_CASE("shared in-process references retain work until the final owner releases", "[svcref][tier1]")
{
    constexpr uint16_t Index = 500;
    SvcRef::SvrStart(Index);
    auto       held  = SvcRef::AcquireInProc("save", SvcRef::AllSessions, Index);
    auto       copy  = held;
    auto const token = SvcRef::SvrAcquireLocalProcRef(Pid(), Index);
    SvcRef::SvrReleaseLocalProcRef(Pid(), token, Index);
    Shutdown shutdown(Index);
    held.reset();
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    copy.reset();
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("live PID references require explicit release", "[svcref][tier1]")
{
    constexpr uint16_t Index = 501;
    SvcRef::SvrStart(Index);
    auto const token = SvcRef::SvrAcquireLocalProcRef(Pid(), Index);
    Shutdown   shutdown(Index);
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    SvcRef::SvrReleaseLocalProcRef(Pid(), token, Index);
    SvcRef::SvrReleaseLocalProcRef(Pid(), token, Index);
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("zombie PIDs do not retain the daemon", "[svcref][tier1]")
{
    auto const child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) { ::_exit(0); }
    siginfo_t exited{};
    REQUIRE(::waitid(P_PID, static_cast<id_t>(child), &exited, WEXITED | WNOWAIT) == 0);
    constexpr uint16_t Index = 502;
    SvcRef::SvrStart(Index);
    static_cast<void>(SvcRef::SvrAcquireLocalProcRef(static_cast<uint64_t>(child), Index));
    Shutdown shutdown(Index);
    CHECK(shutdown.FinishedWithin(1500ms));
    int status{};
    REQUIRE(::waitpid(child, &status, 0) == child);
}

TEST_CASE("self references avoid unnecessary peer probes", "[svcref][tier1]")
{
    constexpr uint16_t Index = 503;
    Peer               peer;
    peer.available = false;
    SvcRef::SvrStart(Index);
    static_cast<void>(SvcRef::SvrAcquireRemoteProcRef(peer.Address(), peer.generation.load(), Index));
    auto     held = SvcRef::AcquireInProc("pending request", SvcRef::AllSessions, Index);
    Shutdown shutdown(Index);
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    CHECK(peer.probes.load() == 0);
    held.reset();
    CHECK(shutdown.FinishedWithin(1500ms));
    CHECK(peer.probes.load() > 0);
}

TEST_CASE("peer startup generation must match its reference", "[svcref][tier1]")
{
    constexpr uint16_t Index = 504;
    Peer               peer;
    SvcRef::SvrStart(Index);
    static_cast<void>(SvcRef::SvrAcquireRemoteProcRef(peer.Address(), peer.generation.load() + 1, Index));
    Shutdown shutdown(Index);
    CHECK(shutdown.FinishedWithin(1500ms));
    CHECK(peer.probes.load() > 0);
}

TEST_CASE("failed peer references retry while another peer remains alive", "[svcref][tier1]")
{
    constexpr uint16_t Index = 505;
    Peer               recovering;
    Peer               keeper;
    recovering.available = false;
    SvcRef::SvrStart(Index);
    auto const first = SvcRef::SvrAcquireRemoteProcRef(recovering.Address(), recovering.generation.load(), Index);
    auto const last  = SvcRef::SvrAcquireRemoteProcRef(keeper.Address(), 0, Index);
    Shutdown   shutdown(Index);
    REQUIRE(Eventually([&] { return recovering.probes.load() > 0 && keeper.probes.load() > 0; }));
    auto const before    = recovering.probes.load();
    recovering.available = true;
    SvcRef::SvrReleaseRemoteProcRef(keeper.Address(), last, Index);
    REQUIRE(Eventually([&] { return recovering.probes.load() > before; }));
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    SvcRef::SvrReleaseRemoteProcRef(recovering.Address(), first, Index);
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("session coordinator links drain and retry releases independently", "[svcref][tier1]")
{
    constexpr uint16_t Index = 506;
    Peer               coordinator;
    SvcRef::SvrStart(Index);
    auto const            first      = SvcRef::SvrAcquireLocalProcRef(Pid(), Index, 42, coordinator.Address(), "peer:4100");
    auto const            other      = SvcRef::SvrAcquireLocalProcRef(Pid(), Index, 43);
    auto const            firstToken = coordinator.acquiredToken.load();
    std::atomic<bool>     busy{true}, drained{false};
    std::atomic<unsigned> drainCalls{};
    auto                  work = SvcRef::AcquireInProc("pending", 42, Index);
    Shutdown              shutdown(
        Index,
        [&](uint32_t session) { return session == 42 && busy.load(); },
        [&] {
            ++drainCalls;
            return drained.load();
        });
    SvcRef::SvrReleaseLocalProcRef(Pid(), first, Index);
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    CHECK(coordinator.releaseAttempts.load() == 0);
    work.reset();
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    CHECK(coordinator.releaseAttempts.load() == 0);
    busy = false;
    REQUIRE(Eventually([&] { return drainCalls.load() > 0; }));
    CHECK(coordinator.releaseAttempts.load() == 0);
    coordinator.rejectReleases = true;
    drained                    = true;
    REQUIRE(Eventually([&] { return coordinator.releaseAttempts.load() > 0; }));
    CHECK(coordinator.releases.load() == 0);
    auto const next      = SvcRef::SvrAcquireLocalProcRef(Pid(), Index, 42, coordinator.Address(), "peer:4100");
    auto const nextToken = coordinator.acquiredToken.load();
    CHECK(coordinator.acquisitions.load() == 2);
    CHECK(nextToken != firstToken);
    coordinator.rejectReleases = false;
    REQUIRE(Eventually([&] { return coordinator.releases.load() == 1; }));
    CHECK(coordinator.releasedToken.load() == firstToken);
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    SvcRef::SvrReleaseLocalProcRef(Pid(), next, Index);
    REQUIRE(Eventually([&] { return coordinator.releases.load() == 2; }));
    CHECK(coordinator.releasedToken.load() == nextToken);
    SvcRef::SvrReleaseLocalProcRef(Pid(), other, Index);
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("peer endpoints retain the coordinator through peer shutdown", "[svcref][server][tier1]")
{
    auto const telemetryFile = std::filesystem::temp_directory_path() / "svcref-telemetry.jsonl";
    ::setenv("NATIVE_CKPT_OTEL_EXPORT", "file", 1);                    // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_FILE", telemetryFile.c_str(), 1);       // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS", "1", 1);    // NOLINT(concurrency-mt-unsafe)
    struct Cleanup
    {
        ~Cleanup()
        {
            SpinDownAllInprocServers();
            ::unsetenv("NATIVE_CKPT_OTEL_EXPORT");                       // NOLINT(concurrency-mt-unsafe)
            ::unsetenv("NATIVE_CKPT_OTEL_FILE");                         // NOLINT(concurrency-mt-unsafe)
            ::unsetenv("NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS");    // NOLINT(concurrency-mt-unsafe)
        }
    } cleanup;
    RunInprocServer(2);
    REQUIRE(checkpoint::telemetry::IsEnabled());
    auto     client = SvcRef::CliAcquireLocalProc(1, 42, "127.0.0.1:4100", "127.0.0.1:4101");
    Shutdown coordinator(0);
    Shutdown peer(1);
    CHECK_FALSE(coordinator.FinishedWithin(50ms));
    CHECK_FALSE(peer.FinishedWithin(50ms));
    client.reset();
    CHECK(peer.FinishedWithin(1500ms));
    CHECK_FALSE(coordinator.FinishedWithin(50ms));
    SvcRef::ReleaseRemoteClients(1);
    CHECK(coordinator.FinishedWithin(1500ms));
}

TEST_CASE("local references start the selected daemon", "[svcref][launch][tier1]")
{
    constexpr uint16_t Index     = 2;
    auto               reference = SvcRef::CliAcquireLocalProc(Index);
    CHECK(TryConnectToCheckpointService(Index));
    reference.reset();
    CHECK(Eventually([] { return !TryConnectToCheckpointService(Index); }));
    WaitForOwnedInprocServer();
}
