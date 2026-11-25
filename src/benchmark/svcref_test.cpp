#include "svcref_test.h"

#include "CommonMacros.h"
#include "checkpoint.h"
#include "messages.h"
#include "server.h"
#include "state.h"
#include "svcref.h"
#include "tensor.h"

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/http.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

// NOLINTBEGIN(readability-magic-numbers)

using namespace std::chrono_literals;
namespace http = boost::beast::http;
using checkpoint::ipc::SessionId;

namespace
{
constexpr uint16_t BasePort = checkpoint::ipc::Message::Port;

template <typename TPredicate> bool WaitUntil(TPredicate predicate, std::chrono::milliseconds timeout)
{
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate())
    {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::sleep_for(10ms);
    }
    return true;
}

bool ServiceAnswers(uint16_t port)
{ return SvcRef::Answers(port); }

std::filesystem::path TestDirectory(std::string_view stem)
{
    auto const directory = std::filesystem::temp_directory_path() / "phoenix_ckpttest" / stem;
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

// Answers HTTP requests on a loopback port with the status the responder picks, recording each target and peer header.
class FakePeer
{
    public:
    explicit FakePeer(std::function<http::status(std::string const& target)> responder LFTBND) :
        _responder(std::move(responder)), _acceptor(_ioc, {boost::asio::ip::make_address("127.0.0.1"), 0})
    {
        _thread = std::thread([this] { Serve_(); });
    }

    ~FakePeer()
    {
        _stopped = true;
        boost::system::error_code    ec;
        boost::asio::io_context      ioc;
        boost::asio::ip::tcp::socket wake(ioc);
        static_cast<void>(wake.connect(_acceptor.local_endpoint(), ec));
        static_cast<void>(wake.close(ec));
        _thread.join();
    }

    FakePeer(FakePeer const&)            = delete;
    FakePeer& operator=(FakePeer const&) = delete;
    FakePeer(FakePeer&&)                 = delete;
    FakePeer& operator=(FakePeer&&)      = delete;

    [[nodiscard]] std::string Address() const { return fmt::format("127.0.0.1:{}", _acceptor.local_endpoint().port()); }

    std::vector<std::string> Requests()
    {
        std::scoped_lock lock(_mutex);
        return _requests;
    }

    private:
    void Serve_()
    {
        while (!_stopped)
        {
            boost::asio::ip::tcp::socket socket(_ioc);
            boost::system::error_code    ec;
            static_cast<void>(_acceptor.accept(socket, ec));
            if (ec || _stopped) { return; }
            boost::beast::flat_buffer        buffer;
            http::request<http::string_body> request;
            http::read(socket, buffer, request, ec);
            if (ec) { continue; }
            auto const target = std::string{request.target()};
            {
                std::scoped_lock lock(_mutex);
                _requests.push_back(fmt::format("{} {}", target, std::string{request[SvcRef::PeerHeader]}));
            }
            http::response<http::string_body> response{_responder(target), request.version()};
            response.body() = "7";
            response.prepare_payload();
            http::write(socket, response, ec);
        }
    }

    std::function<http::status(std::string const&)> _responder;
    boost::asio::io_context                         _ioc;
    boost::asio::ip::tcp::acceptor                  _acceptor;
    std::atomic<bool>                               _stopped{false};
    std::mutex                                      _mutex;
    std::vector<std::string>                        _requests;
    std::thread                                     _thread;
};

// A ckpttest child in one of the service reference roles, with stdout piped back and stderr in a file.
class ChildProcess
{
    public:
    ChildProcess(std::vector<std::string> const&           args,
                 std::map<std::string, std::string> const& environment,
                 std::filesystem::path const&              errorLog)
    {
        auto const               executable = std::filesystem::read_symlink("/proc/self/exe").string();
        std::vector<std::string> argStorage{executable, std::string{SvcRefChildArg}};
        argStorage.insert(argStorage.end(), args.begin(), args.end());
        std::vector<std::string> envStorage;
        for (char** entry = environ; *entry != nullptr; ++entry)
        {
            std::string_view const variable{*entry};
            if (!environment.contains(std::string{variable.substr(0, variable.find('='))})) { envStorage.emplace_back(variable); }
        }
        for (auto const& [name, value] : environment) { envStorage.push_back(name + "=" + value); }
        std::vector<char*> argPointers;
        std::vector<char*> envPointers;
        for (auto& arg : argStorage) { argPointers.push_back(arg.data()); }
        for (auto& variable : envStorage) { envPointers.push_back(variable.data()); }
        argPointers.push_back(nullptr);
        envPointers.push_back(nullptr);

        std::array<int, 2> pipeFds{-1, -1};
        if (::pipe2(pipeFds.data(), O_CLOEXEC) != 0) { throw std::runtime_error("pipe2 failed"); }
        int const errorFd
            = ::open(errorLog.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);    // NOLINT(cppcoreguidelines-pro-type-vararg)
        _pid = ::fork();
        if (_pid == 0)
        {
            ::dup2(pipeFds[1], STDOUT_FILENO);
            if (errorFd >= 0) { ::dup2(errorFd, STDERR_FILENO); }
            ::execve(executable.c_str(), argPointers.data(), envPointers.data());
            ::_exit(127);
        }
        ::close(pipeFds[1]);
        if (errorFd >= 0) { ::close(errorFd); }
        _output = pipeFds[0];
        if (_pid < 0) { throw std::runtime_error("fork failed"); }
    }

    ~ChildProcess()
    {
        if (_pid > 0)
        {
            ::kill(_pid, SIGKILL);
            ::waitpid(_pid, nullptr, 0);
        }
        if (_output >= 0) { ::close(_output); }
    }

    ChildProcess(ChildProcess const&)            = delete;
    ChildProcess& operator=(ChildProcess const&) = delete;
    ChildProcess(ChildProcess&&)                 = delete;
    ChildProcess& operator=(ChildProcess&&)      = delete;

    bool WaitForLine(std::string_view line, std::chrono::milliseconds timeout)
    {
        auto const deadline = std::chrono::steady_clock::now() + timeout;
        auto const expected = fmt::format("{}\n", line);
        while (true)
        {
            if (auto const position = _buffer.find(expected); position != std::string::npos)
            {
                _buffer.erase(0, position + expected.size());
                return true;
            }
            auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            if (remaining <= 0ms) { return false; }
            pollfd descriptor{.fd = _output, .events = POLLIN, .revents = 0};
            if (::poll(&descriptor, 1, static_cast<int>(remaining.count())) <= 0) { continue; }
            std::array<char, 256> chunk{};
            auto const            count = ::read(_output, chunk.data(), chunk.size());
            if (count <= 0) { return false; }
            _buffer.append(chunk.data(), static_cast<size_t>(count));
        }
    }

    bool Running() { return !Reap_(WNOHANG); }

    std::optional<int> WaitForExit(std::chrono::milliseconds timeout)
    {
        if (!WaitUntil([this] { return Reap_(WNOHANG); }, timeout)) { return std::nullopt; }
        return _status;
    }

    void Kill() const
    {
        if (_pid > 0) { ::kill(_pid, SIGKILL); }
    }

    private:
    bool Reap_(int options)
    {
        if (_pid <= 0) { return true; }
        int status = 0;
        if (::waitpid(_pid, &status, options) != _pid) { return false; }
        _pid    = -1;
        _status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        return true;
    }

    pid_t       _pid{-1};
    int         _output{-1};
    int         _status{-1};
    std::string _buffer;
};

void Emit(std::string_view marker)
{
    fmt::print(stdout, "{}\n", marker);
    std::fflush(stdout);
}

// One rank's root over a four-float slice of a shared weight.
struct RankState
{
    std::array<float, 4>       values{};
    std::unique_ptr<StateTree> tree;
};

std::unique_ptr<RankState> MakeRank(uint32_t rank, uint32_t worldSize, uint32_t ranksPerPeer, std::vector<std::string> const& peers)
{
    auto state = std::make_unique<RankState>();
    state->values.fill(static_cast<float>(rank));
    state->tree = std::make_unique<StateTree>(SessionId::Default,
                                              RankTopology{
                                                  .rank         = Rank{rank},
                                                  .worldSize    = Rank{worldSize},
                                                  .ranksPerPeer = ranksPerPeer,
                                                  .peersPerNode = static_cast<uint32_t>(peers.size()),
                                                  .peers        = peers,
                                              });
    Tensor::Dim fulldim;
    fulldim.push_back(size_t{4} * worldSize);
    DTensor          tensor(Tensor::DType::Float32, fulldim);
    Tensor::DimRange sharddim;
    sharddim.push_back({.begin = size_t{4} * rank, .end = size_t{4} * (rank + 1)});
    tensor.AddShard({
        .rank          = Rank{rank},
        .snapshotDtype = Tensor::DType::Float32,
        .handle        = reinterpret_cast<uint64_t>(state->values.data()),    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        .storageSize   = sizeof(state->values),
        .sharddim      = sharddim,
        .device        = DeviceId{"cpu"},
    });
    state->tree->AddChild(StateNode::Handle::Root, "weight").data = std::move(tensor);
    return state;
}

void SaveRank(RankState& state, std::filesystem::path const& directory)
{
    FinalizeStateRegistry(*state.tree);
    auto  snapshotRegistry = GetSnapshotRegistry(*state.tree);
    auto& shards           = std::get<DTensor>(snapshotRegistry->GetNode(snapshotRegistry->FindNodeByFQN("weight")).data).shards;
    if (shards.size() != 1) { throw std::runtime_error("the snapshot plan should keep each rank's own slice"); }
    auto source = shards.extract(shards.begin());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    source.value().handle = reinterpret_cast<uint64_t>(state.values.data());
    shards.insert(std::move(source));
    SaveCheckpoint(*snapshotRegistry, checkpoint::ipc::SnapshotId{1}, directory);
}

int RunPeer(uint32_t                     peerIndex,
            uint32_t                     peerCount,
            uint32_t                     ranksPerPeer,
            std::filesystem::path const& directory,
            std::chrono::milliseconds    hold,
            std::chrono::milliseconds    firstRankDelay)
{
    std::vector<std::string> peers;
    for (uint32_t index = 0; index < peerCount; ++index) { peers.push_back(fmt::format("localhost:{}", BasePort + index)); }
    std::vector<std::unique_ptr<RankState>> ranks;
    for (uint32_t local = 0; local < ranksPerPeer; ++local)
    {
        ranks.push_back(MakeRank((peerIndex * ranksPerPeer) + local, peerCount * ranksPerPeer, ranksPerPeer, peers));
    }
    std::vector<std::future<void>> saves;
    for (uint32_t local = 0; local < ranksPerPeer; ++local)
    {
        saves.push_back(std::async(std::launch::async, [&, local] {
            if (local == 0) { std::this_thread::sleep_for(firstRankDelay); }
            SaveRank(*ranks[local], directory);
        }));
    }
    for (auto& save : saves) { save.get(); }
    WaitForCkptFinalize(0, 1);
    Emit("SAVED");
    std::this_thread::sleep_for(hold);
    Emit("RELEASING");
    ranks.clear();
    Emit("EXITING");
    return EXIT_SUCCESS;
}

int RunSequentialRoots(std::filesystem::path const& directory)
{
    std::vector<std::string> const peers{fmt::format("localhost:{}", BasePort)};
    for (int round = 1; round <= 2; ++round)
    {
        auto rank = MakeRank(0, 1, 1, peers);
        SaveRank(*rank, directory / fmt::format("round{}", round));
        WaitForCkptFinalize(0, 1);
        rank.reset();
        // The next root needs a fresh daemon, so the previous one finishes its queued work and stops first.
        if (!WaitUntil([] { return !ServiceAnswers(BasePort); }, 30s)) { throw std::runtime_error("the daemon did not stop"); }
        Emit(fmt::format("ROUND{}", round));
    }
    return EXIT_SUCCESS;
}

int RunLeakedRoot(std::filesystem::path const& directory)
{
    // The leaked root keeps its reference on the hosted service through exit.
    auto* const rank = MakeRank(0, 1, 1, {fmt::format("localhost:{}", BasePort)}).release();
    SaveRank(*rank, directory);
    WaitForCkptFinalize(0, 1);
    Emit("LEAKED");
    return EXIT_SUCCESS;
}

std::map<std::string, std::string> Environment(std::filesystem::path const& directory)
{
    return {
        {"NATIVE_CKPT_SERVER_DISABLE_INPROC", "0"},
        {"NATIVE_CKPT_SERVER_REFERENCE_SCAN_INTERVAL_SECONDS", "1"},
        {"NATIVE_CKPT_SERVER_LOG", (directory / "server.%p.log").string()},
    };
}

// Serves references for one test and closes the service when the test ends, so the next test starts afresh.
struct Service
{
    Service(SvcRef::Clock::duration startupGrace, SvcRef::Clock::duration scanInterval)
    { SvcRef::BeginService(startupGrace, scanInterval); }
    ~Service()
    {
        std::atomic<bool> const stop{true};
        SvcRef::WaitUntilIdle(stop);
    }
    Service(Service const&)            = delete;
    Service& operator=(Service const&) = delete;
    Service(Service&&)                 = delete;
    Service& operator=(Service&&)      = delete;
};
}    // namespace

int RunSvcRefChild(int argc, char* argv[])
try
{
    std::vector<std::string_view> const args(argv + 2, argv + argc);
    if (args.size() == 1 && args[0] == "server")
    {
        RunServer(0);
        return EXIT_SUCCESS;
    }
    if (args.size() == 2 && args[0] == "holder")
    {
        auto const reference = SvcRef::CliAcquireLocalProc(static_cast<uint16_t>(std::stoul(std::string{args[1]})));
        Emit(!reference ? "UNAVAILABLE" : reference->token != 0 ? "HOLDING" : "UNTRACKED");
        while (true) { std::this_thread::sleep_for(1h); }
    }
    if (args.size() == 7 && args[0] == "peer")
    {
        return RunPeer(static_cast<uint32_t>(std::stoul(std::string{args[1]})),
                       static_cast<uint32_t>(std::stoul(std::string{args[2]})),
                       static_cast<uint32_t>(std::stoul(std::string{args[3]})),
                       std::filesystem::path{args[4]},
                       std::chrono::milliseconds{std::stol(std::string{args[5]})},
                       std::chrono::milliseconds{std::stol(std::string{args[6]})});
    }
    if (args.size() == 2 && args[0] == "sequential") { return RunSequentialRoots(std::filesystem::path{args[1]}); }
    if (args.size() == 2 && args[0] == "leaked") { return RunLeakedRoot(std::filesystem::path{args[1]}); }
    fmt::print(stderr, "unknown service reference child role\n");
    return EXIT_FAILURE;
} catch (std::exception const& error)
{
    fmt::print(stderr, "service reference child failed: {}\n", error.what());
    return EXIT_FAILURE;
}

TEST_CASE("an in-process reference holds the service until it is released", "[reference][tier1]")
{
    Service const     service{0s, 50ms};
    auto              held = SvcRef::AcquireInProc();
    std::atomic<bool> stop{false};
    auto              drained = std::async(std::launch::async, [&] { SvcRef::WaitUntilIdle(stop); });
    CHECK(drained.wait_for(300ms) == std::future_status::timeout);
    CHECK(SvcRef::HasInProc());
    held.reset();
    REQUIRE(drained.wait_for(5s) == std::future_status::ready);
    CHECK(SvcRef::Closing());
    CHECK_FALSE(SvcRef::HasInProc());
}

TEST_CASE("a process reference counts until its process exits", "[reference][tier1]")
{
    Service const service{0s, 50ms};
    pid_t const   child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0)
    {
        ::pause();
        ::_exit(0);
    }
    REQUIRE(SvcRef::SvrAcquireLocalProcRef(static_cast<uint64_t>(child)) != 0);
    std::atomic<bool> stop{false};
    auto              drained = std::async(std::launch::async, [&] { SvcRef::WaitUntilIdle(stop); });
    CHECK(drained.wait_for(300ms) == std::future_status::timeout);
    ::kill(child, SIGKILL);
    ::waitpid(child, nullptr, 0);
    REQUIRE(drained.wait_for(5s) == std::future_status::ready);
}

TEST_CASE("a released process reference no longer counts and the release is idempotent", "[reference][tier1]")
{
    Service const service{0s, 50ms};
    auto const    pid   = SvcRef::CurrentPid();
    auto const    token = SvcRef::SvrAcquireLocalProcRef(pid);
    REQUIRE(token != 0);
    std::atomic<bool> stop{false};
    auto              drained = std::async(std::launch::async, [&] { SvcRef::WaitUntilIdle(stop); });
    CHECK(drained.wait_for(300ms) == std::future_status::timeout);
    SvcRef::SvrReleaseLocalProcRef(pid, token + 1);
    CHECK(drained.wait_for(300ms) == std::future_status::timeout);
    SvcRef::SvrReleaseLocalProcRef(pid, token);
    SvcRef::SvrReleaseLocalProcRef(pid, token);
    REQUIRE(drained.wait_for(5s) == std::future_status::ready);
}

TEST_CASE("an exiting host drops every reference its pid holds", "[reference][tier1]")
{
    Service const service{0s, 50ms};
    auto const    pid = SvcRef::CurrentPid();
    REQUIRE(SvcRef::SvrAcquireLocalProcRef(pid) != 0);
    REQUIRE(SvcRef::SvrAcquireLocalProcRef(pid) != 0);
    std::atomic<bool> stop{false};
    auto              drained = std::async(std::launch::async, [&] { SvcRef::WaitUntilIdle(stop); });
    CHECK(drained.wait_for(300ms) == std::future_status::timeout);
    SvcRef::SvrReleaseLocalProcRefs(pid);
    REQUIRE(drained.wait_for(5s) == std::future_status::ready);
}

TEST_CASE("a peer reference counts only while the peer answers pings", "[reference][tier1]")
{
    Service const service{0s, 50ms};
    auto          peer = std::make_unique<FakePeer>([](std::string const& /* target */) { return http::status::ok; });
    REQUIRE(SvcRef::SvrAcquireRemoteProcRef(peer->Address()) != 0);
    std::atomic<bool> stop{false};
    auto              drained = std::async(std::launch::async, [&] { SvcRef::WaitUntilIdle(stop); });
    CHECK(drained.wait_for(500ms) == std::future_status::timeout);
    auto const pings = peer->Requests();
    REQUIRE_FALSE(pings.empty());
    CHECK(pings.front().starts_with("/api/peer/ping "));
    peer.reset();
    REQUIRE(drained.wait_for(10s) == std::future_status::ready);
}

TEST_CASE("a closing service refuses new references", "[reference][tier1]")
{
    Service const     service{0s, 50ms};
    std::atomic<bool> stop{true};
    SvcRef::WaitUntilIdle(stop);
    CHECK(SvcRef::Closing());
    CHECK(SvcRef::SvrAcquireLocalProcRef(SvcRef::CurrentPid()) == 0);
    CHECK(SvcRef::SvrAcquireRemoteProcRef("peer:4101") == 0);
    SvcRef::BeginService(0s, 50ms);
    CHECK_FALSE(SvcRef::Closing());
    auto const token = SvcRef::SvrAcquireLocalProcRef(SvcRef::CurrentPid());
    CHECK(token != 0);
    SvcRef::SvrReleaseLocalProcRef(SvcRef::CurrentPid(), token);
}

TEST_CASE("draining waits out the startup grace and then ends without references", "[reference][tier1]")
{
    Service const     service{500ms, 50ms};
    std::atomic<bool> stop{false};
    auto const        started = std::chrono::steady_clock::now();
    SvcRef::WaitUntilIdle(stop);
    CHECK(std::chrono::steady_clock::now() - started >= 450ms);
}

TEST_CASE("a stop request ends the drain wait", "[reference][tier1]")
{
    Service const     service{0s, 1h};
    auto const        held = SvcRef::AcquireInProc();
    std::atomic<bool> stop{false};
    auto              drained = std::async(std::launch::async, [&] { SvcRef::WaitUntilIdle(stop); });
    CHECK(drained.wait_for(200ms) == std::future_status::timeout);
    stop = true;
    SvcRef::Wake();
    REQUIRE(drained.wait_for(5s) == std::future_status::ready);
    CHECK(SvcRef::Closing());
}

TEST_CASE("a coordinator reference waits for the coordinator and lasts until its last handle is gone", "[reference][tier1]")
{
    Service const    service{0s, 50ms};
    std::atomic<int> refusals{2};
    FakePeer         coordinator([&](std::string const& target) {
        if (target == "/api/peer/acquireref" && refusals.fetch_sub(1) > 0) { return http::status::service_unavailable; }
        return http::status::ok;
    });
    SvcRef::RecordSelfAddress("peer:4101");
    auto held = SvcRef::CliAcquireRemoteProc(coordinator.Address());
    REQUIRE(held != nullptr);
    CHECK(held->token == 7);
    auto requests = coordinator.Requests();
    REQUIRE(requests.size() == 3);
    CHECK(requests.back() == "/api/peer/acquireref peer:4101");
    auto copy = held;
    held.reset();
    CHECK(coordinator.Requests().size() == 3);
    copy.reset();
    requests = coordinator.Requests();
    REQUIRE(requests.size() == 4);
    CHECK(requests.back() == "/api/peer/releaseref/7 peer:4101");

    CHECK(SvcRef::CliAcquireRemoteProc("peer:4101") == nullptr);
    FakePeer untracked([](std::string const& /* target */) { return http::status::not_found; });
    auto     unheld = SvcRef::CliAcquireRemoteProc(untracked.Address());
    REQUIRE(unheld != nullptr);
    CHECK(unheld->token == 0);
    unheld.reset();
    CHECK(untracked.Requests().size() == 1);
}

TEST_CASE("the process probe treats a killed child as exited before it is reaped", "[reference][tier1]")
{
    CHECK(SvcRef::ProcessAlive(SvcRef::CurrentPid()));
    CHECK_FALSE(SvcRef::ProcessAlive(0));
    pid_t const child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0)
    {
        ::pause();
        ::_exit(0);
    }
    CHECK(SvcRef::ProcessAlive(static_cast<uint64_t>(child)));
    ::kill(child, SIGKILL);
    CHECK(WaitUntil([&] { return !SvcRef::ProcessAlive(static_cast<uint64_t>(child)); }, 5s));
    int status = 0;
    ::waitpid(child, &status, 0);
    CHECK_FALSE(SvcRef::ProcessAlive(static_cast<uint64_t>(child)));
}

TEST_CASE("a standalone daemon waits out its startup grace and stops with its last reference", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort)) { SKIP("a checkpoint service already listens on the test port"); }
    auto const directory = TestDirectory("reference_standalone_lifetime");
    {
        ChildProcess unreferenced({"server"}, Environment(directory), directory / "unreferenced.stderr");
        REQUIRE(unreferenced.WaitForLine("READY", 30s));
        auto const ready  = std::chrono::steady_clock::now();
        auto const status = unreferenced.WaitForExit(15s);
        REQUIRE(status.has_value());
        CHECK(*status == EXIT_SUCCESS);
        CHECK(std::chrono::steady_clock::now() - ready >= 3s);
    }
    ChildProcess server({"server"}, Environment(directory), directory / "server.stderr");
    REQUIRE(server.WaitForLine("READY", 30s));
    {
        auto const reference = SvcRef::CliAcquireLocalProc(0);
        REQUIRE(reference != nullptr);
        REQUIRE(reference->token != 0);
        std::this_thread::sleep_for(6s);
        CHECK(server.Running());
    }
    auto const released = std::chrono::steady_clock::now();
    auto const status   = server.WaitForExit(15s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::chrono::steady_clock::now() - released < 3s);
}

TEST_CASE("a standalone daemon drops the reference of a killed client", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort)) { SKIP("a checkpoint service already listens on the test port"); }
    auto const   directory = TestDirectory("reference_killed_client");
    ChildProcess server({"server"}, Environment(directory), directory / "server.stderr");
    REQUIRE(server.WaitForLine("READY", 30s));
    ChildProcess holder({"holder", "0"}, {}, directory / "holder.stderr");
    REQUIRE(holder.WaitForLine("HOLDING", 30s));
    std::this_thread::sleep_for(6s);
    CHECK(server.Running());
    holder.Kill();
    REQUIRE(holder.WaitForExit(10s).has_value());
    auto const killed = std::chrono::steady_clock::now();
    auto const status = server.WaitForExit(15s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::chrono::steady_clock::now() - killed < 5s);
}

TEST_CASE("an in-process coordinator outlives the peers that reference it", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort) || ServiceAnswers(BasePort + 1)) { SKIP("a checkpoint service already listens on a test port"); }
    auto const   directory   = TestDirectory("reference_inproc_hosts");
    auto const   environment = Environment(directory);
    ChildProcess peer({"peer", "1", "2", "2", directory.string(), "3000", "1000"}, environment, directory / "peer1.stderr");
    ChildProcess coordinator({"peer", "0", "2", "2", directory.string(), "0", "0"}, environment, directory / "peer0.stderr");
    REQUIRE(coordinator.WaitForLine("SAVED", 60s));
    REQUIRE(coordinator.WaitForLine("EXITING", 10s));
    REQUIRE(peer.WaitForLine("SAVED", 60s));
    std::this_thread::sleep_for(1500ms);
    CHECK(coordinator.Running());
    REQUIRE(peer.WaitForLine("RELEASING", 10s));
    auto const coordinatorStatus = coordinator.WaitForExit(30s);
    auto const peerStatus        = peer.WaitForExit(30s);
    REQUIRE(coordinatorStatus.has_value());
    REQUIRE(peerStatus.has_value());
    CHECK(*coordinatorStatus == EXIT_SUCCESS);
    CHECK(*peerStatus == EXIT_SUCCESS);
    CHECK(std::filesystem::exists(directory / ".metadata"));
    for (uint32_t rank = 0; rank < 4; ++rank) { CHECK(std::filesystem::exists(directory / fmt::format("__{}_0.distcp", rank))); }
    CHECK_FALSE(std::filesystem::exists("/dev/shm/checkpoint_request_0_0"));
    CHECK_FALSE(std::filesystem::exists("/dev/shm/checkpoint_request_1_0"));
}

TEST_CASE("an in-process coordinator stops waiting on a killed peer", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort) || ServiceAnswers(BasePort + 1)) { SKIP("a checkpoint service already listens on a test port"); }
    auto const   directory   = TestDirectory("reference_inproc_killed_peer");
    auto const   environment = Environment(directory);
    ChildProcess peer({"peer", "1", "2", "1", directory.string(), "600000", "0"}, environment, directory / "peer1.stderr");
    ChildProcess coordinator({"peer", "0", "2", "1", directory.string(), "0", "0"}, environment, directory / "peer0.stderr");
    REQUIRE(coordinator.WaitForLine("EXITING", 60s));
    REQUIRE(peer.WaitForLine("SAVED", 60s));
    std::this_thread::sleep_for(2s);
    CHECK(coordinator.Running());
    peer.Kill();
    REQUIRE(peer.WaitForExit(10s).has_value());
    auto const killed = std::chrono::steady_clock::now();
    auto const status = coordinator.WaitForExit(30s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::chrono::steady_clock::now() - killed < 10s);
    for (int queue = 0; queue < 8; ++queue)
    {
        boost::interprocess::shared_memory_object::remove(fmt::format("checkpoint_request_1_{}", queue).c_str());
    }
}

TEST_CASE("sequential roots in one process register again with a fresh in-process daemon", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort)) { SKIP("a checkpoint service already listens on the test port"); }
    auto const   directory = TestDirectory("reference_sequential_roots");
    ChildProcess child({"sequential", directory.string()}, Environment(directory), directory / "child.stderr");
    REQUIRE(child.WaitForLine("ROUND1", 60s));
    REQUIRE(child.WaitForLine("ROUND2", 60s));
    auto const status = child.WaitForExit(30s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::filesystem::exists(directory / "round1" / ".metadata"));
    CHECK(std::filesystem::exists(directory / "round2" / ".metadata"));
}

TEST_CASE("an in-process host exits promptly while it still holds its own reference", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort)) { SKIP("a checkpoint service already listens on the test port"); }
    auto const   directory = TestDirectory("reference_leaked_root");
    ChildProcess child({"leaked", directory.string()}, Environment(directory), directory / "child.stderr");
    REQUIRE(child.WaitForLine("LEAKED", 60s));
    auto const status = child.WaitForExit(30s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::filesystem::exists(directory / ".metadata"));
}

TEST_CASE("in-process hosts with telemetry enabled exit cleanly", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort) || ServiceAnswers(BasePort + 1)) { SKIP("a checkpoint service already listens on a test port"); }
    auto const directory   = TestDirectory("reference_inproc_telemetry");
    auto       environment = Environment(directory);
    environment.emplace("NATIVE_CKPT_OTEL_FILE", (directory / "otel.jsonl").string());
    ChildProcess peer({"peer", "1", "2", "1", directory.string(), "1000", "0"}, environment, directory / "peer1.stderr");
    ChildProcess coordinator({"peer", "0", "2", "1", directory.string(), "0", "0"}, environment, directory / "peer0.stderr");
    auto const   coordinatorStatus = coordinator.WaitForExit(60s);
    auto const   peerStatus        = peer.WaitForExit(60s);
    REQUIRE(coordinatorStatus.has_value());
    REQUIRE(peerStatus.has_value());
    CHECK(*coordinatorStatus == EXIT_SUCCESS);
    CHECK(*peerStatus == EXIT_SUCCESS);
    CHECK(std::filesystem::exists(directory / "otel.jsonl"));
}

// NOLINTEND(readability-magic-numbers)
