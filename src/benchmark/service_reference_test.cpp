#include "service_reference_test.h"

#include "checkpoint.h"
#include "messages.h"
#include "server.h"
#include "service_reference.h"
#include "state.h"
#include "tensor.h"

#include <boost/beast/http/status.hpp>
#include <boost/beast/http/verb.hpp>
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
namespace service = checkpoint::service;
namespace http    = boost::beast::http;
using checkpoint::ipc::SessionId;

namespace
{
constexpr uint16_t BasePort = checkpoint::ipc::Message::Port;

class FakeProbes
{
    public:
    void SetProcess(int64_t pid, bool alive)
    {
        std::scoped_lock lock(_mutex);
        _processes[pid] = alive;
    }

    void SetPeer(std::string const& address, bool alive)
    {
        std::scoped_lock lock(_mutex);
        _peers[address] = alive;
    }

    size_t ProcessProbesFromThisThread()
    {
        std::scoped_lock lock(_mutex);
        return _processProbes[std::this_thread::get_id()];
    }

    size_t PeerProbesFromThisThread()
    {
        std::scoped_lock lock(_mutex);
        return _peerProbes[std::this_thread::get_id()];
    }

    std::vector<std::string> Notifications()
    {
        std::scoped_lock lock(_mutex);
        return _notifications;
    }

    service::Probes Make()
    {
        return {
            .processAlive =
                [this](int64_t pid) {
                    std::scoped_lock lock(_mutex);
                    ++_processProbes[std::this_thread::get_id()];
                    auto const it = _processes.find(pid);
                    return it != _processes.end() && it->second;
                },
            .peerAlive =
                [this](std::string const& address) {
                    std::scoped_lock lock(_mutex);
                    ++_peerProbes[std::this_thread::get_id()];
                    auto const it = _peers.find(address);
                    return it != _peers.end() && it->second;
                },
            .notify =
                [this](http::verb method, std::string const& address, std::string const& target, std::string const& body) {
                    std::scoped_lock lock(_mutex);
                    _notifications.push_back(fmt::format("{} {}{} {}", std::string{http::to_string(method)}, address, target, body));
                    return true;
                },
        };
    }

    private:
    std::mutex                        _mutex;
    std::map<int64_t, bool>           _processes;
    std::map<std::string, bool>       _peers;
    std::map<std::thread::id, size_t> _processProbes;
    std::map<std::thread::id, size_t> _peerProbes;
    std::vector<std::string>          _notifications;
};

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
{
    try
    {
        service::WaitForService(port, 0s);
        return true;
    } catch (std::runtime_error const&) { return false; }
}

std::filesystem::path TestDirectory(std::string_view stem)
{
    auto const directory = std::filesystem::temp_directory_path() / "phoenix_ckpttest" / stem;
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

// A ckpttest child in one of the service reference roles, with stdout piped back and stderr in a file.
class ChildProcess
{
    public:
    ChildProcess(std::vector<std::string> const&           args,
                 std::map<std::string, std::string> const& environment,
                 std::filesystem::path const&              errorLog)
    {
        auto const               executable = std::filesystem::read_symlink("/proc/self/exe").string();
        std::vector<std::string> argStorage{executable, std::string{ServiceReferenceChildArg}};
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

int RunPeer(uint32_t                     peerIndex,
            uint32_t                     peerCount,
            uint32_t                     ranksPerPeer,
            std::filesystem::path const& directory,
            std::chrono::milliseconds    hold,
            std::chrono::milliseconds    hostDelay)
{
    std::vector<std::string> peers;
    for (uint32_t index = 0; index < peerCount; ++index) { peers.push_back(fmt::format("localhost:{}", BasePort + index)); }
    auto const                              worldSize = peerCount * ranksPerPeer;
    std::vector<std::array<float, 4>>       values(ranksPerPeer);
    std::vector<std::unique_ptr<StateTree>> trees;
    for (uint32_t local = 0; local < ranksPerPeer; ++local)
    {
        auto const rank = (peerIndex * ranksPerPeer) + local;
        values[local].fill(static_cast<float>(rank));
        trees.push_back(std::make_unique<StateTree>(SessionId::Default,
                                                    RankTopology{
                                                        .rank         = Rank{rank},
                                                        .worldSize    = Rank{worldSize},
                                                        .ranksPerPeer = ranksPerPeer,
                                                        .peersPerNode = peerCount,
                                                        .peers        = peers,
                                                    }));
        Tensor::Dim fulldim;
        fulldim.push_back(size_t{4} * worldSize);
        DTensor          tensor(Tensor::DType::Float32, fulldim);
        Tensor::DimRange sharddim;
        sharddim.push_back({.begin = size_t{4} * rank, .end = size_t{4} * (rank + 1)});
        tensor.AddShard({
            .rank          = Rank{rank},
            .snapshotDtype = Tensor::DType::Float32,
            .handle        = reinterpret_cast<uint64_t>(values[local].data()),    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            .storageSize   = sizeof(values[local]),
            .sharddim      = sharddim,
            .device        = DeviceId{"cpu"},
        });
        trees.back()->AddChild(StateNode::Handle::Root, "weight").data = std::move(tensor);
    }

    std::vector<std::future<void>> ranks;
    for (uint32_t local = 0; local < ranksPerPeer; ++local)
    {
        ranks.push_back(std::async(std::launch::async, [&, local] {
            if (local == 0) { std::this_thread::sleep_for(hostDelay); }
            FinalizeStateRegistry(*trees[local]);
            auto  snapshotRegistry = GetSnapshotRegistry(*trees[local]);
            auto& shards           = std::get<DTensor>(snapshotRegistry->GetNode(snapshotRegistry->FindNodeByFQN("weight")).data).shards;
            if (shards.size() != 1) { throw std::runtime_error("the snapshot plan should keep each rank's own slice"); }
            auto source = shards.extract(shards.begin());
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            source.value().handle = reinterpret_cast<uint64_t>(values[local].data());
            shards.insert(std::move(source));
            SaveCheckpoint(*snapshotRegistry, checkpoint::ipc::SnapshotId{1}, directory);
        }));
    }
    for (auto& rank : ranks) { rank.get(); }
    WaitForCkptFinalize(0, 1);
    Emit("SAVED");
    std::this_thread::sleep_for(hold);
    Emit("RELEASING");
    trees.clear();
    Emit("EXITING");
    return EXIT_SUCCESS;
}

std::map<std::string, std::string> InprocEnvironment(std::filesystem::path const& directory)
{
    return {
        {"NATIVE_CKPT_SERVER_INPROC", "1"},
        {"NATIVE_CKPT_SERVER_REFERENCE_SCAN_INTERVAL_SECONDS", "1"},
        {"NATIVE_CKPT_SERVER_LOG", (directory / "server.log").string()},
    };
}

std::map<std::string, std::string> ServerEnvironment(std::filesystem::path const& directory)
{
    return {
        {"NATIVE_CKPT_SERVER_AUTO_SHUTDOWN_TIMEOUT_SECONDS", "2"},
        {"NATIVE_CKPT_SERVER_REFERENCE_SCAN_INTERVAL_SECONDS", "1"},
        {"NATIVE_CKPT_SERVER_LOG", (directory / "server.log").string()},
    };
}
}    // namespace

int RunServiceReferenceChild(int argc, char* argv[])
try
{
    std::vector<std::string_view> const args(argv + 2, argv + argc);
    if (args.size() == 1 && args[0] == "server")
    {
        RunServer();
        return EXIT_SUCCESS;
    }
    if (args.size() == 2 && args[0] == "holder")
    {
        auto const reference = service::AcquireClientReference(static_cast<uint16_t>(std::stoul(std::string{args[1]})), SessionId::Default);
        Emit(reference != nullptr ? "HOLDING" : "UNTRACKED");
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
    fmt::print(stderr, "unknown service reference child role\n");
    return EXIT_FAILURE;
} catch (std::exception const& error)
{
    fmt::print(stderr, "service reference child failed: {}\n", error.what());
    return EXIT_FAILURE;
}

TEST_CASE("self references count holders per context and release on scope exit", "[reference][tier1]")
{
    FakeProbes        probes;
    service::Registry registry(1h, probes.Make());
    CHECK_FALSE(registry.HasValidReference());
    {
        auto       first  = registry.AcquireSelf("request");
        auto const second = registry.AcquireSelf("request");
        CHECK(registry.HasValidReference(service::Kinds::Self));
        CHECK(registry.Census().contains("self:requestx2"));
        {
            auto const moved = std::move(first);
        }
        CHECK(registry.Census().contains("self:requestx1"));
    }
    CHECK_FALSE(registry.HasValidReference());
    CHECK_THROWS_AS(([&registry] {
                        auto const held = registry.AcquireSelf("request");
                        throw std::runtime_error("handler failed");
                    }()),
                    std::runtime_error);
    CHECK_FALSE(registry.HasValidReference());
}

TEST_CASE("pid references are counted per pid and session", "[reference][tier1]")
{
    FakeProbes probes;
    probes.SetProcess(100, true);
    service::Registry registry(1h, probes.Make());
    registry.AcquirePid(100, SessionId::Default);
    registry.AcquirePid(100, SessionId::Default);
    registry.AcquirePid(100, SessionId{7});
    registry.ReleasePid(100, SessionId::Default);
    CHECK(registry.Census().contains("pid:100@0x1"));
    registry.ReleasePid(100, SessionId::Default);
    registry.ReleasePid(100, SessionId::Default);
    CHECK(registry.HasValidReference(service::Kinds::Pid));
    registry.ReleasePid(100, SessionId{7});
    CHECK_FALSE(registry.HasValidReference(service::Kinds::Pid));
    registry.AcquirePid(100, SessionId::Default);
    registry.AcquirePid(100, SessionId{7});
    registry.ForgetPid(100);
    CHECK_FALSE(registry.HasValidReference());
}

TEST_CASE("validity walks stop at the first valid reference in self, pid, remote order", "[reference][tier1]")
{
    FakeProbes probes;
    probes.SetProcess(100, true);
    probes.SetPeer("peer:1", true);
    service::Registry registry(1h, probes.Make());
    registry.AcquirePid(100, SessionId::Default);
    registry.AcquireRemote("peer:1", SessionId::Default);
    {
        auto const held = registry.AcquireSelf("request");
        CHECK(registry.HasValidReference());
        CHECK(probes.ProcessProbesFromThisThread() == 0);
    }
    CHECK(registry.HasValidReference());
    CHECK(probes.ProcessProbesFromThisThread() == 1);
    CHECK(probes.PeerProbesFromThisThread() == 0);
    probes.SetProcess(100, false);
    CHECK(registry.HasValidReference());
    CHECK(probes.PeerProbesFromThisThread() == 0);
    CHECK_FALSE(registry.HasValidReference(service::Kinds::Pid));
}

TEST_CASE("exited processes lose their pid references", "[reference][tier1]")
{
    FakeProbes probes;
    probes.SetProcess(100, false);
    probes.SetProcess(200, true);
    service::Registry registry(1h, probes.Make());
    registry.AcquirePid(100, SessionId::Default);
    registry.AcquirePid(200, SessionId::Default);
    CHECK(registry.HasValidReference(service::Kinds::Pid));
    CHECK_FALSE(registry.Census().contains("pid:100"));
    probes.SetProcess(200, false);
    CHECK_FALSE(registry.HasValidReference(service::Kinds::Pid));
    CHECK(registry.Census() == "none");
}

TEST_CASE("remote references survive two failed pings and drop on the third", "[reference][tier1]")
{
    constexpr auto Interval = 100ms;
    FakeProbes     probes;
    probes.SetPeer("peer:1", false);
    probes.SetPeer("peer:2", false);
    service::Registry registry(Interval, probes.Make());

    auto const acquired = std::chrono::steady_clock::now();
    registry.AcquireRemote("peer:1", SessionId::Default);
    CHECK(registry.HasValidReference(service::Kinds::Remote));
    REQUIRE(WaitUntil([&] { return !registry.Census().contains("peer:peer:1"); }, 5s));
    CHECK(std::chrono::steady_clock::now() - acquired >= 3 * Interval);

    registry.AcquireRemote("peer:2", SessionId::Default);
    std::this_thread::sleep_for(Interval * 5 / 2);
    probes.SetPeer("peer:2", true);
    std::this_thread::sleep_for(Interval * 2);
    CHECK(registry.Census().contains("peer:peer:2@0x1 failures=0"));
    probes.SetPeer("peer:2", false);
    auto const failing = std::chrono::steady_clock::now();
    REQUIRE(WaitUntil([&] { return !registry.HasValidReference(service::Kinds::Remote); }, 5s));
    CHECK(std::chrono::steady_clock::now() - failing >= 2 * Interval);
}

TEST_CASE("session links hold the coordinator while the session has references", "[reference][tier1]")
{
    constexpr auto Interval = 50ms;
    FakeProbes     probes;
    probes.SetProcess(100, true);
    service::Registry registry(Interval, probes.Make());
    std::atomic<bool> busy{false};
    std::string const acquire = "POST coordinator:4100/api/reference/peer/0 peer:4101";
    std::string const release = "DELETE coordinator:4100/api/reference/peer/0 peer:4101";
    {
        auto const link = registry.AttachSession(
            SessionId::Default, [&busy] { return busy ? std::string{"busy"} : std::string{}; }, "coordinator:4100", "peer:4101");
        registry.AcquirePid(100, SessionId{9});
        std::this_thread::sleep_for(4 * Interval);
        CHECK(probes.Notifications().empty());

        registry.AcquirePid(100, SessionId::Default);
        REQUIRE(WaitUntil([&] { return probes.Notifications().size() == 1; }, 5s));
        CHECK(probes.Notifications().back() == acquire);
        registry.ReleasePid(100, SessionId::Default);
        REQUIRE(WaitUntil([&] { return probes.Notifications().size() == 2; }, 5s));
        CHECK(probes.Notifications().back() == release);

        busy = true;
        REQUIRE(WaitUntil([&] { return probes.Notifications().size() == 3; }, 5s));
        CHECK(probes.Notifications().back() == acquire);
    }
    auto const notifications = probes.Notifications();
    REQUIRE(notifications.size() == 4);
    CHECK(notifications.back() == release);
}

TEST_CASE("draining waits wake on release and give up at their deadline", "[reference][tier1]")
{
    FakeProbes                            probes;
    service::Registry                     registry(1h, probes.Make());
    std::optional<service::SelfReference> held = registry.AcquireSelf("request");
    auto drained = std::async(std::launch::async, [&registry] { return registry.WaitUntilDrained(service::Clock::now() + 10s); });
    CHECK(drained.wait_for(200ms) == std::future_status::timeout);
    held.reset();
    REQUIRE(drained.wait_for(5s) == std::future_status::ready);
    CHECK(drained.get());

    auto const stuck   = registry.AcquireSelf("request");
    auto const started = service::Clock::now();
    CHECK_FALSE(registry.WaitUntilDrained(started + 200ms));
    CHECK(service::Clock::now() - started >= 200ms);
}

TEST_CASE("drain timing arms with the first client reference", "[reference][tier1]")
{
    FakeProbes probes;
    probes.SetProcess(100, true);
    service::Registry registry(20ms, probes.Make());
    std::this_thread::sleep_for(100ms);
    CHECK(registry.DrainedFor() == service::Clock::duration::zero());
    registry.AcquirePid(100, SessionId::Default);
    std::this_thread::sleep_for(100ms);
    CHECK(registry.DrainedFor() == service::Clock::duration::zero());
    registry.ReleasePid(100, SessionId::Default);
    REQUIRE(WaitUntil([&] { return registry.DrainedFor() > 50ms; }, 5s));
    registry.AcquirePid(100, SessionId::Default);
    CHECK(registry.DrainedFor() == service::Clock::duration::zero());
}

TEST_CASE("a session callback in flight finishes before its link detaches", "[reference][tier1]")
{
    FakeProbes         probes;
    service::Registry  registry(1h, probes.Make());
    std::atomic<int>   calls{0};
    std::atomic<bool>  returned{false};
    std::promise<void> entered;
    std::promise<void> proceed;
    auto               proceeding = proceed.get_future().share();

    std::optional<service::SessionReferences> link = registry.AttachSession(SessionId::Default, [&] {
        if (calls.fetch_add(1) == 0)
        {
            entered.set_value();
            proceeding.wait();
            returned = true;
        }
        return std::string{};
    });
    auto walker = std::async(std::launch::async, [&registry] { return registry.HasValidReference(service::Kinds::Self); });
    REQUIRE(entered.get_future().wait_for(5s) == std::future_status::ready);
    auto detached = std::async(std::launch::async, [&link] { link.reset(); });
    CHECK(detached.wait_for(200ms) == std::future_status::timeout);
    proceed.set_value();
    REQUIRE(detached.wait_for(5s) == std::future_status::ready);
    CHECK(returned);
    CHECK_FALSE(walker.get());
}

TEST_CASE("reference requests acquire and release pid and peer references", "[reference][tier1]")
{
    FakeProbes probes;
    probes.SetProcess(100, true);
    service::Registry registry(1h, probes.Make());
    CHECK(registry.Handle(http::verb::post, "/api/reference/pid/100/0", "") == http::status::ok);
    CHECK(registry.HasValidReference(service::Kinds::Pid));
    CHECK(registry.Handle(http::verb::delete_, "/api/reference/pid/100/0", "") == http::status::ok);
    CHECK_FALSE(registry.HasValidReference(service::Kinds::Pid));
    CHECK(registry.Handle(http::verb::post, "/api/reference/peer/3", "node7:4101") == http::status::ok);
    CHECK(registry.Census().contains("peer:node7:4101@3x1"));
    CHECK(registry.Handle(http::verb::delete_, "/api/reference/peer/3", "node7:4101") == http::status::ok);
    CHECK(registry.Census() == "none");
    CHECK(registry.Handle(http::verb::post, "/api/reference/pid/x/0", "") == http::status::bad_request);
    CHECK(registry.Handle(http::verb::post, "/api/reference/pid/0/0", "") == http::status::bad_request);
    CHECK(registry.Handle(http::verb::post, "/api/reference/peer/0", "") == http::status::bad_request);
    CHECK(registry.Handle(http::verb::get, "/api/reference/pid/100/0", "") == http::status::method_not_allowed);
}

TEST_CASE("the process probe treats a killed child as exited before it is reaped", "[reference][tier1]")
{
    CHECK(service::ProcessAlive(service::CurrentPid()));
    CHECK_FALSE(service::ProcessAlive(0));
    pid_t const child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0)
    {
        ::pause();
        ::_exit(0);
    }
    CHECK(service::ProcessAlive(child));
    ::kill(child, SIGKILL);
    CHECK(WaitUntil([&] { return !service::ProcessAlive(child); }, 5s));
    int status = 0;
    ::waitpid(child, &status, 0);
    CHECK_FALSE(service::ProcessAlive(child));
}

TEST_CASE("an out-of-process daemon stops after its last reference is released", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort)) { SKIP("a checkpoint service already listens on the test port"); }
    auto const   directory = TestDirectory("reference_last_release");
    ChildProcess server({"server"}, ServerEnvironment(directory), directory / "server.stderr");
    REQUIRE(server.WaitForLine("READY", 30s));
    std::this_thread::sleep_for(3500ms);
    CHECK(server.Running());
    {
        auto const reference = service::AcquireClientReference(BasePort, SessionId::Default);
        REQUIRE(reference != nullptr);
        std::this_thread::sleep_for(3500ms);
        CHECK(server.Running());
    }
    auto const released = std::chrono::steady_clock::now();
    auto const status   = server.WaitForExit(15s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::chrono::steady_clock::now() - released >= 2s);
}

TEST_CASE("an out-of-process daemon reaps the reference of a killed client", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort)) { SKIP("a checkpoint service already listens on the test port"); }
    auto const   directory = TestDirectory("reference_killed_client");
    ChildProcess server({"server"}, ServerEnvironment(directory), directory / "server.stderr");
    REQUIRE(server.WaitForLine("READY", 30s));
    ChildProcess holder({"holder", std::to_string(BasePort)}, {}, directory / "holder.stderr");
    REQUIRE(holder.WaitForLine("HOLDING", 30s));
    std::this_thread::sleep_for(3500ms);
    CHECK(server.Running());
    holder.Kill();
    REQUIRE(holder.WaitForExit(10s).has_value());
    auto const killed = std::chrono::steady_clock::now();
    auto const status = server.WaitForExit(15s);
    REQUIRE(status.has_value());
    CHECK(*status == EXIT_SUCCESS);
    CHECK(std::chrono::steady_clock::now() - killed >= 2s);
}

TEST_CASE("an in-process coordinator outlives the peers that reference it", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort) || ServiceAnswers(BasePort + 1)) { SKIP("a checkpoint service already listens on a test port"); }
    auto const   directory   = TestDirectory("reference_inproc_hosts");
    auto const   environment = InprocEnvironment(directory);
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

TEST_CASE("an in-process coordinator reaps a killed peer", "[reference][tier2]")
{
    if (ServiceAnswers(BasePort) || ServiceAnswers(BasePort + 1)) { SKIP("a checkpoint service already listens on a test port"); }
    auto const   directory   = TestDirectory("reference_inproc_killed_peer");
    auto const   environment = InprocEnvironment(directory);
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
    CHECK(std::chrono::steady_clock::now() - killed < 20s);
    for (int queue = 0; queue < 8; ++queue)
    {
        boost::interprocess::shared_memory_object::remove(fmt::format("checkpoint_request_1_{}", queue).c_str());
    }
}

// NOLINTEND(readability-magic-numbers)
