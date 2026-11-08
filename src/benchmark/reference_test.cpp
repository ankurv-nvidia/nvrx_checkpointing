#include "CommonMacros.h"
#include "messages.h"
#include "reference_registry.h"
#include "stateserdes.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#ifdef __linux__
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#endif

namespace
{
class ChildProcess
{
    public:
    ChildProcess()
    {
        _pid = ::fork();
        if (_pid == 0)
        {
            while (true) { ::pause(); }
        }
        REQUIRE(_pid > 0);
    }

    ~ChildProcess()
    {
        if (_pid > 0)
        {
            ::kill(_pid, SIGKILL);
            ::waitpid(_pid, nullptr, 0);
        }
    }

    CLASS_DELETE_COPY_AND_MOVE(ChildProcess);

    [[nodiscard]] pid_t Pid() const { return _pid; }

    void MakeZombie() const
    {
        REQUIRE(::kill(_pid, SIGKILL) == 0);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
        while (std::chrono::steady_clock::now() < deadline)
        {
            siginfo_t information{};
            REQUIRE(::waitid(P_PID, static_cast<id_t>(_pid), &information, WEXITED | WNOHANG | WNOWAIT) == 0);
            if (information.si_pid == _pid) { return; }
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        FAIL("child process did not reach zombie state");
    }

    void Reap()
    {
        REQUIRE(::waitpid(_pid, nullptr, 0) == _pid);
        _pid = -1;
    }

    private:
    pid_t _pid{-1};
};

#ifdef __linux__
namespace http = boost::beast::http;
using Response = http::response<http::string_body>;

std::optional<Response> Request(uint16_t index, http::verb method, std::string const& target, std::string body = {})
{
    boost::asio::io_context          ioc;
    boost::beast::tcp_stream         stream{ioc};
    boost::beast::flat_buffer        buffer;
    http::request<http::string_body> request{method, target, 11};
    request.set(http::field::host, "127.0.0.1");
    request.body() = std::move(body);
    request.prepare_payload();
    Response                  response;
    boost::system::error_code result = boost::asio::error::operation_aborted;
    stream.expires_after(std::chrono::seconds{1});
    auto const port = static_cast<uint16_t>(checkpoint::ipc::Message::Port + index);
    stream.async_connect({boost::asio::ip::address_v4::loopback(), port}, [&](boost::system::error_code error) {
        result = error;
        if (error) { return; }
        http::async_write(stream, request, [&](boost::system::error_code writeError, size_t) {
            result = writeError;
            if (writeError) { return; }
            http::async_read(stream, buffer, response, [&](boost::system::error_code readError, size_t) { result = readError; });
        });
    });
    ioc.run();
    if (result) { return std::nullopt; }
    return response;
}

bool Ping(uint16_t index)
{
    auto const response = Request(index, http::verb::get, "/api/ping");
    return response && response->result() == http::status::ok;
}

class ManagedServerProcess
{
    public:
    ManagedServerProcess() = default;
    ~ManagedServerProcess()
    {
        if (_pid > 0)
        {
            ::kill(_pid, SIGKILL);
            ::waitpid(_pid, nullptr, 0);
        }
    }
    CLASS_DELETE_COPY_AND_MOVE(ManagedServerProcess);

    void Start(uint16_t index)
    {
        auto                 executable    = std::filesystem::read_symlink("/proc/self/exe").string();
        auto                 argument      = std::string{"--shmem-test-server"};
        auto                 indexArgument = std::to_string(index);
        std::array<char*, 4> arguments{executable.data(), argument.data(), indexArgument.data(), nullptr};
        _pid = ::fork();
        if (_pid == 0)
        {
            ::execv(executable.c_str(), arguments.data());
            ::_exit(EXIT_FAILURE);
        }
        REQUIRE(_pid > 0);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (Ping(index)) { return; }
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        FAIL("managed daemon did not become ready");
    }

    void WaitForCleanExit()
    {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
        while (std::chrono::steady_clock::now() < deadline)
        {
            int        status{};
            auto const result = ::waitpid(_pid, &status, WNOHANG);
            REQUIRE(result >= 0);
            if (result == _pid)
            {
                _pid = -1;
                CHECK(WIFEXITED(status));
                CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        FAIL("managed daemon did not shut down after its final reference");
    }

    private:
    pid_t _pid{-1};
};

std::string ActivationBody(uint32_t rank)
{
    RankTopology const topology{
        .rank = Rank{rank}, .worldSize = Rank{2}, .ranksPerPeer = 1, .peersPerNode = 2, .peers = {"127.0.0.1:4100", "127.0.0.1:4101"}};
    std::ostringstream body;
    serdes::WriteBlobWithSize<uint16_t>(body, "");
    StateSerDes::WriteTopology(body, topology);
    return body.str();
}

constexpr uint32_t TestSession = 7301;

void AcquireClient(uint16_t index, pid_t owner, uint32_t session = TestSession)
{
    auto const target   = "/api/reference/acquire/" + std::to_string(session) + "/client/" + std::to_string(owner);
    auto const response = Request(index, http::verb::post, target, ActivationBody(index));
    REQUIRE(response);
    INFO(response->body());
    REQUIRE(response->result() == http::status::ok);
}

void ReleaseClient(uint16_t index, uint32_t session = TestSession)
{
    auto const response = Request(index, http::verb::post, "/api/reference/release/" + std::to_string(session) + "/client");
    REQUIRE(response);
    INFO(response->body());
    REQUIRE(response->result() == http::status::ok);
}
#endif
}    // namespace

TEST_CASE("daemon references prioritize tasks and local processes over peers", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireRemote(0, "remote", "coordinator");
    references.AcquireLocal(1, "local", ::getpid());
    auto self = references.AcquireSelf(2);
    REQUIRE(self);
    size_t pings = 0;
    auto   ping  = [&](std::string const&) {
        ++pings;
        return true;
    };

    CHECK(references.HasValidReferences(ping));
    CHECK(pings == 0);
    self.reset();
    CHECK(references.HasValidReferences(ping));
    CHECK(pings == 0);
    references.ReleaseLocal(1, "local");
    CHECK(references.HasValidReferences(ping));
    CHECK(pings == 1);
    references.ReleaseRemote(0, "remote");
    CHECK_FALSE(references.HasValidReferences(ping));
    CHECK(references.TryRetire());
}

TEST_CASE("daemon reaping stops at the first live peer and removes failed peers", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireRemote(0, "a", "dead");
    references.AcquireRemote(0, "b", "first");
    references.AcquireRemote(0, "c", "second");
    std::vector<std::string> checked;
    auto const               ping = [&](std::string const& peer) {
        checked.push_back(peer);
        return peer != "dead";
    };

    CHECK(references.HasValidReferences(ping));
    CHECK(checked == std::vector<std::string>{"dead", "first"});
    checked.clear();
    references.ReleaseRemote(0, "b");
    CHECK(references.HasValidReferences(ping));
    CHECK(checked == std::vector<std::string>{"second"});
    references.ReleaseRemote(0, "c");
    CHECK_FALSE(references.HasValidReferences(ping));
    CHECK(references.TryRetire());
}

TEST_CASE("daemon self references defer dead process reaping", "[server][reference][tier1]")
{
    ChildProcess                  child;
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(0, "child", child.Pid());
    auto task = references.AcquireSelf(1);
    REQUIRE(task);
    child.MakeZombie();
    CHECK(references.HasValidReferences([](std::string const&) { return false; }));
    task.reset();
    CHECK_FALSE(references.TryRetire());
    CHECK_FALSE(references.HasValidReferences([](std::string const&) { return false; }));
    CHECK(references.TryRetire());
}

TEST_CASE("daemon references remain isolated by session", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(1, "same-token", ::getpid());
    references.AcquireRemote(2, "same-token", "peer");
    auto self  = references.AcquireSelf(3);
    auto other = references.AcquireSelf(4);
    REQUIRE(self);
    REQUIRE(other);
    size_t pings = 0;
    auto   ping  = [&](std::string const&) {
        ++pings;
        return true;
    };

    CHECK(references.HasValidReferences(ping, 1));
    CHECK(pings == 0);
    CHECK(references.HasValidReferences(ping, 2));
    CHECK(pings == 1);
    CHECK(references.HasValidReferences(ping, 3));
    CHECK_FALSE(references.HasValidReferences(ping, 5));
    references.ReleaseLocal(1, "same-token");
    CHECK_FALSE(references.HasValidReferences(ping, 1));
    CHECK(references.HasValidReferences(ping, 2));
    self.reset();
    CHECK_FALSE(references.HasValidReferences(ping, 3));
    CHECK(references.HasValidReferences(ping, 4));
    references.ReleaseRemote(2, "same-token");
    other.reset();
    CHECK_FALSE(references.HasValidReferences(ping));
}

TEST_CASE("daemon client references support retried acquisition and release", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(1, "local", ::getpid());
    CHECK_NOTHROW(references.AcquireLocal(1, "local", ::getpid()));
    references.AcquireRemote(1, "remote", "peer");
    CHECK_NOTHROW(references.AcquireRemote(1, "remote", "peer"));
    CHECK_THROWS_AS(references.AcquireRemote(1, "remote", "different-peer"), std::runtime_error);
    ChildProcess child;
    CHECK_THROWS_AS(references.AcquireLocal(1, "local", child.Pid()), std::runtime_error);
    references.ReleaseLocal(1, "local");
    CHECK_NOTHROW(references.ReleaseLocal(1, "local"));
    references.ReleaseRemote(1, "remote");
    CHECK_NOTHROW(references.ReleaseRemote(1, "remote"));
    CHECK_FALSE(references.HasValidReferences([](std::string const&) { return true; }));
    CHECK(references.TryRetire());
}

TEST_CASE("daemon process identity rejects stale incarnations and zombie owners", "[server][reference][tier1]")
{
    auto const current = checkpoint::ProcessIdentity::Read(::getpid());
    REQUIRE(current);
    CHECK(current->IsAlive());
    auto stale = *current;
    ++stale.started;
    CHECK_FALSE(stale.IsAlive());
    CHECK_FALSE(checkpoint::ProcessIdentity::Read(-1));

    ChildProcess child;
    auto const   identity = checkpoint::ProcessIdentity::Read(child.Pid());
    REQUIRE(identity);
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(1, "child", child.Pid());
    CHECK(references.HasValidReferences([](std::string const&) { return false; }));
    child.MakeZombie();
    CHECK_FALSE(identity->IsAlive());
    CHECK_FALSE(checkpoint::ProcessIdentity::Read(child.Pid()));
    CHECK_FALSE(references.HasValidReferences([](std::string const&) { return false; }));
    CHECK_THROWS_AS(references.AcquireLocal(1, "zombie", child.Pid()), std::runtime_error);
    CHECK(references.TryRetire());
    child.Reap();
    CHECK_FALSE(identity->IsAlive());
}

TEST_CASE("daemon startup grace ends after the first client reference", "[server][reference][tier1]")
{
    using Clock = checkpoint::ReferenceRegistry::Clock;
    checkpoint::ReferenceRegistry unused;
    CHECK_FALSE(unused.TryRetire());
    CHECK(unused.TryRetire(Clock::now() + std::chrono::seconds{6}));

    checkpoint::ReferenceRegistry local;
    local.AcquireLocal(1, "client", ::getpid());
    local.ReleaseLocal(1, "client");
    CHECK(local.TryRetire());

    checkpoint::ReferenceRegistry remote;
    remote.AcquireRemote(1, "node", "peer");
    remote.ReleaseRemote(1, "node");
    CHECK(remote.TryRetire());
}

TEST_CASE("daemon task leases block retirement until the final copy is released", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(1, "client", ::getpid());
    auto task     = references.AcquireSelf(1);
    auto retained = task;
    REQUIRE(task);
    references.ReleaseLocal(1, "client");
    task.reset();
    CHECK_FALSE(references.TryRetire());
    CHECK(references.HasValidReferences([](std::string const&) { return false; }, 1));
    retained.reset();
    CHECK(references.TryRetire());
    CHECK_FALSE(references.AcquireSelf(1));
    CHECK_THROWS_AS(references.AcquireLocal(1, "new-client", ::getpid()), std::runtime_error);
    CHECK_THROWS_AS(references.AcquireRemote(1, "new-node", "peer"), std::runtime_error);
}

TEST_CASE("daemon references reject delayed acquisition after release", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(1, "local", ::getpid());
    references.ReleaseLocal(1, "local");
    CHECK_THROWS_AS(references.AcquireLocal(1, "local", ::getpid()), std::runtime_error);
    references.AcquireRemote(1, "remote", "peer");
    references.ReleaseRemote(1, "remote");
    CHECK_THROWS_AS(references.AcquireRemote(1, "remote", "peer"), std::runtime_error);
    CHECK_FALSE(references.HasValidReferences([](std::string const&) { return true; }));
    CHECK(references.TryRetire());
}

TEST_CASE("daemon session closure serializes task admission", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireLocal(1, "client", ::getpid());
    CHECK_FALSE(references.TryCloseSession(1));
    references.ReleaseLocal(1, "client");
    auto request = references.AcquireSelf();
    REQUIRE(request);
    CHECK(references.HasUnassignedWork());
    CHECK_FALSE(references.TryCloseSession(1));
    request.reset();
    CHECK_FALSE(references.HasUnassignedWork());
    CHECK(references.TryCloseSession(1));
    CHECK_FALSE(references.AcquireSelf(1));
    references.AcquireLocal(1, "fresh-client", ::getpid());
    auto task = references.AcquireSelf(1);
    REQUIRE(task);
    references.ReleaseLocal(1, "fresh-client");
    CHECK_FALSE(references.TryCloseSession(1));
    task.reset();
    CHECK(references.TryCloseSession(1));
    CHECK(references.TryRetire());
    CHECK_FALSE(references.IsAccepting());
    CHECK_FALSE(references.AcquireSelf());
    auto control = references.AcquireSelf(checkpoint::ReferenceRegistry::NoSession, true);
    REQUIRE(control);
    CHECK_FALSE(references.TryRetire());
    control.reset();
    CHECK(references.TryRetire());
}

TEST_CASE("daemon peer validation respects a release during its ping", "[server][reference][tier1]")
{
    checkpoint::ReferenceRegistry references;
    references.AcquireRemote(1, "peer", "coordinator");
    CHECK_FALSE(references.HasValidReferences([&](std::string const&) {
        references.ReleaseRemote(1, "peer");
        return true;
    }));
    CHECK(references.TryRetire());
}

TEST_CASE("coordinator lifetime follows the peer session reference", "[server][reference][coordinator][tier2]")
{
#ifdef __linux__
    if (Ping(0) || Ping(1)) { SKIP("checkpoint service ports are already occupied"); }
    for (bool const ownerDies : {false, true})
    {
        DYNAMIC_SECTION("peer client exits abruptly=" << ownerDies)
        {
            ManagedServerProcess coordinator;
            coordinator.Start(0);
            AcquireClient(0, ::getpid());
            ManagedServerProcess peer;
            peer.Start(1);
            ChildProcess peerClient;
            AcquireClient(1, peerClient.Pid());
            AcquireClient(1, ::getpid(), TestSession + 1);
            ReleaseClient(0);
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
            REQUIRE(Ping(0));
            REQUIRE(Ping(1));
            auto const released = std::chrono::steady_clock::now();
            if (ownerDies)
            {
                peerClient.MakeZombie();
                peerClient.Reap();
            }
            else
            {
                ReleaseClient(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
            REQUIRE(Ping(0));
            REQUIRE(Ping(1));
            ReleaseClient(1, TestSession + 1);
            peer.WaitForCleanExit();
            coordinator.WaitForCleanExit();
            CHECK(std::chrono::steady_clock::now() - released < std::chrono::seconds{4});
            CHECK_FALSE(Ping(0));
            CHECK_FALSE(Ping(1));
        }
    }
#else
    SKIP("independent managed daemons require Linux process execution");
#endif
}
