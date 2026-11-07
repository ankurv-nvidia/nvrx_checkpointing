#include "CommonMacros.h"
#include "messages.h"
#include "stateserdes.h"

#include <catch2/catch_test_macros.hpp>

#ifdef __linux__
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

namespace
{
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

class ClientProcess
{
    public:
    ClientProcess()
    {
        _pid = ::fork();
        if (_pid == 0)
        {
            while (true) { ::pause(); }
        }
        REQUIRE(_pid > 0);
    }
    ~ClientProcess() { Stop(); }
    CLASS_DELETE_COPY_AND_MOVE(ClientProcess);

    [[nodiscard]] pid_t Pid() const { return _pid; }

    void Stop()
    {
        if (_pid <= 0) { return; }
        ::kill(_pid, SIGKILL);
        ::waitpid(_pid, nullptr, 0);
        _pid = -1;
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

void AcquireClient(uint16_t index, pid_t owner)
{
    auto const target   = "/api/reference/acquire/7301/client/" + std::to_string(owner);
    auto const response = Request(index, http::verb::post, target, ActivationBody(index));
    REQUIRE(response);
    INFO(response->body());
    REQUIRE(response->result() == http::status::ok);
}

void ReleaseClient(uint16_t index)
{
    auto const response = Request(index, http::verb::post, "/api/reference/release/7301/client");
    REQUIRE(response);
    INFO(response->body());
    REQUIRE(response->result() == http::status::ok);
}
}    // namespace
#endif

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
            ClientProcess peerClient;
            AcquireClient(1, peerClient.Pid());
            ReleaseClient(0);
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
            REQUIRE(Ping(0));
            REQUIRE(Ping(1));
            auto const released = std::chrono::steady_clock::now();
            if (ownerDies) { peerClient.Stop(); }
            else
            {
                ReleaseClient(1);
            }
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
