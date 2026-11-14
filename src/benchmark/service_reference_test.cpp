#include "service_reference.h"

#include <boost/asio/ip/tcp.hpp>
#include <boost/process/v2/pid.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
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
using Response = http::response<http::string_body>;

class ReferenceEndpoint
{
    public:
    explicit ReferenceEndpoint(bool peer = false) : _acceptor(_ioc, Tcp::endpoint{boost::asio::ip::address_v4::loopback(), 0}), _peer(peer)
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
                    boost::beast::tcp_stream               stream(std::move(socket));
                    boost::beast::flat_buffer              buffer;
                    http::request_parser<http::empty_body> parser;
                    http::read_header(stream, buffer, parser);
                    if (_peer)
                    {
                        ++probes;
                        Response response{available ? http::status::ok : http::status::service_unavailable, 11};
                        response.body() = std::to_string(generation.load());
                        response.prepare_payload();
                        http::write(stream, response);
                    }
                    else
                    {
                        static_cast<void>(references.HandleRequest(stream, buffer, parser));
                    }
                } catch (boost::system::system_error const&) {}
            }
        });
    }

    [[nodiscard]] std::string Address() const { return "127.0.0.1:" + std::to_string(_acceptor.local_endpoint().port()); }

    Response Post(std::string const& path, std::string body = {}) const
    {
        boost::asio::io_context  ioc;
        boost::beast::tcp_stream stream(ioc);
        stream.connect(_acceptor.local_endpoint());
        http::request<http::string_body> request{http::verb::post, "/api/reference/" + path, 11};
        request.body() = std::move(body);
        request.prepare_payload();
        http::write(stream, request);
        boost::beast::flat_buffer buffer;
        Response                  response;
        http::read(stream, buffer, response);
        return response;
    }

    Response Process(std::string const& token, boost::process::v2::pid_type pid) const
    { return Post("process/42/" + std::to_string(pid) + '/' + token); }

    Response Remote(std::string const& token, ReferenceEndpoint const& peer, uint64_t owner = 1) const
    { return Post("remote/42/" + std::to_string(owner) + '/' + token, peer.Address()); }

    checkpoint::ServiceReferences references;
    std::atomic<bool>             available{true};
    std::atomic<uint64_t>         generation{1};
    std::atomic<unsigned>         probes{};

    private:
    boost::asio::io_context _ioc;
    Tcp::acceptor           _acceptor;
    bool const              _peer;
    std::jthread            _thread;
};

class ShutdownWaiter
{
    public:
    explicit ShutdownWaiter(checkpoint::ServiceReferences& references) :
        _finished(std::async(std::launch::async, [&references, this] { references.WaitForEmpty(_stop); }))
    {}
    ~ShutdownWaiter()
    {
        _stop = true;
        _finished.wait();
    }

    [[nodiscard]] bool FinishedWithin(std::chrono::milliseconds timeout)
    { return _finished.wait_for(timeout) == std::future_status::ready; }

    private:
    std::atomic<bool> _stop{false};
    std::future<void> _finished;
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

TEST_CASE("self references keep unrelated asynchronous work alive", "[reference][tier1]")
{
    ReferenceEndpoint endpoint;
    auto              client = endpoint.references.Acquire("training", checkpoint::ipc::SessionId{1});
    REQUIRE(endpoint.Process("startup", boost::process::v2::current_pid()).result() == http::status::ok);
    REQUIRE(endpoint.Post("release/startup").result() == http::status::ok);
    std::promise<void> complete;
    auto               finished = complete.get_future().share();
    auto               task     = endpoint.references.Async("pending work", [finished] { finished.wait(); }, checkpoint::ipc::SessionId{2});
    ShutdownWaiter     shutdown(endpoint.references);

    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    client.Reset();
    client.Reset();
    CHECK_FALSE(shutdown.FinishedWithin(50ms));
    complete.set_value();
    task.get();
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("process references stay alive until explicit release", "[reference][tier1]")
{
    ReferenceEndpoint endpoint;
    REQUIRE(endpoint.Process("client", boost::process::v2::current_pid()).result() == http::status::ok);
    ShutdownWaiter shutdown(endpoint.references);
    CHECK_FALSE(shutdown.FinishedWithin(50ms));

    REQUIRE(endpoint.Post("release/client").result() == http::status::ok);
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("zombie process references do not keep the service alive", "[reference][tier1]")
{
    auto const pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) { ::_exit(0); }
    siginfo_t exitInfo{};
    REQUIRE(::waitid(P_PID, static_cast<id_t>(pid), &exitInfo, WEXITED | WNOWAIT) == 0);
    ReferenceEndpoint endpoint;
    REQUIRE(endpoint.Process("dead-client", static_cast<boost::process::v2::pid_type>(pid)).result() == http::status::ok);
    ShutdownWaiter shutdown(endpoint.references);
    CHECK(shutdown.FinishedWithin(1500ms));
    int status{};
    REQUIRE(::waitpid(pid, &status, 0) == pid);
}

TEST_CASE("a failed remote reference can recover while another peer stays alive", "[reference][tier1]")
{
    ReferenceEndpoint failed(true);
    ReferenceEndpoint keeper(true);
    failed.available = false;
    ReferenceEndpoint endpoint;
    REQUIRE(endpoint.Remote("a-recovering", failed).result() == http::status::ok);
    REQUIRE(endpoint.Remote("z-keeper", keeper).result() == http::status::ok);
    ShutdownWaiter shutdown(endpoint.references);
    REQUIRE(Eventually([&] { return failed.probes > 0 && keeper.probes > 0; }));
    CHECK_FALSE(shutdown.FinishedWithin(50ms));

    auto const initialProbes = failed.probes.load();
    failed.available         = true;
    REQUIRE(endpoint.Post("release/z-keeper").result() == http::status::ok);
    REQUIRE(Eventually([&] { return failed.probes > initialProbes; }));
    CHECK_FALSE(shutdown.FinishedWithin(50ms));

    REQUIRE(endpoint.Post("release/a-recovering").result() == http::status::ok);
    CHECK(shutdown.FinishedWithin(1500ms));
}

TEST_CASE("a different peer generation invalidates its remote reference", "[reference][tier1]")
{
    ReferenceEndpoint peer(true);
    ReferenceEndpoint endpoint;
    REQUIRE(endpoint.Remote("old-generation", peer, peer.generation + 1).result() == http::status::ok);
    ShutdownWaiter shutdown(endpoint.references);
    CHECK(shutdown.FinishedWithin(1500ms));
    CHECK(peer.probes > 0);
}

TEST_CASE("duplicate remote acquisition and release are idempotent", "[reference][tier1]")
{
    ReferenceEndpoint peer(true);
    ReferenceEndpoint endpoint;
    REQUIRE(endpoint.Remote("same-peer", peer).result() == http::status::ok);
    REQUIRE(endpoint.Remote("same-peer", peer).result() == http::status::ok);
    ShutdownWaiter shutdown(endpoint.references);
    REQUIRE(Eventually([&] { return peer.probes > 0; }));
    CHECK_FALSE(shutdown.FinishedWithin(50ms));

    REQUIRE(endpoint.Post("release/same-peer").result() == http::status::ok);
    CHECK(shutdown.FinishedWithin(1500ms));
    REQUIRE(endpoint.Post("release/same-peer").result() == http::status::ok);
}
