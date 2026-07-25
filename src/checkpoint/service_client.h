#pragma once
#include "messages.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <fmt/format.h>

#include <chrono>
#include <string>
#include <thread>

// A short readiness probe shared by the training client and losing daemon
// candidates. EINTR retries protect one socket operation; the two outer attempts
// cover the narrow interval between instance election and the winner binding.
inline bool TryConnectToCheckpointService()
{
    using namespace std::chrono_literals;
    namespace cipc = checkpoint::ipc;

    static constexpr int      MaxInterruptedRetries = 5;
    static constexpr int      MaxConnectAttempts    = 2;
    boost::system::error_code ec;

    for (int connectAttempt = 0; connectAttempt < MaxConnectAttempts; ++connectAttempt)
    {
        ec = boost::asio::error::interrupted;
        for (int interruptedAttempt = 0; interruptedAttempt < MaxInterruptedRetries && ec == boost::asio::error::interrupted;
             ++interruptedAttempt)
        {
            boost::asio::io_context        ioc;
            boost::beast::tcp_stream       stream(ioc);
            boost::asio::ip::tcp::resolver resolver(ioc);

            auto results = resolver.resolve("localhost", std::to_string(cipc::Message::Port), ec);
            if (ec) { continue; }

            [[maybe_unused]] auto endpoint = stream.connect(results, ec);
            if (ec) { continue; }

            static constexpr int                                         HttpVersion11 = 11;
            boost::beast::http::request<boost::beast::http::buffer_body> req{boost::beast::http::verb::get, "/api/ping", HttpVersion11};
            req.set(boost::beast::http::field::host, fmt::format("{}:{}", "localhost", cipc::Message::Port));
            req.body().data = nullptr;
            req.body().size = 0;
            req.body().more = false;
            req.prepare_payload();

            boost::beast::http::write(stream, req, ec);
            if (ec) { continue; }

            boost::beast::flat_buffer                                     buffer;
            boost::beast::http::response<boost::beast::http::string_body> res;
            boost::beast::http::read(stream, buffer, res, ec);
            if (ec) { continue; }
            if (res.result() != boost::beast::http::status::ok) { break; }

            ec = stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
            return true;
        }
        if (connectAttempt + 1 < MaxConnectAttempts) { std::this_thread::sleep_for(100ms); }
    }

    if (ec != boost::asio::error::connection_refused)
    {
        fmt::print(stderr, "[native_ckpt:cli][ERROR] Attempting to connect with checkpoint service failed with: {}\n", ec.what());
    }
    return false;
}
