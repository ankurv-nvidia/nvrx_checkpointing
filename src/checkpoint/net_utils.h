#pragma once

#include "CommonMacros.h"
#include "logging.h"
#include "messages.h"

#include <fmt/ostream.h>

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/buffer_body.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/beast/version.hpp>
#include <boost/url.hpp>

#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

template <> struct fmt::formatter<boost::asio::ip::tcp::endpoint> : fmt::ostream_formatter
{};
template <> struct fmt::formatter<boost::asio::ip::address> : fmt::ostream_formatter
{};

inline auto ResolveEndpoint(boost::asio::io_context& ioc, std::string_view const& address)
{
    // Cached results avoid intermittent host-not-found errors on repeated resolution.
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    SUPPRESS_CLANG_WARNING("-Wunique-object-duplication")
    static std::unordered_map<boost::urls::url, boost::asio::ip::tcp::resolver::results_type> EndpointCache;
    static std::mutex                                                                         Mutex;
    SUPPRESS_WARNINGS_END

    boost::urls::url endpoint;

    auto it = address.find(':');
    if (it != std::string::npos)
    {
        endpoint.set_host(address.substr(0, it));
        endpoint.set_port(address.substr(it + 1));
    }
    else
    {
        endpoint.set_host(address);
        endpoint.set_port(std::to_string(checkpoint::ipc::Message::Port));
    }

    std::scoped_lock lock(Mutex);
    if (auto it1 = EndpointCache.find(endpoint); it1 != EndpointCache.end()) { return std::make_pair(it1->first, it1->second); }

    boost::asio::ip::tcp::resolver resolver(ioc);
    auto                           results = resolver.resolve(endpoint.host(), endpoint.port());
    if (results.empty()) { throw std::runtime_error(fmt::format("[native_ckpt:svr] Failed to resolve endpoint={}", address)); }
    EndpointCache[endpoint] = results;
    return std::make_pair(endpoint, results);
}

[[nodiscard]] inline auto CreateRequestForPeer(boost::asio::io_context&  ioc,
                                               boost::beast::tcp_stream& stream,
                                               std::string_view const&   peer,
                                               std::string_view const&   target)
{
    auto [peerAddr, resolveResults] = ResolveEndpoint(ioc, peer);
    auto const endpoint             = stream.connect(resolveResults);
    INFO("Connected to endpoint={} resolved from peer={}", endpoint, peer);

    using Request = boost::beast::http::request<boost::beast::http::buffer_body>;

    Request req;
    req.method(boost::beast::http::verb::post);
    static constexpr int HttpVersion11 = 11;
    req.version(HttpVersion11);
    req.set(boost::beast::http::field::host, fmt::format("{}:{}", peerAddr.host(), peerAddr.port()));
    req.set(boost::beast::http::field::content_type, "application/octet-stream");
    req.target(target);
    return req;
}
