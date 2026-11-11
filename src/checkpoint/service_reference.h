#pragma once

#include "messages.h"
#include "tensor.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <string_view>
#include <utility>

namespace checkpoint
{
class ServiceReference
{
    public:
    ServiceReference() = default;
    ServiceReference(std::shared_ptr<void> lease, uint64_t generation) : _lease(std::move(lease)), _generation(generation) {}
    void                   Reset() { _lease.reset(); }
    [[nodiscard]] uint64_t Generation() const { return _generation; }

    private:
    std::shared_ptr<void> _lease;
    uint64_t              _generation{};
};

class ServiceReferences
{
    public:
    static constexpr auto NoSession = ipc::SessionId::Invalid;

    explicit ServiceReferences(uint16_t index = 0);
    ~ServiceReferences();
    ServiceReferences(ServiceReferences const&)            = delete;
    ServiceReferences& operator=(ServiceReferences const&) = delete;

    [[nodiscard]] ServiceReference Acquire(std::string_view context, ipc::SessionId session = NoSession, bool control = false);

    template <typename TCall>
    auto Bind(std::string_view context, TCall&& operation, ipc::SessionId session = NoSession, bool control = false)
    {
        return [reference = Acquire(context, session, control), call = std::forward<TCall>(operation)] mutable {
            auto lifetime = std::move(reference);
            call();
        };
    }

    template <typename TCall> auto Async(std::string_view context, TCall&& call, ipc::SessionId session = NoSession)
    { return std::async(std::launch::async, Bind(context, std::forward<TCall>(call), session)); }

    bool               HandleRequest(boost::beast::tcp_stream&                                           stream,
                                     boost::beast::flat_buffer&                                          buffer,
                                     boost::beast::http::request_parser<boost::beast::http::empty_body>& parser);
    void               WaitForEmpty(std::atomic<bool> const& stopRequested);
    [[nodiscard]] bool Accepting() const;
    void               ReleaseRemoteReferences();

    [[nodiscard]] static ServiceReference AcquireProcess(ipc::SessionId session, RankTopology const& topology);

    private:
    struct State;
    std::shared_ptr<State> _state;
};
}    // namespace checkpoint
