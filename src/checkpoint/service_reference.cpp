#include "service_reference.h"

#include "CommonMacros.h"
#include "net_utils.h"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_future.hpp>
#include <boost/process/v2/ext/exe.hpp>
#include <boost/process/v2/pid.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace checkpoint
{
namespace
{
using namespace std::chrono_literals;
namespace http                          = boost::beast::http;
using Response                          = http::response<http::string_body>;
using Clock                             = std::chrono::steady_clock;
constexpr auto             Timeout      = 5s;
constexpr auto             ScanInterval = 1s;
constexpr std::string_view Prefix       = "/api/reference/";

uint64_t NewGeneration()
{ return static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()); }

std::string NewToken()
{
    static std::atomic<uint64_t> Sequence{};
    return fmt::format("{}-{}-{}", boost::process::v2::current_pid(), NewGeneration(), Sequence.fetch_add(1));
}

Response Request(std::string const& peer, std::string const& target, std::string body = {}, std::chrono::seconds timeout = Timeout)
{
    boost::asio::io_context ioc;
    auto                    result = boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<Response> {
            boost::beast::tcp_stream stream(ioc);
            auto const [address, endpoints] = ResolveEndpoint(ioc, peer);
            stream.expires_after(timeout);
            co_await stream.async_connect(endpoints, boost::asio::use_awaitable);
            http::request<http::string_body> request{http::verb::post, target, HttpVersion11};
            request.set(http::field::host, address.host());
            request.body() = std::move(body);
            request.prepare_payload();
            co_await http::async_write(stream, request, boost::asio::use_awaitable);
            boost::beast::flat_buffer buffer;
            Response                  response;
            co_await http::async_read(stream, buffer, response, boost::asio::use_awaitable);
            if (response.result() == http::status::service_unavailable)
            {
                throw boost::system::system_error(boost::asio::error::try_again, response.body());
            }
            if (response.result() != http::status::ok) { throw std::runtime_error(response.body()); }
            co_return response;
        },
        boost::asio::use_future);
    ioc.run();
    return result.get();
}

void Release(std::string const& peer, std::string const& token) noexcept
{
    try
    {
        static_cast<void>(Request(peer, std::string(Prefix) + "release/" + token));
    } catch (std::exception const& error) { WARN("Reference release to {} failed: {}", peer, error.what()); }
}

struct Process
{
    explicit Process(boost::process::v2::pid_type value) : pid(value) {}
    [[nodiscard]] bool Valid() const
    {
        boost::system::error_code error;
        static_cast<void>(boost::process::v2::ext::exe(pid, error));
        return error != boost::system::errc::no_such_file_or_directory && error != boost::system::errc::no_such_process
               && error != boost::system::errc::invalid_argument;
    }
    boost::process::v2::pid_type pid;
};

struct LocatedState
{
    uint16_t            port;
    uint64_t            generation;
    std::weak_ptr<void> state;
};

struct Locator
{
    boost::process::v2::pid_type const creator = boost::process::v2::current_pid();
    std::mutex                         mutex;
    std::vector<LocatedState>          states;
};

Locator& LocalStates()
{
    static auto* States = new Locator;
    if (States->creator != boost::process::v2::current_pid()) { States = new Locator; }
    return *States;
}
}    // namespace

struct ServiceReferences::State
{
    enum class Kind : uint8_t
    {
        Process,
        Remote
    };
    struct Reference
    {
        Kind                     kind;
        ipc::SessionId           session;
        std::string              context;
        std::shared_ptr<Process> process;
        uint64_t                 generation{};

        [[nodiscard]] bool Valid() const
        {
            if (kind == Kind::Process) { return process->Valid(); }
            try
            {
                return std::stoull(Request(context, std::string(Prefix) + "ping").body()) == generation;
            } catch (std::exception const&) { return false; }
        }
    };
    struct Session
    {
        std::mutex  mutex;
        std::string coordinator;
        std::string token;
    };

    std::mutex                                                   mutex;
    std::condition_variable                                      changed;
    std::map<std::pair<ipc::SessionId, std::string>, size_t>     selfReferences;
    std::map<std::string, std::shared_ptr<Reference>>            references;
    std::unordered_set<std::string>                              released;
    std::unordered_map<ipc::SessionId, std::shared_ptr<Session>> sessions;
    uint64_t                                                     generation{NewGeneration()};
    uint64_t                                                     revision{};
    bool                                                         owned{false};
    bool                                                         closing{false};
    bool                                                         telemetryClosed{false};
    bool                                                         sealed{false};

    void Add(std::string const& token, Reference reference)
    {
        std::scoped_lock lock(mutex);
        if (closing) { throw boost::system::system_error(boost::asio::error::try_again, "Checkpoint service is shutting down"); }
        if (released.contains(token)) { throw std::runtime_error("Reference is already released"); }
        owned = true;
        references.try_emplace(token, std::make_shared<Reference>(std::move(reference)));
        ++revision;
        changed.notify_all();
    }

    void Remove(std::string const& token, bool external)
    {
        std::scoped_lock lock(mutex);
        references.erase(token);
        if (external) { released.insert(token); }
        ++revision;
        changed.notify_all();
    }

    bool Live(ipc::SessionId session = NoSession)
    {
        std::vector<std::pair<std::string, std::shared_ptr<Reference>>> candidates;
        {
            std::scoped_lock lock(mutex);
            for (auto const& [context, count] : selfReferences)
            {
                if (count && (session == NoSession || context.first == session || context.first == NoSession)) { return true; }
            }
            for (auto const& item : references)
            {
                if (session == NoSession || item.second->session == session || item.second->session == NoSession)
                {
                    candidates.push_back(item);
                }
            }
        }
        std::ranges::sort(candidates, {}, [](auto const& item) { return item.second->kind; });
        for (auto const& [token, reference] : candidates)
        {
            if (reference->Valid()) { return true; }
            if (reference->kind == Kind::Process) { Remove(token, false); }
        }
        return false;
    }

    void Coordinate(ipc::SessionId id, std::string const& peers)
    {
        if (peers.empty()) { return; }
        auto const separator = peers.find('\n');
        if (separator == std::string::npos) { throw std::invalid_argument("Missing coordinator reference peer"); }
        std::shared_ptr<Session> session;
        {
            std::scoped_lock lock(mutex);
            auto&            entry = sessions[id];
            if (!entry) { entry = std::make_shared<Session>(); }
            session = entry;
        }
        std::scoped_lock lock(session->mutex);
        if (!session->token.empty()) { return; }
        session->coordinator = peers.substr(0, separator);
        auto       token     = NewToken();
        auto const target    = fmt::format("{}remote/{}/{}/{}", Prefix, static_cast<uint32_t>(id), generation, token);
        auto const deadline  = Clock::now() + 900s;
        try
        {
            while (true)
            {
                try
                {
                    static_cast<void>(Request(session->coordinator, target, peers.substr(separator + 1)));
                    session->token = std::move(token);
                    return;
                } catch (std::exception const&)
                {
                    if (Clock::now() >= deadline) { throw; }
                    std::this_thread::sleep_for(100ms);
                }
            }
        } catch (...)
        {
            Release(session->coordinator, token);
            throw;
        }
    }

    void ReleaseSessions(bool all)
    {
        std::vector<std::pair<ipc::SessionId, std::shared_ptr<Session>>> candidates;
        {
            std::scoped_lock lock(mutex);
            for (auto const& session : sessions) { candidates.push_back(session); }
        }
        for (auto const& [id, session] : candidates)
        {
            std::scoped_lock lock(session->mutex);
            if (session->token.empty()) { continue; }
            if (!all && (!Live() || Live(id))) { continue; }
            Release(session->coordinator, session->token);
            session->token.clear();
        }
    }
};

ServiceReferences::ServiceReferences(uint16_t index) : _state(std::make_shared<State>())
{
    auto&            locator = LocalStates();
    std::scoped_lock lock(locator.mutex);
    std::erase_if(locator.states, [](auto const& item) { return item.state.expired(); });
    locator.states.push_back({static_cast<uint16_t>(ipc::Message::Port + index), _state->generation, _state});
}

ServiceReferences::~ServiceReferences()
{ ReleaseRemoteReferences(); }

ServiceReference ServiceReferences::Acquire(std::string_view context, ipc::SessionId session, bool control)
{
    auto const key     = std::make_pair(session, std::string(context));
    auto const creator = boost::process::v2::current_pid();
    {
        std::scoped_lock lock(_state->mutex);
        if (_state->sealed && control) { return {}; }
        if (_state->closing && !control) { throw std::runtime_error("Checkpoint service is shutting down"); }
        ++_state->selfReferences[key];
        ++_state->revision;
        _state->changed.notify_all();
    }
    return {std::shared_ptr<void>(nullptr,
                                  [state = std::weak_ptr<State>(_state), key, creator](void*) {
                                      if (boost::process::v2::current_pid() != creator) { return; }
                                      if (auto owner = state.lock())
                                      {
                                          std::scoped_lock lock(owner->mutex);
                                          if (--owner->selfReferences.at(key) == 0) { owner->selfReferences.erase(key); }
                                          ++owner->revision;
                                          owner->changed.notify_all();
                                      }
                                  }),
            _state->generation};
}

void ServiceReferences::DrainRequests()
{
    std::unique_lock lock(_state->mutex);
    _state->closing         = true;
    _state->telemetryClosed = true;
    _state->changed.wait(lock, [&] { return _state->selfReferences.empty(); });
}

void ServiceReferences::Seal()
{
    std::scoped_lock lock(_state->mutex);
    _state->sealed = true;
}

void ServiceReferences::WaitForEmpty(std::atomic<bool> const& stopRequested)
{
    auto const startupDeadline = Clock::now() + Timeout;
    while (true)
    {
        uint64_t revision;
        {
            std::scoped_lock lock(_state->mutex);
            revision = _state->revision;
        }
        auto const live = _state->Live();
        {
            std::unique_lock lock(_state->mutex);
            if (stopRequested || (!live && revision == _state->revision && (_state->owned || Clock::now() >= startupDeadline)))
            {
                _state->closing = true;
                return;
            }
        }
        _state->ReleaseSessions(false);
        std::unique_lock lock(_state->mutex);
        _state->changed.wait_for(lock, ScanInterval, [&] { return _state->revision != revision || stopRequested; });
    }
}

void ServiceReferences::ReleaseRemoteReferences()
{ _state->ReleaseSessions(true); }

bool ServiceReferences::Accepting() const
{
    std::scoped_lock lock(_state->mutex);
    return !_state->closing;
}

bool ServiceReferences::HandleRequest(boost::beast::tcp_stream&               stream,
                                      boost::beast::flat_buffer&              buffer,
                                      http::request_parser<http::empty_body>& parser)
{
    auto const target = std::string(parser.get().target());
    if (!target.starts_with(Prefix))
    {
        {
            std::scoped_lock lock(_state->mutex);
            if (!_state->closing || (!_state->telemetryClosed && (target.starts_with("/api/telemetry") || target.starts_with("/v1/"))))
            {
                return false;
            }
        }
        Response response{http::status::service_unavailable, parser.get().version()};
        response.body() = "Checkpoint service is shutting down";
        response.prepare_payload();
        http::write(stream, response);
        return true;
    }
    Response response{http::status::ok, parser.get().version()};
    try
    {
        auto const path = target.substr(Prefix.size());
        if (path == "ping") { response.body() = std::to_string(_state->generation); }
        else if (path.starts_with("release/")) { _state->Remove(path.substr(std::string_view("release/").size()), true); }
        else
        {
            http::request_parser<http::string_body> bodyParser{std::move(parser)};
            bodyParser.body_limit(1024 * 1024);
            http::read(stream, buffer, bodyParser);
            auto const operationEnd = path.find('/');
            auto const sessionEnd   = path.find('/', operationEnd + 1);
            auto const ownerEnd     = path.find('/', sessionEnd + 1);
            if (operationEnd == std::string::npos || sessionEnd == std::string::npos || ownerEnd == std::string::npos)
            {
                throw std::invalid_argument("Malformed reference request");
            }
            auto const session = static_cast<ipc::SessionId>(std::stoul(path.substr(operationEnd + 1, sessionEnd - operationEnd - 1)));
            auto const owner   = std::stoull(path.substr(sessionEnd + 1, ownerEnd - sessionEnd - 1));
            auto const token   = path.substr(ownerEnd + 1);
            if (path.starts_with("process/"))
            {
                _state->Add(token,
                            {.kind    = State::Kind::Process,
                             .session = session,
                             .context = {},
                             .process = std::make_shared<Process>(static_cast<boost::process::v2::pid_type>(owner))});
                try
                {
                    _state->Coordinate(session, bodyParser.get().body());
                } catch (...)
                {
                    _state->Remove(token, true);
                    throw;
                }
            }
            else if (path.starts_with("remote/"))
            {
                _state->Add(token,
                            {.kind       = State::Kind::Remote,
                             .session    = session,
                             .context    = bodyParser.get().body(),
                             .process    = {},
                             .generation = owner});
            }
            else
            {
                throw std::invalid_argument("Unknown reference kind");
            }
            response.body() = std::to_string(_state->generation);
        }
    } catch (boost::system::system_error const& error)
    {
        response.result(http::status::service_unavailable);
        response.body() = error.what();
    } catch (std::exception const& error)
    {
        response.result(http::status::bad_request);
        response.body() = error.what();
    }
    response.prepare_payload();
    http::write(stream, response);
    return true;
}

ServiceReference ServiceReferences::AcquireProcess(ipc::SessionId session, RankTopology const& topology)
{
    if (topology.ranksPerPeer == 0 || topology.peersPerNode == 0
        || static_cast<uint32_t>(topology.rank) >= static_cast<uint32_t>(topology.worldSize))
    {
        throw std::invalid_argument("Invalid checkpoint reference topology");
    }
    auto const index = (static_cast<uint32_t>(topology.rank) / topology.ranksPerPeer) % topology.peersPerNode;
    auto const node  = static_cast<uint32_t>(topology.rank) / topology.ranksPerPeer;
    return AcquireProcess(session, static_cast<uint16_t>(index), node == 0 ? std::string{} : topology.peers.at(0), topology.peers.at(node));
}

ServiceReference ServiceReferences::AcquireProcess(ipc::SessionId session, uint16_t index, std::string coordinator, std::string address)
{
    auto const creator = boost::process::v2::current_pid();
    auto const port    = static_cast<uint16_t>(ipc::Message::Port + index);
    auto const peer    = fmt::format("localhost:{}", port);
    auto const token   = NewToken();
    auto const target  = fmt::format("{}process/{}/{}/{}", Prefix, static_cast<uint32_t>(session), creator, token);
    auto const body    = coordinator.empty() ? std::string{} : coordinator + '\n' + address;
    uint64_t   generation{};
    try
    {
        generation = std::stoull(Request(peer, target, body, 900s).body());
    } catch (...)
    {
        Release(peer, token);
        throw;
    }
    std::weak_ptr<State> local;
    {
        auto&            locator = LocalStates();
        std::scoped_lock lock(locator.mutex);
        for (auto const& item : locator.states)
        {
            if (item.port == port && item.generation == generation)
            {
                local = std::static_pointer_cast<State>(item.state.lock());
                break;
            }
        }
    }
    return {std::shared_ptr<void>(nullptr,
                                  [local, peer, token, creator](void*) {
                                      if (boost::process::v2::current_pid() != creator) { return; }
                                      if (auto state = local.lock()) { state->Remove(token, true); }
                                      else
                                      {
                                          Release(peer, token);
                                      }
                                  }),
            generation};
}
}    // namespace checkpoint
