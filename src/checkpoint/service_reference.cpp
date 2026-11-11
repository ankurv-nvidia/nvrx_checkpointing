#include "service_reference.h"

#include "SetThreadName.h"
#include "config.h"
#include "logging.h"
#include "net_utils.h"

#include <boost/asio/error.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/process/v2/ext/exe.hpp>
#include <boost/process/v2/pid.hpp>
#include <boost/system/error_code.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <charconv>
#include <exception>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <system_error>

using namespace std::chrono_literals;
namespace cipc = checkpoint::ipc;
namespace http = boost::beast::http;

struct checkpoint::service::SessionReferences::State
{
    cipc::SessionId              session{cipc::SessionId::Invalid};
    std::function<std::string()> pendingWork;
    std::string                  coordinator;
    std::string                  address;
    std::mutex                   mutex;
    bool                         open{true};
    bool                         upstreamHeld{false};
};

namespace checkpoint::service
{
namespace
{
constexpr uint32_t FailureLimit    = 3;
constexpr auto     ProbeTimeout    = 2s;
constexpr auto     ReleaseTimeout  = 1s;
constexpr int      AcquireAttempts = 5;
constexpr auto     RetryInterval   = 200ms;

constexpr bool Includes(Kinds set, Kinds kind)
{ return (std::to_underlying(set) & std::to_underlying(kind)) != 0; }

std::optional<http::status>
Request(http::verb method, std::string const& address, std::string const& target, std::string const& body, Clock::duration timeout)
{
    try
    {
        boost::asio::io_context ioc;
        auto [endpoint, results] = ResolveEndpoint(ioc, address);
        boost::beast::tcp_stream         stream(ioc);
        http::request<http::string_body> request{method, target, HttpVersion11};
        request.set(http::field::host, fmt::format("{}:{}", endpoint.host(), endpoint.port()));
        request.body() = body;
        request.prepare_payload();
        boost::beast::flat_buffer         buffer;
        http::response<http::string_body> response;
        boost::system::error_code         result = boost::asio::error::timed_out;
        stream.expires_after(timeout);
        stream.async_connect(results, [&](boost::system::error_code connectError, auto const& /* connected */) {
            if (connectError)
            {
                result = connectError;
                return;
            }
            http::async_write(stream, request, [&](boost::system::error_code writeError, size_t /* written */) {
                if (writeError)
                {
                    result = writeError;
                    return;
                }
                http::async_read(
                    stream, buffer, response, [&](boost::system::error_code readError, size_t /* read */) { result = readError; });
            });
        });
        ioc.run();
        boost::system::error_code ignored;
        static_cast<void>(stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ignored));
        if (result) { return std::nullopt; }
        return response.result();
    } catch (std::exception const&) { return std::nullopt; }
}

template <typename TValue> std::optional<TValue> ParseInteger(std::string_view text)
{
    TValue     value{};
    auto const parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) { return std::nullopt; }
    return value;
}

std::vector<std::string_view> PathSegments(std::string_view path)
{
    std::vector<std::string_view> segments;
    for (auto const segment : std::views::split(path, '/'))
    {
        if (!segment.empty()) { segments.emplace_back(segment.begin(), segment.end()); }
    }
    return segments;
}

std::string PendingReason(SessionReferences::State& state)
{
    std::scoped_lock lock(state.mutex);
    return state.open && state.pendingWork ? state.pendingWork() : std::string{};
}

class ClientReference
{
    public:
    ClientReference(std::string address, std::string target, int64_t pid) :
        _address(std::move(address)), _target(std::move(target)), _pid(pid)
    {}
    ~ClientReference()
    {
        // A forked child inherits the handle but not the reference.
        if (CurrentPid() != _pid) { return; }
        static_cast<void>(Request(http::verb::delete_, _address, _target, {}, ReleaseTimeout));
    }
    CLASS_DELETE_COPY_AND_MOVE(ClientReference);

    private:
    std::string _address;
    std::string _target;
    int64_t     _pid;
};
}    // namespace

int64_t CurrentPid()
{ return static_cast<int64_t>(boost::process::v2::current_pid()); }

bool ProcessAlive(int64_t pid)
{
    if (pid <= 0) { return false; }
    boost::system::error_code error;
    static_cast<void>(boost::process::v2::ext::exe(static_cast<boost::process::v2::pid_type>(pid), error));
    // Only a definite lookup failure counts as exited; a zombie has no executable and reads as exited.
    return !(error == boost::system::errc::no_such_file_or_directory || error == boost::system::errc::no_such_process
             || error == boost::system::errc::invalid_argument);
}

Probes DefaultProbes()
{
    return {
        .processAlive = &ProcessAlive,
        .peerAlive
        = [](std::string const& address) { return Request(http::verb::get, address, "/api/ping", {}, ProbeTimeout) == http::status::ok; },
        .notify =
            [](http::verb method, std::string const& address, std::string const& target, std::string const& body) {
                auto const status = Request(method, address, target, body, ProbeTimeout);
                return status == http::status::ok || status == http::status::not_found;
            },
    };
}

SelfReference::~SelfReference()
{
    if (_registry != nullptr) { _registry->ReleaseSelf_(_count); }
}

SelfReference::SelfReference(SelfReference&& other) noexcept :
    _registry(std::exchange(other._registry, nullptr)), _count(std::exchange(other._count, nullptr))
{}

SelfReference& SelfReference::operator=(SelfReference&& other) noexcept
{
    if (this != &other)
    {
        if (_registry != nullptr) { _registry->ReleaseSelf_(_count); }
        _registry = std::exchange(other._registry, nullptr);
        _count    = std::exchange(other._count, nullptr);
    }
    return *this;
}

SessionReferences::~SessionReferences()
{
    if (_registry != nullptr) { _registry->Detach_(_state); }
}

SessionReferences::SessionReferences(SessionReferences&& other) noexcept :
    _registry(std::exchange(other._registry, nullptr)), _state(std::move(other._state))
{}

SessionReferences& SessionReferences::operator=(SessionReferences&& other) noexcept
{
    if (this != &other)
    {
        if (_registry != nullptr) { _registry->Detach_(_state); }
        _registry = std::exchange(other._registry, nullptr);
        _state    = std::move(other._state);
    }
    return *this;
}

Clock::duration Registry::ConfiguredScanInterval()
{ return std::chrono::seconds{checkpoint::config::GetConfigValue<checkpoint::config::ServerReferenceScanIntervalSeconds>()}; }

Registry::Registry(Clock::duration scanInterval, Probes probes) :
    _scanInterval(scanInterval), _probes(std::move(probes)), _reaper([this] { Run_(); })
{}

Registry::~Registry()
{
    {
        std::scoped_lock lock(_mutex);
        _stopping = true;
    }
    _wake.notify_all();
    _changed.notify_all();
    if (_reaper.joinable()) { _reaper.join(); }
}

SelfReference Registry::AcquireSelf(std::string_view context) LFTBND
{
    std::scoped_lock lock(_mutex);
    auto             it = _selfCounts.find(context);
    if (it == _selfCounts.end()) { it = _selfCounts.emplace(std::string{context}, 0).first; }
    ++it->second;
    return SelfReference{this, &it->second};
}

void Registry::ReleaseSelf_(size_t* count)
{
    {
        std::scoped_lock lock(_mutex);
        --*count;
        ++_releases;
    }
    _changed.notify_all();
}

SessionReferences Registry::AttachSession(cipc::SessionId              session,
                                          std::function<std::string()> pendingWork,
                                          std::string                  coordinator,
                                          std::string                  address) LFTBND
{
    auto state         = std::make_shared<SessionReferences::State>();
    state->session     = session;
    state->pendingWork = std::move(pendingWork);
    state->coordinator = std::move(coordinator);
    state->address     = std::move(address);
    {
        std::scoped_lock lock(_mutex);
        _sessions.push_back(state);
        _poked = true;
    }
    _wake.notify_one();
    return SessionReferences{this, std::move(state)};
}

void Registry::Detach_(std::shared_ptr<SessionReferences::State> const& state)
{
    {
        std::scoped_lock lock(_mutex);
        std::erase(_sessions, state);
    }
    bool release = false;
    {
        // Waits out a pending-work call or coordinator exchange that is in flight.
        std::scoped_lock lock(state->mutex);
        state->open         = false;
        release             = state->upstreamHeld;
        state->upstreamHeld = false;
    }
    if (release)
    {
        static_cast<void>(
            _probes.notify(http::verb::delete_, state->coordinator, fmt::format("/api/reference/peer/{}", state->session), state->address));
        INFO("Coordinator reference released on {} for session={}", state->coordinator, state->session);
    }
    Changed_(false);
}

void Registry::Changed_(bool wakeReaper)
{
    {
        std::scoped_lock lock(_mutex);
        ++_releases;
        if (wakeReaper) { _poked = true; }
    }
    if (wakeReaper) { _wake.notify_one(); }
    _changed.notify_all();
}

void Registry::AcquirePid(int64_t pid, cipc::SessionId session)
{
    size_t holders = 0;
    {
        std::scoped_lock lock(_mutex);
        auto&            entry = _pids[{pid, session}];
        holders                = ++entry.count;
        entry.generation       = ++_generation;
        _armed                 = true;
        _drainedSince.reset();
        _poked = true;
    }
    _wake.notify_one();
    INFO("Service reference acquired by pid={} session={} holders={}", pid, session, holders);
}

void Registry::ReleasePid(int64_t pid, cipc::SessionId session)
{
    {
        std::scoped_lock lock(_mutex);
        auto const       it = _pids.find({pid, session});
        if (it == _pids.end()) { return; }
        if (--it->second.count == 0) { _pids.erase(it); }
    }
    INFO("Service reference released by pid={} session={}", pid, session);
    Changed_(true);
}

void Registry::ForgetPid(int64_t pid)
{
    size_t forgotten = 0;
    {
        std::scoped_lock lock(_mutex);
        forgotten = std::erase_if(_pids, [pid](auto const& entry) { return entry.first.first == pid; });
    }
    if (forgotten == 0) { return; }
    INFO("Service references of pid={} forgotten for {} session(s)", pid, forgotten);
    Changed_(true);
}

void Registry::AcquireRemote(std::string const& address, cipc::SessionId session)
{
    size_t holders = 0;
    {
        std::scoped_lock lock(_mutex);
        auto&            entry = _remotes[{address, session}];
        holders                = ++entry.count;
        entry.generation       = ++_generation;
        entry.failures         = 0;
        entry.lastProbe        = Clock::now();
        _armed                 = true;
        _drainedSince.reset();
    }
    INFO("Service reference acquired by peer={} session={} holders={}", address, session, holders);
}

void Registry::ReleaseRemote(std::string const& address, cipc::SessionId session)
{
    {
        std::scoped_lock lock(_mutex);
        auto const       it = _remotes.find({address, session});
        if (it == _remotes.end()) { return; }
        if (--it->second.count == 0) { _remotes.erase(it); }
    }
    INFO("Service reference released by peer={} session={}", address, session);
    Changed_(true);
}

http::status Registry::Handle(http::verb method, std::string_view path, std::string_view body)
{
    bool const acquire = method == http::verb::post;
    if (!acquire && method != http::verb::delete_) { return http::status::method_not_allowed; }
    auto const segments = PathSegments(path);
    if (segments.size() == 5 && segments[2] == "pid")
    {
        auto const pid     = ParseInteger<int64_t>(segments[3]);
        auto const session = ParseInteger<uint32_t>(segments[4]);
        if (!pid || !session || *pid <= 0) { return http::status::bad_request; }
        if (acquire) { AcquirePid(*pid, cipc::SessionId{*session}); }
        else
        {
            ReleasePid(*pid, cipc::SessionId{*session});
        }
        return http::status::ok;
    }
    if (segments.size() == 4 && segments[2] == "peer" && !body.empty())
    {
        auto const session = ParseInteger<uint32_t>(segments[3]);
        if (!session) { return http::status::bad_request; }
        if (acquire) { AcquireRemote(std::string{body}, cipc::SessionId{*session}); }
        else
        {
            ReleaseRemote(std::string{body}, cipc::SessionId{*session});
        }
        return http::status::ok;
    }
    return http::status::bad_request;
}

bool Registry::HasValidReference(Kinds kinds)
{ return Walk_(kinds, std::nullopt); }

bool Registry::Walk_(Kinds kinds, std::optional<cipc::SessionId> session)
{
    if (Includes(kinds, Kinds::Self) && SelfValid_(session)) { return true; }
    if (Includes(kinds, Kinds::Pid) && PidsValid_(session)) { return true; }
    return Includes(kinds, Kinds::Remote) && RemotesValid_(session);
}

bool Registry::SelfValid_(std::optional<cipc::SessionId> session)
{
    std::vector<std::shared_ptr<SessionReferences::State>> sessions;
    {
        std::scoped_lock lock(_mutex);
        if (!session && std::ranges::any_of(_selfCounts, [](auto const& entry) { return entry.second > 0; })) { return true; }
        std::ranges::copy_if(
            _sessions, std::back_inserter(sessions), [&](auto const& state) { return !session || state->session == *session; });
    }
    return std::ranges::any_of(sessions, [](auto const& state) { return !PendingReason(*state).empty(); });
}

bool Registry::PidsValid_(std::optional<cipc::SessionId> session)
{
    std::vector<std::pair<PidKey, uint64_t>> candidates;
    {
        std::scoped_lock lock(_mutex);
        for (auto const& [key, entry] : _pids)
        {
            if (!session || key.second == *session) { candidates.emplace_back(key, entry.generation); }
        }
    }
    for (auto const& [key, generation] : candidates)
    {
        if (_probes.processAlive(key.first)) { return true; }
        bool reaped = false;
        {
            std::scoped_lock lock(_mutex);
            auto const       it = _pids.find(key);
            if (it != _pids.end() && it->second.generation == generation)
            {
                _pids.erase(it);
                reaped = true;
            }
        }
        if (reaped)
        {
            INFO("Service reference of exited pid={} session={} reaped", key.first, key.second);
            Changed_(true);
        }
    }
    return false;
}

bool Registry::RemotesValid_(std::optional<cipc::SessionId> session)
{
    std::vector<std::pair<RemoteKey, uint64_t>> candidates;
    {
        std::scoped_lock lock(_mutex);
        auto const       now = Clock::now();
        for (auto const& [key, entry] : _remotes)
        {
            if (session && key.second != *session) { continue; }
            // Each remote is probed at most once per scan interval; until then its last verdict stands.
            if (entry.probing || now - entry.lastProbe < _scanInterval) { return true; }
            candidates.emplace_back(key, entry.generation);
        }
    }
    for (auto const& [key, generation] : candidates)
    {
        {
            std::scoped_lock lock(_mutex);
            auto const       it = _remotes.find(key);
            if (it == _remotes.end()) { continue; }
            if (it->second.probing) { return true; }
            it->second.probing = true;
        }
        bool const alive    = _probes.peerAlive(key.first);
        bool       reaped   = false;
        uint32_t   failures = 0;
        {
            std::scoped_lock lock(_mutex);
            auto const       it = _remotes.find(key);
            if (it == _remotes.end()) { continue; }
            auto& entry     = it->second;
            entry.probing   = false;
            entry.lastProbe = Clock::now();
            if (alive)
            {
                entry.failures = 0;
                return true;
            }
            failures = ++entry.failures;
            if (failures < FailureLimit || entry.generation != generation) { return true; }
            _remotes.erase(it);
            reaped = true;
        }
        if (reaped)
        {
            WARN("Service reference of unreachable peer={} session={} reaped after {} failed pings", key.first, key.second, failures);
            Changed_(true);
        }
    }
    return false;
}

void Registry::Reconcile_()
{
    std::vector<std::shared_ptr<SessionReferences::State>> linked;
    {
        std::scoped_lock lock(_mutex);
        std::ranges::copy_if(_sessions, std::back_inserter(linked), [](auto const& state) { return !state->coordinator.empty(); });
    }
    for (auto const& state : linked)
    {
        bool const       wanted = Walk_(Kinds::All, state->session);
        std::scoped_lock lock(state->mutex);
        if (!state->open || wanted == state->upstreamHeld) { continue; }
        auto const method = wanted ? http::verb::post : http::verb::delete_;
        if (_probes.notify(method, state->coordinator, fmt::format("/api/reference/peer/{}", state->session), state->address))
        {
            state->upstreamHeld = wanted;
            INFO("Coordinator reference {} on {} for session={}", wanted ? "acquired" : "released", state->coordinator, state->session);
        }
        else
        {
            WARN("Coordinator reference {} on {} for session={} failed; retrying next scan",
                 wanted ? "acquire" : "release",
                 state->coordinator,
                 state->session);
        }
    }
}

void Registry::Scan_()
{
    Reconcile_();
    bool const       valid = Walk_(Kinds::All, std::nullopt);
    std::scoped_lock lock(_mutex);
    if (valid) { _drainedSince.reset(); }
    else if (_armed && !_drainedSince) { _drainedSince = Clock::now(); }
}

void Registry::Run_()
{
    SetThreadName("ncs:refs");
    std::unique_lock lock(_mutex);
    while (!_stopping)
    {
        _wake.wait_for(lock, _scanInterval, [this] { return _stopping || _poked; });
        if (_stopping) { break; }
        _poked = false;
        lock.unlock();
        try
        {
            Scan_();
        } catch (std::exception const& e) { ERR("Service reference scan failed: {}", e.what()); }
        lock.lock();
    }
}

bool Registry::WaitUntilDrained(Clock::time_point deadline, Kinds kinds)
{
    auto nextCensus = Clock::now() + _scanInterval;
    while (true)
    {
        uint64_t seen = 0;
        {
            std::scoped_lock lock(_mutex);
            seen = _releases;
        }
        if (!Walk_(kinds, std::nullopt)) { return true; }
        auto const now = Clock::now();
        if (now >= deadline)
        {
            WARN("Service references still held at the drain deadline: {}", Census());
            return false;
        }
        if (now >= nextCensus)
        {
            INFO("Waiting for service references: {}", Census());
            nextCensus = now + _scanInterval;
        }
        std::unique_lock lock(_mutex);
        _changed.wait_until(lock, std::min(deadline, now + _scanInterval), [&] { return _releases != seen || _stopping; });
        if (_stopping) { return false; }
    }
}

Clock::duration Registry::DrainedFor()
{
    std::scoped_lock lock(_mutex);
    return _drainedSince ? Clock::now() - *_drainedSince : Clock::duration::zero();
}

std::string Registry::Census()
{
    std::vector<std::string>                               parts;
    std::vector<std::shared_ptr<SessionReferences::State>> sessions;
    {
        std::scoped_lock lock(_mutex);
        for (auto const& [context, count] : _selfCounts)
        {
            if (count > 0) { parts.push_back(fmt::format("self:{}x{}", context, count)); }
        }
        for (auto const& [key, entry] : _pids) { parts.push_back(fmt::format("pid:{}@{}x{}", key.first, key.second, entry.count)); }
        for (auto const& [key, entry] : _remotes)
        {
            parts.push_back(fmt::format("peer:{}@{}x{} failures={}", key.first, key.second, entry.count, entry.failures));
        }
        sessions = _sessions;
    }
    for (auto const& state : sessions)
    {
        auto reason = PendingReason(*state);
        if (!reason.empty()) { parts.push_back(fmt::format("session:{} {}", state->session, reason)); }
    }
    return parts.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(parts, ", "));
}

std::shared_ptr<void> AcquireClientReference(uint16_t port, cipc::SessionId session)
{
    auto const pid     = CurrentPid();
    auto       address = fmt::format("localhost:{}", port);
    auto       target  = fmt::format("/api/reference/pid/{}/{}", pid, session);
    for (int attempt = 0; attempt < AcquireAttempts; ++attempt)
    {
        auto const status = Request(http::verb::post, address, target, {}, ProbeTimeout);
        if (status == http::status::ok) { return std::make_shared<ClientReference>(std::move(address), std::move(target), pid); }
        if (status == http::status::not_found)
        {
            fmt::print(stderr, "[native_ckpt:cli][WARN] Checkpoint service on port {} does not track service references\n", port);
            return nullptr;
        }
        std::this_thread::sleep_for(RetryInterval);
    }
    throw std::runtime_error(fmt::format("Failed to acquire a checkpoint service reference on port {}", port));
}

void WaitForService(uint16_t port, Clock::duration timeout)
{
    auto const address  = fmt::format("localhost:{}", port);
    auto const deadline = Clock::now() + timeout;
    while (Request(http::verb::get, address, "/api/ping", {}, ProbeTimeout) != http::status::ok)
    {
        if (Clock::now() >= deadline) { throw std::runtime_error(fmt::format("Checkpoint service on port {} did not become ready", port)); }
        std::this_thread::sleep_for(RetryInterval);
    }
}
}    // namespace checkpoint::service
