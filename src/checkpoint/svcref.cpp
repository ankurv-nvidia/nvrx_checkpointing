#include "svcref.h"

#include "net_utils.h"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_future.hpp>
#include <boost/process/v2/ext/exe.hpp>
#include <boost/process/v2/pid.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace
{
using namespace std::chrono_literals;
namespace http = boost::beast::http;
using Response = http::response<http::string_body>;

uint64_t Pid()
{ return static_cast<uint64_t>(boost::process::v2::current_pid()); }

uint64_t Generation()
{
    std::random_device random;
    auto const         value = (static_cast<uint64_t>(random()) << 32U) ^ random()
                               ^ static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return value == 0 ? 1 : value;
}

std::string LocalAddress(uint16_t index)
{ return fmt::format("localhost:{}", checkpoint::ipc::Message::Port + index); }

Response
Request(std::string const& peer, std::string const& target, std::string body = {}, uint64_t token = 0, std::chrono::seconds timeout = 5s)
{
    boost::asio::io_context ioc;
    auto                    result = boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<Response> {
            auto const [address, endpoints] = ResolveEndpoint(ioc, peer);
            boost::beast::tcp_stream stream(ioc);
            stream.expires_after(timeout);
            co_await stream.async_connect(endpoints, boost::asio::use_awaitable);
            http::request<http::string_body> request{http::verb::post, target, HttpVersion11};
            request.set(http::field::host, address.host());
            if (token != 0) { request.set("X-Checkpoint-Token", std::to_string(token)); }
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

bool ProcessAlive(uint64_t pid)
{
    boost::system::error_code error;
    static_cast<void>(boost::process::v2::ext::exe(static_cast<boost::process::v2::pid_type>(pid), error));
    return error != boost::system::errc::no_such_file_or_directory && error != boost::system::errc::no_such_process
           && error != boost::system::errc::invalid_argument;
}

bool Matches(uint32_t reference, uint32_t session)
{ return session == SvcRef::AllSessions || reference == session || reference == SvcRef::AllSessions; }
}    // namespace

struct SvcRef::Instances
{
    uint64_t const                              creator{Pid()};
    std::mutex                                  mutex;
    std::map<uint16_t, std::unique_ptr<SvcRef>> values;
};

SvcRef::Instances& SvcRef::Instances_()
{
    static auto* instances = new Instances;    // NOLINT(cppcoreguidelines-owning-memory)
    return *instances;
}

SvcRef& SvcRef::Instance_(uint16_t index)
{
    auto&            instances = Instances_();
    std::scoped_lock lock(instances.mutex);
    auto&            instance = instances.values[index];
    if (!instance) { instance.reset(new SvcRef(index)); }    // NOLINT(cppcoreguidelines-owning-memory)
    return *instance;
}

SvcRef::SvcRef(uint16_t index) : _index(index), _generation(Generation())
{}

uint64_t SvcRef::Token_()
{
    std::scoped_lock lock(_mutex);
    auto             value = _generation ^ (static_cast<uint64_t>(++_token) << 32U);
    if (value == 0) { value = _generation ^ (static_cast<uint64_t>(++_token) << 32U); }
    return value;
}

void SvcRef::CheckOpen_()
{
    std::scoped_lock lock(_mutex);
    if (_closing) { throw boost::system::system_error(boost::asio::error::try_again, "Checkpoint service is shutting down"); }
}

std::shared_ptr<SvcRef::InProc> SvcRef::AcquireInProc_(std::string_view context, uint32_t session, bool control)
{
    auto const       creator   = Pid();
    auto             reference = std::shared_ptr<InProc>(new InProc{std::string(context), session}, [this, creator](InProc* value) {
        delete value;    // NOLINT(cppcoreguidelines-owning-memory)
        if (creator != Pid()) { return; }
        std::scoped_lock released(_mutex);
        ++_revision;
        _changed.notify_all();
    });
    std::scoped_lock lock(_mutex);
    if (_sealed && control) { return {}; }
    if (_closing && !control) { throw std::runtime_error("Checkpoint service is shutting down"); }
    _inProc.push_back(reference);
    ++_revision;
    _changed.notify_all();
    return reference;
}

SvcRef::LocalProcCli::~LocalProcCli()
{ Close(); }
void SvcRef::LocalProcCli::Close() noexcept
{
    if (pid != Pid()) { return; }
    uint64_t released{};
    {
        std::scoped_lock lock(_mutex);
        released = std::exchange(token, 0);
    }
    if (released != 0) { Instance_(_index).Release_(LocalAddress(_index), fmt::format("/api/releaselocalref/{}/{}", pid, released)); }
}

SvcRef::RemoteProcCli::~RemoteProcCli()
{ Close(); }
void SvcRef::RemoteProcCli::Close() noexcept
{
    if (_pid != Pid()) { return; }
    uint64_t released{};
    {
        std::scoped_lock lock(_mutex);
        released = std::exchange(token, 0);
    }
    if (released != 0) { Instance_(_index).Release_(peer, fmt::format("/api/peer/releaseref/{}", released), _address); }
}

void SvcRef::Release_(std::string peer, std::string target, std::string body) noexcept
{
    try
    {
        static_cast<void>(Request(peer, target, body));
        return;
    } catch (std::exception const&) {}
    try
    {
        std::scoped_lock lock(_mutex);
        _pending.emplace_back(std::move(peer), std::move(target), std::move(body));
    } catch (std::exception const&) {}
}

void SvcRef::RetryReleases_()
{
    decltype(_pending) pending;
    {
        std::scoped_lock lock(_mutex);
        pending.swap(_pending);
    }
    for (auto& [peer, target, body] : pending) { Release_(std::move(peer), std::move(target), std::move(body)); }
}

std::shared_ptr<SvcRef::LocalProcCli> SvcRef::CliAcquireLocalProc_(uint32_t session, std::string coordinator, std::string address)
{
    auto reference        = std::make_shared<LocalProcCli>();
    reference->pid        = Pid();
    reference->token      = Token_();
    reference->_index     = _index;
    reference->_session   = session;
    auto const response   = Request(LocalAddress(_index),
                                    fmt::format("/api/acquirelocalref/{}", reference->pid),
                                    fmt::format("{}\n{}\n{}", session, coordinator, address),
                                    reference->token,
                                    900s);
    reference->generation = std::stoull(std::string(response["X-Checkpoint-Generation"]));
    {
        std::scoped_lock lock(_mutex);
        _localProcCli.push_back(reference);
        if (_exitGenerations.insert(reference->generation).second && std::atexit(CloseLocalProcClients) != 0)
        {
            throw std::runtime_error("Cannot register checkpoint client cleanup");
        }
    }
    return reference;
}

std::shared_ptr<SvcRef::RemoteProcCli> SvcRef::CliAcquireRemoteProc_(std::string peer, std::string address, uint32_t session, bool managed)
{
    if (peer.empty() || peer == address) { return {}; }
    auto reference      = std::make_shared<RemoteProcCli>();
    reference->token    = Token_();
    reference->peer     = std::move(peer);
    reference->_pid     = Pid();
    reference->_index   = _index;
    reference->_session = session;
    reference->_address = std::move(address);
    reference->_managed = managed;
    auto const body     = fmt::format("{}\n{}\n{}", session, reference->_address, SvrGeneration(_index));
    auto const deadline = Clock::now() + 900s;
    while (true)
    {
        CheckOpen_();
        try
        {
            static_cast<void>(Request(reference->peer, "/api/peer/acquireref", body, reference->token));
            CheckOpen_();
            break;
        } catch (std::exception const&)
        {
            CheckOpen_();
            if (Clock::now() >= deadline) { throw; }
            std::this_thread::sleep_for(100ms);
        }
    }
    {
        std::scoped_lock lock(_mutex);
        _remoteProcCli.push_back(reference);
    }
    return reference;
}

uint64_t SvcRef::SvrAcquireLocalProcRef_(uint64_t pid, uint32_t session, std::string coordinator, std::string address, uint64_t token)
{
    if (token == 0) { token = Token_(); }
    std::shared_ptr<Link> link;
    {
        std::scoped_lock lock(_mutex);
        if (_closing) { throw boost::system::system_error(boost::asio::error::try_again, "Checkpoint service is shutting down"); }
        if (_released.contains(token)) { throw std::runtime_error("Reference is already released"); }
        if (std::ranges::none_of(_localProcSrv, [token](auto const& value) { return value.token == token; }))
        {
            _localProcSrv.push_back({pid, token, session});
        }
        _owned = true;
        ++_revision;
        auto& entry = _links[session];
        if (!entry) { entry = std::make_shared<Link>(); }
        link = entry;
        _changed.notify_all();
    }
    try
    {
        if (!coordinator.empty() && coordinator != address)
        {
            std::scoped_lock lock(link->mutex);
            if (!link->reference) { link->reference = CliAcquireRemoteProc_(std::move(coordinator), std::move(address), session, true); }
        }
    } catch (...)
    {
        SvrReleaseLocalProcRef_(pid, token);
        throw;
    }
    return token;
}

void SvcRef::SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token)
{
    std::scoped_lock lock(_mutex);
    std::erase_if(_localProcSrv, [pid, token](auto const& reference) { return reference.pid == pid && reference.token == token; });
    _released.insert(token);
    ++_revision;
    _changed.notify_all();
}

uint64_t SvcRef::SvrAcquireRemoteProcRef_(std::string peer, uint64_t generation, uint32_t session, uint64_t token)
{
    if (generation == 0) { generation = std::stoull(Request(peer, "/api/ping").body()); }
    if (token == 0) { token = Token_(); }
    std::scoped_lock lock(_mutex);
    if (_closing) { throw boost::system::system_error(boost::asio::error::try_again, "Checkpoint service is shutting down"); }
    if (_released.contains(token)) { throw std::runtime_error("Reference is already released"); }
    if (std::ranges::none_of(_remoteProcSrv, [token](auto const& value) { return value.token == token; }))
    {
        _remoteProcSrv.push_back({token, std::move(peer), generation, session});
    }
    _owned = true;
    ++_revision;
    _changed.notify_all();
    return token;
}

void SvcRef::SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token)
{
    std::scoped_lock lock(_mutex);
    std::erase_if(_remoteProcSrv, [peer, token](auto const& reference) { return reference.peer == peer && reference.token == token; });
    _released.insert(token);
    ++_revision;
    _changed.notify_all();
}

bool SvcRef::HasReferences_(uint32_t session)
{
    decltype(_inProc)        inProc;
    decltype(_localProcCli)  clients;
    decltype(_remoteProcCli) peers;
    decltype(_localProcSrv)  local;
    decltype(_remoteProcSrv) remote;
    uint64_t                 generation{};
    {
        std::scoped_lock lock(_mutex);
        std::erase_if(_inProc, [](auto const& reference) { return reference.expired(); });
        std::erase_if(_localProcCli, [](auto const& reference) { return reference.expired(); });
        std::erase_if(_remoteProcCli, [](auto const& reference) { return reference.expired(); });
        inProc     = _inProc;
        clients    = _localProcCli;
        peers      = _remoteProcCli;
        local      = _localProcSrv;
        remote     = _remoteProcSrv;
        generation = _generation;
    }
    for (auto const& weak : inProc)
    {
        if (auto reference = weak.lock(); reference && Matches(reference->session, session)) { return true; }
    }
    for (auto const& reference : local)
    {
        if (!Matches(reference.session, session)) { continue; }
        if (ProcessAlive(reference.pid)) { return true; }
        std::scoped_lock lock(_mutex);
        std::erase_if(_localProcSrv, [&](auto const& value) { return value.token == reference.token; });
        ++_revision;
    }
    for (auto const& weak : clients)
    {
        if (auto reference = weak.lock())
        {
            std::scoped_lock lock(reference->_mutex);
            if (reference->token != 0 && reference->generation == generation && Matches(reference->_session, session)) { return true; }
        }
    }
    for (auto const& reference : remote)
    {
        if (!Matches(reference.session, session)) { continue; }
        try
        {
            if (std::stoull(Request(reference.peer, "/api/ping").body()) == reference.generation) { return true; }
        } catch (std::exception const&) {}
    }
    for (auto const& weak : peers)
    {
        if (auto reference = weak.lock())
        {
            std::scoped_lock lock(reference->_mutex);
            if (reference->token != 0 && !reference->_managed && Matches(reference->_session, session)) { return true; }
        }
    }
    return false;
}

void SvcRef::ReleaseRemoteClients_(bool all, Busy const& busy, std::function<bool()> const& drained)
{
    decltype(_links) links;
    {
        std::scoped_lock lock(_mutex);
        links = _links;
    }
    for (auto const& [session, link] : links)
    {
        std::unique_lock lock(link->mutex, std::defer_lock);
        if (all) { lock.lock(); }
        else if (!lock.try_lock()) { continue; }
        if (!link->reference) { continue; }
        if (!all && (HasReferences_(session) || (busy && busy(session)))) { continue; }
        if (!all && drained && (!drained() || HasReferences_(session) || (busy && busy(session)))) { continue; }
        link->reference->Close();
        link->reference.reset();
    }
    RetryReleases_();
}

void SvcRef::SvrStart(uint16_t index)
{
    auto& state = Instance_(index);
    state.ReleaseRemoteClients_(true);
    std::scoped_lock lock(state._mutex);
    state._generation = Generation();
    state._started    = Clock::now();
    state._localProcSrv.clear();
    state._remoteProcSrv.clear();
    state._owned   = false;
    state._closing = false;
    state._sealed  = false;
    ++state._revision;
}

uint64_t SvcRef::SvrGeneration(uint16_t index)
{
    auto&            state = Instance_(index);
    std::scoped_lock lock(state._mutex);
    return state._generation;
}

bool SvcRef::SvrClosing(uint16_t index)
{
    auto&            state = Instance_(index);
    std::scoped_lock lock(state._mutex);
    return state._closing;
}

void SvcRef::WaitForShutdown(uint16_t index, std::atomic<bool> const& stopRequested, Busy const& busy, std::function<bool()> const& drained)
{
    auto& state = Instance_(index);
    while (true)
    {
        uint64_t revision{};
        {
            std::scoped_lock lock(state._mutex);
            revision = state._revision;
        }
        bool const live = (busy && busy(AllSessions)) || state.HasReferences_(AllSessions);
        {
            std::unique_lock lock(state._mutex);
            if (stopRequested || (!live && revision == state._revision && (state._owned || Clock::now() >= state._started + 5s)))
            {
                state._closing = true;
                return;
            }
        }
        state.ReleaseRemoteClients_(false, busy, drained);
        std::unique_lock lock(state._mutex);
        state._changed.wait_for(lock, 1s, [&] { return revision != state._revision || stopRequested; });
    }
}

void SvcRef::DrainRequests(uint16_t index)
{
    auto&            state = Instance_(index);
    std::unique_lock lock(state._mutex);
    state._closing = true;
    state._changed.wait(lock,
                        [&] { return std::ranges::none_of(state._inProc, [](auto const& reference) { return !reference.expired(); }); });
}

void SvcRef::Seal(uint16_t index)
{
    auto&            state = Instance_(index);
    std::scoped_lock lock(state._mutex);
    state._sealed = true;
}

void SvcRef::CloseLocalProcClients()
{
    auto& instances = Instances_();
    if (instances.creator != Pid()) { return; }
    std::vector<SvcRef*> states;
    {
        std::scoped_lock lock(instances.mutex);
        for (auto const& [index, state] : instances.values) { states.push_back(state.get()); }
    }
    for (auto* state : states)
    {
        decltype(_localProcCli) clients;
        {
            std::scoped_lock lock(state->_mutex);
            clients = state->_localProcCli;
        }
        for (auto const& weak : clients)
        {
            if (auto reference = weak.lock()) { reference->Close(); }
        }
        state->RetryReleases_();
    }
}
