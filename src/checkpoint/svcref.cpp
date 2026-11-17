#include "svcref.h"
#include "CommonMacros.h"

#include "config.h"
#include "logging.h"
#include "net_utils.h"
#include "server.h"

#include <boost/asio/error.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/process.hpp>
#include <boost/process/v2/ext/exe.hpp>
#include <boost/process/v2/pid.hpp>
#include <boost/system/error_code.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <charconv>
#include <exception>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
namespace http = boost::beast::http;

namespace
{
constexpr auto ProbeTimeout   = 2s;
constexpr auto ReleaseTimeout = 1s;
constexpr auto RetryInterval  = 200ms;

struct Response
{
    http::status status{};
    std::string  body;
};

std::optional<Response>
Exchange(http::verb method, std::string const& address, std::string const& target, std::string_view peer, SvcRef::Clock::duration timeout)
{
    try
    {
        boost::asio::io_context ioc;
        auto [endpoint, results] = ResolveEndpoint(ioc, address);
        boost::beast::tcp_stream         stream(ioc);
        http::request<http::string_body> request{method, target, HttpVersion11};
        request.set(http::field::host, fmt::format("{}:{}", endpoint.host(), endpoint.port()));
        if (!peer.empty()) { request.set(SvcRef::PeerHeader, peer); }
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
        return Response{.status = response.result(), .body = std::move(response.body())};
    } catch (std::exception const&) { return std::nullopt; }
}

template <typename TValue> std::optional<TValue> ParseInteger(std::string_view text)
{
    TValue     value{};
    auto const parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) { return std::nullopt; }
    return value;
}

uint64_t NewGeneration()
{
    std::random_device device;
    auto const         random = (static_cast<uint64_t>(device()) << 32U) | device();
    auto const         value  = random ^ static_cast<uint64_t>(SvcRef::Clock::now().time_since_epoch().count());
    return value == 0 ? 1 : value;
}

std::optional<Response> PingPeer(std::string const& peer)
{ return Exchange(http::verb::get, peer, "/api/peer/ping", {}, ProbeTimeout); }

// True when a service that is not closing answers on the port; a second short attempt covers a listener that is just starting.
bool Serves(uint16_t port)
{
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        if (attempt != 0) { std::this_thread::sleep_for(100ms); }
        auto const response = Exchange(http::verb::get, fmt::format("localhost:{}", port), "/api/ping", {}, ProbeTimeout);
        if (response) { return response->status == http::status::ok; }
    }
    return false;
}

std::shared_ptr<SvcRef::LocalProcCli> TryAcquireLocalProc(uint16_t port)
{
    auto const pid      = SvcRef::CurrentPid();
    auto       address  = fmt::format("localhost:{}", port);
    auto const response = Exchange(http::verb::post, address, fmt::format("/api/acquirelocalref/{}", pid), {}, ProbeTimeout);
    if (!response) { return nullptr; }
    if (response->status == http::status::not_found)
    {
        fmt::print(stderr, "[native_ckpt:cli][WARN] Checkpoint service on port {} does not track service references\n", port);
        return std::make_shared<SvcRef::LocalProcCli>(SvcRef::LocalProcCli{.pid = pid});
    }
    if (response->status != http::status::ok) { return nullptr; }
    auto const separator = response->body.find(' ');
    auto const token     = ParseInteger<uint64_t>(std::string_view{response->body}.substr(0, separator));
    if (!token || separator == std::string::npos) { return nullptr; }
    auto const generation = ParseInteger<uint64_t>(std::string_view{response->body}.substr(separator + 1)).value_or(0);
    return {new SvcRef::LocalProcCli{.pid = pid, .token = *token, .generation = generation},    // NOLINT(cppcoreguidelines-owning-memory)
            [service = std::move(address)](SvcRef::LocalProcCli* held) {
                // A forked child inherits the handle but not the reference.
                if (SvcRef::CurrentPid() == held->pid)
                {
                    static_cast<void>(Exchange(
                        http::verb::post, service, fmt::format("/api/releaselocalref/{}/{}", held->pid, held->token), {}, ReleaseTimeout));
                }
                delete held;    // NOLINT(cppcoreguidelines-owning-memory)
            }};
}
}    // namespace

SvcRef& SvcRef::Instance_()
{
    // Its ok all local servers can share the same svcref
    // They all spin down together
    static auto* Instance = new SvcRef();    // NOLINT(cppcoreguidelines-owning-memory)
    return *Instance;
}

void SvcRef::Changed_()
{
    {
        std::scoped_lock lock(_mutex);
        ++_revision;
    }
    _changed.notify_all();
}

std::shared_ptr<SvcRef::InProc> SvcRef::AcquireInProc_()
{
    auto reference = std::shared_ptr<InProc>(new InProc{}, [](InProc* held) {
        delete held;    // NOLINT(cppcoreguidelines-owning-memory)
        Instance_().Changed_();
    });
    {
        std::scoped_lock lock(_mutex);
        std::erase_if(_inProc, [](auto const& item) { return item.expired(); });
        _inProc.emplace_back(reference);
        ++_revision;
    }
    return reference;
}

std::shared_ptr<SvcRef::LocalProcCli> SvcRef::CliAcquireLocalProc_(uint16_t serverIndex)
{
    auto const port     = static_cast<uint16_t>(checkpoint::ipc::Message::Port + serverIndex);
    auto const deadline = Clock::now() + ServerWaitTimeout;
    while (true)
    {
        CliEnsureLocalProc(serverIndex);
        if (auto reference = TryAcquireLocalProc(port))
        {
            std::scoped_lock lock(_mutex);
            std::erase_if(_localProcCli, [](auto const& item) { return item.expired(); });
            _localProcCli.emplace_back(reference);
            return reference;
        }
        // The service closed after it answered, so a live one is ensured again.
        if (Clock::now() >= deadline)
        {
            throw std::runtime_error(fmt::format("Failed to acquire a checkpoint service reference on port {}", port));
        }
        std::this_thread::sleep_for(RetryInterval);
    }
}

void SvcRef::CliEnsureLocalProc(uint16_t serverIndex)
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex EnsureMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(EnsureMutex);
    auto const       port = static_cast<uint16_t>(checkpoint::ipc::Message::Port + serverIndex);
    if (Serves(port)) { return; }
    if (checkpoint::config::GetConfigValue<checkpoint::config::SvcDisableAutoLaunch>())
    {
        throw std::runtime_error("Externally managed checkpoint service is unavailable");
    }
    if (!checkpoint::config::GetConfigValue<checkpoint::config::ServerDisableInproc>())
    {
        EnsureInprocServer(serverIndex);
        return;
    }

    auto const selfLoc   = boost::dll::this_line_location();
    auto const svcExeLoc = selfLoc.parent_path() / "checkpoint_server";
    auto const deadline  = Clock::now() + ServerWaitTimeout;
    while (true)
    {
        std::string line;
        try
        {
            boost::asio::io_context    ctx;
            boost::asio::readable_pipe output(ctx);
            boost::process::process    proc(ctx,
                                            svcExeLoc,
                                            {"--index", std::to_string(serverIndex)},
                                            boost::process::process_stdio{.in = {}, .out = output, .err = stderr});
            boost::system::error_code  ec;
            do
            {
                boost::asio::read_until(output, boost::asio::dynamic_buffer(line), '\n', ec);
            } while (ec == boost::asio::error::interrupted);
            proc.detach();
        } catch (std::exception const& ex)
        {
            fmt::print(stderr,
                       "[native_ckpt:cli][ERROR] Exception launching the checkpoint service from {} svcbin={}: {}\n",
                       selfLoc.string(),
                       svcExeLoc.string(),
                       ex.what());
            throw;
        }
        // A candidate that lost the election to a closing daemon asks to be launched again.
        if (line != "RETRY\n" || Clock::now() >= deadline) { break; }
        std::this_thread::sleep_for(RetryInterval);
    }

    // Both the elected daemon and losing candidates write READY only after the elected service answers a ping.
    if (!Serves(port)) { throw std::runtime_error("Failed to start or connect to checkpoint service"); }
}

std::shared_ptr<SvcRef::RemoteProcCli> SvcRef::CliAcquireRemoteProc_(std::string peer, std::string const& address, Clock::duration timeout)
{
    {
        std::scoped_lock lock(_mutex);
        if (!_linkUpstream || peer.empty() || peer == address) { return nullptr; }
    }
    auto const deadline = Clock::now() + timeout;
    while (true)
    {
        auto const response = Exchange(http::verb::post, peer, "/api/peer/acquireref", address, ProbeTimeout);
        auto const token    = response && response->status == http::status::ok ? ParseInteger<uint64_t>(response->body) : std::nullopt;
        if (token)
        {
            INFO("Coordinator reference acquired on {}", peer);
            auto reference = std::shared_ptr<RemoteProcCli>(
                new RemoteProcCli{.token = *token, .peer = std::move(peer)},    // NOLINT(cppcoreguidelines-owning-memory)
                [address, creator = CurrentPid()](RemoteProcCli* held) {
                    // A forked child inherits the handle but not the reference.
                    if (CurrentPid() == creator)
                    {
                        static_cast<void>(Exchange(
                            http::verb::post, held->peer, fmt::format("/api/peer/releaseref/{}", held->token), address, ReleaseTimeout));
                    }
                    delete held;    // NOLINT(cppcoreguidelines-owning-memory)
                });
            std::scoped_lock lock(_mutex);
            std::erase_if(_remoteProcCli, [](auto const& item) { return item.expired(); });
            _remoteProcCli.emplace_back(reference);
            return reference;
        }
        if (response && response->status == http::status::not_found)
        {
            INFO("Coordinator {} does not track service references", peer);
            return nullptr;
        }
        if (Closing()) { throw std::runtime_error("Checkpoint service is shutting down"); }
        if (Clock::now() >= deadline)
        {
            throw std::runtime_error(fmt::format("Coordinator {} did not accept a service reference within {}s",
                                                 peer,
                                                 std::chrono::duration_cast<std::chrono::seconds>(timeout).count()));
        }
        std::this_thread::sleep_for(RetryInterval);
    }
}

uint64_t SvcRef::SvrAcquireLocalProcRef_(uint64_t pid)
{
    uint64_t token = 0;
    {
        std::scoped_lock lock(_mutex);
        if (_closing) { return 0; }
        token = ++_token;
        _localProcSrv.push_back({.pid = pid, .token = token});
        _owned = true;
        ++_revision;
    }
    _changed.notify_all();
    INFO("Service reference acquired by pid={}", pid);
    return token;
}

void SvcRef::SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token)
{
    size_t released = 0;
    {
        std::scoped_lock lock(_mutex);
        released = std::erase_if(_localProcSrv, [&](auto const& item) { return item.pid == pid && item.token == token; });
        if (released != 0) { ++_revision; }
    }
    _changed.notify_all();
    if (released != 0) { INFO("Service reference released by pid={}", pid); }
}

uint64_t SvcRef::SvrAcquireRemoteProcRef_(std::string peer)
{
    uint64_t token = 0;
    {
        std::scoped_lock lock(_mutex);
        if (_closing) { return 0; }
        token = ++_token;
        _remoteProcSrv.push_back({.token = token, .peer = peer});
        _owned = true;
        ++_revision;
    }
    _changed.notify_all();
    INFO("Service reference acquired by peer={}", peer);
    return token;
}

void SvcRef::SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token)
{
    size_t released = 0;
    {
        std::scoped_lock lock(_mutex);
        released
            = std::erase_if(_remoteProcSrv, [&](auto const& item) { return item.token == token && (peer.empty() || item.peer == peer); });
        if (released != 0) { ++_revision; }
    }
    _changed.notify_all();
    if (released != 0) { INFO("Service reference released by peer={}", peer); }
}

void SvcRef::SvrReleaseLocalProcRefs(uint64_t pid)
{
    auto&  self     = Instance_();
    size_t released = 0;
    {
        std::scoped_lock lock(self._mutex);
        released = std::erase_if(self._localProcSrv, [pid](auto const& item) { return item.pid == pid; });
        if (released != 0) { ++self._revision; }
    }
    self._changed.notify_all();
    if (released != 0) { INFO("Released {} service reference(s) held by exiting pid={}", released, pid); }
}

void SvcRef::BeginService(Clock::duration startupGrace, Clock::duration scanInterval)
{
    auto& self = Instance_();
    {
        std::scoped_lock lock(self._mutex);
        self._startupGrace = startupGrace;
        self._scanInterval = scanInterval;
        self._linkUpstream = true;
        if (self._serving) { return; }
        self._serving    = true;
        self._owned      = false;
        self._closing    = false;
        self._woken      = false;
        self._generation = NewGeneration();
        self._localProcSrv.clear();
        self._remoteProcSrv.clear();
        ++self._revision;
    }
    self._changed.notify_all();
}

bool SvcRef::Live_()
{
    std::vector<LocalProcSrv>  processes;
    std::vector<RemoteProcSrv> peers;
    {
        std::scoped_lock lock(_mutex);
        std::erase_if(_inProc, [](auto const& item) { return item.expired(); });
        if (!_inProc.empty()) { return true; }
        processes.assign(_localProcSrv.begin(), _localProcSrv.end());
        peers.assign(_remoteProcSrv.begin(), _remoteProcSrv.end());
    }
    for (auto const& process : processes)
    {
        if (ProcessAlive(process.pid)) { return true; }
        size_t reaped = 0;
        {
            std::scoped_lock lock(_mutex);
            reaped = std::erase_if(_localProcSrv, [&](auto const& item) { return item.token == process.token; });
            if (reaped != 0) { ++_revision; }
        }
        if (reaped != 0) { INFO("Service reference of exited pid={} reaped", process.pid); }
    }
    for (auto const& peer : peers)
    {
        auto const response = PingPeer(peer.peer);
        bool const answered = response && response->status == http::status::ok;
        bool       changed  = false;
        {
            std::scoped_lock lock(_mutex);
            for (auto& item : _remoteProcSrv)
            {
                if (item.token != peer.token || item.answering == answered) { continue; }
                item.answering = answered;
                changed        = true;
            }
        }
        if (changed && answered) { INFO("Peer {} answers again", peer.peer); }
        else if (changed) { WARN("Peer {} does not answer; its reference does not count until it does", peer.peer); }
        if (answered) { return true; }
    }
    return false;
}

void SvcRef::WaitUntilIdle(std::atomic<bool> const& stopRequested)
{
    auto&             self = Instance_();
    auto const        now  = Clock::now();
    Clock::time_point graceEnd;
    {
        std::scoped_lock lock(self._mutex);
        graceEnd = now + self._startupGrace;
    }
    auto nextCensus = now;
    while (true)
    {
        uint64_t revision = 0;
        {
            std::scoped_lock lock(self._mutex);
            revision = self._revision;
        }
        bool const       live = !stopRequested && self.Live_();
        std::unique_lock lock(self._mutex);
        auto const       current   = Clock::now();
        bool const       graceOver = self._owned || current >= graceEnd;
        if (stopRequested || (!live && graceOver && revision == self._revision))
        {
            self._closing = true;
            self._serving = false;
            if (!stopRequested) { INFO("No service reference remains; the checkpoint service is winding down"); }
            return;
        }
        if (self._inProc.empty() && self._localProcSrv.empty() && !self._remoteProcSrv.empty() && current >= nextCensus)
        {
            std::vector<std::string> peers;
            for (auto const& item : self._remoteProcSrv) { peers.push_back(item.answering ? item.peer : item.peer + " unanswered"); }
            lock.unlock();
            INFO("Waiting for peer references: {}", fmt::join(peers, ", "));
            lock.lock();
            nextCensus = current + self._scanInterval;
        }
        auto const wakeAt = graceOver ? current + self._scanInterval : std::min(current + self._scanInterval, graceEnd);
        self._changed.wait_until(lock, wakeAt, [&] { return self._revision != revision || self._woken; });
        self._woken = false;
    }
}

bool SvcRef::Closing()
{
    auto&            self = Instance_();
    std::scoped_lock lock(self._mutex);
    return self._closing;
}

bool SvcRef::HasInProc()
{
    auto&            self = Instance_();
    std::scoped_lock lock(self._mutex);
    std::erase_if(self._inProc, [](auto const& item) { return item.expired(); });
    return !self._inProc.empty();
}

uint64_t SvcRef::Generation()
{
    auto&            self = Instance_();
    std::scoped_lock lock(self._mutex);
    return self._generation;
}

void SvcRef::Wake()
{
    auto& self = Instance_();
    {
        std::scoped_lock lock(self._mutex);
        self._woken = true;
    }
    self._changed.notify_all();
}

uint64_t SvcRef::CurrentPid()
{ return static_cast<uint64_t>(boost::process::v2::current_pid()); }

bool SvcRef::Answers(uint16_t port)
{ return PingPeer(fmt::format("localhost:{}", port)).has_value(); }

bool SvcRef::ProcessAlive(uint64_t pid)
{
    if (pid == 0) { return false; }
    boost::system::error_code error;
    static_cast<void>(boost::process::v2::ext::exe(static_cast<boost::process::v2::pid_type>(pid), error));
    // Only a definite lookup failure counts as exited; a zombie has no executable and reads as exited.
    return !(error == boost::system::errc::no_such_file_or_directory || error == boost::system::errc::no_such_process
             || error == boost::system::errc::invalid_argument);
}

SvcRef::Clock::duration SvcRef::ConfiguredScanInterval()
{ return std::chrono::seconds{checkpoint::config::GetConfigValue<checkpoint::config::ServerReferenceScanIntervalSeconds>()}; }
