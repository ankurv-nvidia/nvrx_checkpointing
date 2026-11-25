#include "svcref.h"
#include "config.h"
#include "logging.h"
#include "net_utils.h"
#include "server.h"
#include "service_client.h"

#include <algorithm>
#include <boost/asio/error.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/process.hpp>
#include <boost/process/execute.hpp>
#include <boost/process/v2/ext/exe.hpp>
#include <boost/process/v2/pid.hpp>
#include <boost/system/error_code.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <atomic>
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

// A service that cannot take a reference now, because it is closing or not yet listening.
struct ServiceUnavailable : std::runtime_error
{ using std::runtime_error::runtime_error; };

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

std::string LocalAddress(uint16_t port)
{ return fmt::format("localhost:{}", port); }

std::optional<Response> PingPeer(std::string const& peer)
{ return Exchange(http::verb::get, peer, "/api/peer/ping", {}, ProbeTimeout); }
}    // namespace

static void SvcPing(uint16_t index)
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex PingMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(PingMutex);
    auto             selfLoc   = boost::dll::this_line_location();
    auto             svcExeLoc = selfLoc.parent_path() / "checkpoint_server";
    if (TryConnectToCheckpointService(LocalSvrPeer, index)) { return; }
    if (checkpoint::config::GetConfigValue<checkpoint::config::SvcDisableAutoLaunch>())
    {
        throw std::runtime_error("Externally managed checkpoint service is unavailable");
    }
    if (!checkpoint::config::GetConfigValue<checkpoint::config::ServerDisableInproc>())
    {
        EnsureInprocServer(index);
        return;
    }

    auto const deadline = SvcRef::Clock::now() + ServerWaitTimeout;
    while (true)
    {
        std::string line;
        try
        {
            boost::asio::io_context    ctx;
            boost::asio::readable_pipe output(ctx);
            boost::process::process    proc(
                ctx, svcExeLoc, {"--index", std::to_string(index)}, boost::process::process_stdio{.in = {}, .out = output, .err = stderr});
            boost::system::error_code ec;
            do
            {
                boost::asio::read_until(output, boost::asio::dynamic_buffer(line), '\n', ec);
            } while (ec == boost::asio::error::interrupted);
            proc.detach();
        } catch (std::exception const& ex)
        {
            fmt::print(stderr,
                       "[native_ckpt:cli][ERROR] Exception in SvcPing from {} svcbin={}: {}\n",
                       selfLoc.string(),
                       svcExeLoc.string(),
                       ex.what());
            throw;
        }
        // A candidate that lost the election to a closing daemon asks to be launched again.
        if (line != "RETRY\n" || SvcRef::Clock::now() >= deadline) { break; }
        std::this_thread::sleep_for(RetryInterval);
    }

    // Both the elected daemon and losing candidates write READY only after the
    // elected service answers a ping. The two short attempts inside TryConnect
    // cover a final scheduling edge without a second readiness policy here.
    if (!TryConnectToCheckpointService(LocalSvrPeer, index))
    {
        throw std::runtime_error("Failed to start or connect to checkpoint service");
    }
}

static std::atomic<uint64_t> InProcCounter{0};           // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> LocalProcCounterCli{0};     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> RemoteProcCounterCli{0};    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> LocalProcCounterSrv{0};     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> RemoteProcCounterSrv{0};    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

SvcRef::InProc::InProc()
{ InProcCounter++; }
SvcRef::InProc::~InProc()
{
    InProcCounter--;
    Instance_().Changed_();
}

SvcRef::LocalProcSrv::LocalProcSrv()
{ LocalProcCounterSrv++; }
SvcRef::LocalProcSrv::~LocalProcSrv()
{ LocalProcCounterSrv--; }
SvcRef::RemoteProcSrv::RemoteProcSrv()
{ RemoteProcCounterSrv++; }
SvcRef::RemoteProcSrv::~RemoteProcSrv()
{ RemoteProcCounterSrv--; }

SvcRef::LocalProcCli::LocalProcCli(uint16_t serverIndex) :
    pid(CurrentPid()), port(static_cast<uint16_t>(checkpoint::ipc::Message::Port + serverIndex))
{
    if (LocalProcCounterCli == 0) { SvcPing(serverIndex); }
    auto const response = Exchange(http::verb::post, LocalAddress(port), fmt::format("/api/acquirelocalref/{}", pid), {}, ProbeTimeout);
    if (response && response->status == http::status::not_found)
    {
        fmt::print(stderr, "[native_ckpt:cli][WARN] Checkpoint service on port {} does not track service references\n", port);
    }
    else
    {
        if (!response || response->status != http::status::ok)
        {
            throw ServiceUnavailable(fmt::format("Checkpoint service on port {} took no reference", port));
        }
        auto const separator = response->body.find(' ');
        auto const acquired  = ParseInteger<uint64_t>(std::string_view{response->body}.substr(0, separator));
        if (!acquired || separator == std::string::npos)
        {
            throw ServiceUnavailable(fmt::format("Checkpoint service on port {} answered {:?}", port, response->body));
        }
        token      = *acquired;
        generation = ParseInteger<uint64_t>(std::string_view{response->body}.substr(separator + 1)).value_or(0);
    }
    LocalProcCounterCli++;
}

SvcRef::LocalProcCli::~LocalProcCli()
{
    LocalProcCounterCli--;
    // A forked child inherits the handle but not the reference.
    if (token != 0 && CurrentPid() == pid)
    {
        static_cast<void>(
            Exchange(http::verb::post, LocalAddress(port), fmt::format("/api/releaselocalref/{}/{}", pid, token), {}, ReleaseTimeout));
    }
}

SvcRef::RemoteProcCli::RemoteProcCli(std::string peerIn) : peer(std::move(peerIn)), pid(CurrentPid())
{
    {
        auto&            service = Instance_();
        std::scoped_lock lock(service._mutex);
        self = service._selfAddress;
    }
    auto const deadline = Clock::now() + ServerWaitTimeout;
    while (true)
    {
        auto const response = Exchange(http::verb::post, peer, "/api/peer/acquireref", self, ProbeTimeout);
        if (response && response->status == http::status::not_found)
        {
            INFO("Coordinator {} does not track service references", peer);
            break;
        }
        auto const acquired = response && response->status == http::status::ok ? ParseInteger<uint64_t>(response->body) : std::nullopt;
        if (acquired)
        {
            token = *acquired;
            INFO("Coordinator reference acquired on {}", peer);
            break;
        }
        if (Closing()) { throw std::runtime_error("Checkpoint service is shutting down"); }
        if (Clock::now() >= deadline)
        {
            throw std::runtime_error(fmt::format("Coordinator {} did not accept a service reference within {}s",
                                                 peer,
                                                 std::chrono::duration_cast<std::chrono::seconds>(ServerWaitTimeout).count()));
        }
        std::this_thread::sleep_for(RetryInterval);
    }
    RemoteProcCounterCli++;
}

SvcRef::RemoteProcCli::~RemoteProcCli()
{
    RemoteProcCounterCli--;
    // A forked child inherits the handle but not the reference.
    if (token != 0 && CurrentPid() == pid)
    {
        static_cast<void>(Exchange(http::verb::post, peer, fmt::format("/api/peer/releaseref/{}", token), self, ReleaseTimeout));
    }
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
    std::scoped_lock lock(_mutex);
    ++_revision;
    return std::make_shared<SvcRef::InProc>();
}

std::shared_ptr<SvcRef::LocalProcCli> SvcRef::CliAcquireLocalProc_(uint16_t serverIndex)
{
    auto const deadline = Clock::now() + ServerWaitTimeout;
    while (true)
    {
        try
        {
            return std::make_shared<SvcRef::LocalProcCli>(serverIndex);
        } catch (ServiceUnavailable const&)
        {
            // The service closed after it answered, so a live one is ensured again.
            if (Clock::now() >= deadline) { throw; }
            std::this_thread::sleep_for(RetryInterval);
            SvcPing(serverIndex);
        }
    }
}

std::shared_ptr<SvcRef::RemoteProcCli> SvcRef::CliAcquireRemoteProc_(std::string peer)
{
    {
        std::scoped_lock lock(_mutex);
        if (!_linkUpstream || peer.empty() || _selfAddress.empty() || peer == _selfAddress) { return nullptr; }
    }
    return std::make_shared<SvcRef::RemoteProcCli>(std::move(peer));
}

void SvcRef::RecordSelfAddress(std::string address)
{
    auto&            self = Instance_();
    std::scoped_lock lock(self._mutex);
    if (self._selfAddress.empty()) { self._selfAddress = std::move(address); }
}

uint64_t SvcRef::SvrAcquireLocalProcRef_(uint64_t pid)
{
    uint64_t token = 0;
    {
        std::scoped_lock lock(_mutex);
        if (_closing) { return 0; }
        auto ref   = std::make_shared<SvcRef::LocalProcSrv>();
        ref->pid   = pid;
        ref->token = token = ++_token;
        _localProcSrv.push_back(std::move(ref));
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
        released = std::erase_if(_localProcSrv, [&](auto const& ref) { return ref->pid == pid && ref->token == token; });
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
        auto ref   = std::make_shared<SvcRef::RemoteProcSrv>();
        ref->peer  = peer;
        ref->token = token = ++_token;
        _remoteProcSrv.push_back(std::move(ref));
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
            = std::erase_if(_remoteProcSrv, [&](auto const& ref) { return ref->token == token && (peer.empty() || ref->peer == peer); });
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
        released = std::erase_if(self._localProcSrv, [pid](auto const& ref) { return ref->pid == pid; });
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
        self._selfAddress.clear();
        self._localProcSrv.clear();
        self._remoteProcSrv.clear();
        ++self._revision;
    }
    self._changed.notify_all();
}

bool SvcRef::Live_()
{
    if (InProcCounter.load() != 0) { return true; }
    std::vector<std::shared_ptr<LocalProcSrv>>  processes;
    std::vector<std::shared_ptr<RemoteProcSrv>> peers;
    {
        std::scoped_lock lock(_mutex);
        processes.assign(_localProcSrv.begin(), _localProcSrv.end());
        peers.assign(_remoteProcSrv.begin(), _remoteProcSrv.end());
    }
    for (auto const& process : processes)
    {
        if (ProcessAlive(process->pid)) { return true; }
        size_t reaped = 0;
        {
            std::scoped_lock lock(_mutex);
            reaped = std::erase(_localProcSrv, process);
            if (reaped != 0) { ++_revision; }
        }
        if (reaped != 0) { INFO("Service reference of exited pid={} reaped", process->pid); }
    }
    for (auto const& peer : peers)
    {
        auto const response = PingPeer(peer->peer);
        bool const answered = response && response->status == http::status::ok;
        bool       changed  = false;
        {
            std::scoped_lock lock(_mutex);
            changed         = peer->answering != answered;
            peer->answering = answered;
        }
        if (changed && answered) { INFO("Peer {} answers again", peer->peer); }
        else if (changed) { WARN("Peer {} does not answer; its reference does not count until it does", peer->peer); }
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
        if (InProcCounter.load() == 0 && self._localProcSrv.empty() && !self._remoteProcSrv.empty() && current >= nextCensus)
        {
            std::vector<std::string> peers;
            for (auto const& ref : self._remoteProcSrv) { peers.push_back(ref->answering ? ref->peer : ref->peer + " unanswered"); }
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
{ return InProcCounter.load() != 0; }

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
{ return PingPeer(LocalAddress(port)).has_value(); }

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

SvcRef& SvcRef::Instance_()
{
    // Its ok all local servers can share the same svcref
    // They all spin down together
    static auto* Instance = new SvcRef();    // NOLINT(cppcoreguidelines-owning-memory)
    return *Instance;
}
