#include "CommonMacros.h"
#include "SetThreadName.h"

#include "byte_storage.h"
#include "config.h"
#include "logging.h"
#include "messages.h"
#include "net_utils.h"
#include "numa_utils.h"
#include "serdes_utils.h"
#include "server.h"
#include "session.h"
#include "shm_ring_buffer.h"
#include "state.h"
#include "stateserdes.h"
#include "telemetry.h"
#include "tensor.h"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/buffer_body.hpp>
#include <boost/beast/http/empty_body.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/beast/http/file_body.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/counter.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/url.hpp>
#include <hwloc.h>

#include <latch>

using namespace std::chrono_literals;
namespace bip  = boost::interprocess;
namespace cipc = checkpoint::ipc;
namespace tel  = checkpoint::telemetry;

using PerfClock = std::chrono::system_clock;
using Request   = boost::beast::http::request<boost::beast::http::buffer_body>;
using boost::asio::ip::tcp;
using tcp_stream
    = boost::beast::tcp_stream::rebind_executor<boost::asio::use_awaitable_t<>::executor_with_default<boost::asio::any_io_executor>>::other;

struct Server
{
    static constexpr uint16_t Port = cipc::Message::Port;

    public:
    explicit Server(uint16_t index = 0) : _index(index) {}
    ~Server() { Stop(); }
    CLASS_DELETE_COPY_AND_MOVE(Server);

    void Start();
    bool AllSessionsComplete(std::scoped_lock<std::mutex> const& /* lock */);
    bool AllSessionsComplete();
    bool HasSessionsAndAllSessionsComplete();
    void WaitForShutdown();
    void Ping();
    void HandleMessage(cipc::Message::Header const& hdr, std::span<char const> buf);
    void Stop();
    void QuerySessions(ServerSessionCallback callback, void* cbData);

    private:
    Session& SessionById_(cipc::SessionId sessionId) LFTBND;

    template <typename TLambda>
    void FindOrWaitForSessionAndSnapshot_(std::string_view context,
                                          cipc::SessionId  sessionId,
                                          cipc::SnapshotId snapshotId,
                                          TLambda const&   lambda);

    template <typename TLambda> void FindOrWaitForSession_(std::string_view context, cipc::SessionId sessionId, TLambda const& lambda);

    void NotifySnapshotComplete_(cipc::SessionId sessionId, cipc::SnapshotId snapshotId, cipc::Rank fromRank, std::string failure);
    void NotifyWriteCkptComplete_(cipc::SessionId                           sessionId,
                                  cipc::SnapshotId                          snapshotId,
                                  cipc::Rank                                fromRank,
                                  std::string                               failure,
                                  std::unordered_map<std::string, uint64_t> dynamicBlobSizes);
    void ReceiveGlobalRegistry_(cipc::SessionId sessionId, std::string_view const& body, size_t fwdStart, size_t fwdEnd);

    void ReceiveLocalStateRegistry_(cipc::SessionId const&                                               sessionId,
                                    cipc::Rank                                                           fromRank,
                                    tcp_stream&                                                          stream,
                                    boost::beast::flat_buffer&                                           buffer,
                                    boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser);

    void InitializeShm_(RankTopology const& topo, Rank rank, uint64_t snapshotSize, int32_t numaNode);
    void ReceiveLocalRegistry_(cipc::SessionId sessionId, std::string_view const& buf);
    void ActivateSession_(cipc::SessionId sessionId, uint32_t numActiveSnapshots, std::string_view const& buf);

    void SpinDownWaitLoop_(tcp_stream stream);
    void FinalizeWaitLoop_(tcp_stream stream, cipc::SessionId sessionId, int64_t iteration);
    void HandleRequest_(tcp_stream& stream);

#ifdef SO_REUSEPORT
    void AcquireInstanceLock_();
    class ReusePortOption
    {
        public:
        explicit ReusePortOption(bool enabled) : _value(enabled ? 1 : 0) {}

        template <typename Protocol> int         Level(Protocol const& /*unused*/) const { return SOL_SOCKET; }
        template <typename Protocol> int         Name(Protocol const& /*unused*/) const { return SO_REUSEPORT; }
        template <typename Protocol> void const* Data(Protocol const& /*unused*/) const { return &_value; }
        template <typename Protocol> std::size_t Size(Protocol const& /*unused*/) const { return sizeof(_value); }

        private:
        int _value;
    };
#endif

    void StartTcpListener_(bool reusePort, size_t acceptorCount);
    void StartTcpAcceptors_();

    // The accept loop only accepts and hands each connection to `_handlerPool`, then
    // returns to async_accept at once. HandleRequest_ does synchronous, sometimes
    // long-running work (peer forwards, file writes, a bounded wait for a session);
    // running it inline serializes accepts and lets the kernel listen backlog overflow
    // under a many-node connect storm (the "connection refused on node:0" symptom).
    boost::asio::awaitable<void> DoListen_(boost::asio::ip::tcp::acceptor& acceptor);

    private:
    uint16_t   _index{};
    std::mutex _mutex;

    PerfClock::time_point        _lastPingTime{PerfClock::now()};
    std::atomic<bool>            _stopRequested{false};
    std::vector<cipc::MsgQueue*> _msgQueue;
    bool                         _running{false};
    std::vector<std::thread>     _threads;
    // Live spin-down waiter threads, each holding one client connection open until idle.
    std::vector<std::thread> _spinDownThreads;
    // Live wait-for-ckpt-finalize handlers, one per client wait connection. Reaped when new waits
    // arrive and drained on Stop().
    std::vector<std::future<void>> _finalizeWaits;

    boost::asio::io_context                     _ioc;
    std::vector<boost::asio::ip::tcp::acceptor> _tcpAcceptors;

    // Held across Stop()/Start() cycles so a rival daemon cannot elect itself mid-teardown.
    std::unique_ptr<bip::file_lock> _instanceLock;

    // Connection handlers run here, off the accept loop. Declared after _ioc so it is
    // joined (destroyed) before the io_context the handlers' sockets are bound to.
    std::unique_ptr<boost::asio::thread_pool> _handlerPool;
    uint32_t                                  _maxActiveSnapshots{};

    // TODO_NOPENDING_STRUCT
    struct PendingSessionData
    {
        Rank                       recvFrom{Rank::Invalid};
        std::shared_ptr<StateTree> registry;
    };

    std::unordered_map<cipc::SessionId, std::vector<std::unique_ptr<PendingSessionData>>> _pendingSessionData;

    std::unordered_map<cipc::SessionId, std::unique_ptr<Session>> _activeSessions;
    friend struct Session;
};

// Formatter for Server*
template <> struct fmt::formatter<Server> : fmt::formatter<std::string_view>
{
    auto format(Server const& server, fmt::format_context& ctx) const    // NOLINT
    {
        uint32_t ptrVal = reinterpret_cast<uintptr_t>(&server) % std::numeric_limits<uint32_t>::max();    // NOLINT
        return fmt::format_to(ctx.out(), "{:x}", ptrVal);
    }
};

using BufStream = boost::iostreams::stream<boost::iostreams::array_source>;

void Server::Start()
{
    checkpoint::logging::Init();
    checkpoint::telemetry::Init();
    checkpoint::config::OtelConfig const otelConfig;
    if (otelConfig.enabled && otelConfig.RoutesThroughDaemon() && !otelConfig.collector.empty())
    {
        if (otelConfig.collector == "file") { checkpoint::telemetry::InitLocalCollector(otelConfig.CollectorFile()); }
        else
        {
            checkpoint::telemetry::InitRelayCollector(otelConfig.collector);
        }
    }

    std::scoped_lock lock(_mutex);
    _stopRequested = false;
    if (_running) { return; }
    _ioc.restart();

    auto const handlerThreads     = checkpoint::config::GetConfigValue<checkpoint::config::ServerHandlerThreads>();
    auto const reusePort          = checkpoint::config::GetConfigValue<checkpoint::config::ServerTcpReusePort>();
    auto const acceptorCount      = checkpoint::config::GetConfigValue<checkpoint::config::ServerTcpAcceptors>();
    auto const maxActiveSnapshots = checkpoint::config::GetConfigValue<checkpoint::config::ServerMaxActiveSnapshots>();
    // static_cast<void>(checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferWriteTimeoutSeconds>());

    _maxActiveSnapshots = maxActiveSnapshots;
    _handlerPool        = std::make_unique<boost::asio::thread_pool>(handlerThreads);
    StartTcpListener_(reusePort, acceptorCount);
    _running = true;
    StartTcpAcceptors_();
}

void Server::InitializeShm_(RankTopology const& topo, Rank rank, uint64_t snapshotSize, int32_t numaNode)
{
    auto const configuredSize       = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferSize>();
    auto       configuredQueueCount = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferNumQueues>();
    if (configuredQueueCount == 0 && configuredSize != 0) { configuredQueueCount = cipc::ExplicitSizeDefaultQueueCount; }
    auto const queueCount    = cipc::ResolveQueueCount(topo.ranksPerPeer, configuredQueueCount);
    auto const localRank     = static_cast<uint32_t>(rank) % topo.ranksPerPeer;
    auto const queueIndex    = cipc::QueueIndex(localRank, topo.ranksPerPeer, queueCount);
    auto const producerCount = cipc::QueueProducerCount(queueIndex, topo.ranksPerPeer, queueCount);

    std::scoped_lock lock(_mutex);
    if (_stopRequested) { throw std::runtime_error("Checkpoint service is stopping"); }
    if (_msgQueue.size() <= queueIndex) { _msgQueue.resize(static_cast<size_t>(queueIndex) + 1); }
    if (_msgQueue[queueIndex] != nullptr) { return; }

    auto queueSize = configuredSize / queueCount;
    if (configuredSize == 0)
    {
        auto const snapshotSizePerProducer = cipc::RoundSnapshotShmSize(static_cast<size_t>(snapshotSize));
        if (snapshotSizePerProducer > std::numeric_limits<size_t>::max() / producerCount)
        {
            throw std::overflow_error("automatic shared-memory size exceeds the supported range");
        }
        queueSize = snapshotSizePerProducer * producerCount;
    }
    auto slotCount = static_cast<size_t>(checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferSlots>());
    if (slotCount == 0)
    {
        slotCount = configuredSize != 0 ? cipc::ExplicitSizeDefaultSlotCount
                                        : cipc::MsgQueue::SlotCountForShmSize(queueSize, cipc::MsgQueue::DefaultSlotCount, producerCount);
    }
    queueSize = cipc::MsgQueue::DetectShmSize(
        cipc::Message::RequestChannelName(_index, queueIndex), queueSize, cipc::MinShmSize / queueCount, slotCount);

    std::latch         queueReady{1};
    std::exception_ptr initFailure;
    _threads.emplace_back([this, queueIndex, numaNode, queueSize, slotCount, &queueReady, &initFailure] {
        try
        {
            SetThreadName(fmt::format("ncs:svrq:{}", queueIndex).c_str());
            BindThreadCpuToNumaNode(numaNode);
            auto msgQueue = cipc::MsgQueue::Create(cipc::Message::RequestChannelName(_index, queueIndex), queueSize, slotCount);
            if (numaNode >= 0)
            {
                auto region = msgQueue.GetShmDataRegion();
                BindRegionToNumaNode(numaNode, region.data(), region.size());
                std::fill(region.begin() + static_cast<std::ptrdiff_t>(msgQueue.HeaderSize()), region.end(), char{});
            }
            INFO("[{}] Checkpoint server started QueueSize={} Waiting for messages...", *this, queueSize);
            std::fflush(stderr);
            _msgQueue[queueIndex] = &msgQueue;
            queueReady.count_down();
            [&]() noexcept {
                while (true) try
                    {
                        auto const [stop, hdr, buf] = msgQueue.BeginRead();
                        if (stop)
                        {
                            std::scoped_lock lock1(_mutex);
                            INFO("[{}] Stop requested. Shutting down server thread.", *this);
                            _msgQueue[queueIndex] = nullptr;
                            return;
                        }
                        // The slot returns to the writer even on a throw; a skipped EndRead
                        // leaks it permanently and a saturated leak wedges the queue.
                        try
                        {
                            HandleMessage(hdr, buf);
                        } catch (std::exception const& e) { ERR("[{}] Exception handling message (slot released): {}", *this, e.what()); }
                        msgQueue.EndRead();

                    } catch (std::exception const& e) { ERR("[{}] Exception: {}", *this, e.what()); }
            }();
        } catch (...)
        {
            initFailure = std::current_exception();
            queueReady.count_down();
        }
    });
    queueReady.wait();
    if (initFailure) { std::rethrow_exception(initFailure); }
}

bool Server::AllSessionsComplete(std::scoped_lock<std::mutex> const& /* lock */)
{
    return std::ranges::all_of(_activeSessions, [](auto const& pair) { return pair.second->AllSnapshotsComplete(); });
}

bool Server::AllSessionsComplete()
{
    std::scoped_lock<std::mutex> lock(_mutex);
    return AllSessionsComplete(lock);
}

bool Server::HasSessionsAndAllSessionsComplete()
{
    std::scoped_lock<std::mutex> lock(_mutex);
    return !_activeSessions.empty() && AllSessionsComplete(lock);
}

void Server::WaitForShutdown()
{
    auto const autoShutdownTimeout
        = std::chrono::seconds{checkpoint::config::GetConfigValue<checkpoint::config::ServerAutoShutdownTimeoutSeconds>()};
    while (true)
    {
        auto const autoShutdown
            = autoShutdownTimeout != 0s && (_lastPingTime + autoShutdownTimeout) < PerfClock::now() && HasSessionsAndAllSessionsComplete();
        if (_stopRequested || autoShutdown) { break; }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    Stop();
}

void Server::Ping()
{
    std::scoped_lock lock(_mutex);
    _lastPingTime = PerfClock::now();
}

void Server::HandleMessage(cipc::Message::Header const& hdr, std::span<char const> buf)
{
    switch (static_cast<cipc::Message::Type>(hdr.index()))
    {
    case cipc::Message::Type::Ping:
    {
        Ping();
        return;
    }
    case cipc::Message::Type::Shutdown:
    {
        Stop();
    }
    break;
    // break;
    case cipc::Message::Type::ActivateSession:
    {
        auto const& hdr1 = std::get<cipc::Message::ActivateSession>(hdr);
        ActivateSession_(hdr1.sessionId, hdr1.numActiveSnapshots, {buf.data(), buf.size()});
    }
    break;
    case cipc::Message::Type::FinalizeSession:
    {
        auto const& hdr1 = std::get<cipc::Message::FinalizeSession>(hdr);
        INFO("[{}#{}] FinalizeSession    ", *this, hdr1.sessionId);
        SessionById_(hdr1.sessionId).Handle(hdr1, buf);
    }
    break;
    case cipc::Message::Type::SnapshotData:
    {
        auto const& hdr1 = std::get<cipc::Message::SnapshotData>(hdr);
        SessionById_(hdr1.sessionId).Handle(hdr1, buf);
    }
    break;
    case cipc::Message::Type::ActivateSnapshot:
    {
        auto const& hdr1 = std::get<cipc::Message::ActivateSnapshot>(hdr);
        SessionById_(hdr1.sessionId).Handle(hdr1, buf);
    }
    break;
    case cipc::Message::Type::FinalizeSnapshot:
    {
        // INFO("{}", _msgQueue->rlatency);
        auto const& hdr1 = std::get<cipc::Message::FinalizeSnapshot>(hdr);
        SessionById_(hdr1.sessionId).Handle(hdr1, buf);
    }
    break;
    case cipc::Message::Type::LocalStateRegistry:
        throw std::logic_error("LocalStateRegistry messages are not expected via shared memory-queue at the server");
    case cipc::Message::Type::StateRegistryFragment:
        throw std::logic_error("StateRegistryFragment messages are not expected at the server");
    // case cipc::Message::Type::ShardedStateData: [[fallthrough]];
    case cipc::Message::Type::Unknown: [[fallthrough]];
    case cipc::Message::Type::Invalid: [[fallthrough]];
    default: throw std::logic_error("[native_ckpt:svr] Unexpected message type");
    }
}

void Server::Stop()
{
    checkpoint::logging::Shutdown();
    checkpoint::telemetry::Shutdown();
    bool wasRunning = false;
    {
        std::scoped_lock lock(_mutex);
        wasRunning     = _running;
        _stopRequested = true;
        // Diagnostic: a graceful stop should only happen once every snapshot is both finalized
        // and write-completed across all ranks. Warn (loudly) if we are stopping while any is
        // still incomplete — that flags a premature "idle" decision (e.g. counting only
        // write-complete and missing a peer node's still-in-flight finalize).
        if (wasRunning)
        {
            for (auto const& [sessionId, session] : _activeSessions)
            {
                auto const       expected = session->ExpectedWriteCompletions();
                std::scoped_lock slock(session->mutex);
                for (auto const& [snapshotId, snapshot] : session->snapshots)
                {
                    auto const fin = snapshot->finalizedRanks.size();
                    auto const wr  = snapshot->writeCkptCompletedRanks.size();
                    if (fin < expected || wr < expected)
                    {
                        WARN("[{}#{}] stopping with snapshot {} INCOMPLETE: finalized {}/{}, write-completed {}/{}",
                             *this,
                             sessionId,
                             snapshotId,
                             fin,
                             expected,
                             wr,
                             expected);
                    }
                }
            }
            std::fflush(stderr);
        }
        for (auto const& msgQueue : _msgQueue)
        {
            if (msgQueue != nullptr) { msgQueue->Stop(); }
        }
    }
    {
        std::vector<std::thread> spinThreads;
        {
            std::scoped_lock lock(_mutex);
            spinThreads.swap(_spinDownThreads);
        }
        for (auto& t : spinThreads)
        {
            if (t.joinable()) { t.join(); }
        }
    }
    {
        // Finalization waiters finish before session cleanup.
        std::vector<std::future<void>> finalizeWaits;
        {
            std::scoped_lock lock(_mutex);
            finalizeWaits.swap(_finalizeWaits);
        }
        finalizeWaits.clear();
    }
    {
        std::vector<std::vector<std::future<void>>> allTasks;
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            for (auto& [id, session] : _activeSessions)
            {
                std::scoped_lock sessionLock(session->mutex);    // handlers push under this lock
                allTasks.push_back(std::move(session->asyncTasks));
            }
        }
    }

    if (!wasRunning) { return; }

    boost::system::error_code ec;
    for (auto& acceptor : _tcpAcceptors)
    {
        if (acceptor.close(ec)) { WARN("[{}][TCP] Failed to close acceptor: {}", *this, ec.message()); }
    }
    // Drain in-flight connection handlers before tearing down the io_context their
    // accepted sockets are bound to.
    if (_handlerPool)
    {
        _handlerPool->stop();
        _handlerPool->join();
    }
    // Reset the session so they wait for the async ops to finish
    _ioc.stop();
    for (auto& t : _threads)
    {
        if (t.joinable()) { t.join(); }
    }
    {
        std::scoped_lock<std::mutex> lock(_mutex);
        _msgQueue.clear();
        _threads.clear();
        _running = false;
    }
}

void Server::QuerySessions(ServerSessionCallback callback, void* cbData)
{
    std::scoped_lock lock(_mutex);
    for (auto const& [sessionId, session] : _activeSessions)
    {
        std::scoped_lock sessionLock(session->mutex);
        for (auto const& [snapshotId, snapshot] : session->snapshots)
        {
            for (auto const& [rank, snapshotState] : snapshot->snapshotStateForRank)
            {
                callback(cbData,
                         sessionId,
                         snapshotId,
                         snapshot->name,
                         snapshot->path,
                         *session->sessionState.globalRegistry,
                         *session->sessionState.localRegistryForRank.at(rank),
                         *snapshotState->snapshotRegistry,
                         *session->sessionState.dedupedGlobalRegistry,
                         snapshotState->snapshotData);
            }
        }
    }
}

Session& Server::SessionById_(cipc::SessionId sessionId) LFTBND
{
    std::scoped_lock lock(_mutex);
    _lastPingTime = PerfClock::now();
    auto it       = _activeSessions.find(sessionId);
    if (it == _activeSessions.end())
    {
        throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Session id={} not found", *this, sessionId));
    }
    return *it->second;
}

template <typename TLambda>
void Server::FindOrWaitForSessionAndSnapshot_(std::string_view context,
                                              cipc::SessionId  sessionId,
                                              cipc::SnapshotId snapshotId,
                                              TLambda const&   lambda)
{
    unsigned attempt = 0;
    auto     start   = PerfClock::now();
    // TODO_NOSTALL : avoid a stall. Use pending queues that are processed when session+snapshot is activated.
    while (true)
    {
        if (attempt > 0)
        {
            if (attempt == 1) { INFO("[{}#{}] {} snapshot={} Waiting to be activated", *this, sessionId, context, snapshotId); }
            if (PerfClock::now() - start > ThreadStallWaitForEventTimeout)
            {
                throw std::runtime_error(
                    fmt::format("[native_ckpt:svr][{}#{}] {} snapshot={} never activated.\n", *this, sessionId, context, snapshotId));
            }
            std::this_thread::sleep_for(IntervalBetweenRetries);
        }
        attempt++;
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            auto                         it = _activeSessions.find(sessionId);
            if (it == _activeSessions.end()) { continue; }

            std::scoped_lock<std::mutex> lock2(it->second->mutex);
            auto&                        snapshot = it->second->snapshots[snapshotId];
            if (!snapshot)
            {
                snapshot             = std::make_unique<Session::Snapshot>();
                snapshot->snapshotId = snapshotId;
            }

            lambda(*it->second, *snapshot);
        }
        INFO("[{}#{}] {} snapshot={} complete", *this, sessionId, context, snapshotId);
        return;
    }
}

template <typename TLambda> void Server::FindOrWaitForSession_(std::string_view context, cipc::SessionId sessionId, TLambda const& lambda)
{
    unsigned attempt = 0;
    auto     start   = PerfClock::now();
    // TODO_NOSTALL : avoid a stall. Use pending queues that are processed when session+snapshot is activated.
    while (true)
    {
        if (attempt > 0)
        {
            if (attempt == 1) { INFO("[{}#{}] {} Waiting to be activated", *this, sessionId, context); }
            if (PerfClock::now() - start > ThreadStallWaitForEventTimeout)
            {
                throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}] {} never activated.\n", *this, sessionId, context));
            }
            std::this_thread::sleep_for(IntervalBetweenRetries);
        }
        attempt++;
        Session* session = nullptr;
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            auto                         it = _activeSessions.find(sessionId);
            if (it == _activeSessions.end()) { continue; }
            session = it->second.get();
        }
        lambda(*session);
        INFO("[{}#{}] {} complete", *this, sessionId, context);
        return;
    }
}

void Server::NotifySnapshotComplete_(cipc::SessionId sessionId, cipc::SnapshotId snapshotId, cipc::Rank fromRank, std::string failure)
{
    FindOrWaitForSessionAndSnapshot_(fmt::format("NotifyFinalizeComplete rank={}", fromRank),
                                     sessionId,
                                     snapshotId,
                                     [&](Session& /* session */, Session::Snapshot& snapshot) {
                                         snapshot.finalizedRanks.insert(fromRank);
                                         if (!failure.empty())
                                         {
                                             snapshot.failures.push_back(fmt::format("rank={} Finalize failed: {}", fromRank, failure));
                                         }
                                     });
}

void Server::NotifyWriteCkptComplete_(cipc::SessionId                           sessionId,
                                      cipc::SnapshotId                          snapshotId,
                                      cipc::Rank                                fromRank,
                                      std::string                               failure,
                                      std::unordered_map<std::string, uint64_t> dynamicBlobSizes)
{
    FindOrWaitForSessionAndSnapshot_(fmt::format("NotifyWriteCkptComplete rank={}", fromRank),
                                     sessionId,
                                     snapshotId,
                                     [&](Session& session, Session::Snapshot& snapshot) {
                                         auto const onCoordinator = session.IsCoordinatorPeer();
                                         if (onCoordinator)
                                         {
                                             snapshot.dynamicBlobSizes.merge(dynamicBlobSizes);
                                             snapshot.writeCkptCompletedRanks.insert(fromRank);
                                         }
                                         if (!failure.empty())
                                         {
                                             snapshot.failures.push_back(
                                                 onCoordinator ? fmt::format("rank={} WriteCkpt failed: {}", fromRank, failure)
                                                               : std::move(failure));
                                         }
                                         if (!onCoordinator) { snapshot.complete = true; }
                                     });
}

void Server::ActivateSession_(cipc::SessionId sessionId, uint32_t numActiveSnapshots, std::string_view const& buf)
{
    std::unique_lock lock(_mutex);
    _lastPingTime = PerfClock::now();

    BufStream bufStream{buf.data(), buf.size()};
    auto      workDirStr    = serdes::ReadBlobWithSize<uint16_t>(bufStream);
    auto      workDir       = workDirStr.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(workDirStr);
    auto      createSession = [&] {
        auto topo                   = StateSerDes::ReadTopology(bufStream);
        auto session                = std::make_unique<Session>(this, sessionId, std::move(workDir), std::move(topo));
        session->numActiveSnapshots = std::min(numActiveSnapshots, _maxActiveSnapshots);
        session->Activate();
        return session;
    };

    INFO("[{}#{}] ActivateSession workDir={}", *this, sessionId, workDir);
    auto it = _activeSessions.find(sessionId);
    it      = it != _activeSessions.end() ? it : _activeSessions.insert({sessionId, createSession()}).first;
    checkpoint::config::OtelConfig const otelConfig;
    if (otelConfig.enabled && otelConfig.RoutesThroughDaemon())
    {
        if (it->second->IsCoordinatorPeer()) { checkpoint::telemetry::InitLocalCollector(otelConfig.CollectorFile()); }
        else if (!it->second->topo.peers.empty()) { checkpoint::telemetry::InitRelayCollector(it->second->topo.peers[0]); }
    }
    auto*                                            session = it->second.get();
    std::vector<std::unique_ptr<PendingSessionData>> pending;
    auto const                                       queued = _pendingSessionData.find(sessionId);
    if (queued != _pendingSessionData.end())
    {
        pending = std::move(queued->second);
        _pendingSessionData.erase(queued);
    }
    lock.unlock();
    for (auto& fragment : pending) { session->HandleLocalStateRegistry(fragment->recvFrom, std::move(fragment->registry)); }
}

void Server::ReceiveLocalRegistry_(cipc::SessionId sessionId, std::string_view const& buf)
{
    FindOrWaitForSession_("ReceiveLocalRegistry", sessionId, [&](Session& session) {
        session.Handle(cipc::Message::LocalStateRegistry{.sessionId = sessionId, .messageSize = buf.size()}, buf);
    });
}

static std::shared_ptr<StateTree> InflateRegistry(std::string_view bytes)
{
    boost::iostreams::filtering_istream istrm;
    istrm.push(boost::iostreams::zlib_decompressor());
    istrm.push(boost::iostreams::array_source(bytes.data(), bytes.size()));
    return StateSerDes::Read(istrm);
}

void Server::ReceiveGlobalRegistry_(cipc::SessionId sessionId, std::string_view const& body, size_t fwdStart, size_t fwdEnd)
{
    FindOrWaitForSession_("ReceiveGlobalRegistry", sessionId, [&](Session& session) {
        BufStream result{body.data(), body.size()};
        result.exceptions(std::ios::failbit | std::ios::badbit);
        auto failure = serdes::ReadBlobWithSize<uint32_t>(result);
        if (!failure.empty())
        {
            session.HandleGlobalRegistry({}, std::move(failure));
            return;
        }
        if (session.IsCoordinatorPeer()) { throw std::logic_error("The coordinator peer never receives the global registry"); }
        if (session.PeerIndex() != fwdStart) { throw std::invalid_argument("Global registry forwarding rank does not match daemon"); }
        auto registry = InflateRegistry(body.substr(sizeof(uint32_t)));
        session.RelayGlobalRegistryToPeers(body, fwdStart + 1, fwdEnd);
        session.HandleGlobalRegistry(std::move(registry), {});
    });
}

void Server::ReceiveLocalStateRegistry_(cipc::SessionId const&                                               sessionId,
                                        Rank                                                                 fromRank,
                                        tcp_stream&                                                          stream,
                                        boost::beast::flat_buffer&                                           buffer,
                                        boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser)
{
    std::string             body;
    std::array<char, 65536> chunk{};
    while (!parser.is_done())
    {
        parser.get().body().data = chunk.data();
        parser.get().body().size = chunk.size();
        parser.get().body().more = true;
        boost::system::error_code error;
        boost::beast::http::read_some(stream, buffer, parser, error);
        if (error && error != boost::beast::http::error::need_buffer) { throw boost::system::system_error(error); }
        body.append(chunk.data(), chunk.size() - parser.get().body().size);
    }
    BufStream metadata{body.data(), body.size()};
    auto      registry = StateSerDes::Read(metadata);
    if (registry->SessionId() != sessionId || registry->GlobalRank() != fromRank)
    {
        throw std::invalid_argument("Peer registry does not match its session or rank");
    }
    Session* session = nullptr;
    {
        std::scoped_lock lock(_mutex);
        auto             item = _activeSessions.find(sessionId);
        if (item == _activeSessions.end())
        {
            auto pending      = std::make_unique<PendingSessionData>();
            pending->recvFrom = fromRank;
            pending->registry = std::move(registry);
            _pendingSessionData[sessionId].push_back(std::move(pending));
            return;
        }
        session = item->second.get();
    }
    session->HandleLocalStateRegistry(fromRank, std::move(registry));
}

void Server::SpinDownWaitLoop_(tcp_stream stream)
{
    SetThreadName("ncs:spindown");
    static constexpr auto HeartbeatInterval = 2s;
    // The bounded wait prevents a wedged server from hanging a caller forever.

    auto const deadline = PerfClock::now() + std::chrono::duration_cast<PerfClock::duration>(ServerWaitTimeout);

    // No deadline on the synchronous writes; the heartbeat cadence, not a socket timeout,
    // bounds how long we hold the connection.
    stream.expires_never();

    boost::system::error_code ec;
    // NOLINTNEXTLINE(readability-magic-numbers)
    boost::beast::http::response<boost::beast::http::empty_body> res{boost::beast::http::status::ok, 11};
    res.set(boost::beast::http::field::server, "NativeCheckpointServer");
    res.chunked(true);
    boost::beast::http::response_serializer<boost::beast::http::empty_body> sr{res};
    boost::beast::http::write_header(stream, sr, ec);

    bool hardStop = false;
    while (!ec)
    {
        if (AllSessionsComplete()) { break; }
        if (_stopRequested) { break; }    // another path already began shutdown
        if (PerfClock::now() >= deadline)
        {
            hardStop = true;
            break;
        }
        // A non-empty chunk keeps the client read alive while we wait for idle. The body
        // is opaque to the caller, which only inspects the status line.
        static constexpr std::string_view Heartbeat = " ";
        boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(Heartbeat)), ec);
        if (ec) { break; }    // client hung up — stop heartbeating, still spin down below
        std::this_thread::sleep_for(HeartbeatInterval);
    }

    if (hardStop)
    {
        ERR("[{}] spindown timed out before idle; hard-stopping (aborting in-flight work)", *this);
        std::scoped_lock lock(_mutex);
        for (auto& session : _activeSessions)
        {
            std::scoped_lock slock(session.second->mutex);
            session.second->stopRequested = true;
        }
    }
    else
    {
        INFO("[{}] spindown: all sessions idle; spinning down", *this);
    }

    // Final empty "{}" payload, matching the hard-stop reply, then the terminating chunk.
    static constexpr std::string_view Final = "{}";
    boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(Final)), ec);
    boost::asio::write(stream.socket(), boost::beast::http::make_chunk_last(), ec);
    TryCleanShutdown(stream);

    _stopRequested = true;    // WaitForShutdown observes this and calls Stop()
}

// A dedicated waiter keeps peer completion requests progressing during the global durability wait.
void Server::FinalizeWaitLoop_(tcp_stream stream, cipc::SessionId sessionId, int64_t iteration)
{
    SetThreadName("ncs:finwait");
    static constexpr auto PollInterval = 2s;
    auto const            deadline     = std::chrono::steady_clock::now() + ServerWaitTimeout;
    stream.expires_never();

    std::string      failure;
    cipc::SnapshotId snapshotId{cipc::SnapshotId::Invalid};
    bool             resolved = false;
    while (true)
    {
        if (_stopRequested)
        {
            failure = "checkpoint finalization aborted because the server is stopping";
            break;
        }
        try
        {
            std::scoped_lock lock(_mutex);
            auto             it = _activeSessions.find(sessionId);
            if (it != _activeSessions.end())
            {
                auto& session = *it->second;
                if (!resolved) { resolved = session.ResolveSnapshotId(iteration, snapshotId); }
                std::scoped_lock sessionLock(session.mutex);
                auto const       snapshot = session.snapshots.find(snapshotId);
                if (resolved && snapshot != session.snapshots.end() && snapshot->second->complete)
                {
                    if (!snapshot->second->failures.empty())
                    {
                        failure = fmt::format("snapshot={} has failures: {}", snapshotId, fmt::join(snapshot->second->failures, ", "));
                    }
                    break;
                }
            }
        } catch (std::exception const& ex)
        {
            failure = ex.what();
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            failure = fmt::format("checkpoint finalization timed out after {} seconds", ServerWaitTimeout.count());
            ERR("[{}#{}] wait-for-ckpt-finalize timed out for iteration={}", *this, sessionId, iteration);
            break;
        }
        std::this_thread::sleep_for(PollInterval);
    }

    auto const status = failure.empty() ? boost::beast::http::status::ok : boost::beast::http::status::internal_server_error;
    boost::beast::http::response<boost::beast::http::string_body> res{status, HttpVersion11};
    res.set(boost::beast::http::field::server, "NativeCheckpointServer");
    res.set(boost::beast::http::field::content_type, "text/plain; charset=utf-8");
    if (!failure.empty())
    {
        res.body() = fmt::format("Checkpoint finalization failed for session={} iteration={}: {}", sessionId, iteration, failure);
    }
    res.prepare_payload();
    boost::system::error_code ec;
    boost::beast::http::write(stream, res, ec);
    TryCleanShutdown(stream);
}

void Server::HandleRequest_(tcp_stream& stream)    // NOLINT(readability-function-cognitive-complexity)
{
    boost::beast::flat_buffer buffer;
    // First, read only the headers to determine the endpoint
    boost::beast::http::request_parser<boost::beast::http::empty_body> headerParser;
    headerParser.body_limit(std::numeric_limits<std::uint64_t>::max());
    boost::beast::http::read_header(stream, buffer, headerParser);

    auto const& headers = headerParser.get();
    auto        target  = headers.target();
    SetThreadName(fmt::format("ncs:w:{}", target).c_str());

    auto const getOrPost = headers.method() == boost::beast::http::verb::get || headers.method() == boost::beast::http::verb::post;

    auto url     = boost::urls::parse_origin_form(target).value();
    auto segs    = url.segments();
    auto it      = segs.begin();
    auto matches = [&](std::initializer_list<std::string> const& patterns) -> bool {
        if (segs.size() < patterns.size()) { return false; }
        it = segs.begin();
        for (auto const& pattern : patterns)
        {
            if (*(it++) != pattern) { return false; }
        }
        return true;
    };

    // Check if this is the state upload endpoint: POST /api/peer/localstateregistry/{sessionId}/{rank}
    if (matches({"api", "peer", "localstateregistry"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto rank      = cipc::Rank{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in state registry upload endpoint"); }

        boost::beast::http::request_parser<boost::beast::http::buffer_body> bufferParser{std::move(headerParser)};
        ReceiveLocalStateRegistry_(sessionId, rank, stream, buffer, bufferParser);
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    else if (matches({"api", "peer", "snapshotcomplete"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId  = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto rank       = cipc::Rank{boost::lexical_cast<uint32_t>(*(it++))};
        auto snapshotId = cipc::SnapshotId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in snapshot complete endpoint"); }
        boost::beast::http::request_parser<boost::beast::http::string_body> parser{std::move(headerParser)};
        boost::beast::http::read(stream, buffer, parser);

        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        // TODO_NOSTALL : move it before we send 200 once we dont stall for session+snaphot anymore
        NotifySnapshotComplete_(sessionId, snapshotId, rank, parser.get().body());
    }
    else if (matches({"api", "peer", "writeckptcomplete"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId  = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto rank       = cipc::Rank{boost::lexical_cast<uint32_t>(*(it++))};
        auto snapshotId = cipc::SnapshotId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in state registry upload endpoint"); }

        boost::beast::http::response<boost::beast::http::string_body>       res{boost::beast::http::status::ok, headers.version()};
        boost::beast::http::request_parser<boost::beast::http::string_body> parser{std::move(headerParser)};
        boost::beast::http::read(stream, buffer, parser);
        BufStream completion{parser.get().body().data(), parser.get().body().size()};
        completion.exceptions(std::ios::failbit | std::ios::badbit);
        auto                                      failure = serdes::ReadBlobWithSize<uint32_t>(completion);
        std::unordered_map<std::string, uint64_t> dynamicBlobSizes;
        auto const                                count = serdes::ReadTrivial<uint32_t>(completion);
        for (uint32_t index = 0; index < count; ++index)
        {
            auto fqn  = serdes::ReadBlobWithSize<uint32_t>(completion);
            auto size = serdes::ReadTrivial<uint64_t>(completion);
            dynamicBlobSizes.emplace(std::move(fqn), size);
        }
        NotifyWriteCkptComplete_(sessionId, snapshotId, rank, std::move(failure), std::move(dynamicBlobSizes));
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);

        if (auto const fwdParam = url.params().find("fwd"); fwdParam != url.params().end())
        {
            auto const range     = (*fwdParam).value;
            auto const separator = range.find(':');
            auto const fwdStart  = boost::lexical_cast<size_t>(range.substr(0, separator));
            auto const fwdEnd    = boost::lexical_cast<size_t>(range.substr(separator + 1));
            FindOrWaitForSession_("ReceiveWriteCkptComplete", sessionId, [&](Session& session) {
                session.RelayWriteCkptCompleteToPeers(snapshotId, parser.get().body(), fwdStart + 1, fwdEnd);
            });
        }
    }
    else if (matches({"api", "peer", "globalregistry"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end())
        {
            throw std::invalid_argument("Unexpected extra URL segments in globalregistry endpoint; globalregistry requires a "
                                        "/{sessionId}?fwd=<start>:<end> path");
        }
        auto [fwdRangeStart, fwdRangeEnd] = [&] -> std::pair<size_t, size_t> {
            auto const fwdParam = url.params().find("fwd");
            if (fwdParam == url.params().end()) { return {0, 0}; }
            auto const fwdValue = (*fwdParam).value;
            auto const sepPos   = fwdValue.find(':');
            if (sepPos == std::string::npos)
            {
                throw std::invalid_argument(fmt::format("Malformed fwd query parameter '{}', expected '<start>:<end>'", fwdValue));
            }
            return {boost::lexical_cast<size_t>(fwdValue.substr(0, sepPos)), boost::lexical_cast<size_t>(fwdValue.substr(sepPos + 1))};
        }();

        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(std::numeric_limits<std::uint64_t>::max());
        boost::beast::http::read(stream, buffer, bodyParser);
        ReceiveGlobalRegistry_(sessionId, bodyParser.get().body(), fwdRangeStart, fwdRangeEnd);
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    else if (matches({"api", "activatesession"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto const sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in local registry endpoint"); }
        auto const version   = headers.version();
        auto const keepAlive = headers.keep_alive();

        stream.expires_never();
        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(std::numeric_limits<std::uint64_t>::max());
        boost::beast::http::read(stream, buffer, bodyParser);
        auto const& body = bodyParser.get().body();

        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, version};
        try
        {
            ActivateSession_(sessionId, 1, body);
            res.body() = "{}";
        } catch (std::exception const& e)
        {
            res.result(boost::beast::http::status::insufficient_storage);
            res.body() = e.what();
        }
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(keepAlive);
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        Ping();
    }
    else if (matches({"api", "activate-snapshot"}) && headers.method() == boost::beast::http::verb::post)
    {
        auto const version   = headers.version();
        auto const keepAlive = headers.keep_alive();
        stream.expires_never();
        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(std::numeric_limits<std::uint64_t>::max());
        boost::beast::http::read(stream, buffer, bodyParser);
        boost::beast::http::response<boost::beast::http::string_body> response{boost::beast::http::status::ok, version};
        try
        {
            if (it == segs.end()) { throw std::invalid_argument("Snapshot activation requires a session ID"); }
            auto const sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
            if (it == segs.end()) { throw std::invalid_argument("Snapshot activation requires a rank"); }
            auto const rank = Rank{boost::lexical_cast<uint32_t>(*(it++))};
            if (it == segs.end()) { throw std::invalid_argument("Snapshot activation requires a snapshot ID"); }
            auto const snapshotId = cipc::SnapshotId{boost::lexical_cast<uint32_t>(*(it++))};
            if (it != segs.end()) { throw std::invalid_argument("Unexpected snapshot activation URL segments"); }
            auto const& body = bodyParser.get().body();
            if (body.size() < sizeof(cipc::Message::ActivateSnapshot))
            {
                throw std::invalid_argument("Snapshot activation header is truncated");
            }
            cipc::Message::ActivateSnapshot activation;
            std::memcpy(&activation, body.data(), sizeof(activation));
            auto const descriptor = std::span<char const>(body.data() + sizeof(activation), body.size() - sizeof(activation));
            if (activation.sessionId != sessionId || activation.globalRank != rank || activation.snapshotId != snapshotId)
            {
                throw std::invalid_argument("Snapshot activation header does not match its URL");
            }
            if (activation.messageSize != descriptor.size())
            {
                throw std::invalid_argument("Snapshot activation size does not match its descriptor");
            }
            FindOrWaitForSession_("ActivateSnapshot", sessionId, [&](Session& session) { session.Handle(activation, descriptor); });
            response.body() = "{}";
        } catch (std::exception const& error)
        {
            response.result(boost::beast::http::status::internal_server_error);
            response.body() = error.what();
        }
        response.set(boost::beast::http::field::server, "NativeCheckpointServer");
        response.keep_alive(keepAlive);
        response.prepare_payload();
        boost::beast::http::write(stream, response);
        Ping();
    }
    else if (matches({"api", "localstateregistry"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto const sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in local registry endpoint"); }
        auto const version   = headers.version();
        auto const keepAlive = headers.keep_alive();

        // Registries are no longer constrained to a shared-memory slot and may
        // legitimately take longer than the ordinary control-request timeout.
        stream.expires_never();
        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(std::numeric_limits<std::uint64_t>::max());
        boost::beast::http::read(stream, buffer, bodyParser);
        auto const& body = bodyParser.get().body();

        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, version};
        try
        {
            ReceiveLocalRegistry_(sessionId, body);
            res.body() = "{}";
        } catch (std::exception const& e)
        {
            res.result(boost::beast::http::status::insufficient_storage);
            res.body() = e.what();
        }
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(keepAlive);
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        Ping();
    }
    else if (matches({"api", "shmqueue"}) && headers.method() == boost::beast::http::verb::post)
    {
        auto const version   = headers.version();
        auto const keepAlive = headers.keep_alive();
        stream.expires_never();
        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        boost::beast::http::read(stream, buffer, bodyParser);
        boost::beast::http::response<boost::beast::http::string_body> response{boost::beast::http::status::ok, version};
        try
        {
            if (it == segs.end()) { throw std::invalid_argument("Shared-memory queue requires a session ID"); }
            auto const sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
            if (it == segs.end()) { throw std::invalid_argument("Shared-memory queue requires a rank"); }
            auto const rank = Rank{boost::lexical_cast<uint32_t>(*(it++))};
            if (it != segs.end()) { throw std::invalid_argument("Unexpected shared-memory queue URL segments"); }
            BufStream request{bodyParser.get().body().data(), bodyParser.get().body().size()};
            request.exceptions(std::ios::failbit | std::ios::badbit);
            auto const snapshotSize = serdes::ReadTrivial<uint64_t>(request);
            auto const numaNode     = serdes::ReadTrivial<int32_t>(request);
            FindOrWaitForSession_(
                "InitializeShm", sessionId, [&](Session& session) { InitializeShm_(session.topo, rank, snapshotSize, numaNode); });
            response.body() = "{}";
        } catch (std::exception const& error)
        {
            response.result(boost::beast::http::status::insufficient_storage);
            response.body() = error.what();
        }
        response.set(boost::beast::http::field::server, "NativeCheckpointServer");
        response.keep_alive(keepAlive);
        response.prepare_payload();
        boost::beast::http::write(stream, response);
        Ping();
    }
    else if (matches({"api", "snapshot-plan"}) && getOrPost && it != segs.end())
    {
        auto const sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it == segs.end()) { throw std::invalid_argument("Snapshot plan requires a rank"); }
        auto const rank = Rank{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected snapshot plan URL segments"); }
        std::scoped_lock waitersLock(_mutex);
        std::erase_if(_finalizeWaits,
                      [](std::future<void>& task) { return task.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
        _finalizeWaits.emplace_back(std::async(std::launch::async, [this, socket = std::move(stream), sessionId, rank] mutable {
            auto connection = std::move(socket);
            SetThreadName("ncs:planwait");
            connection.expires_never();
            auto const                 deadline = std::chrono::steady_clock::now() + ServerWaitTimeout;
            std::shared_ptr<StateTree> plan;
            std::string                failure;
            while (true)
            {
                if (_stopRequested)
                {
                    failure = "Server is stopping";
                    break;
                }
                {
                    std::scoped_lock lock(_mutex);
                    auto const       item = _activeSessions.find(sessionId);
                    if (item != _activeSessions.end())
                    {
                        auto&            session = *item->second;
                        std::scoped_lock sessionLock(session.mutex);
                        auto const       firstRank = static_cast<uint32_t>(session.PeerIndex()) * session.topo.ranksPerPeer;
                        auto const       lastRank
                            = std::min(firstRank + session.topo.ranksPerPeer, static_cast<uint32_t>(session.topo.worldSize));
                        if (static_cast<uint32_t>(rank) < firstRank || static_cast<uint32_t>(rank) >= lastRank)
                        {
                            failure = "Requested snapshot plan rank is not local to this daemon";
                            break;
                        }
                        if (session.stopRequested)
                        {
                            failure = "Session is stopping";
                            break;
                        }
                        if (!session.sessionState.planningFailure.empty())
                        {
                            failure = session.sessionState.planningFailure;
                            break;
                        }
                        if (session.sessionState.planningReady)
                        {
                            auto const selected = session.sessionState.snapshotRegistryForRank.find(rank);
                            if (selected == session.sessionState.snapshotRegistryForRank.end())
                            {
                                failure = "Local snapshot plan is unavailable";
                            }
                            else
                            {
                                plan = selected->second;
                            }
                            break;
                        }
                    }
                }
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    failure = "Snapshot planning timed out";
                    break;
                }
                std::this_thread::sleep_for(IntervalBetweenRetries);
            }
            auto const status = failure.empty() ? boost::beast::http::status::ok : boost::beast::http::status::internal_server_error;
            boost::beast::http::response<boost::beast::http::string_body> response{status, HttpVersion11};
            response.set(boost::beast::http::field::server, "NativeCheckpointServer");
            if (plan != nullptr)
            {
                std::ostringstream metadata;
                StateSerDes::Write(metadata, *plan, StateNode::Handle::Root);
                response.body() = metadata.str();
                response.set(boost::beast::http::field::content_type, "application/octet-stream");
            }
            else
            {
                response.body() = failure;
            }
            response.prepare_payload();
            boost::system::error_code error;
            boost::beast::http::write(connection, response, error);
            TryCleanShutdown(connection);
        }));
    }
    else if (matches({"api", "ping"}) && getOrPost && (it == segs.end()))
    {
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        Ping();
    }
    else if (matches({"api", "stop"}) && getOrPost && (it == segs.end()))
    {
        // Reply OK first, then signal shutdown via the flag that WaitForShutdown polls.
        // We can't call Stop() inline: it joins _threads and we're running on one of them.
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        _stopRequested = true;
        {
            for (auto& session : _activeSessions)
            {
                std::scoped_lock lock(session.second->mutex);
                session.second->stopRequested = true;
            }
        }
    }
    else if (matches({"api", "spindown"}) && getOrPost)
    {
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in spindown endpoint"); }
        // Hand the connection off to a dedicated waiter thread and return at once: the
        // accept loop must stay free to service the peer writeckptcomplete/snapshotcomplete
        // notifications that idle-ness depends on, otherwise the wait would deadlock.
        std::scoped_lock lock(_mutex);
        _spinDownThreads.emplace_back([this, s = std::move(stream)] mutable { SpinDownWaitLoop_(std::move(s)); });
    }
    // The response completes when the selected checkpoint is durable.
    // `iteration` is signed — negative values index back from the newest snapshot (-1 = latest).
    else if (matches({"api", "wait-for-ckpt-finalize"}) && getOrPost && (it != segs.end()))
    {
        auto sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it == segs.end()) { throw std::invalid_argument("wait-for-ckpt-finalize requires a /{sessionId}/{iteration} path"); }
        auto iteration = boost::lexical_cast<int64_t>(*(it++));
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in wait-for-ckpt-finalize endpoint"); }
        // Hand off to a dedicated waiter thread and return at once, like spin-down: the handler
        // pool must stay free to service the peer notifications durability depends on. Reap
        // finished waits opportunistically so the tracking vector does not grow across a run.
        std::scoped_lock lock(_mutex);
        std::erase_if(_finalizeWaits,
                      [](std::future<void>& f) { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
        _finalizeWaits.emplace_back(std::async(std::launch::async, [this, s = std::move(stream), sessionId, iteration] mutable {
            FinalizeWaitLoop_(std::move(s), sessionId, iteration);
        }));
    }
    else if ((url.path() == "/v1/traces" || url.path() == "/v1/metrics" || url.path() == "/v1/logs")
             && headers.method() == boost::beast::http::verb::post)
    {
        std::string const path            = url.path();
        std::string const contentType     = headers[boost::beast::http::field::content_type];
        std::string const contentEncoding = headers[boost::beast::http::field::content_encoding];
        unsigned const    version         = headers.version();

        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(checkpoint::telemetry::MaxOtlpRequestBytes);
        boost::beast::http::read(stream, buffer, bodyParser);
        auto result = checkpoint::telemetry::ReceiveOtlp(path, contentType, contentEncoding, bodyParser.get().body());

        boost::beast::http::response<boost::beast::http::string_body> res{static_cast<boost::beast::http::status>(result.status), version};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.set(boost::beast::http::field::content_type, "application/x-protobuf");
        res.keep_alive(false);
        res.body() = std::move(result.body);
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    else if ((url.path() == "/api/telemetry" || url.path() == "/api/telemetry/remote")
             && headers.method() == boost::beast::http::verb::post)
    {
        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(checkpoint::telemetry::MaxOtlpRequestBytes);
        boost::beast::http::read(stream, buffer, bodyParser);
        checkpoint::telemetry::Collect(bodyParser.get().body());
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    else
    {
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::not_found, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "Endpoint not found";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }

    SetThreadName("ncs:w");
}

#ifdef SO_REUSEPORT
void Server::AcquireInstanceLock_()
{
    if (_instanceLock) { return; }

    auto const lockPath = std::filesystem::temp_directory_path() / fmt::format("nvrx_ckpt_server.{}.lock", Port + _index);
    {
        std::ofstream lockFile(lockPath, std::ios::app);
        if (!lockFile.good()) { throw std::runtime_error(fmt::format("Instance lock create failed: {}", lockPath)); }
    }

    try
    {
        auto lock = std::make_unique<bip::file_lock>(lockPath.string().c_str());
        if (!lock->try_lock()) { throw cipc::ErrorAlreadyRunning(); }
        _instanceLock = std::move(lock);
    } catch (bip::interprocess_exception const& e) { throw std::runtime_error(fmt::format("Instance lock {}: {}", lockPath, e.what())); }
}

#endif

void Server::StartTcpListener_(bool reusePort, size_t acceptorCount)
{
    using boost::asio::ip::tcp;
    auto const    address = boost::asio::ip::make_address("0.0.0.0");
    tcp::endpoint endpoint(address, Port + _index);

#ifdef SO_REUSEPORT
    // When SO_REUSEPORT is enabled, the TCP port no longer elects a single owner.
    // Losers must throw before touching acceptors or shm rings; Start() spawns the
    // shm consumer threads only after this returns.
    if (reusePort) { AcquireInstanceLock_(); }
#else
    reusePort = false;
#endif

    if (!reusePort) { acceptorCount = 1; }
    _tcpAcceptors.clear();
    _tcpAcceptors.reserve(acceptorCount);
    for (size_t i = 0; i < acceptorCount; ++i)
    {
        auto& acceptor = _tcpAcceptors.emplace_back(_ioc);
        acceptor.open(endpoint.protocol());
        acceptor.set_option(boost::asio::socket_base::reuse_address(true));
#ifdef SO_REUSEPORT
        if (reusePort)
        {
            int const enable = 1;
            ::setsockopt(acceptor.native_handle(), SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable));
        }
#endif
        boost::system::error_code ec;
        ec = acceptor.bind(endpoint, ec);
        if (ec) { throw cipc::ErrorAlreadyRunning(); }
        ec = acceptor.listen(boost::asio::socket_base::max_listen_connections, ec);
        if (ec) { throw cipc::ErrorAlreadyRunning(); }
    }
}

void Server::StartTcpAcceptors_()
{
    for (auto& acceptor : _tcpAcceptors)
    {
        boost::asio::co_spawn(_ioc, DoListen_(acceptor), [](std::exception_ptr const& e) {
            if (e) try
                {
                    std::rethrow_exception(e);
                } catch (boost::system::system_error& e)
                {
                    if (e.code() != boost::asio::error::operation_aborted) { throw; }
                } catch (std::exception& e) { ERR("Error in acceptor: {}", e.what()); }
        });
    }

    for (size_t i = 0; i < _tcpAcceptors.size(); i++)
    {
        _threads.emplace_back([this, i] {
            SetThreadName(fmt::format("ncs:ios:{}", i).c_str());
            _ioc.run();
        });
    }
}

// The accept loop only accepts and hands each connection to `_handlerPool`, then
// returns to async_accept at once. HandleRequest_ does synchronous, sometimes
// long-running work (peer forwards, file writes, a bounded wait for a session);
// running it inline serializes accepts and lets the kernel listen backlog overflow
// under a many-node connect storm (the "connection refused on node:0" symptom).
boost::asio::awaitable<void>
Server::DoListen_(boost::asio::ip::tcp::acceptor& acceptor)    // NOLINT(cppcoreguidelines-avoid-reference-coroutine-parameters)
{
    while (true)
    {
        boost::system::error_code ec;
        auto                      socket = co_await acceptor.async_accept(boost::asio::redirect_error(boost::asio::use_awaitable, ec));
        if (ec == boost::asio::error::operation_aborted)
        {    //
            co_return;
        }
        if (ec)
        {
            ERR("accept error: {}", ec.message());
            co_return;
        }
        boost::asio::post(*_handlerPool, [this, sock = std::move(socket)] mutable {
            static constexpr auto Timeout = std::chrono::seconds{30};
            tcp_stream            stream(std::move(sock));
            stream.expires_after(Timeout);
            try
            {
                HandleRequest_(stream);
            } catch (boost::system::system_error const& e)
            {
                TryCleanShutdown(stream);
                if (e.code() != boost::beast::http::error::end_of_stream) { ERR("Error handling request: {}", e.what()); }
            } catch (std::exception const& e)
            {
                TryCleanShutdown(stream);
                ERR("Error: {}", e.what());
            }
        });
    }
}

SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
static std::vector<std::unique_ptr<Server>> ServerSingletons;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
SUPPRESS_WARNINGS_END

void RunInprocServer(uint16_t nodes)
{
    checkpoint::logging::Init();
    for (uint16_t i = 0; i < nodes; i++)
    {
        INFO("Starting inproc server instance for node={}", i);
        ServerSingletons.push_back(std::make_unique<Server>(i));
        ServerSingletons.back()->Start();
    }
    INFO("Inproc server started.");
    std::fflush(stderr);
}

void RunServer()    // NOLINT(misc-use-internal-linkage)
{
    SetThreadName("ncs:main");
    Server server;
    server.Start();

    // Signal to the launching process (SvcPing) that we are ready to accept connections.
    fmt::print(stdout, "READY\n");
    std::fflush(stdout);

    server.WaitForShutdown();
}

void StopInprocServer()
{
    for (auto const& server : ServerSingletons)
    {
        if (server) { server->Stop(); }
    }
    ServerSingletons.clear();
}

void WaitForInprocServerWrites()
{
    static constexpr auto WaitDuration = 50ms;
    auto const            deadline     = PerfClock::now() + ServerWaitTimeout;
    while (true)
    {
        bool foundServer = false;
        bool allComplete = true;
        for (auto const& server : ServerSingletons)
        {
            if (!server) { continue; }
            foundServer = true;
            if (!server->HasSessionsAndAllSessionsComplete()) { allComplete = false; }
        }
        if (foundServer && allComplete) { return; }
        if (PerfClock::now() >= deadline)
        {
            throw std::runtime_error("Timed out waiting for in-process checkpoint server writes to become durable");
        }
        std::this_thread::sleep_for(WaitDuration);
    }
}

void QueryServerSessions(ServerSessionCallback callback, void* cbData)
{
    for (auto& server : ServerSingletons)
    {
        if (server) { server->QuerySessions(callback, cbData); }
    }
}
