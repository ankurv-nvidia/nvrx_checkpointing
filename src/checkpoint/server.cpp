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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <latch>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
namespace bip  = boost::interprocess;
namespace cipc = checkpoint::ipc;
namespace tel  = checkpoint::telemetry;

// Save timestamps cross the Python/daemon process boundary as Unix-epoch nanoseconds.
// Keep every server-side calculation on the same wall-clock epoch.
using PerfClock = std::chrono::system_clock;

static constexpr auto ServerWaitTimeout = 900s;

// uint32_t ranks
// uint32_t iterations
using Request = boost::beast::http::request<boost::beast::http::buffer_body>;

using boost::asio::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
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

    void ReceiveStateRegistryFragments_(cipc::SessionId const&                                               sessionId,
                                        cipc::Rank                                                           fromRank,
                                        tcp_stream&                                                          stream,
                                        boost::beast::flat_buffer&                                           buffer,
                                        boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser);

    void InitializeShm_(std::scoped_lock<std::mutex> const& /* lock */, size_t totalSize, uint8_t queueCount, size_t slotCount);
    void InitializeShmFromRegistry_(uint64_t snapshotSize, RankTopology const& topo);
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
template <typename TStream> static void TryCleanShutdown(TStream& stream)
{
    boost::system::error_code ec;
    ec = stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    if (ec && ec != boost::system::errc::not_connected)
    {    //
        ERR("Shutdown Error: {}", ec.what());
    }
}

using BufStream = boost::iostreams::stream<boost::iostreams::array_source>;

// Every rank of a run pads to the digit count of the world size so a plain name sort orders the dumps by rank.
static std::string RankRegistryDumpName(Rank rank, Rank worldSize)
{
    auto const digits = static_cast<size_t>(std::log10(std::max(static_cast<double>(static_cast<uint32_t>(worldSize)), 1.0))) + 1;
    return fmt::format("registry.{:0{}}.bin", static_cast<uint32_t>(rank), digits);
}

static void DumpRegistry(std::string_view fileName, std::function<void(std::ostream&)> const& write)
{
    std::error_code ec;
    auto const      logDir = std::filesystem::absolute(checkpoint::config::GetConfigValue<checkpoint::config::LogFile>(), ec).parent_path();
    if (!std::filesystem::is_directory(logDir, ec)) { return; }
    auto const path    = logDir / fileName;
    auto const tmpPath = logDir / fmt::format("{}.tmp", fileName);
    try
    {
        std::ofstream ostrm(tmpPath, std::ios::binary | std::ios::trunc);
        write(ostrm);
        ostrm.close();
        if (!ostrm) { throw std::runtime_error(fmt::format("cannot write {}", tmpPath)); }

        std::filesystem::rename(tmpPath, path);
    } catch (std::exception const& ex) { WARN("Failed to dump registry {}: {}", fileName, ex.what()); }
}

static constexpr int  MaxRetries                     = 20;
static constexpr auto IntervalBetweenRetries         = std::chrono::milliseconds(100);
static constexpr auto ThreadStallWaitForEventTimeout = std::chrono::seconds(30);

static void SendToPeerWithRetries(Server const* const     server,
                                  cipc::SessionId const   sessionId,
                                  std::string_view const& peer,
                                  std::string_view const& body,
                                  std::string_view const& target,
                                  std::string_view const& contentEncoding)
{
    std::string lastError;
    for (int attempt = 1; attempt <= MaxRetries; ++attempt)
    {
        try
        {
            boost::asio::io_context  ioc;
            boost::beast::tcp_stream stream(ioc);
            auto                     req = CreateRequestForPeer(ioc, stream, peer, target);
            if (!contentEncoding.empty()) { req.set(boost::beast::http::field::content_encoding, contentEncoding); }
            req.body().data = const_cast<char*>(body.data());    // NOLINT — beast buffer_body is non-owning; body outlives the write
            req.body().size = body.size();
            req.body().more = false;
            req.prepare_payload();
            boost::beast::http::write(stream, req);

#ifdef USE_EXPECT_CONTINUE
            if (expectContinue) { req.set(boost::beast::http::field::expect, "100-continue"); }

            boost::beast::flat_buffer                                               buffer;
            boost::beast::http::request_serializer<boost::beast::http::buffer_body> serializer{req};

            // Offer the body rather than upload it. A peer that already holds the registry
            // answers 204 here and the payload never goes on the wire. Declining mid-upload
            // instead would race a connection reset against the response.
            if (expectContinue)
            {
                boost::beast::http::write_header(stream, serializer);
                boost::beast::http::response_parser<boost::beast::http::empty_body> interim;
                boost::beast::http::read_header(stream, buffer, interim);
                auto const offerResult = interim.get().result();
                if (offerResult == boost::beast::http::status::no_content)
                {
                    TryCleanShutdown(stream);
                    return false;
                }
                if (offerResult != boost::beast::http::status::continue_)
                {
                    throw std::runtime_error(fmt::format("peer answered the body offer with HTTP {}", interim.get().result_int()));
                }
            }
            boost::beast::http::write(stream, serializer);
#endif
            boost::beast::flat_buffer                                     buffer;
            boost::beast::http::response<boost::beast::http::string_body> res;
            boost::beast::http::read(stream, buffer, res);
            if (res.result() != boost::beast::http::status::ok)
            {
                throw std::runtime_error(fmt::format("peer returned HTTP {}", res.result_int()));
            }
            TryCleanShutdown(stream);
            return;
        } catch (std::exception const& ex)
        {
            lastError = ex.what();
            if (attempt < MaxRetries)
            {
                WARN("[{}#{}] Attempt {} peer={} failed target={} {}", *server, sessionId, attempt, peer, target, ex.what());
            }
            else
            {
                ERR("[{}#{}] Attempt {} peer={} failed target={} {}", *server, sessionId, attempt, peer, target, ex.what());
            }
            if (attempt < MaxRetries) { std::this_thread::sleep_for(IntervalBetweenRetries); }
        }
    }
    throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Failed to notify peer={} target={} after {} attempts: {}",
                                         *server,
                                         sessionId,
                                         peer,
                                         target,
                                         MaxRetries,
                                         lastError));
}

static std::string WriteCkptCompleteBody(std::string_view failure, std::map<std::string, uint64_t> const& dynamicBlobSizes)
{
    std::ostringstream body;
    serdes::WriteBlobWithSize<uint32_t>(body, failure);
    serdes::WriteTrivial(body, static_cast<uint32_t>(dynamicBlobSizes.size()));
    for (auto const& [fqn, size] : dynamicBlobSizes)
    {
        serdes::WriteBlobWithSize<uint32_t>(body, fqn);
        serdes::WriteTrivial(body, size);
    }
    return body.str();
}

static void SendWriteCkptCompleteToPeers(Server const* const     server,
                                         cipc::SessionId         sessionId,
                                         RankTopology const&     topo,
                                         cipc::SnapshotId        snapshotId,
                                         std::string_view const& body,
                                         size_t                  startPeerIndex,
                                         size_t                  endPeerIndex)
{
    if (startPeerIndex >= endPeerIndex) { return; }
    auto        delegateStart = (startPeerIndex + endPeerIndex) / 2;
    auto const& peer          = topo.peers[delegateStart];
    auto const  postTarget
        = fmt::format("/api/peer/writeckptcomplete/{}/0/{}?fwd={}:{}", sessionId, snapshotId, delegateStart, endPeerIndex);
    SendToPeerWithRetries(server, sessionId, peer, body, postTarget, {});
    SendWriteCkptCompleteToPeers(server, sessionId, topo, snapshotId, body, startPeerIndex, delegateStart);
}

static void FailPlanning(Session& session, std::string failure)
{
    std::scoped_lock lock(session.mutex);
    if (!session.sessionState.planningFailure.empty()) { return; }
    session.sessionState.planningFailure = std::move(failure);
    if (session.stopRequested) { return; }
    ERR("[{}#{}] Snapshot planning failed: {}", *session.server, session.sessionId, session.sessionState.planningFailure);
    auto const message = session.sessionState.planningFailure;
    session.asyncTasks.emplace_back(std::async(std::launch::async, [&session, message] {
        std::ostringstream result;
        serdes::WriteBlobWithSize<uint32_t>(result, message);
        auto const body      = result.str();
        auto const firstPeer = session.IsCoordinatorPeer() ? size_t{1} : size_t{0};
        auto const lastPeer  = session.IsCoordinatorPeer() ? session.topo.peers.size() : size_t{1};
        for (size_t peer = firstPeer; peer < lastPeer; ++peer)
        {
            try
            {
                SendToPeerWithRetries(session.server,
                                      session.sessionId,
                                      session.topo.peers[peer],
                                      body,
                                      fmt::format("/api/peer/globalregistry/{}", session.sessionId),
                                      {});
            } catch (std::exception const& error)
            {
                ERR("[{}#{}] Failed to propagate planning error: {}", *session.server, session.sessionId, error.what());
            }
        }
    }));
}

static void InstallGlobalRegistry(Session& session, std::shared_ptr<StateTree> globalRegistry)
{

    struct Assignment
    {
        StateNode::Handle node{StateNode::Handle::Invalid};
        std::string       fqn;
        Tensor::DimRange  ranges;
        uint64_t          size{};
        std::vector<Rank> candidates;
        Rank              owner{Rank::Invalid};
    };
    globalRegistry->FinalizeRegistry();
    auto const firstRank = static_cast<uint32_t>(session.PeerIndex()) * session.topo.ranksPerPeer;
    auto const rankCount = std::min(session.topo.ranksPerPeer, static_cast<uint32_t>(session.topo.worldSize) - firstRank);
    auto const worldSize = static_cast<uint32_t>(globalRegistry->GlobalWorldSize());
    if (firstRank > worldSize || rankCount > worldSize - firstRank)
    {
        throw std::runtime_error("Snapshot registry rank range exceeds the world");
    }
    if (worldSize == 0) { throw std::runtime_error("Cannot select owners for a zero-rank state registry"); }
    std::vector<Assignment> assignments;
    std::as_const(*globalRegistry).WalkNodes([&](StateNode const& node) {
        auto const type = node.GetType();
        if (type == StateNode::Type::Dict || type == StateNode::Type::Invalid) { return; }
        auto const fqn = globalRegistry->GetNodeId(node.handle);
        if (type == StateNode::Type::Scalar || type == StateNode::Type::DynamicSizeBlob)
        {
            auto const& replicas = type == StateNode::Type::Scalar ? std::get<StateNode::Scalar>(node.data).replicas
                                                                   : std::get<StateNode::DynamicSizeBlob>(node.data).replicas;
            if (replicas.empty()) { throw std::runtime_error(fmt::format("State '{}' has no replica ranks", fqn)); }
            assignments.push_back(Assignment{.node       = node.handle,
                                             .fqn        = fqn,
                                             .ranges     = {},
                                             .size       = type == StateNode::Type::Scalar ? sizeof(Primitives64Bit) : 0,
                                             .candidates = {replicas.begin(), replicas.end()}});
            return;
        }
        if (type != StateNode::Type::DTensor && type != StateNode::Type::FixedSizeBlob)
        {
            throw std::runtime_error(fmt::format("State '{}' is not normalized for owner selection", fqn));
        }
        auto const& tensor = type == StateNode::Type::FixedSizeBlob
                                 ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                 : std::get<DTensor>(node.data);
        if (!tensor.holes.empty()) { throw std::runtime_error(fmt::format("State '{}' has incomplete global shard coverage", fqn)); }
        std::map<Tensor::DimRange, std::vector<TensorShard const*>> groups;
        for (auto const& shard : tensor.shards)
        {
            if (shard.sharddim.size() != tensor.fulldim.size())
            {
                throw std::runtime_error(fmt::format("State '{}' has incompatible shard dimensions", fqn));
            }
            for (size_t dim = 0; dim < tensor.fulldim.size(); ++dim)
            {
                auto const range = shard.sharddim[dim];
                if (range.begin >= range.end || range.end > tensor.fulldim[dim])
                {
                    throw std::runtime_error(fmt::format("State '{}' has an invalid shard range", fqn));
                }
                if (type == StateNode::Type::FixedSizeBlob && range.end - range.begin != 1)
                {
                    throw std::runtime_error(fmt::format("Fixed blob '{}' shards contain one logical object", fqn));
                }
            }
            groups[shard.sharddim].push_back(&shard);
        }
        for (auto const& [ranges, copies] : groups)
        {
            Assignment entry{.node = node.handle, .fqn = fqn, .ranges = ranges, .size = copies.front()->storageSize, .candidates = {}};
            for (auto const* copy : copies)
            {
                if (type == StateNode::Type::FixedSizeBlob && copy->storageSize != entry.size)
                {
                    throw std::runtime_error(fmt::format("Fixed blob '{}' replicas have inconsistent sizes", fqn));
                }
                entry.candidates.push_back(copy->rank);
            }
            assignments.push_back(std::move(entry));
        }
    });
    std::ranges::sort(assignments, [](Assignment const& lhs, Assignment const& rhs) {
        if (lhs.candidates.size() != rhs.candidates.size()) { return lhs.candidates.size() < rhs.candidates.size(); }
        if (lhs.size != rhs.size) { return lhs.size > rhs.size; }
        if (lhs.fqn != rhs.fqn) { return lhs.fqn < rhs.fqn; }
        return lhs.ranges < rhs.ranges;
    });
    std::vector<uint64_t> assignedBytes(worldSize, 0);
    for (auto& item : assignments)
    {
        auto selected = item.candidates.front();
        for (auto const candidate : item.candidates)
        {
            auto const candidateIndex = static_cast<uint32_t>(candidate);
            auto const selectedIndex  = static_cast<uint32_t>(selected);
            if (assignedBytes[candidateIndex] < assignedBytes[selectedIndex]
                || (assignedBytes[candidateIndex] == assignedBytes[selectedIndex] && candidate < selected))
            {
                selected = candidate;
            }
        }
        item.owner       = selected;
        auto const index = static_cast<uint32_t>(selected);
        if (item.size > std::numeric_limits<uint64_t>::max() - assignedBytes[index])
        {
            throw std::overflow_error("Assigned snapshot payload size overflow");
        }
        assignedBytes[index] += item.size;
    }

    std::map<Rank, std::shared_ptr<StateTree>> registries;
    for (uint32_t index = 0; index < rankCount; ++index)
    {
        auto topology = globalRegistry->Topo();
        topology.rank = Rank{firstRank + index};
        registries.emplace(topology.rank, std::make_shared<StateTree>(globalRegistry->SessionId(), std::move(topology)));
    }
    auto dedupedGlobalRegistry = std::make_shared<StateTree>(globalRegistry->SessionId(), session.topo);
    std::map<StateNode::Handle, std::vector<Assignment const*>> byNode;
    for (auto const& item : assignments)
    {
        if (session.IsCoordinatorPeer() || registries.contains(item.owner)) { byNode[item.node].push_back(&item); }
    }
    std::as_const(*globalRegistry).WalkNodes([&](StateNode const& node) {
        auto const entries = byNode.find(node.handle);
        if (entries == byNode.end()) { return; }
        auto appendNode = [&](StateTree& registry, std::vector<Assignment const*> const& selected) {
            auto const fqn            = globalRegistry->GetNodeId(node.handle);
            auto const [name, parent] = registry.FindOrAddNodeByFQN(fqn);
            auto& output              = registry.AddChild(parent, name);
            output.data               = node.data;
            if (node.GetType() == StateNode::Type::Scalar)
            {
                std::get<StateNode::Scalar>(output.data).replicas = {selected.front()->owner};
            }
            else if (node.GetType() == StateNode::Type::DynamicSizeBlob)
            {
                std::get<StateNode::DynamicSizeBlob>(output.data).replicas = {selected.front()->owner};
            }
            else
            {
                auto const& metadata = node.GetType() == StateNode::Type::FixedSizeBlob
                                           ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                           : std::get<DTensor>(node.data);
                DTensor     chosen(metadata.durableDtype, metadata.fulldim);
                for (auto const* entry : selected)
                {
                    auto const sourceShard = std::ranges::find_if(metadata.shards, [&](TensorShard const& shard) {
                        return shard.rank == entry->owner && shard.sharddim == entry->ranges;
                    });
                    chosen.AddShard(*sourceShard);
                }
                if (node.GetType() == StateNode::Type::FixedSizeBlob) { output.data = StateNode::FixedSizeBlob(chosen); }
                else
                {
                    output.data = std::move(chosen);
                }
            }
        };
        appendNode(*dedupedGlobalRegistry, entries->second);
        std::map<Rank, std::vector<Assignment const*>> byRank;
        for (auto const* entry : entries->second)
        {
            if (registries.contains(entry->owner)) { byRank[entry->owner].push_back(entry); }
        }
        for (auto const& [rank, selected] : byRank) { appendNode(*registries.at(rank), selected); }
    });
    for (auto const& [rank, registry] : registries)
    {
        static_cast<void>(rank);
        registry->FinalizeRegistry();
    }

    dedupedGlobalRegistry->FinalizeRegistry();
    std::scoped_lock lock(session.mutex);
    session.sessionState.dedupedGlobalRegistry   = std::move(dedupedGlobalRegistry);
    session.sessionState.globalRegistry          = std::move(globalRegistry);
    session.sessionState.snapshotRegistryForRank = std::move(registries);
    session.sessionState.planningReady           = true;
}

Session::Session(Server* serverIn LFTBND, cipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn) :
    server(serverIn), sessionId(sessionIdIn), topo(std::move(topoIn)), workDir(std::move(workDirIn))
{}

Session::Snapshot& Session::GetSnapshot_(cipc::SnapshotId snapshotId) LFTBND
{
    std::scoped_lock lock(mutex);
    return GetSnapshot_(lock, snapshotId);
}

Session::Snapshot& Session::GetSnapshot_(std::scoped_lock<std::mutex> const& /* lock */, cipc::SnapshotId snapshotId) LFTBND
{
    auto it = snapshots.find(snapshotId);
    if (it == snapshots.end())
    {
        throw std::runtime_error(
            fmt::format("[native_ckpt:svr][{}#{}][ERROR] GetSnapshot_: no snapshot with id={}", *server, sessionId, snapshotId));
    }
    return *it->second;
}

struct WaitingForRanks
{
    std::unordered_set<Rank> const& receivedFrom;    // NOLINT
    Rank                            begin;
    Rank                            end;
};

template <> struct fmt::formatter<WaitingForRanks> : fmt::formatter<std::string_view>
{
    auto format(WaitingForRanks const& wfr, fmt::format_context& ctx) const    // NOLINT
    {
        static constexpr size_t MaxWaitingForVerbose = 10;

        auto needed = static_cast<uint32_t>(wfr.end) - static_cast<uint32_t>(wfr.begin) - wfr.receivedFrom.size();

        if (needed > MaxWaitingForVerbose) { return format_to(ctx.out(), "waitingFor=[{} ranks]", needed); }

        std::vector<Rank> waitingFor;
        for (auto rank = static_cast<uint32_t>(wfr.begin); rank < static_cast<uint32_t>(wfr.end); ++rank)
        {
            if (!wfr.receivedFrom.contains(Rank{rank})) { waitingFor.push_back(Rank{rank}); }
        }
        return format_to(ctx.out(), "{}", fmt::format("waitingFor=[{}]", fmt::join(waitingFor, ",")));
    }
};

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Session::OnFinalizeSnapshotAsync_(cipc::Message::FinalizeSnapshot const& hdr) noexcept
{
    // Ranks on the coordinator peer (peers[0]) record their finalize/write-complete
    // locally; only ranks on other peers forward those notifications to the coordinator.
    bool const onCoordinatorPeer = static_cast<uint32_t>(hdr.globalRank) / topo.ranksPerPeer == 0;
    bool const notifyCoordinator = !onCoordinatorPeer;

    Snapshot*      snapshotPtr      = nullptr;
    SnapshotState* snapshotStatePtr = nullptr;
    {
        std::scoped_lock lock(mutex);
        snapshotPtr      = snapshots.at(hdr.snapshotId).get();
        snapshotStatePtr = snapshotPtr->snapshotStateForRank.at(hdr.globalRank).get();
    }
    auto& snapshot      = *snapshotPtr;
    auto& snapshotState = *snapshotStatePtr;

    auto const startedAt                   = snapshotState.startedAt;
    auto const numTensors                  = snapshotState.snapshotRegistry->GetTotalNumTensors();
    auto const totalTensorStateSizeInBytes = snapshotState.snapshotRegistry->GetTotalTensorStateSizeInBytes();

    try
    {
        {
            std::scoped_lock lock(mutex);
            if (!snapshot.failures.empty()) { throw std::runtime_error("Snapshot already has failures"); }
            if (!sessionState.planningFailure.empty()) { throw std::runtime_error(sessionState.planningFailure); }
        }
        if (notifyCoordinator)
        {
            SendToPeerWithRetries(server,
                                  sessionId,
                                  topo.peers[0],
                                  {},
                                  fmt::format("/api/peer/snapshotcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                  {});
        }
        checkpoint::distcp::WriteRankShard(
            snapshot.path.parent_path(), hdr.globalRank, *snapshotState.snapshotRegistry, snapshotState.snapshotData);
        std::map<std::string, uint64_t> dynamicBlobSizes;
        std::as_const(*snapshotState.snapshotRegistry).WalkNodes([&](StateNode const& node) {
            if (auto const* blob = std::get_if<StateNode::DynamicSizeBlob>(&node.data))
            {
                dynamicBlobSizes.emplace(snapshotState.snapshotRegistry->GetNodeId(node.handle), blob->storageSize);
            }
        });
        if (notifyCoordinator)
        {
            SendToPeerWithRetries(server,
                                  sessionId,
                                  topo.peers[0],
                                  WriteCkptCompleteBody({}, dynamicBlobSizes),
                                  fmt::format("/api/peer/writeckptcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                  {});
        }
        // A peer may declare its local shard complete only after the coordinator has
        // acknowledged the notification. Rank 0 is delayed further, until metadata is durable.
        {
            std::scoped_lock lock(mutex);
            if (!notifyCoordinator) { snapshot.dynamicBlobSizes.insert(dynamicBlobSizes.begin(), dynamicBlobSizes.end()); }
            if (hdr.globalRank != Rank{0}) { snapshot.writeCkptCompletedRanks.insert(hdr.globalRank); }
        }

        // Rank 0 emits metadata once every other rank has finished its shard. Its own
        // shard was written synchronously above; rank 0 joins the completion set only
        // after metadata is fsync'd, making a world-size set the global durability marker.
        if (hdr.globalRank == Rank{0})
        {
            std::map<std::string, uint64_t> blobSizes;
            auto const                      otherRanks = static_cast<size_t>(topo.worldSize) - 1;
            while (true)
            {
                {
                    std::scoped_lock<std::mutex> lock(mutex);
                    if (!snapshot.failures.empty()) { throw std::runtime_error("Aborting Wait for snapshot write completion"); }
                    if (snapshot.writeCkptCompletedRanks.size() >= otherRanks)
                    {
                        blobSizes = snapshot.dynamicBlobSizes;
                        break;
                    }
                    INFO("[{}#{}] FinalizeSnapshot waiting for snapshot={} write-completed ranks {} of {} remaining...",
                         *server,
                         sessionId,
                         hdr.snapshotId,
                         otherRanks - snapshot.writeCkptCompletedRanks.size(),
                         otherRanks);
                }
                if (stopRequested) { throw std::runtime_error("Stop requested while waiting for ranks to write checkpoint"); }

                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            try
            {
                auto const id = tel::Id{.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId};
                PERF() << tel::span::SvrWriteMetadata::Begin{.id = id};
                checkpoint::distcp::WriteMetadata(snapshot.path.parent_path(), *sessionState.dedupedGlobalRegistry, blobSizes);
                PERF() << tel::span::SvrWriteMetadata::End{.id = id};
            } catch (std::exception const& ex)
            {
                throw std::runtime_error(fmt::format("[{}#{}] WriteMetadata failed for snapshot={} rank={}: {}",
                                                     *server,
                                                     sessionId,
                                                     hdr.snapshotId,
                                                     hdr.globalRank,
                                                     ex.what()));
            }
            std::chrono::microseconds durableUs{};
            {
                std::scoped_lock lock(mutex);
                snapshot.writeCkptCompletedRanks.insert(Rank{0});
                durableUs = std::chrono::duration_cast<std::chrono::microseconds>(PerfClock::now() - snapshot.activatedAt);
            }
            // Rank 0 joins last, so the set is now world-size: every shard and the metadata
            // are on disk and the checkpoint is loadable.
            // This raw marker has a strict schema consumed by CI.
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][PERF] Snapshot={} Durable ranks={} | Time(us): activate_to_durable_us={}\n",
                       *server,
                       sessionId,
                       hdr.snapshotId,
                       static_cast<size_t>(topo.worldSize),
                       durableUs.count());
            PERF() << tel::span::SvrDurable::End{.id = {.sessionId = sessionId, .rank = Rank{0}, .snapshotId = hdr.snapshotId}};
            BroadcastWriteCkptCompleteToPeers_(snapshot);
            {
                std::scoped_lock lock(mutex);
                snapshot.complete = true;
            }
        }
    } catch (std::exception const& ex)
    {
        std::string failure;
        {
            std::scoped_lock lock(mutex);
            snapshot.failures.emplace_back(
                fmt::format("snapshot={} rank={} Finalize failed: {}", hdr.snapshotId, hdr.globalRank, ex.what()));
            failure = fmt::format("{}", fmt::join(snapshot.failures, ", "));
        }
        ERR("[{}#{}] Snapshot finalization failed for snapshot={} rank={}: {}",
            *server,
            sessionId,
            hdr.snapshotId,
            hdr.globalRank,
            ex.what());
        if (hdr.globalRank == Rank{0})
        {
            try
            {
                BroadcastWriteCkptCompleteToPeers_(snapshot);
            } catch (std::exception const& notifyError)
            {
                ERR("[{}#{}] Failed to broadcast checkpoint failure: {}", *server, sessionId, notifyError.what());
            }
            std::scoped_lock lock(mutex);
            snapshot.complete = true;
        }
        else if (notifyCoordinator)
        {
            bool reportFailure = false;
            {
                std::scoped_lock lock(mutex);
                reportFailure = !snapshot.complete;
            }
            if (reportFailure)
            {
                try
                {
                    SendToPeerWithRetries(server,
                                          sessionId,
                                          topo.peers[0],
                                          WriteCkptCompleteBody(failure, {}),
                                          fmt::format("/api/peer/writeckptcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                          {});
                } catch (std::exception const& notifyError)
                {
                    ERR("[{}#{}] Failed to report checkpoint failure: {}", *server, sessionId, notifyError.what());
                }
            }
        }
    }

    std::string failure;
    {
        std::scoped_lock lock(mutex);
        if (!snapshot.failures.empty()) { failure = snapshot.failures.front(); }
    }
    INFO("[{}#{}] OnFinalizeSnapshotAsync_ for rank={}, snapshotId={} failure={}",
         *server,
         sessionId,
         hdr.globalRank,
         hdr.snapshotId,
         failure.empty() ? "<none>" : failure);
    auto const totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(PerfClock::now().time_since_epoch() - startedAt);
    PERF() << tel::span::SvrFinalize::End{.id = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId}};
    PERF() << tel::span::CkptSave::End{
        .id                = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId},
        .d2hDuration       = std::chrono::duration_cast<std::chrono::microseconds>(hdr.d2hDuration),
        .snapshotDuration  = std::chrono::duration_cast<std::chrono::microseconds>(hdr.snapshotDuration),
        .totalDuration     = totalDuration,
        .numTensors        = numTensors,
        .d2hRequestCount   = hdr.d2hRequests,
        .totalTensorBytes  = totalTensorStateSizeInBytes,
        .totalWrittenBytes = hdr.totalSnapshotBytes,
    };

    try
    {
        // Completed local writes release tensor buffers while retaining their finalization state.
        if (hdr.globalRank == Rank{0} || !onCoordinatorPeer)
        {
            // Large buffers are freed outside the session mutex.
            std::vector<std::unique_ptr<Snapshot>> evicted;
            size_t                                 remainingSnapshots = 0;
            {
                std::scoped_lock lock(mutex);
                remainingSnapshots = static_cast<size_t>(
                    std::ranges::count_if(snapshots, [](auto const& entry) { return !entry.second->snapshotStateForRank.empty(); }));
                for (auto const& [id, candidate] : snapshots)
                {
                    if (remainingSnapshots <= numActiveSnapshots) { break; }
                    if (candidate->snapshotStateForRank.empty()) { continue; }
                    // Local writes finish before their buffers are released.
                    if (!candidate->failures.empty() || candidate->finalizedRanks.size() < ExpectedWriteCompletions()
                        || candidate->writeCkptCompletedRanks.size() < ExpectedWriteCompletions())
                    {
                        break;
                    }
                    auto& released = *evicted.emplace_back(std::make_unique<Snapshot>());
                    released.snapshotStateForRank.swap(candidate->snapshotStateForRank);
                    released.dynamicBlobSizes.swap(candidate->dynamicBlobSizes);
                    --remainingSnapshots;
                }
            }
            INFO("[{}#{}] Released {} snapshot buffers after finalizing snapshot={} rank={}. Remaining snapshots with buffers={} "
                 "numActiveSnapshots={}",
                 *server,
                 sessionId,
                 evicted.size(),
                 hdr.snapshotId,
                 hdr.globalRank,
                 remainingSnapshots,
                 numActiveSnapshots);
        }
    } catch (std::exception const& ex)
    {
        ERR("[{}#{}] Exception occurred while releasing snapshot buffers: {}", *server, sessionId, ex.what());
    }
}

void Session::SendLocalShardedStateRegistryToPeer_(cipc::Rank rank, std::string const& peer)
{
    std::shared_ptr<StateTree> registry;
    {
        std::scoped_lock lock(mutex);
        registry = sessionState.localRegistryForRank.at(rank);
    }
    std::ostringstream body;
    StateSerDes::Write(body, *registry, StateNode::Handle::Root);
    SendToPeerWithRetries(server, sessionId, peer, body.str(), fmt::format("/api/peer/shardedstateregistry/{}/{}", sessionId, rank), {});
}

static std::string DeflateRegistry(StateTree const& registry, uint64_t& rawSize)
{
    static constexpr int RegistryCompressionLevel = 1;

    std::string out;
    {
        boost::iostreams::filtering_ostream ostrm;
        ostrm.push(boost::iostreams::counter());
        ostrm.push(boost::iostreams::zlib_compressor(boost::iostreams::zlib_params(RegistryCompressionLevel)));
        ostrm.push(boost::iostreams::back_inserter(out));
        StateSerDes::Write(ostrm, registry, StateNode::Handle::Root);
        ostrm.strict_sync();
        rawSize = static_cast<uint64_t>(ostrm.component<boost::iostreams::counter>(0)->characters());
    }
    return out;
}

static std::shared_ptr<StateTree> InflateRegistry(std::string_view bytes)
{
    boost::iostreams::filtering_istream istrm;
    istrm.push(boost::iostreams::zlib_decompressor());
    istrm.push(boost::iostreams::array_source(bytes.data(), bytes.size()));
    return StateSerDes::Read(istrm);
}

void Session::SendLocalShardedStateRegistryToPeerWithRetries_(cipc::Rank rank, std::string const& peer)
{
    try
    {
        SendLocalShardedStateRegistryToPeer_(rank, peer);
    } catch (std::exception const& error) { FailPlanning(*this, error.what()); }
}

void Session::BroadcastWriteCkptCompleteToPeers_(Snapshot const& snapshot) const
{
    if (topo.peers.size() <= 1) { return; }
    std::string failure;
    {
        std::scoped_lock lock(mutex);
        failure = fmt::format("{}", fmt::join(snapshot.failures, ","));
    }
    SendWriteCkptCompleteToPeers(server, sessionId, topo, snapshot.snapshotId, WriteCkptCompleteBody(failure, {}), 1, topo.peers.size());

    INFO("[{}#{}] Broadcast global registry to {} peer(s)", *server, sessionId, topo.peers.size() - 1);
}

void Session::BroadcastGlobalRegistryToPeers_(StateTree const& globalRegistry) const
{
    if (topo.peers.size() <= 1) { return; }
    uint64_t           rawSize  = 0;
    std::string const  registry = DeflateRegistry(globalRegistry, rawSize);
    std::ostringstream body;
    serdes::WriteBlobWithSize<uint32_t>(body, {});
    body.write(registry.data(), static_cast<std::streamsize>(registry.size()));

    INFO("[{}#{}] Global registry {} bytes -> {} deflated ({:.1f}x)",
         *server,
         sessionId,
         rawSize,
         registry.size(),
         static_cast<double>(rawSize) / static_cast<double>(registry.size()));

    SendGlobalRegistryToPeers(body.str(), 1, topo.peers.size());

    INFO("[{}#{}] Broadcast global registry to {} peer(s)", *server, sessionId, topo.peers.size() - 1);
}

void Session::SendGlobalRegistryToPeers(std::string_view const& registry, size_t startPeerIndex, size_t endPeerIndex) const
{
    if (startPeerIndex >= endPeerIndex) { return; }
    auto        delegateStart = (startPeerIndex + endPeerIndex) / 2;
    auto const& peer          = topo.peers[delegateStart];
    auto const  postTarget    = fmt::format("/api/peer/globalregistry/{}?fwd={}:{}", sessionId, delegateStart, endPeerIndex);
    SendToPeerWithRetries(server, sessionId, peer, registry, postTarget, {});
    SendGlobalRegistryToPeers(registry, startPeerIndex, delegateStart);
}

void Session::Activate()
{ start = PerfClock::now(); }

void Session::Handle(cipc::Message::LocalStateRegistry const& hdr, std::span<char const> buf)
{
    if (hdr.messageSize > buf.size())
    {
        throw std::invalid_argument(
            fmt::format("Local state registry declares {} bytes but only {} bytes were received", hdr.messageSize, buf.size()));
    }
    BufStream                  istrm{buf.data(), static_cast<size_t>(hdr.messageSize)};
    std::shared_ptr<StateTree> localRegistry = StateSerDes::Read(istrm);
    if (localRegistry->SessionId() != sessionId)
    {
        throw std::invalid_argument(
            fmt::format("Local registry session {} does not match request session {}", localRegistry->SessionId(), sessionId));
    }
    Rank rank         = localRegistry->Topo().rank;
    auto snapshotSize = localRegistry->GetTotalSnapshotStateSizeInBytes();
    auto registryTopo = localRegistry->Topo();

    localRegistry->WalkNodes([rank](StateNode& node) {
        auto const type = node.GetType();
        if (type == StateNode::Type::Scalar)
        {
            auto& scalar = std::get<StateNode::Scalar>(node.data);
            Primitives64Bit::Traits<uint64_t>::Assign(scalar, uint64_t{});
        }
        else if (type == StateNode::Type::DynamicSizeBlob)
        {
            auto& blob       = std::get<StateNode::DynamicSizeBlob>(node.data);
            blob.handle      = 0;
            blob.storageSize = 0;
        }
        else if (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob)
        {
            auto& tensor = type == StateNode::Type::FixedSizeBlob ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                                                  : std::get<DTensor>(node.data);
            std::set<TensorShard> cleared;
            for (auto shard : tensor.shards)
            {
                if (shard.rank != rank) { throw std::runtime_error("Local state contains another rank's shard"); }
                shard.handle = 0;
                cleared.insert(shard);
            }
            tensor.shards = std::move(cleared);
        }
    });
    localRegistry->FinalizeRegistry();
    INFO("[{}#{}] Received local registry rank={} metadata bytes={} payload bytes={}",
         *server,
         sessionId,
         rank,
         hdr.messageSize,
         snapshotSize);
    bool registryAccepted = false;
    {
        // Concurrent handlers push here; an unlocked emplace corrupts sibling futures.
        std::scoped_lock lock(mutex);
        auto&            registeredRegistry = sessionState.localRegistryForRank[rank];
        if (registeredRegistry != nullptr)
        {
            WARN("[{}#{}] Received a duplicate local state registry from rank={} for an unknown reason; ignoring it",
                 *server,
                 sessionId,
                 rank);
            registryTopo = registeredRegistry->Topo();
        }
        else
        {
            registeredRegistry  = std::move(localRegistry);
            largestSnapshotSize = std::max(largestSnapshotSize, snapshotSize);
            registryAccepted    = true;
            asyncTasks.emplace_back(
                std::async(std::launch::async | std::launch::deferred, &Session::OnReceiveLocalStateRegistryAsync_, this, rank));
            if (static_cast<uint32_t>(rank) / topo.ranksPerPeer != 0)    // TODO_MULTILOCAL_CKPT
            {
                asyncTasks.push_back(std::async(std::launch::async | std::launch::deferred,
                                                &Session::SendLocalShardedStateRegistryToPeerWithRetries_,
                                                this,
                                                rank,
                                                topo.peers[0]));
            }
        }
    }
    if (registryAccepted)
    {
        DumpRegistry(RankRegistryDumpName(rank, registryTopo.worldSize),
                     [&](std::ostream& ostrm) { ostrm.write(buf.data(), static_cast<std::streamsize>(buf.size())); });
    }
    // Each reply waits for the node's last registry, so every rank opens finished queues sized for the largest local snapshot.
    auto const firstLocalRank = static_cast<uint32_t>(PeerIndex()) * topo.ranksPerPeer;
    auto const localRanks     = std::min(topo.ranksPerPeer, static_cast<uint32_t>(topo.worldSize) - firstLocalRank);
    auto const deadline       = PerfClock::now() + std::chrono::duration_cast<PerfClock::duration>(ServerWaitTimeout);
    auto const waitId         = tel::Id{.sessionId = sessionId, .rank = rank, .snapshotId = cipc::SnapshotId::Invalid};
    uint64_t   maxSnapshotSize{};
    PERF() << tel::span::SvrWaitLocalRegistries::Begin{.id = waitId};
    while (true)
    {
        {
            std::scoped_lock lock(mutex);
            maxSnapshotSize = largestSnapshotSize;
            if (PerfClock::now() >= deadline)
            {
                WARN("[native_ckpt:svr][{}#{}][WARN] Wait for local registries timed out", *server, sessionId);
                break;
            }
            if (sessionState.localRegistryForRank.size() >= localRanks || stopRequested) { break; }
        }
        std::this_thread::sleep_for(IntervalBetweenRetries);
    }
    PERF() << tel::span::SvrWaitLocalRegistries::End{.id = waitId};
    server->InitializeShmFromRegistry_(maxSnapshotSize, registryTopo);
}

void Session::Handle(cipc::Message::LocalStateData const& hdr1, std::span<char const> buf)
{
    void* ptr      = nullptr;
    int   numaNode = -1;
    {
        std::scoped_lock lock(mutex);
        auto&            snapshot      = GetSnapshot_(lock, hdr1.snapshotId);
        auto&            snapshotState = *snapshot.snapshotStateForRank.at(hdr1.globalRank);
        auto             endOffset     = hdr1.offset + hdr1.messageSize;
        if (endOffset > snapshotState.snapshotData.size)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}#{}][ERROR] Received snapshot data exceeds allocated size={} for session id={} offset={} size={}",
                *server,
                sessionId,
                snapshotState.snapshotData.size,
                hdr1.sessionId,
                hdr1.offset,
                hdr1.messageSize));
        }

        snapshotState.writtenPayloadSize += hdr1.messageSize;
        snapshotState.writtenPayloadRangeBegin = std::min(snapshotState.writtenPayloadRangeBegin, static_cast<uint64_t>(hdr1.offset));
        snapshotState.writtenPayloadRangeEnd   = std::max(snapshotState.writtenPayloadRangeEnd, endOffset);
        if (snapshotState.writtenPayloadSize > snapshotState.snapshotData.size)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Received snapshot data size={} (range {}-{}) exceeds "
                                                 "total snapshot size={} for session id={}",
                                                 *server,
                                                 sessionId,
                                                 snapshotState.writtenPayloadSize,
                                                 snapshotState.writtenPayloadRangeBegin,
                                                 snapshotState.writtenPayloadRangeEnd,
                                                 snapshotState.snapshotData.size,
                                                 hdr1.sessionId));
        }

        ptr      = snapshotState.snapshotData[hdr1.offset];
        numaNode = snapshotState.numaNode;
    }
    BindThreadCpuToNumaNode(numaNode);
    std::memcpy(ptr, buf.data(), hdr1.messageSize);
}

void Session::Handle(cipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf)
{

    auto AddSize = [](uint64_t total, uint64_t size) -> uint64_t {
        if (size > std::numeric_limits<uint64_t>::max() - total) { throw std::overflow_error("Snapshot payload size overflow"); }
        return total + size;
    };

    BufStream                  stream{buf.data(), buf.size()};
    auto const                 ckptPath = std::filesystem::path(serdes::ReadBlobWithSize<uint16_t>(stream));
    std::shared_ptr<StateTree> registry = StateSerDes::Read(stream);
    if (registry->SessionId() != sessionId || registry->GlobalRank() != hdr.globalRank)
    {
        throw std::invalid_argument("Snapshot registry session or rank does not match activation");
    }
    {
        registry->FinalizeRegistry();
        uint64_t offset = 0;
        for (auto const group : {StateNode::Type::DTensor, StateNode::Type::FixedSizeBlob, StateNode::Type::DynamicSizeBlob})
        {
            std::as_const(*registry).WalkNodes([&](StateNode const& stored) {
                auto&      node = registry->GetNode(stored.handle);
                auto const type = node.GetType();
                if (type == group && (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob))
                {
                    auto&                 tensor = type == StateNode::Type::FixedSizeBlob
                                                       ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                                       : std::get<DTensor>(node.data);
                    std::set<TensorShard> packed;
                    for (auto shard : tensor.shards)
                    {
                        shard.handle = offset;
                        offset       = AddSize(offset, shard.storageSize);
                        packed.insert(shard);
                    }
                    tensor.shards = std::move(packed);
                }
                else if (type == group && type == StateNode::Type::DynamicSizeBlob)
                {
                    auto& blob  = std::get<StateNode::DynamicSizeBlob>(node.data);
                    blob.handle = offset;
                    offset      = AddSize(offset, blob.storageSize);
                }
            });
        }
    }
    bool                                  beginDurableSpan = false;
    std::chrono::system_clock::time_point durableStartedAt;
    {
        std::scoped_lock lock(mutex);
        if (!sessionState.planningReady) { throw std::runtime_error("Snapshot plan is not ready"); }
        if (!sessionState.planningFailure.empty()) { throw std::runtime_error(sessionState.planningFailure); }
        if (!sessionState.snapshotRegistryForRank.contains(hdr.globalRank))
        {
            throw std::invalid_argument("Snapshot rank is not local to this daemon");
        }
        auto const payloadSize = registry->GetTotalSnapshotStateSizeInBytes();
        if (hdr.snapshotSize != payloadSize) { throw std::invalid_argument("Snapshot payload size does not match its registry"); }
        auto& snapshot = snapshots[hdr.snapshotId];
        if (snapshot == nullptr)
        {
            snapshot             = std::make_unique<Snapshot>();
            snapshot->snapshotId = hdr.snapshotId;
        }
        if (snapshot->snapshotStateForRank.contains(hdr.globalRank) || snapshot->finalizedRanks.contains(hdr.globalRank))
        {
            throw std::runtime_error("Snapshot rank is already activated or finalized");
        }
        if (snapshot->path.empty())
        {
            snapshot->name        = fmt::format("snapshot_{}", hdr.snapshotId);
            snapshot->path        = ckptPath / fmt::format("checkpoint_session_{}.snapshot_{}.bin", sessionId, hdr.snapshotId);
            snapshot->activatedAt = PerfClock::now();
            durableStartedAt      = snapshot->activatedAt;
            beginDurableSpan      = true;
        }
        auto snapshotState              = std::make_unique<SnapshotState>();
        snapshotState->snapshotRegistry = registry;
        snapshotState->startedAt        = hdr.startedAt;
        snapshotState->numaNode         = hdr.numaNode;
        snapshotState->snapshotData     = ByteStorage(payloadSize);
        if (snapshotState->snapshotData.data != nullptr)
        {
            BindRegionToNumaNode(hdr.numaNode, snapshotState->snapshotData.data, snapshotState->snapshotData.size);
        }
        snapshot->snapshotStateForRank.emplace(hdr.globalRank, std::move(snapshotState));
    }
    if (beginDurableSpan && IsCoordinatorPeer())
    {
        PERF() << tel::span::SvrDurable::Begin{
            .id        = {.sessionId = sessionId, .rank = Rank{0}, .snapshotId = hdr.snapshotId},
            .startedAt = durableStartedAt,
        };
    }
    PERF() << tel::span::CkptSave::Begin{
        .id        = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId},
        .startedAt = hdr.startedAt,
    };
    INFO("[{}#{}] Activated snapshot={} rank={} payload bytes={} path={}",
         *server,
         sessionId,
         hdr.snapshotId,
         hdr.globalRank,
         hdr.snapshotSize,
         ckptPath);
}    // namespace

void Session::Handle(cipc::Message::FinalizeSnapshot const& hdr, std::span<char const> /* buf */)
{
    PERF() << tel::span::SvrFinalize::Begin{.id = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId}};
    {
        std::scoped_lock lock(mutex);

        auto& snapshot = GetSnapshot_(lock, hdr.snapshotId);
        try
        {
            auto& snapshotState = *snapshot.snapshotStateForRank.at(hdr.globalRank);

            if (snapshotState.writtenPayloadSize != snapshotState.snapshotData.size)
            {
                throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Incomplete snapshot data received. sessionid={}. "
                                                     "rcvd-size={}(range{} - {}) expected={} ",
                                                     *server,
                                                     sessionId,
                                                     hdr.sessionId,
                                                     snapshotState.writtenPayloadSize,
                                                     snapshotState.writtenPayloadRangeBegin,
                                                     snapshotState.writtenPayloadRangeEnd,
                                                     snapshotState.snapshotData.size));
            }
            if (snapshotState.writtenPayloadRangeBegin != 0 || snapshotState.writtenPayloadRangeEnd != snapshotState.snapshotData.size)
            {
                throw std::runtime_error(
                    fmt::format("[native_ckpt:svr][{}#{}][ERROR] Snapshot data received has invalid range. sessionid={}. "
                                "rcvd-size={}(range{} - {}) expected range=0 - {}",
                                *server,
                                sessionId,
                                hdr.sessionId,
                                snapshotState.writtenPayloadSize,
                                snapshotState.writtenPayloadRangeBegin,
                                snapshotState.writtenPayloadRangeEnd,
                                snapshotState.snapshotData.size));
            }
            if (sessionState.globalRegistry == nullptr)
            {
                WARN("[{}#{}] Global state registry is not initialized when finalizing rank={} snapshot={}",
                     *server,
                     sessionId,
                     hdr.globalRank,
                     hdr.snapshotId);
            }
            snapshot.finalizedRanks.insert(hdr.globalRank);
        } catch (std::exception const& e) { snapshot.failures.push_back(fmt::format("Error finalizing snapshot: {}", e.what())); }
    }
    {
        std::scoped_lock lock(mutex);    // local registry uploads can arrive on concurrent HTTP handlers
        asyncTasks.emplace_back(std::async(std::launch::async | std::launch::deferred, &Session::OnFinalizeSnapshotAsync_, this, hdr));
    }
}

void Session::HandleStateRegistryFragment(Rank recvRank, std::shared_ptr<StateTree> registry)
{
    try
    {
        {
            std::scoped_lock lock(mutex);
            if (receivedFrom.contains(recvRank)) { return; }
            if (sessionState.planningStarted) { throw std::runtime_error("Registry arrived after planning started"); }
            if (sessionState.globalRegistry == nullptr) { sessionState.globalRegistry = std::make_shared<StateTree>(sessionId, topo); }
            auto&       global   = *sessionState.globalRegistry;
            auto const& incoming = *registry;
            if (global.SessionId() != incoming.SessionId() || global.GlobalWorldSize() != incoming.GlobalWorldSize()
                || global.Topo().peers != incoming.Topo().peers || global.Topo().ranksPerPeer != incoming.Topo().ranksPerPeer
                || global.Topo().peersPerNode != incoming.Topo().peersPerNode)
            {
                throw std::runtime_error("Cannot merge state registries from incompatible sessions or topologies");
            }
            if (global.IsFinalized()) { throw std::logic_error("Cannot merge into a finalized registry"); }
            incoming.WalkNodes([&](StateNode const& supplied) {
                auto const type = supplied.GetType();
                if (type == StateNode::Type::Invalid || supplied.handle == StateNode::Handle::Root) { return; }
                auto const fqn      = incoming.GetNodeId(supplied.handle);
                auto const previous = global.FindNodeByFQN(fqn);
                if (previous == StateNode::Handle::Invalid)
                {
                    auto const [name, parent] = global.FindOrAddNodeByFQN(fqn);
                    if (type == StateNode::Type::Dict) { global.FindOrAddDictChild(parent, name); }
                    else
                    {
                        global.AddChild(parent, name).data = supplied.data;
                    }
                    return;
                }
                auto& registered = global.GetNode(previous);
                if (registered.GetType() != type)
                {
                    throw std::runtime_error(fmt::format("State '{}' has conflicting registered types", fqn));
                }
                if (type == StateNode::Type::Dict) { return; }
                if (type == StateNode::Type::Scalar)
                {
                    auto&       target = std::get<StateNode::Scalar>(registered.data);
                    auto const& value  = std::get<StateNode::Scalar>(supplied.data);
                    if (target.GetType() != value.GetType())
                    {
                        throw std::runtime_error(fmt::format("Scalar '{}' has conflicting types", fqn));
                    }
                    target.replicas.insert(value.replicas.begin(), value.replicas.end());
                }
                else if (type == StateNode::Type::DynamicSizeBlob)
                {
                    auto&       target = std::get<StateNode::DynamicSizeBlob>(registered.data);
                    auto const& value  = std::get<StateNode::DynamicSizeBlob>(supplied.data);
                    target.replicas.insert(value.replicas.begin(), value.replicas.end());
                }
                else if (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob)
                {
                    auto&       target = type == StateNode::Type::FixedSizeBlob
                                             ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(registered.data))
                                             : std::get<DTensor>(registered.data);
                    auto const& value  = type == StateNode::Type::FixedSizeBlob
                                             ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(supplied.data))
                                             : std::get<DTensor>(supplied.data);
                    if (target.durableDtype != value.durableDtype || target.fulldim != value.fulldim)
                    {
                        throw std::runtime_error(fmt::format("State '{}' has incompatible dtype or global shape", fqn));
                    }
                    for (auto const& shard : value.shards)
                    {
                        auto const existing = target.shards.find(shard);
                        if (existing != target.shards.end())
                        {
                            if (*existing != shard)
                            {
                                throw std::runtime_error(fmt::format("State '{}' has conflicting shard metadata", fqn));
                            }
                            continue;
                        }
                        target.AddShard(shard);
                    }
                }
            });

            receivedFrom.insert(recvRank);
        }
        std::scoped_lock lock(mutex);
        if (!IsCoordinatorPeer() || stopRequested || sessionState.planningStarted || !sessionState.planningFailure.empty()
            || receivedFrom.size() != static_cast<size_t>(topo.worldSize))
        {
            return;
        }
        sessionState.planningStarted = true;
        auto globalRegistry          = sessionState.globalRegistry;
        asyncTasks.emplace_back(std::async(std::launch::async, [this, globalRegistry] {
            try
            {
                globalRegistry->FinalizeRegistry();
                DumpRegistry("registry.global.bin",
                             [&](std::ostream& stream) { StateSerDes::Write(stream, *globalRegistry, StateNode::Handle::Root); });
                BroadcastGlobalRegistryToPeers_(*globalRegistry);
                InstallGlobalRegistry(*this, globalRegistry);
            } catch (std::exception const& error) { FailPlanning(*this, error.what()); }
        }));
    } catch (std::exception const& error) { FailPlanning(*this, error.what()); }
}

void Session::Handle(cipc::Message::FinalizeSession const& /* hdr */, std::span<char const> /* buf */)
{}

bool Session::ResolveSnapshotId(int64_t iteration, cipc::SnapshotId& out) const
{
    if (iteration >= 0)
    {
        if (static_cast<uint64_t>(iteration) >= static_cast<uint32_t>(cipc::SnapshotId::Invalid))
        {
            throw std::invalid_argument(fmt::format("checkpoint iteration {} is outside the supported snapshot range", iteration));
        }
        out = static_cast<cipc::SnapshotId>(static_cast<uint32_t>(iteration));
        return true;
    }
    // Negative: 1-based index back from the newest known snapshot.
    std::scoped_lock lock(mutex);
    auto const       idx = static_cast<uint64_t>(-(iteration + 1)) + 1;
    if (idx <= snapshots.size())
    {
        auto it = snapshots.rbegin();
        std::advance(it, static_cast<std::ptrdiff_t>(idx - 1));
        out = it->first;
        return true;
    }
    return false;
}

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
    auto const shmSize = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferSize>();
    if (shmSize != 0)
    {
        auto queueCount = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferNumQueues>();
        auto slotCount  = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferSlots>();
        if (queueCount == 0) { queueCount = cipc::ExplicitSizeDefaultQueueCount; }
        if (slotCount == 0) { slotCount = cipc::ExplicitSizeDefaultSlotCount; }
        InitializeShm_(lock, shmSize, queueCount, slotCount);
    }
}

void Server::InitializeShm_(std::scoped_lock<std::mutex> const& /* lock */,
                            size_t  configuredTotalSize,
                            uint8_t queueCount,
                            size_t  slotCount)
{
    if (queueCount == 0) { throw std::invalid_argument("at least one shared-memory queue is required"); }
    if (queueCount <= _msgQueue.size()) { return; }
    if (cipc::MsgQueue::MinimumShmSize(slotCount) > std::numeric_limits<size_t>::max() / queueCount)
    {
        throw std::overflow_error("configured shared-memory ring-buffer geometry is too large");
    }

    auto const minimumTotal = std::max(cipc::MinShmSize, cipc::MsgQueue::MinimumShmSize(slotCount) * queueCount);
    auto const totalSize    = cipc::MsgQueue::DetectShmSize(
        cipc::Message::RequestChannelName(_index, queueCount), configuredTotalSize, minimumTotal, slotCount);
    auto const queueSize = totalSize / queueCount;

    _running              = true;
    auto       neededMore = queueCount - _msgQueue.size();
    std::latch queuesReady{static_cast<ptrdiff_t>(neededMore)};
    // Start server thread
    //_serverThread = std::thread([this]() {
    auto origSize = _msgQueue.size();
    _msgQueue.resize(queueCount);
    for (auto i = origSize; i < queueCount; ++i)
    {
        _threads.emplace_back([this, i, queueSize, slotCount, queueCount, &queuesReady] {
            SetThreadName(fmt::format("ncs:svrq:{}", i).c_str());
            // Map queues across host-memory NUMA domains. Pinning around Create
            // places the control page; materializing the data pages in the owner
            // mapping prevents CUDA registration in a restricted client process
            // from faulting every shared page onto one socket.
            auto const numaNodes = DetectHostMemoryNumaNodes();
            int        queueNuma = numaNodes.front();
            if (numaNodes.size() > 1)
            {
                size_t queuesPerDomain = queueCount / numaNodes.size();
                if (queuesPerDomain == 0) { queuesPerDomain = 1; }
                size_t const domain = std::min(numaNodes.size() - 1, static_cast<size_t>(i) / queuesPerDomain);
                queueNuma           = numaNodes[domain];
                BindThreadCpuToNumaNode(queueNuma);
            }
            auto const queueIndex = static_cast<uint8_t>(i);
            auto       msgQueue   = cipc::MsgQueue::Create(cipc::Message::RequestChannelName(_index, queueIndex), queueSize, slotCount);
            if (numaNodes.size() > 1)
            {
                auto region = msgQueue.GetShmDataRegion();
                BindRegionToNumaNode(queueNuma, region.data(), region.size());
                std::fill(region.begin() + static_cast<std::ptrdiff_t>(msgQueue.HeaderSize()), region.end(), char{});
            }
            UnbindThreadCpu();
            // The caller holds _mutex until every queue publishes its geometry.
            // Each thread owns one pre-sized vector element, and the latch makes
            // those writes visible before InitializeShm_ returns.
            _msgQueue[i] = &msgQueue;
            queuesReady.count_down();
            INFO("[{}] Checkpoint server started QueueSize={} Waiting for messages...", *this, queueSize);
            std::fflush(stderr);
            while (true) try
                {
                    auto const [stop, hdr, buf] = msgQueue.BeginRead();
                    if (stop)
                    {
                        std::scoped_lock lock1(_mutex);
                        INFO("[{}] Stop requested. Shutting down server thread.", *this);
                        _msgQueue[i] = nullptr;
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
        });
    }
    queuesReady.wait();
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
    case cipc::Message::Type::LocalStateData:
    {
        auto const& hdr1 = std::get<cipc::Message::LocalStateData>(hdr);
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

void Server::NotifyWriteCkptComplete_(cipc::SessionId                 sessionId,
                                      cipc::SnapshotId                snapshotId,
                                      cipc::Rank                      fromRank,
                                      std::string                     failure,
                                      std::map<std::string, uint64_t> dynamicBlobSizes)
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
    std::scoped_lock lock(_mutex);
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
}

void Server::ReceiveLocalRegistry_(cipc::SessionId sessionId, std::string_view const& buf)
{
    FindOrWaitForSession_("ReceiveLocalRegistry", sessionId, [&](Session& session) {
        session.Handle(cipc::Message::LocalStateRegistry{.sessionId = sessionId, .messageSize = buf.size()}, buf);
    });
}

void Server::InitializeShmFromRegistry_(uint64_t snapshotSize, RankTopology const& topo)
{
    if (topo.ranksPerPeer == 0) { throw std::invalid_argument("local registry has no ranks per peer"); }
    auto const configuredQueueCount = checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferNumQueues>();
    auto const queueCount           = cipc::ResolveQueueCount(topo.ranksPerPeer, configuredQueueCount);
    auto const maxProducerCount     = (topo.ranksPerPeer + queueCount - 1) / queueCount;
    auto const snapshotSizePerQueue = cipc::RoundSnapshotShmSize(static_cast<size_t>(snapshotSize));
    if (snapshotSizePerQueue > std::numeric_limits<size_t>::max() / maxProducerCount / queueCount)
    {
        throw std::overflow_error("automatic shared-memory size exceeds the supported range");
    }
    auto const totalSize = snapshotSizePerQueue * maxProducerCount * queueCount;

    auto slotCount = static_cast<size_t>(checkpoint::config::GetConfigValue<checkpoint::config::ShmRingBufferSlots>());
    if (slotCount == 0)
    {
        auto const queueSize = totalSize / queueCount;
        slotCount            = cipc::MsgQueue::SlotCountForShmSize(queueSize, cipc::MsgQueue::DefaultSlotCount, maxProducerCount);
    }

    std::scoped_lock<std::mutex> lock(_mutex);
    InitializeShm_(lock, totalSize, queueCount, slotCount);
}

void Server::ReceiveGlobalRegistry_(cipc::SessionId sessionId, std::string_view const& body, size_t fwdStart, size_t fwdEnd)
{
    FindOrWaitForSession_("ReceiveGlobalRegistry", sessionId, [&](Session& session) {
        try
        {
            BufStream result{body.data(), body.size()};
            result.exceptions(std::ios::failbit | std::ios::badbit);
            auto failure = serdes::ReadBlobWithSize<uint32_t>(result);
            if (!failure.empty())
            {
                if (session.IsCoordinatorPeer()) { FailPlanning(session, std::move(failure)); }
                else
                {
                    std::scoped_lock lock(session.mutex);
                    session.sessionState.planningFailure = std::move(failure);
                }
                return;
            }
            if (session.PeerIndex() != fwdStart) { throw std::invalid_argument("Global registry forwarding rank does not match daemon"); }
            session.SendGlobalRegistryToPeers(body, fwdStart + 1, fwdEnd);
            auto registry = InflateRegistry(body.substr(sizeof(uint32_t)));
            InstallGlobalRegistry(session, std::move(registry));
        } catch (std::exception const& error) { FailPlanning(session, error.what()); }
    });
}

void Server::ReceiveStateRegistryFragments_(cipc::SessionId const&                                               sessionId,
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
    registry->FinalizeRegistry();
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
    session->HandleStateRegistryFragment(fromRank, std::move(registry));
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

    // Check if this is the state upload endpoint: POST /api/peer/shardedstateregistry/{sessionId}/{rank}
    if (matches({"api", "peer", "shardedstateregistry"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto rank      = cipc::Rank{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in state registry upload endpoint"); }

        boost::beast::http::request_parser<boost::beast::http::buffer_body> bufferParser{std::move(headerParser)};
        ReceiveStateRegistryFragments_(sessionId, rank, stream, buffer, bufferParser);
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
        auto                            failure = serdes::ReadBlobWithSize<uint32_t>(completion);
        std::map<std::string, uint64_t> dynamicBlobSizes;
        auto const                      count = serdes::ReadTrivial<uint32_t>(completion);
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
                SendWriteCkptCompleteToPeers(this, sessionId, session.topo, snapshotId, parser.get().body(), fwdStart + 1, fwdEnd);
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
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);

        ReceiveGlobalRegistry_(sessionId, bodyParser.get().body(), fwdRangeStart, fwdRangeEnd);
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

void Session::OnReceiveLocalStateRegistryAsync_(Rank const& rank)
{
    std::shared_ptr<StateTree> registry;
    {
        std::scoped_lock lock(mutex);
        registry = sessionState.localRegistryForRank.at(rank);
        if (!IsCoordinatorPeer()) { receivedFrom.insert(rank); }
    }
    if (IsCoordinatorPeer()) { HandleStateRegistryFragment(rank, std::move(registry)); }
    std::vector<std::unique_ptr<Server::PendingSessionData>> pending;
    {
        std::scoped_lock lock(server->_mutex);
        auto             item = server->_pendingSessionData.find(sessionId);
        if (item != server->_pendingSessionData.end())
        {
            pending = std::move(item->second);
            server->_pendingSessionData.erase(item);
        }
    }
    for (auto& fragment : pending) { HandleStateRegistryFragment(fragment->recvFrom, std::move(fragment->registry)); }
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
