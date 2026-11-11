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
        PerfClock::time_point                         receivedAt;
        Rank                                          recvFrom{Rank::Invalid};
        std::vector<TensorShardInfo>                  tensorShards;
        std::vector<std::pair<std::string, uint64_t>> blobMetas;
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

[[maybe_unused]] static void DebugDumpLocalStateTreeKVP(std::filesystem::path const& fname, StateTree const& state)
{

    std::ofstream ostrm(fname);
    if (!ostrm.good()) { throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Failed to open file for writing: {}", fname)); }

    state.WalkNodes([&](auto& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::Invalid || type >= StateNode::Type::Unknown)
        {
            throw std::runtime_error(fmt::format("Invalid node type={}", static_cast<uint8_t>(type)));
        }
        if (type == StateNode::Type::Dict) { return; }
        auto const name = state.GetNodeId(node.handle);

        if (type == StateNode::Type::Scalar)
        {
            auto const& scalar = std::get<StateNode::Scalar>(node.data);
            auto const& t      = scalar.GetType();
            if (StateNode::Scalar::Type::IsFloat(t)) { fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<double>()); }
            else if (StateNode::Scalar::Type::IsSigned(t)) { fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<int64_t>()); }
            else if (StateNode::Scalar::Type::IsUnsigned(t)) { fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<uint64_t>()); }
            else
            {
                fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<bool>());
            }
        }
        else if (type == StateNode::Type::DynamicSizeBlob) { fmt::print(ostrm, "{}: [blob]\n", name); }
        else if (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob)
        {
            auto const& tnsr = type == StateNode::Type::FixedSizeBlob
                                   ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                   : std::get<DTensor>(node.data);
            if (tnsr.shards.size() == 0) throw std::runtime_error(fmt::format("Empty Sharded tensor={}", name));
            fmt::print(ostrm, "{}.fulldim: [{}]\n", name, tnsr.fulldim);
            fmt::print(ostrm, "{}.durableDtype: {}\n", name, tnsr.durableDtype);

            for (auto const& rshard : tnsr.shards)
            {
                // fmt::print(ostrm, "{}.sharddim@{}.device: {}\n", name, rshard.rank, rshard.device);
                fmt::print(ostrm, "{}.sharddim@{}.shards: [{}]\n", name, rshard.rank, rshard.sharddim);
            }
        }
    });
}

// Peer registry-fragment record tags. A fragment stream is a sequence of
// tag-prefixed records terminated by a clean end-of-stream at a record boundary:
// tensor shards (the merged global tensor view) followed by ShardedObject metadata
// (fqn + length; the owning rank writes the bytes itself).
static constexpr uint8_t PeerRecordTensor    = 0;
static constexpr uint8_t PeerRecordBlob      = 1;
static constexpr uint8_t PeerRecordFixedBlob = 2;

static void WriteTensorShardInfo(std::ostream& ostrm, TensorShardInfo const& info)
{
    serdes::WriteTrivial(ostrm, info.durableDtype);
    serdes::WriteTrivial(ostrm, info.fulldim);
    serdes::WriteTrivial(ostrm, info.tensorInfo);
    serdes::WriteBlobWithSize<uint32_t>(ostrm, info.nodeId);
}

static void ValidateLocalTensorMetadata(StateTree const& registry)
{
    std::unordered_set<StateTree::NodeId> tensorNodeIds;
    registry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::FixedSizeBlob && type != StateNode::Type::DTensor) { return; }

        auto const nodeId = registry.GetNodeId(node.handle);
        if (!tensorNodeIds.insert(nodeId).second)
        {
            throw std::runtime_error(
                fmt::format("Local tensor '{}' is registered more than once; represent multiple shards with one DTensor node", nodeId));
        }
        auto const& tensor = type == StateNode::Type::FixedSizeBlob
                                 ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                 : std::get<DTensor>(node.data);
        for (auto const& shard : tensor.shards)
        {
            if (shard.rank != registry.Topo().rank)
            {
                throw std::runtime_error(
                    fmt::format("Local tensor '{}' has shard rank={} but registry rank={}", nodeId, shard.rank, registry.Topo().rank));
            }
        }
    });
}

template <typename TLambda> static void ForeachTensorShardInfo(StateTree const& registry, TLambda const& lambda)
{
    registry.WalkNodes([&](auto& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob)
        {
            auto const& tnsr = type == StateNode::Type::FixedSizeBlob
                                   ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                   : std::get<DTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                shard.AssertShardIsValid(tnsr.fulldim);
                lambda({tnsr.durableDtype, tnsr.fulldim, shard, registry.GetNodeId(node.handle), type});
            }
        }
    });
}

static constexpr std::string_view RegistryContentEncoding = "deflate";

void Session::SendLocalShardedStateRegistryToPeer_(cipc::Rank rank, std::string const& peer)
{
    boost::asio::io_context  ioc;
    boost::beast::tcp_stream stream(ioc);
    {
        auto req = CreateRequestForPeer(ioc, stream, peer, fmt::format("/api/peer/shardedstateregistry/{}/{}", sessionId, rank));
        req.set(boost::beast::http::field::transfer_encoding, "chunked");
        // req.set(boost::beast::http::field::expect, "100-continue");
        boost::beast::http::request_serializer<boost::beast::http::buffer_body, boost::beast::http::fields> sr{req};
        boost::beast::http::write_header(stream, sr);

        // Each record carries logical tensor metadata, a physical shard and its state path.
        size_t numTensors = 0;

        auto localRegistry = GetLocalRegistry_(rank);
        ForeachTensorShardInfo(*localRegistry, [&](TensorShardInfo const& info) {
            // TODO : use constant sized stack based buffer streams later
            std::ostringstream oss;
            serdes::WriteTrivial(oss, info.type == StateNode::Type::FixedSizeBlob ? PeerRecordFixedBlob : PeerRecordTensor);
            WriteTensorShardInfo(oss, info);
            boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(oss.str())));
            numTensors++;
        });

        // Blob metadata: the coordinator already knows the sending rank, so
        // send only fqn + length. The bytes stay on this rank and are written locally.
        size_t numBlobs = 0;
        localRegistry->WalkNodes([&](StateNode const& node) {
            if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::DynamicSizeBlob) { return; }
            auto const&        blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            std::ostringstream oss;
            serdes::WriteTrivial(oss, PeerRecordBlob);
            serdes::WriteTrivial(oss, blob.storageSize);
            serdes::WriteBlobWithSize<uint32_t>(oss, localRegistry->GetNodeId(node.handle));
            boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(oss.str())));
            numBlobs++;
        });

        // Write terminal chunk
        boost::asio::write(stream.socket(), boost::beast::http::make_chunk_last());

        INFO("[{}#{}] Rank={} Successfully sent {} tensor shards and {} Blobs to {}", *server, sessionId, rank, numTensors, numBlobs, peer);
    }
    // Read the response
    boost::beast::flat_buffer                                     buffer;
    boost::beast::http::response<boost::beast::http::string_body> res;
    boost::beast::http::read(stream, buffer, res);

    if (res.result() != boost::beast::http::status::ok)
    {
        throw std::runtime_error(
            fmt::format("[native_ckpt:svr][{}#{}][ERROR] Peer={} returned error={}", *server, sessionId, peer, res.result_int()));
    }
    TryCleanShutdown(stream);
}

static std::shared_ptr<StateTree> InflateRegistry(std::string_view bytes)
{
    boost::iostreams::filtering_istream istrm;
    istrm.push(boost::iostreams::zlib_decompressor());
    istrm.push(boost::iostreams::array_source(bytes.data(), bytes.size()));
    return StateSerDes::Read(istrm);
}

void Session::Handle(cipc::Message::LocalStateRegistry const& hdr, std::span<char const> buf)
{
    if (hdr.messageSize > buf.size())
    {
        throw std::invalid_argument(
            fmt::format("Local state registry declares {} bytes but only {} bytes were received", hdr.messageSize, buf.size()));
    }
    BufStream istrm{buf.data(), static_cast<size_t>(hdr.messageSize)};
    auto      localRegistry = StateSerDes::Read(istrm);
    if (localRegistry->SessionId() != sessionId)
    {
        throw std::invalid_argument(
            fmt::format("Local registry session {} does not match request session {}", localRegistry->SessionId(), sessionId));
    }
    ValidateLocalTensorMetadata(*localRegistry);
    Rank rank         = localRegistry->Topo().rank;
    auto snapshotSize = localRegistry->GetTotalSnapshotStateSizeInBytes();
    auto registryTopo = localRegistry->Topo();

    uint64_t offset = 0;
    for (auto const group : {StateNode::Type::DTensor, StateNode::Type::FixedSizeBlob, StateNode::Type::DynamicSizeBlob})
    {
        localRegistry->WalkNodes([&](StateNode& node) {
            if (static_cast<StateNode::Type>(node.data.index()) != group) { return; }
            if (group == StateNode::Type::DynamicSizeBlob)
            {
                auto& blob  = std::get<StateNode::DynamicSizeBlob>(node.data);
                blob.handle = offset;
                offset += blob.storageSize;
            }
            else
            {
                auto&                 tensor = group == StateNode::Type::FixedSizeBlob
                                                   ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                                   : std::get<DTensor>(node.data);
                std::set<TensorShard> shards;
                for (auto shard : tensor.shards)
                {
                    shard.handle = offset;
                    offset += shard.StorageSizeBytes();
                    shards.insert(shard);
                }
                tensor.shards = std::move(shards);
            }
        });
    }
    INFO("[{}#{}] Received sharded state registry from rank={} data of size={}, Total Tensor Size={}",
         *server,
         sessionId,
         rank,
         hdr.messageSize,
         offset);
    bool registryAccepted = false;
    {
        // Concurrent handlers push here; an unlocked emplace corrupts sibling futures.
        std::scoped_lock lock(mutex);
        auto&            localState = sessionState.localRegistryForRank[rank];
        if (localState != nullptr)
        {
            WARN("[{}#{}] Received a duplicate local state registry from rank={} for an unknown reason; ignoring it",
                 *server,
                 sessionId,
                 rank);
            registryTopo = localState->Topo();
        }
        else
        {
            localState          = std::move(localRegistry);
            largestSnapshotSize = std::max(largestSnapshotSize, snapshotSize);
            registryAccepted    = true;
            asyncTasks.emplace_back(
                std::async(std::launch::async | std::launch::deferred, &Session::AsyncWorkReceiveLocalStateRegistry_, this, rank));
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
        DebugOutputToFile(RankRegistryDumpName(rank, registryTopo.worldSize),
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

void Session::Handle(cipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf)
{
    BufStream bufStream{buf.data(), buf.size()};
    bufStream.exceptions(std::ios::failbit | std::ios::badbit);
    auto ckptPath = std::filesystem::path(serdes::ReadBlobWithSize<uint16_t>(bufStream));

    auto                  numSnapshotBlobs = serdes::ReadTrivial<uint32_t>(bufStream);
    std::vector<uint64_t> snapshotBlobSizes;
    snapshotBlobSizes.reserve(numSnapshotBlobs);
    auto const startedAt = hdr.startedAt;
    for (uint32_t i = 0; i < numSnapshotBlobs; ++i) { snapshotBlobSizes.push_back(serdes::ReadTrivial<uint64_t>(bufStream)); }

    auto                         numSnapshotScalars = serdes::ReadTrivial<uint32_t>(bufStream);
    std::vector<Primitives64Bit> snapshotScalars;
    snapshotScalars.reserve(numSnapshotScalars);
    for (uint32_t i = 0; i < numSnapshotScalars; ++i) { snapshotScalars.emplace_back(serdes::ReadTrivial<Primitives64Bit>(bufStream)); }
    bool                                  beginDurableSpan = false;
    std::chrono::system_clock::time_point durableStartedAt;
    {
        std::scoped_lock lock(mutex);
        auto&            snapshot = snapshots[hdr.snapshotId];
        if (!snapshot)
        {
            snapshot             = std::make_unique<Snapshot>();
            snapshot->snapshotId = hdr.snapshotId;
        }
        if (snapshot->path.empty())
        {
            snapshot->name        = fmt::format("snapshot_{}", hdr.snapshotId);
            snapshot->path        = ckptPath / fmt::format("checkpoint_session_{}.snapshot_{}.bin", sessionId, hdr.snapshotId);
            snapshot->activatedAt = PerfClock::now();
            durableStartedAt      = snapshot->activatedAt;
            beginDurableSpan      = true;
        }

        if (snapshot->localStates.contains(hdr.globalRank) || snapshot->finalizedRanks.contains(hdr.globalRank))
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Snapshot={} for global rank={} is already activated or "
                                                 "finalized when activating snapshot={}",
                                                 *server,
                                                 sessionId,
                                                 hdr.snapshotId,
                                                 hdr.globalRank,
                                                 hdr.snapshotId));
        }

        auto it = sessionState.localRegistryForRank.find(hdr.globalRank);
        if (it == sessionState.localRegistryForRank.end())
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:svr][{}#{}][ERROR] No local state found for global rank={} when activating snapshot={}",
                            *server,
                            sessionId,
                            hdr.globalRank,
                            hdr.snapshotId));
        }
        auto localState           = std::make_unique<LocalState>();
        localState->localRegistry = std::make_shared<StateTree>(it->second->SessionId(), it->second->Topo());
        it->second->CloneInto(*localState->localRegistry);
        localState->startedAt      = startedAt;
        uint32_t registeredBlobs   = 0;
        uint32_t registeredScalars = 0;
        localState->localRegistry->WalkNodes([&](StateNode const& node) {
            auto const type = static_cast<StateNode::Type>(node.data.index());
            if (type == StateNode::Type::DynamicSizeBlob) { ++registeredBlobs; }
            else if (type == StateNode::Type::Scalar) { ++registeredScalars; }
        });
        if (numSnapshotBlobs != registeredBlobs || numSnapshotScalars != registeredScalars)
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:svr][{}#{}][ERROR] Snapshot={} rank={} supplied Blob/Scalar counts {}/{}; expected {}/{}",
                            *server,
                            sessionId,
                            hdr.snapshotId,
                            hdr.globalRank,
                            numSnapshotBlobs,
                            numSnapshotScalars,
                            registeredBlobs,
                            registeredScalars));
        }
        size_t blobIndex   = 0;
        size_t scalarIndex = 0;
        localState->localRegistry->WalkNodes([&](StateNode& node) {
            auto const type = static_cast<StateNode::Type>(node.data.index());
            if (type == StateNode::Type::DynamicSizeBlob)
            {
                std::get<StateNode::DynamicSizeBlob>(node.data).storageSize = snapshotBlobSizes[blobIndex++];
            }
            else if (type == StateNode::Type::Scalar)
            {
                auto&       scalar = std::get<StateNode::Scalar>(node.data);
                auto const& value  = snapshotScalars[scalarIndex++];
                if (scalar.GetType() != value.GetType()) { throw std::runtime_error("Snapshot scalar type does not match its registry"); }
                static_cast<Primitives64Bit&>(scalar) = value;
            }
        });

        uint64_t offset = 0;
        for (auto const group : {StateNode::Type::DTensor, StateNode::Type::FixedSizeBlob, StateNode::Type::DynamicSizeBlob})
        {
            localState->localRegistry->WalkNodes([&](StateNode& node) {
                if (static_cast<StateNode::Type>(node.data.index()) != group) { return; }
                if (group == StateNode::Type::DynamicSizeBlob)
                {
                    auto& blob  = std::get<StateNode::DynamicSizeBlob>(node.data);
                    blob.handle = offset;
                    offset += blob.storageSize;
                }
                else
                {
                    auto&                 tensor = group == StateNode::Type::FixedSizeBlob
                                                       ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                                       : std::get<DTensor>(node.data);
                    std::set<TensorShard> shards;
                    for (auto shard : tensor.shards)
                    {
                        shard.handle = offset;
                        offset += shard.StorageSizeBytes();
                        shards.insert(shard);
                    }
                    tensor.shards = std::move(shards);
                }
            });
        }

        if (hdr.snapshotSize != localState->localRegistry->GetTotalPayloadSizeInBytes())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Snapshot size={} does not match expected total tensor "
                                                 "state size={} from local registry for global rank={} when activating snapshot={}",
                                                 *server,
                                                 sessionId,
                                                 hdr.snapshotSize,
                                                 localState->localRegistry->GetTotalPayloadSizeInBytes(),
                                                 hdr.globalRank,
                                                 hdr.snapshotId));
        }
        localState->tensorStorage = ByteStorage(hdr.snapshotSize);
        // Place the tensor storage on the rank's GPU-local NUMA node (sent by the
        // client) and pin this dequeue thread to the same node, so the shm-to-storage
        // memcpy is on-socket. Authoritative per rank, so it is correct for any
        // layout — multi-peer and ckptbench included.
        if (localState->tensorStorage.data != nullptr)
        {
            BindRegionToNumaNode(hdr.numaNode, localState->tensorStorage.data, localState->tensorStorage.size);
        }
        BindThreadCpuToNumaNode(hdr.numaNode);
        snapshot->localStates[hdr.globalRank] = std::move(localState);
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
        .startedAt = startedAt,
    };

    INFO("[{}#{}] Activated snapshot={} for global rank={} with expected tensor data size={} path={}",
         *server,
         sessionId,
         hdr.snapshotId,
         hdr.globalRank,
         hdr.snapshotSize,
         ckptPath);
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
        for (auto const& [snapshotId, snapshot] : session->snapshots)
        {
            for (auto const& [rank, localState] : snapshot->localStates)
            {
                callback(cbData,
                         sessionId,
                         snapshotId,
                         snapshot->name,
                         snapshot->path,
                         *session->sessionState.globalRegistry,
                         *localState->localRegistry,
                         localState->tensorStorage);
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
        if (session.PeerIndex() != fwdStart)
        {
            throw std::logic_error(fmt::format("[native_ckpt:svr][{}#{}] ReceiveGlobalRegistry expected fwdStart={} but got PeerIndex()={}",
                                               *this,
                                               sessionId,
                                               fwdStart,
                                               session.PeerIndex()));
        }

        session.RelayGlobalRegistryToPeers(body, fwdStart + 1, fwdEnd);
        std::shared_ptr<StateTree> globalRegistry = InflateRegistry(body);
        session.HandleGlobalRegistry(std::move(globalRegistry));
    });
}

void Server::ReceiveStateRegistryFragments_(cipc::SessionId const&                                               sessionId,
                                            cipc::Rank                                                           fromRank,
                                            tcp_stream&                                                          stream,
                                            boost::beast::flat_buffer&                                           buffer,
                                            boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser)
{
    INFO("[{}#{}] Receiving state registry chunks for rank={}", *this, sessionId, fromRank);

    // Reads exactly dest.size() bytes, looping over partial reads until complete.
    // Returns bytes read; caller checks for 0+is_done() at record boundaries.
    auto readSpan = [&](std::span<char> dest, bool allowDone) -> size_t {
        size_t total    = 0;
        auto   readSize = dest.size();
        if (allowDone && parser.is_done()) { return 0; }
        while (total < readSize)
        {
            auto prevRemaining       = readSize - total;
            parser.get().body().data = dest.subspan(total).data();
            parser.get().body().size = prevRemaining;
            parser.get().body().more = true;
            boost::system::error_code   ec;
            [[maybe_unused]] auto const n = boost::beast::http::read_some(stream, buffer, parser, ec);
            if (ec && ec != boost::beast::http::error::need_buffer) { throw boost::system::system_error(ec); }
            auto const bodyBytesWritten = prevRemaining - parser.get().body().size;
            total += bodyBytesWritten;
            if (parser.is_done())
            {
                if (allowDone && bodyBytesWritten == 0) { return 0; }
                if (total != readSize)
                {
                    throw std::runtime_error(fmt::format("Unexpected end of HTTP body: read {} bytes, expected {}", total, readSize));
                }
                break;
            }
        }
        return total;
    };

    auto readTrivial = [&](auto& value, bool allowDone = false) -> bool {
        std::span<char> dest{reinterpret_cast<char*>(&value), sizeof(value)};    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        return readSpan(dest, allowDone) == 0;
    };
    auto readBlob = [&](char* dest, size_t size, bool allowDone = false) -> bool {
        std::span<char> destSpan{dest, size};
        return readSpan(destSpan, allowDone) == 0;
    };

    std::vector<TensorShardInfo>                  tensorShards;
    std::vector<std::pair<std::string, uint64_t>> blobMetas;    // (fqn, length); rank is fromRank

    // Read body in chunks; Beast transparently decodes chunked transfer encoding.
    // Each record is tag-prefixed (PeerRecordTensor / PeerRecordBlob); a clean
    // end-of-stream at a tag boundary terminates the fragment.
    while (true)
    {
        uint8_t recordTag{};
        if (readTrivial(recordTag, true))
        {
            break;    // Clean end-of-stream at a record boundary
        }
        if (recordTag == PeerRecordTensor || recordTag == PeerRecordFixedBlob)
        {
            TensorShardInfo info;
            info.type = recordTag == PeerRecordFixedBlob ? StateNode::Type::FixedSizeBlob : StateNode::Type::DTensor;
            uint32_t nodeIdLen{};
            readTrivial(info.durableDtype);
            readTrivial(info.fulldim);
            readTrivial(info.tensorInfo);
            info.tensorInfo.AssertShardIsValid(info.fulldim);
            readTrivial(nodeIdLen);
            info.nodeId = std::string(nodeIdLen, '\0');
            readBlob(info.nodeId.data(), nodeIdLen);
            if (info.tensorInfo.rank != fromRank)
            {
                throw std::invalid_argument(fmt::format("Peer registry fragment from rank={} contains tensor '{}' owned by rank={}",
                                                        fromRank,
                                                        info.nodeId,
                                                        info.tensorInfo.rank));
            }
            tensorShards.push_back(std::move(info));
        }
        else if (recordTag == PeerRecordBlob)
        {
            uint64_t length{};
            uint32_t fqnLen{};
            readTrivial(length);
            readTrivial(fqnLen);
            std::string fqn(fqnLen, '\0');
            readBlob(fqn.data(), fqnLen);
            blobMetas.emplace_back(std::move(fqn), length);
        }
        else
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}#{}][ERROR] Unknown peer fragment record tag={} from rank={}", *this, sessionId, recordTag, fromRank));
        }
    }

    // Every dispatch takes _mutex; release it before the merge below holds the
    // session mutex for a whole insert batch.
    Session* session = nullptr;
    {
        auto lock = std::scoped_lock<std::mutex>(_mutex);
        auto it   = _activeSessions.find(sessionId);
        if (it == _activeSessions.end() || it->second->sessionState.globalRegistry == nullptr)
        {
            INFO("[{}#{}] Received state registry fragment for non-existent session. Adding to pending data", *this, sessionId, fromRank);
            auto pendingData          = std::make_unique<PendingSessionData>();
            pendingData->tensorShards = std::move(tensorShards);
            pendingData->blobMetas    = std::move(blobMetas);
            pendingData->recvFrom     = fromRank;
            pendingData->receivedAt   = PerfClock::now();
            _pendingSessionData[sessionId].push_back(std::move(pendingData));
            return;
        }
        session = it->second.get();
    }
    session->HandleStateRegistryFragment(fromRank, tensorShards, blobMetas);
    INFO("[{}#{}] Successfully received and processed {} tensor shards from rank={}", *this, sessionId, tensorShards.size(), fromRank);
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
    // Check if this is the state upload endpoint: POST /api/peer/snapshotcomplete/{sessionId}/{rank}/{snapshotId}

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
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        // TODO_NOSTALL : move it before we send 200 once we dont stall for session+snaphot anymore
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
    // The global registry for a snapshot (body = deflated serialized tree).
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

        auto const encoding = headers[boost::beast::http::field::content_encoding];
        if (encoding != RegistryContentEncoding)
        {
            throw std::invalid_argument(
                fmt::format("globalregistry body is '{}'-encoded, expected '{}'", encoding, RegistryContentEncoding));
        }

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
            res.result(boost::beast::http::status::accepted);
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
            res.result(boost::beast::http::status::accepted);
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
    else if (matches({"api", "ping"}) && headers.method() == boost::beast::http::verb::get && (it == segs.end()))
    {
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
        Ping();
    }
    else if (matches({"api", "stop"}) && headers.method() == boost::beast::http::verb::get && (it == segs.end()))
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
    else if (matches({"api", "spindown"}) && headers.method() == boost::beast::http::verb::get)
    {
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in spindown endpoint"); }
        // Hand the connection off to a dedicated waiter thread and return at once: the
        // accept loop must stay free to service the peer writeckptcomplete/snapshotcomplete
        // notifications that idle-ness depends on, otherwise the wait would deadlock.
        std::scoped_lock lock(_mutex);
        _spinDownThreads.emplace_back([this, s = std::move(stream)] mutable { SpinDownWaitLoop_(std::move(s)); });
    }
    // Block until a checkpoint is durable: GET /api/wait-for-ckpt-finalize/{sessionId}/{iteration}.
    // `iteration` is signed — negative values index back from the newest snapshot (-1 = latest).
    else if (matches({"api", "wait-for-ckpt-finalize"}) && headers.method() == boost::beast::http::verb::get && (it != segs.end()))
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

void Session::AsyncWorkReceiveLocalStateRegistry_(Rank const& rank)
{
    {
        std::scoped_lock lock(mutex);
        auto&            localRegistry = *sessionState.localRegistryForRank[rank];
        receivedFrom.insert(rank);
        INFO("[{}#{}] AsyncWorkReceiveLocalStateRegistry_  rank={} {}",
             *server,
             sessionId,
             rank,
             WaitingForRanks{receivedFrom, Rank{0}, topo.worldSize});

        if (sessionState.globalRegistry == nullptr)
        {
            sessionState.globalRegistry = std::make_shared<StateTree>(localRegistry.SessionId(), localRegistry.Topo());
            localRegistry.CloneInto(*sessionState.globalRegistry);
            sessionState.globalRegistry->WalkNodes([&](auto& node) {
                if (node.data.index() != static_cast<size_t>(StateNode::Type::DTensor)
                    && node.data.index() != static_cast<size_t>(StateNode::Type::FixedSizeBlob))
                {
                    return;
                }
                tensorMap[localRegistry.GetNodeId(node.handle)] = node.handle;
            });
        }
        else
        {
            std::vector<TensorShardInfo> tensorShards;
            ForeachTensorShardInfo(localRegistry, [&](TensorShardInfo const& info) { tensorShards.push_back(info); });
            MergeFragmentToGlobalState_(lock, tensorShards);
        }

        if (IsCoordinatorPeer())
        {
            std::vector<std::pair<std::string, uint64_t>> blobMetas;
            localRegistry.WalkNodes([&](StateNode const& node) {
                if (node.data.index() == static_cast<size_t>(StateNode::Type::DynamicSizeBlob))
                {
                    blobMetas.emplace_back(localRegistry.GetNodeId(node.handle),
                                           std::get<StateNode::DynamicSizeBlob>(node.data).storageSize);
                }
            });
            RecordBlobMetasLocked_(lock, rank, blobMetas);
        }
    }
    {
        // TODO_NOPENDING_STRUCT
        auto lock = std::scoped_lock<std::mutex>(server->_mutex);
        auto itp  = server->_pendingSessionData.find(sessionId);
        if (itp != server->_pendingSessionData.end())
        {
            for (auto& p : itp->second)
            {
                WARN("[{}#{}] Found pending session data. activating session with pending data", *server, sessionId);
                HandleStateRegistryFragment(p->recvFrom, p->tensorShards, p->blobMetas);
            }
            server->_pendingSessionData.erase(itp);
        }
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
