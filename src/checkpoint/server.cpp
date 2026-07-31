#include "CommonMacros.h"
#include "SetThreadName.h"

#include "byte_storage.h"
#include "fmttensor.h"
#include "messages.h"
#include "numa_utils.h"
#include "serdes_utils.h"
#include "server.h"
#include "session.h"
#include "shm_ring_buffer.h"
#include "state.h"
#include "stateserdes.h"
#include "tensor.h"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/read_until.hpp>
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
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/url.hpp>
#include <hwloc.h>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
namespace bip   = boost::interprocess;
namespace cipc  = checkpoint::ipc;
using PerfClock = std::chrono::high_resolution_clock;

#ifdef PHOENIX_BUILD_FOR_VALGRIND
static constexpr auto AutoShutdownTimeout = 900s;
#else
// TODO_CLI_SVR_RESPAWN
// Must exceed the longest inter-save training stretch a job produces since the client
// cannot handle cleanly if a daemon spinsdown and respaws.
static constexpr auto AutoShutdownTimeout = 900s;
#endif

// uint32_t ranks
// uint32_t iterations
using Request = boost::beast::http::request<boost::beast::http::buffer_body>;

using tcp = boost::asio::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using tcp_stream
    = boost::beast::tcp_stream::rebind_executor<boost::asio::use_awaitable_t<>::executor_with_default<boost::asio::any_io_executor>>::other;

struct Server
{
    static constexpr uint16_t Port = cipc::Message::Port;

    public:
    Server(uint16_t index = 0) : _index(index) {}
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
    Session& SessionById_(cipc::SessionId sessionId);
    void     NotifyFinalizeComplete_(cipc::SessionId sessionId, cipc::Rank fromRank, cipc::SnapshotId snapshotId);
    void     NotifyWriteCkptComplete_(cipc::SessionId sessionId, cipc::Rank fromRank, cipc::SnapshotId snapshotId);
    void     ReceiveGlobalRegistry_(cipc::SessionId sessionId, cipc::SnapshotId snapshotId, std::string_view body);
    void     RecieveStateRegistryFragments_(cipc::SessionId const&                                               sessionId,
                                            cipc::Rank                                                           fromRank,
                                            tcp_stream&                                                          stream,
                                            boost::beast::flat_buffer&                                           buffer,
                                            boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser);

    void SpinDownWaitLoop_(tcp_stream stream);
    void FinalizeWaitLoop_(tcp_stream stream, cipc::SessionId sessionId, int64_t iteration);
    bool IsFinalizeComplete_(cipc::SessionId sessionId, int64_t iteration, cipc::SnapshotId& snapshotId, bool& resolved);
    void HandleRequest_(tcp_stream& stream);

#if defined(SO_REUSEPORT) && !defined(PHOENIX_BUILD_FOR_VALGRIND)
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

    void StartTcpListener_();
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

    PerfClock::time_point               _lastPingTime{PerfClock::now()};
    std::atomic<bool>                   _stopRequested{false};
    std::unordered_set<cipc::MsgQueue*> _msgQueue{nullptr};
    bool                                _running{false};
    std::vector<std::thread>            _threads;
    // Live spin-down waiter threads, each holding one client connection open until idle.
    std::vector<std::thread> _spinDownThreads;
    // Live wait-for-ckpt-finalize handlers, one per client wait connection. Reaped when new waits
    // arrive and drained on Stop().
    std::vector<std::future<void>> _finalizeWaits;

    boost::asio::io_context _ioc;
    // N acceptors share the port via SO_REUSEPORT where available, so the kernel spreads
    // inbound connections across N accept queues. Without SO_REUSEPORT, fall back to one
    // acceptor and keep the offloaded handler pool.
#if defined(SO_REUSEPORT) && !defined(PHOENIX_BUILD_FOR_VALGRIND)
    static constexpr size_t AcceptorCount = 8;
#else
    static constexpr size_t AcceptorCount = 1;
#endif
    std::vector<boost::asio::ip::tcp::acceptor> _tcpAcceptors;

    // Held across Stop()/Start() cycles so a rival daemon cannot elect itself mid-teardown.
    std::unique_ptr<bip::file_lock> _instanceLock;

    // Connection handlers run here, off the accept loop. Declared after _ioc so it is
    // joined (destroyed) before the io_context the handlers' sockets are bound to.
#ifdef PHOENIX_BUILD_FOR_VALGRIND
    static constexpr size_t HandlerThreads = 4;
#else
    static constexpr size_t HandlerThreads = 64;
#endif
    std::unique_ptr<boost::asio::thread_pool> _handlerPool;

    // TODO_NOPENDING_STRUCT
    struct PendingSessionData
    {
        PerfClock::time_point                         recievedAt;
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

template <> struct fmt::formatter<boost::asio::ip::tcp::endpoint> : fmt::ostream_formatter
{};
template <> struct fmt::formatter<boost::asio::ip::address> : fmt::ostream_formatter
{};

template <typename TStream> static void TryCleanShutdown(TStream& stream)
{
    boost::system::error_code ec;
    ec = stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    if (ec && ec != boost::system::errc::not_connected)
    {    //
        fmt::print(stderr, "Shutdown Error: {}\n", ec.what());
    }
}

using BufStream = boost::iostreams::stream<boost::iostreams::array_source>;

namespace
{
struct TensorReplicaDeduplicationStats
{
    size_t                logicalShards{};
    size_t                replicasRemoved{};
    std::vector<uint64_t> assignedBytesPerRank;
};

struct TensorReplicaDeduplicationResult
{
    std::shared_ptr<StateTree>      dedupedRegistry;
    TensorReplicaDeduplicationStats stats;
};
}    // namespace

// Build a separate storage registry with one deterministic owner for every
// logical tensor shard. The merged source registry remains unchanged. Replica
// identity comes exclusively from normalized server metadata (FQN + dtype +
// full shape + global shard range).
static TensorReplicaDeduplicationResult DeduplicateTensorReplicas(StateTree const& registry, uint32_t worldSize)
{
    if (worldSize == 0) { throw std::invalid_argument("Cannot deduplicate tensor replicas with worldSize=0"); }

    auto dedupedRegistry = registry.Clone();

    using RankedTensorShard = StateNode::ShardedTensor::RankedTensorShard;
    struct LogicalShard
    {
        StateNode::Handle              nodeHandle{StateNode::Handle::Invalid};
        std::string                    fqn;
        DS4Tensor::DimRange            sharddim;
        uint64_t                       sizeBytes{};
        std::vector<RankedTensorShard> candidates;
    };

    std::vector<LogicalShard> logicalShards;
    registry.WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::ShardedTensor) { return; }
        auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);

        std::map<DS4Tensor::DimRange, std::vector<RankedTensorShard>> candidatesByRange;
        for (auto const& shard : tnsr.shards)
        {
            auto const rank = static_cast<uint32_t>(shard.rank);
            if (rank >= worldSize)
            {
                throw std::runtime_error(fmt::format("Tensor '{}' shard {} has owner rank {} outside world size {}",
                                                     registry.GetNodeId(node.handle),
                                                     shard.sharddim,
                                                     rank,
                                                     worldSize));
            }
            candidatesByRange[shard.sharddim].push_back(shard);
        }

        auto const fqn = registry.GetNodeId(node.handle);
        for (auto& [sharddim, candidates] : candidatesByRange)
        {
            logicalShards.push_back({.nodeHandle = node.handle,
                                     .fqn        = fqn,
                                     .sharddim   = sharddim,
                                     .sizeBytes  = candidates.front().ShardSizeBytes(tnsr.dtype),
                                     .candidates = std::move(candidates)});
        }
    });

    // Mirror the fully-parallel writer's deterministic greedy policy without
    // consulting replica_id or process-group hints: constrained shards first,
    // then larger shards, then stable logical identity.
    std::ranges::sort(logicalShards, [](LogicalShard const& lhs, LogicalShard const& rhs) {
        if (lhs.candidates.size() != rhs.candidates.size()) { return lhs.candidates.size() < rhs.candidates.size(); }
        if (lhs.sizeBytes != rhs.sizeBytes) { return lhs.sizeBytes > rhs.sizeBytes; }
        if (lhs.fqn != rhs.fqn) { return lhs.fqn < rhs.fqn; }
        return lhs.sharddim < rhs.sharddim;
    });

    TensorReplicaDeduplicationStats stats;
    stats.logicalShards = logicalShards.size();
    stats.assignedBytesPerRank.resize(worldSize, 0);
    std::map<StateNode::Handle, std::vector<RankedTensorShard>> selectedByNode;

    for (auto const& logicalShard : logicalShards)
    {
        auto const* selected = &logicalShard.candidates.front();
        for (auto const& candidate : logicalShard.candidates)
        {
            auto const candidateRank = static_cast<uint32_t>(candidate.rank);
            auto const selectedRank  = static_cast<uint32_t>(selected->rank);
            if (stats.assignedBytesPerRank[candidateRank] < stats.assignedBytesPerRank[selectedRank]
                || (stats.assignedBytesPerRank[candidateRank] == stats.assignedBytesPerRank[selectedRank] && candidateRank < selectedRank))
            {
                selected = &candidate;
            }
        }

        auto const selectedRank = static_cast<uint32_t>(selected->rank);
        stats.assignedBytesPerRank[selectedRank] += logicalShard.sizeBytes;
        stats.replicasRemoved += logicalShard.candidates.size() - 1;
        selectedByNode[logicalShard.nodeHandle].push_back(*selected);
    }

    dedupedRegistry->WalkNodes([&](StateNode& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::ShardedTensor) { return; }
        auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
        if (tnsr.shards.empty()) { return; }

        StateNode::ShardedTensor selected(tnsr.dtype, tnsr.fulldim);
        for (auto const& shard : selectedByNode.at(node.handle)) { selected.AddShard(shard); }
        tnsr = std::move(selected);
    });

    return {.dedupedRegistry = std::move(dedupedRegistry), .stats = std::move(stats)};
}

static void DebugDumpLocalStateTreeKVP(std::filesystem::path const& fname, StateTree const& state)
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
        else if (type == StateNode::Type::Blob) { fmt::print(ostrm, "{}: [blob]\n", name); }
        else if (type == StateNode::Type::TensorShard)
        {
            auto const& tnsr = std::get<StateNode::TensorShard>(node.data);
            fmt::print(ostrm, "{}.sharddim: [{}]\n", name, tnsr.sharddim);
            fmt::print(ostrm, "{}.fulldim: [{}]\n", name, tnsr.fulldim);
            fmt::print(ostrm, "{}.dtype: {}\n", name, tnsr.dtype);
            fmt::print(ostrm, "{}.device: {}\n", name, tnsr.device);
        }
        else if (type == StateNode::Type::ShardedTensor)
        {
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            if (tnsr.shards.size() == 0) throw std::runtime_error(fmt::format("Empty Sharded tensor={}", name));
            fmt::print(ostrm, "{}.fulldim: [{}]\n", name, tnsr.fulldim);
            fmt::print(ostrm, "{}.dtype: {}\n", name, tnsr.dtype);

            for (auto const& rshard : tnsr.shards)
            {
                fmt::print(ostrm, "{}.sharddim@{}.device: {}\n", name, rshard.rank, rshard.device);
                fmt::print(ostrm, "{}.sharddim@{}.shards: [{}]\n", name, rshard.rank, rshard.sharddim);
            }
        }
    });
}

#ifdef DEBUG_MD5SUM
// Compute a lowercase hex MD5 digest matching Python's hashlib.md5().hexdigest().
static std::string ComputeMd5Hex(char const* data, size_t size)
{
    std::array<uint8_t, 16> digest{};
    auto                    digestLen = static_cast<unsigned int>(digest.size());
    EVP_Digest(data, size, digest.data(), &digestLen, EVP_md5(), nullptr);    // NOLINT
    std::string result;
    result.reserve(digest.size() * 2);
    for (auto const b : digest) { result += fmt::format("{:02x}", b); }
    return result;
}

// Format DS4Tensor::Dim as "[v0, v1, ...]" matching Python's str(list(tensor.size)).
static std::string FormatDim(DS4Tensor::Dim const& dim)
{
    std::string s = "[";
    for (size_t i = 0; i < dim.size(); ++i)
    {
        if (i > 0) { s += ", "; }
        s += std::to_string(dim[i]);    // NOLINT
    }
    return s + "]";
}

// Format sharddim begin values as "[b0, b1, ...]" — matches chunk.offsets in distcp.
static std::string FormatShardOffsets(DS4Tensor::DimRange const& sharddim)
{
    std::string s = "[";
    for (size_t i = 0; i < sharddim.size(); ++i)
    {
        if (i > 0) { s += ", "; }
        s += std::to_string(sharddim[i].begin);    // NOLINT
    }
    return s + "]";
}

// Format sharddim extents as "[e0, e1, ...]" where e = end-begin — matches chunk.sizes in distcp.
static std::string FormatShardSizes(DS4Tensor::DimRange const& sharddim)
{
    std::string s = "[";
    for (size_t i = 0; i < sharddim.size(); ++i)
    {
        if (i > 0) { s += ", "; }
        s += std::to_string(sharddim[i].end - sharddim[i].begin);    // NOLINT
    }
    return s + "]";
}

// Map native DType to the torch.dtype string produced by Python's str(dtype).
static std::string_view DTypeToTorchStr(DS4Tensor::DType dtype)
{
    switch (dtype)
    {
    case DS4Tensor::DType::Float16: return "torch.float16";
    case DS4Tensor::DType::BFloat16: return "torch.bfloat16";
    case DS4Tensor::DType::Float32: return "torch.float32";
    case DS4Tensor::DType::Float64: return "torch.float64";
    case DS4Tensor::DType::Int8: return "torch.int8";
    case DS4Tensor::DType::Int16: return "torch.int16";
    case DS4Tensor::DType::Int32: return "torch.int32";
    case DS4Tensor::DType::Int64: return "torch.int64";
    case DS4Tensor::DType::UInt8: return "torch.uint8";
    case DS4Tensor::DType::UInt16: return "torch.uint16";
    case DS4Tensor::DType::UInt32: return "torch.uint32";
    case DS4Tensor::DType::UInt64: return "torch.uint64";
    case DS4Tensor::DType::Bool: return "torch.bool";
    case DS4Tensor::DType::Invalid: return "torch.unknown";
    default: throw std::runtime_error(fmt::format("Invalid DType={}", static_cast<int>(dtype)));
    }
}
#endif

// Peer registry-fragment record tags. A fragment stream is a sequence of
// tag-prefixed records terminated by a clean end-of-stream at a record boundary:
// tensor shards (the merged global tensor view) followed by ShardedObject metadata
// (fqn + length; the owning rank writes the bytes itself).
static constexpr uint8_t PeerRecordTensor = 0;
static constexpr uint8_t PeerRecordBlob   = 1;

static void WriteTensorShardInfo(std::ostream& ostrm, TensorShardInfo const& info)
{
    serdes::WriteTrivial(ostrm, info.rank);
    serdes::WriteTrivial(ostrm, info.tensorInfo);
    serdes::WriteBlobWithSize<uint32_t>(ostrm, info.nodeId);
}

template <typename TLambda> static void ForeachTensorShardInfo(StateTree const& registry, TLambda const& lambda)
{
    auto rank = registry.Topo().rank;
    registry.WalkNodes([&](auto& node) {
#ifdef NATIVE_CKPT_TRACE_NODE
        auto fqn = registry.GetNodeId(node.handle);
        if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}] ForeachTensorShardInfo NATIVE_CKPT_TRACE_NODE={}. Node data rank={} "
                       "index={}\n",
                       *server,
                       sessionId,
                       fqn,
                       rank,
                       node.data.index());
        }
#endif
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            auto& tnsr = std::get<StateNode::TensorShard>(node.data);
            tnsr.AssertShardIsValid();
            lambda({rank, tnsr, registry.GetNodeId(node.handle)});
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                shard.AssertShardIsValid(tnsr.fulldim);
                lambda({rank,
                        StateNode::TensorShard{.sharddim = shard.sharddim,
                                               .fulldim  = tnsr.fulldim,
                                               .dtype    = tnsr.dtype,
                                               .handle   = shard.handle,
                                               .device   = shard.device},
                        registry.GetNodeId(node.handle)});
            }
        }
    });
}

static auto ResolvePeerEndpoint(boost::asio::io_context& ioc, std::string const& peer)
{
    // For some weird reason resolver does not like quering the same peer multiple times
    // Ends up giving host not found. So just cache the results to avoid duplicate work
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::unordered_map<boost::urls::url, boost::asio::ip::tcp::resolver::results_type> EndpointCache;
    static std::mutex                                                                         Mutex;
    SUPPRESS_WARNINGS_END

    boost::urls::url peerAddr;

    auto it = peer.find(':');
    if (it != std::string::npos)
    {
        peerAddr.set_host(peer.substr(0, it));
        peerAddr.set_port(peer.substr(it + 1));
    }
    else
    {
        peerAddr.set_host(peer);
        peerAddr.set_port(std::to_string(cipc::Message::Port));
    }

    std::scoped_lock lock(Mutex);
    if (auto it1 = EndpointCache.find(peerAddr); it1 != EndpointCache.end()) { return std::make_pair(it1->first, it1->second); }

    boost::asio::ip::tcp::resolver resolver(ioc);
    auto                           results = resolver.resolve(peerAddr.host(), peerAddr.port());
    if (results.empty()) { throw std::runtime_error(fmt::format("[native_ckpt:svr] Failed to resolve address for peer={}", peer)); }
    EndpointCache[peerAddr] = results;
    return std::make_pair(peerAddr, results);
}

[[nodiscard]] static auto CreateRequestForPeer(boost::asio::io_context&  ioc,
                                               boost::beast::tcp_stream& stream,
                                               std::string const&        peer,
                                               std::string_view const&   target)
{
    auto [peerAddr, resolveResults] = ResolvePeerEndpoint(ioc, peer);
    auto const endpoint             = stream.connect(resolveResults);
    fmt::print(stderr, "[native_ckpt:svr] Connected to endpoint={} resolved from peer={}\n", endpoint, peer);

    Request req;
    // Write HTTP/1.1 request headers with Transfer-Encoding: chunked (no Content-Length)
    req.method(boost::beast::http::verb::post);
    static constexpr int HttpVersion11 = 11;    // HTTP/1.1
    req.version(HttpVersion11);
    req.set(boost::beast::http::field::host, fmt::format("{}:{}", peerAddr.host(), peerAddr.port()));
    req.set(boost::beast::http::field::content_type, "application/octet-stream");
    req.target(target);
    return req;
}

Session::Session(Server* serverIn, cipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn) :
    server(serverIn), sessionId(sessionIdIn), topo(std::move(topoIn)), workDir(std::move(workDirIn))
{}

Session::LocalState& Session::GetSessionLocalState_(cipc::Rank rank)
{
    std::scoped_lock lock(mutex);
    return *sessionState.localStates[rank];
}

Session::LocalState& Session::FindOrCreateLocalState_(cipc::Rank rank)
{
    std::scoped_lock lock(mutex);
    auto&            localState = sessionState.localStates[rank];
    if (localState == nullptr) { localState = std::make_unique<LocalState>(); }
    return *localState;
}

Session::Snapshot& Session::GetSnapshot_(cipc::SnapshotId snapshotId)
{
    std::scoped_lock lock(mutex);
    return GetSnapshot_(lock, snapshotId);
}

Session::Snapshot& Session::GetSnapshot_(std::scoped_lock<std::mutex> const& /* lock */, cipc::SnapshotId snapshotId)
{
    auto it = snapshots.find(snapshotId);
    if (it == snapshots.end())
    {
        throw std::runtime_error(
            fmt::format("[native_ckpt:svr][{}#{}][ERROR] GetSnapshot_: no snapshot with id={} (evicted or never activated)",
                        *server,
                        sessionId,
                        snapshotId));
    }
    return *it->second;
}

#ifdef DEBUG_MD5SUM
// Write {peerIndex}.metadata.info: sorted "key=value" pairs per tensor node, aligned with
// reference.metadata.info (from distcp_trace.py) for line-by-line diffing.
void Session::DebugWriteNativeMetadataInfo_(Snapshot const& snapshot) const
{
    auto peerIndex = static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer;

    auto const ckptDir = snapshot.path.parent_path();
    std::filesystem::create_directories(ckptDir);
    auto const    infoFile = ckptDir / fmt::format("{}.metadata.info", peerIndex);
    auto const    tmpFile  = ckptDir / fmt::format("{}.metadata.info.tmp", peerIndex);
    std::ofstream ofs(tmpFile);
    if (!ofs.good())
    {
        throw std::runtime_error(fmt::format(
            "[native_ckpt:svr][{}#{}][ERROR] Failed to open native metadata file for writing: {}", *server, sessionId, tmpFile));
    }

    if (snapshot.globalRegistry == nullptr)
    {
        throw std::runtime_error(
            fmt::format("[native_ckpt:svr][{}#{}][ERROR] Active global state is null when finalizing snapshot={} for session id={}",
                        *server,
                        sessionId,
                        snapshot.name,
                        sessionId));
    }

    auto& globalState = *snapshot.globalRegistry;

    // Collect as a vector so we can sort and emit in stable lexicographic order.
    std::vector<std::pair<std::string, std::string>> entries;
    std::unordered_map<std::string, std::string>     md5Map;    // map["<fqn[range...]>"] = "<md5sum>"

    auto emit   = [&](std::string key, std::string val) { entries.emplace_back(std::move(key), std::move(val)); };
    auto addmd5 = [&](std::string&& fqnShardName, std::string&& md5) {
        auto it = md5Map.find(fqnShardName);
        if (it == md5Map.end()) { md5Map.insert({std::move(fqnShardName), std::move(md5)}); }
        else
        {
            if (it->second != md5)
            {
                // throw std::runtime_error(
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][ERROR] Replicate shard={} md5sums dont match {} != {}\n",
                           *server,
                           sessionId,
                           fqnShardName,
                           it->second,
                           md5) /*)  d*/;
            }
        }
    };

    // localShardHandleMap[(rank, fqn, sharddim)] = handle from the per-rank local registry.
    // Used below to assert that the handle recorded in the merged global registry matches
    // the handle the owning rank actually wrote, catching drift between the two views.
    std::unordered_map<std::string, uint64_t> localShardHandleMap;
    auto                                      makeShardHandleKey    //
        = [](cipc::Rank rank, std::string_view fqn, StateNode::ShardedTensor::DimRange const& sharddim) {
              return fmt::format("rank{}:{}[{}]", rank, fqn, sharddim);
          };

    for (auto const& [localRank, localState] : snapshot.localStates)
    {
        localState->localRegistry->WalkNodes([&](StateNode const& node) {
            auto const type = static_cast<StateNode::Type>(node.data.index());
            if (type == StateNode::Type::TensorShard)
            {
                auto const& tnsr = std::get<StateNode::TensorShard>(node.data);
                auto const  size = tnsr.ShardSizeBytes();
                auto const* data = static_cast<char const*>(localState->tensorStorage.data) + tnsr.handle;    // NOLINT
#ifdef NATIVE_CKPT_TRACE_NODE
                auto fqn = localState->localRegistry->GetNodeId(node.handle);
                if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                {
                    fmt::print(stderr,
                               "[native_ckpt:svr][{}#{}] WriteNativeMetadataInfo_ NATIVE_CKPT_TRACE_NODE={}. ByteStorage Offset={}\n",
                               *server,
                               sessionId,
                               fqn,
                               tnsr.handle);
                }
#endif
                auto const localFqn = localState->localRegistry->GetNodeId(node.handle);
                localShardHandleMap.emplace(makeShardHandleKey(localRank, localFqn, tnsr.sharddim), tnsr.handle);
                addmd5(fmt::format("{}[{}]", localFqn, tnsr.sharddim), ComputeMd5Hex(data, size));
            }
            else if (type == StateNode::Type::ShardedTensor)
            {
                auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
                for (auto const& shard : tnsr.shards)
                {
                    auto const size = shard.ShardSizeBytes(tnsr.dtype);
                    auto const data = localState->tensorStorage.AsSpan<char const>(shard.handle, size);
#ifdef NATIVE_CKPT_TRACE_NODE
                    auto fqn = localState->localRegistry->GetNodeId(node.handle);
                    if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                    {
                        fmt::print(stderr,
                                   "[native_ckpt:svr][{}#{}] WriteNativeMetadataInfo_ NATIVE_CKPT_TRACE_NODE={}. ByteStorage Offset={}\n",
                                   *server,
                                   sessionId,
                                   fqn,
                                   shard.handle);
                    }
#endif
                    auto const localFqn = localState->localRegistry->GetNodeId(node.handle);
                    localShardHandleMap.emplace(makeShardHandleKey(shard.rank, localFqn, shard.sharddim), shard.handle);
                    addmd5(fmt::format("{}[{}]", localFqn, shard.sharddim), ComputeMd5Hex(data.data(), data.size()));
                }
            }
        });
    }
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] DebugWriteNativeMetadataInfo_ for snapshot={}. Writing global metadata\n",
               *server,
               sessionId,
               snapshot.name);
    globalState.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::Blob)
        {
            // ShardedObject bytes. Emit the same {fqn}.bytes.md5sum that distcp_trace
            // derives from the on-disk shard so the server-claim cross-check stays
            // balanced. Restricted to _extra_state, matching distcp_trace: the
            // distributed-optimizer dict and rng_state objects are not byte-identical
            // across checkpoint backends, so they are excluded from the compare.
            auto const nodeId = globalState.GetNodeId(node.handle);
            if (nodeId.contains("_extra_state"))
            {
                auto const& blobBytes = std::get<StateNode::Blob>(node.data);
                emit(nodeId + ".bytes.md5sum", ComputeMd5Hex(blobBytes.data(), blobBytes.size()));
            }
            return;
        }
        if (type != StateNode::Type::ShardedTensor) { return; }
        auto const  nodeName = globalState.GetNodeId(node.handle);
        std::string name     = nodeName.starts_with("model.") ? nodeName.substr(std::string_view("model.").size()) : nodeName;

        auto const& shardedTnsr = std::get<StateNode::ShardedTensor>(node.data);
        if (shardedTnsr.shards.empty())
        {
            throw std::runtime_error(fmt::format("Sharded tensor={} has no shards in metadata output\n", name));
        }
        emit(name + ".size", FormatDim(shardedTnsr.fulldim));
        emit(name + ".properties.dtype", std::string(DTypeToTorchStr(shardedTnsr.dtype)));
        emit(name + ".properties.layout", "torch.strided");
        emit(name + ".properties.memory_format", "torch.contiguous_format");
        emit(name + ".properties.pin_memory", "False");
        emit(name + ".properties.requires_grad", "False");

        StateNode::ShardedTensor::DimRange lastShard{};
        size_t                             i = 0;
        for (auto const& rankShard : shardedTnsr.shards)
        {
            // Verify the per-rank handle recorded in the merged global registry matches
            // the handle the owning rank's local registry actually wrote. Done for every
            // rankShard (including replicas) so each rank's view is checked individually,
            // before the sharddim-dedup skip below collapses replicas for output.
            auto const shardHandleKey = makeShardHandleKey(rankShard.rank, nodeName, rankShard.sharddim);
            auto const localHandleIt  = localShardHandleMap.find(shardHandleKey);
            if (localHandleIt == localShardHandleMap.end())
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][ERROR] Global shard={} has no matching entry in any local registry\n",
                           *server,
                           sessionId,
                           shardHandleKey);
            }
            else if (localHandleIt->second != rankShard.handle)
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][ERROR] Global shard={} handle={} does not match local handle={}\n",
                           *server,
                           sessionId,
                           shardHandleKey,
                           rankShard.handle,
                           localHandleIt->second);
            }

            if (lastShard == rankShard.sharddim) { continue; }
            lastShard         = rankShard.sharddim;
            auto fqnShardName = fmt::format("{}[{}]", globalState.GetNodeId(node.handle), rankShard.sharddim);
            emit(fmt::format("{}.chunks.{}.offsets", name, i), FormatShardOffsets(rankShard.sharddim));
            emit(fmt::format("{}.chunks.{}.sizes", name, i), FormatShardSizes(rankShard.sharddim));
            auto md5It = md5Map.find(fqnShardName);
            if (md5It != md5Map.end()) { emit(fmt::format("{}.chunks.{}.md5sum", name, i), md5It->second); }
            ++i;
        }
    });

    std::ranges::sort(entries);
    for (auto const& [key, val] : entries) { fmt::print(ofs, "{}={}\n", key, val); }
    // Close before rename so the ofstream buffer is flushed. distcp_trace
    // polls for the final path's existence; before this fix the file was
    // created (visible to the poller) before any content was written, so the
    // reader could observe a 0-byte file and diff it against the full
    // reference — manifesting as the gb200 "5313 differences" failure. Atomic
    // rename guarantees the reader only ever sees the file at its final size.
    ofs.close();
    std::filesystem::rename(tmpFile, infoFile);
}
#endif

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Session::OnFinalizeSnapshotAsync_(cipc::Message::FinalizeSnapshot const& hdr)
{
    // Ranks on the coordinator peer (peers[0]) record their finalize/write-complete
    // locally; only ranks on other peers forward those notifications to the coordinator.
    bool const onCoordinatorPeer = static_cast<uint32_t>(hdr.globalRank) / topo.ranksPerPeer == 0;
    bool const notifyCoordinator = !onCoordinatorPeer;
    if (notifyCoordinator) { SendPeerFinalizeSnapshotComplete_(topo.peers[0], hdr); }

    Snapshot*   snapshotPtr   = nullptr;
    LocalState* localStatePtr = nullptr;
    {
        std::scoped_lock lock(mutex);
        snapshotPtr   = snapshots.at(hdr.snapshotId).get();
        localStatePtr = snapshotPtr->localStates.at(hdr.globalRank).get();
    }
    auto& snapshot   = *snapshotPtr;
    auto& localState = *localStatePtr;

    // TODO_GLOBAL_REGISTRY_BCASE
    // Coordinator-peer ranks wait for every fragment: remote merges reallocate the tree's
    // node array, and the WriteRankShard walk below is unlocked.
    auto const localNodeWorldSize = onCoordinatorPeer ? static_cast<uint32_t>(topo.worldSize) : topo.ranksPerPeer;

    while (true)
    {
        {
            std::scoped_lock lock(mutex);
            bool const       snapshotBlobsReady = !onCoordinatorPeer || snapshotPtr->localStates.contains(Rank{0});
            if (recievedFrom.size() >= localNodeWorldSize && recievedFrom.contains(hdr.globalRank) && snapshotBlobsReady)
            {
                if (onCoordinatorPeer && snapshotPtr->globalRegistry == nullptr)
                {
                    // Keep snapshot-scoped Blob values out of the session
                    // template while preserving the complete replica view.
                    snapshotPtr->globalRegistry = sessionState.globalRegistry->Clone();
                    for (auto const& blob : snapshotPtr->snapshotBlobs) { snapshotPtr->globalRegistry->SetBlob(blob.fqn, blob.data); }
                }
                break;
            }
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][INFO] FinalizeSnapshot waiting for snapshot={} local-state-registry {} of {} "
                       "remaining, snapshot-blobs-ready={}... recievedFrom=[{}]\n",
                       *server,
                       sessionId,
                       hdr.snapshotId,
                       localNodeWorldSize - recievedFrom.size(),
                       localNodeWorldSize,
                       snapshotBlobsReady,
                       fmt::join(recievedFrom, ","));
        }
        if (stopRequested) { throw std::runtime_error("Stop requested while waiting for ranks to send local state registry"); }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Rank 0 writes registry.bin once every rank has finalized.
    // TODO_MULTILOCAL_CKPT : For multinode local checkpointing this needs to happen once on each node so designate one rank
    if (hdr.globalRank == Rank{0})
    {
        while (true)
        {
            {
                std::scoped_lock lock(mutex);
                if (snapshot.finalizedRanks.size() >= static_cast<size_t>(topo.worldSize)) { break; }
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][INFO] FinalizeSnapshot waiting for snapshot={} finalized ranks {} of {} remaining... "
                           "Finalized ranks=[{}]\n",
                           *server,
                           sessionId,
                           hdr.snapshotId,
                           static_cast<size_t>(topo.worldSize) - snapshot.finalizedRanks.size(),
                           static_cast<size_t>(topo.worldSize),
                           fmt::join(snapshot.finalizedRanks, ","));
            }
            if (stopRequested) { throw std::runtime_error("Stop requested while waiting for ranks to finalize snapshot"); }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::shared_ptr<StateTree> globalRegistry;
        {
            std::scoped_lock lock(mutex);
            globalRegistry = snapshot.globalRegistry;
        }

        WriteGlobalStateTreeToFile_(snapshot, snapshot.path.parent_path() / "registry.bin");
        // Broadcast the complete replica view. Every daemon derives the same
        // storage plan locally, without mutating this source registry.
        BroadcastGlobalRegistryToPeers_(hdr.snapshotId, *globalRegistry);

        auto dedupResult = DeduplicateTensorReplicas(*globalRegistry, static_cast<uint32_t>(topo.worldSize));
        fmt::print(stderr,
                   "[native_ckpt:svr][{}#{}][INFO] Selected owners for {} logical tensor shards; removed {} replicas; "
                   "assigned bytes by rank=[{}]\n",
                   *server,
                   sessionId,
                   dedupResult.stats.logicalShards,
                   dedupResult.stats.replicasRemoved,
                   fmt::join(dedupResult.stats.assignedBytesPerRank, ","));
        {
            std::scoped_lock lock(mutex);
            snapshot.dedupedGlobalRegistry = std::move(dedupResult.dedupedRegistry);
        }
    }

    // Every writer must consume its daemon's locally derived storage plan. In
    // particular, remote peers may not write while the global-registry
    // broadcast and local derivation are still in flight.
    std::shared_ptr<StateTree> dedupedGlobalRegistry;
    while (true)
    {
        {
            std::scoped_lock lock(mutex);
            dedupedGlobalRegistry = snapshot.dedupedGlobalRegistry;
            if (dedupedGlobalRegistry != nullptr) { break; }
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][INFO] FinalizeSnapshot waiting for deduped global registry for snapshot={} rank={}...\n",
                       *server,
                       sessionId,
                       hdr.snapshotId,
                       hdr.globalRank);
        }
        if (stopRequested) { throw std::runtime_error("Stop requested while waiting for deduped global registry"); }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    PrepareBlobState_(snapshot, localState, hdr.globalRank);
    checkpoint::distcp::WriteRankShard(
        snapshot.path.parent_path(), hdr.globalRank, *dedupedGlobalRegistry, localState.tensorStorage, localState.blobs);
    if (notifyCoordinator) { SendPeerWriteCkptComplete_(topo.peers[0], hdr); }
    // A peer may declare its local shard complete only after the coordinator has
    // acknowledged the notification. Rank 0 is delayed further, until metadata is durable.
    if (hdr.globalRank != Rank{0})
    {
        std::scoped_lock lock(mutex);
        snapshot.writeCkptCompletedRanks.insert(hdr.globalRank);
    }

    // Rank 0 emits metadata once every other rank has finished its shard. Its own
    // shard was written synchronously above; rank 0 joins the completion set only
    // after metadata is fsync'd, making a world-size set the global durability marker.
    if (hdr.globalRank == Rank{0})
    {
        auto const otherRanks = static_cast<size_t>(topo.worldSize) - 1;
        while (true)
        {
            {
                std::scoped_lock<std::mutex> lock(mutex);
                if (snapshot.writeCkptCompletedRanks.size() >= otherRanks) { break; }
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][INFO] FinalizeSnapshot waiting for snapshot={} write-completed ranks {} of {} "
                           "remaining...\n",
                           *server,
                           sessionId,
                           hdr.snapshotId,
                           otherRanks - snapshot.writeCkptCompletedRanks.size(),
                           otherRanks);
            }
            if (stopRequested) { throw std::runtime_error("Stop requested while waiting for ranks to write checkpoint"); }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
#ifdef DEBUG_MD5SUM
        DebugWriteNativeMetadataInfo_(snapshot);
#endif
        try
        {
            checkpoint::distcp::WriteMetadata(snapshot.path.parent_path(), *dedupedGlobalRegistry, snapshot.blobMetadata);
        } catch (std::exception const& ex)
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][ERROR] WriteMetadata failed for snapshot={} rank={}: {}\n",
                       *server,
                       sessionId,
                       hdr.snapshotId,
                       hdr.globalRank,
                       ex.what());
            throw;
        }
        {
            std::scoped_lock lock(mutex);
            snapshot.writeCkptCompletedRanks.insert(Rank{0});
        }
    }

    // A daemon's snapshot data has no use once its local shards are written: the
    // coordinator evicts after rank 0's metadata, other peers once their ranks finish.
    if (hdr.globalRank == Rank{0} || !onCoordinatorPeer)
    {
        // free Snapshot blobs (multi-GB) OUTSIDE the session mutex to avoid mutex blocking
        std::vector<std::unique_ptr<Snapshot>> evicted;
        size_t                                 remainingSnapshots = 0;
        {
            std::scoped_lock lock(mutex);
            while (!snapshots.empty() && snapshots.size() > numActiveSnapshots)
            {
                auto it = snapshots.begin();
                // A non-coordinator peer may finish its local shard writes before rank 0
                // broadcasts the merged global registry. Keep the snapshot addressable until
                // that broadcast marks every global rank finalized; otherwise the peer rejects
                // the late registry and the coordinator can never publish durable metadata.
                if (!IsSnapshotEvictable(*it->second)) { break; }
                // Eviction only drops an evictable (write-complete and finalized) snapshot, so it
                // always drops a durable one: advance the highwater a late wait-for-finalize consults.
                if (highestEvictedDurableSnapshot == cipc::SnapshotId::Invalid
                    || static_cast<uint32_t>(it->first) > static_cast<uint32_t>(highestEvictedDurableSnapshot))
                {
                    highestEvictedDurableSnapshot = it->first;
                }
                evicted.push_back(std::move(it->second));
                snapshots.erase(it);
            }
            remainingSnapshots = snapshots.size();
        }
        fmt::print(stderr,
                   "[native_ckpt:svr][{}#{}][INFO] Evicted {} snapshots after finalizing snapshot={} rank={}. Remaining snapshots "
                   "in memory={} numActiveSnapshots={}\n",
                   *server,
                   sessionId,
                   evicted.size(),
                   hdr.snapshotId,
                   hdr.globalRank,
                   remainingSnapshots,
                   numActiveSnapshots);
    }

    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}][INFO] OnFinalizeSnapshotAsync_ for rank={}, snapshotId={} \n",
               *server,
               sessionId,
               hdr.globalRank,
               hdr.snapshotId);
    // SendLocalShardedStateRegistryToPeer(cipc::Rank rank, std::string const& peer)
}

void Session::MergeFragmentToGlobalState_(std::scoped_lock<std::mutex> const& /* lock */, std::vector<TensorShardInfo> const& tensorShards)
{
    fmt::print(stderr, "[native_ckpt:svr][{}#{}][INFO] MergeFragmentToGlobalState_ shards={}\n", *server, sessionId, tensorShards.size());

    if (sessionState.globalRegistry == nullptr)
    {
        throw std::logic_error("Cannot merge state registry fragments. Global state registry is not initialized");
    }

    for (auto const& tensorShard : tensorShards)
    {
#ifdef NATIVE_CKPT_TRACE_NODE
        if (tensorShard.nodeId.contains(NATIVE_CKPT_TRACE_NODE))
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][INFO] MergeFragmentToGlobalState_ Found tensor shard with nodeId containing "
                       "NATIVE_CKPT_TRACE_NODE={} rank={}\n",
                       *server,
                       sessionId,
                       NATIVE_CKPT_TRACE_NODE,
                       tensorShard.rank);
        }
#endif
        tensorShard.tensorInfo.AssertShardIsValid();
        // Look up the node by its string NodeId
        auto it = tensorMap.find(tensorShard.nodeId);
        if (it == tensorMap.end())
        {
            auto const& stateName        = tensorShard.nodeId;
            auto [childName, parentNode] = sessionState.globalRegistry->FindOrAddNodeByFQN(stateName);
            // Add the final node which contains the handler
            auto& child = sessionState.globalRegistry->AddChild(parentNode, childName, true);
            auto  shTen = StateNode::ShardedTensor(tensorShard.tensorInfo.dtype, tensorShard.tensorInfo.fulldim);
            shTen.AddShard(StateNode::ShardedTensor::RankedTensorShard{.sharddim = tensorShard.tensorInfo.sharddim,
                                                                       .device   = tensorShard.tensorInfo.device,
                                                                       .handle   = tensorShard.tensorInfo.handle,
                                                                       .rank     = tensorShard.rank});
            child.data                    = shTen;
            tensorMap[tensorShard.nodeId] = child.handle;
            continue;
        }
        auto  nodeHandle = it->second;
        auto& node       = sessionState.globalRegistry->GetNode(nodeHandle);
        if (node.data.index() != static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            throw std::runtime_error(fmt::format("Received tensor shard for non-sharded tensor='{}' expected-type={} != actual-type={}",
                                                 tensorShard.nodeId,
                                                 static_cast<size_t>(StateNode::Type::ShardedTensor),
                                                 node.data.index()));
        }
        auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
        if (tnsr.shards.empty())
        {
            throw std::runtime_error(fmt::format("Sharded tensor={} has no shards in metadata output\n", tensorShard.nodeId));
        }

        tensorShard.tensorInfo.AssertShardIsValid();
        if (tnsr.dtype != tensorShard.tensorInfo.dtype || tnsr.fulldim != tensorShard.tensorInfo.fulldim)
        {
            throw std::runtime_error(fmt::format("Received incompatible replicas for tensor='{}': expected dtype={} full-shape=[{}], got "
                                                 "dtype={} full-shape=[{}] on rank={}",
                                                 tensorShard.nodeId,
                                                 tnsr.dtype,
                                                 tnsr.fulldim,
                                                 tensorShard.tensorInfo.dtype,
                                                 tensorShard.tensorInfo.fulldim,
                                                 tensorShard.rank));
        }
        tnsr.AddShard({.sharddim = tensorShard.tensorInfo.sharddim,
                       .device   = tensorShard.tensorInfo.device,
                       .handle   = tensorShard.tensorInfo.handle,
                       .rank     = tensorShard.rank});
    }
}

// TODO_MULTILOCAL_CKPT
void Session::SendPeerFinalizeSnapshotComplete_(std::string const& peer, cipc::Message::FinalizeSnapshot const& msg) const
{ SendToPeerWithRetries_(peer, "", fmt::format("/api/peer/finalizecomplete/{}/{}/{}", sessionId, msg.globalRank, msg.snapshotId)); }

void Session::SendPeerWriteCkptComplete_(std::string const& peer, cipc::Message::FinalizeSnapshot const& msg) const
{ SendToPeerWithRetries_(peer, "", fmt::format("/api/peer/writeckptcomplete/{}/{}/{}", sessionId, msg.globalRank, msg.snapshotId)); }

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

        // Serialize and stream each TensorShardInfo as an individual HTTP chunk
        // Format per entry: [rank: uint32][tensorInfo: DS4Tensor][nodeId_len: uint32][nodeId: bytes]
        size_t numTensors = 0;

        auto& localState = GetSessionLocalState_(rank);
        ForeachTensorShardInfo(*localState.localRegistry, [&](TensorShardInfo const& info) {
            // TODO : use constant sized stack based buffer streams later
            std::ostringstream oss;
            serdes::WriteTrivial(oss, PeerRecordTensor);
            WriteTensorShardInfo(oss, info);
            boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(oss.str())));
            numTensors++;
        });

        // Blob metadata: the coordinator already knows the sending rank, so
        // send only fqn + length. The bytes stay on this rank and are written locally.
        size_t numBlobs = 0;
        localState.localRegistry->WalkNodes([&](StateNode const& node) {
            if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
            auto const&        blob = std::get<StateNode::Blob>(node.data);
            std::ostringstream oss;
            serdes::WriteTrivial(oss, PeerRecordBlob);
            serdes::WriteTrivial(oss, static_cast<uint64_t>(blob.size()));
            serdes::WriteBlobWithSize<uint32_t>(oss, localState.localRegistry->GetNodeId(node.handle));
            boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(oss.str())));
            numBlobs++;
        });

        // Write terminal chunk
        boost::asio::write(stream.socket(), boost::beast::http::make_chunk_last());

        fmt::print(stderr,
                   "[native_ckpt:svr][{}#{}][INFO] Rank={} Successfully sent {} tensor shards and {} Blobs to {}\n",
                   *server,
                   sessionId,
                   rank,
                   numTensors,
                   numBlobs,
                   peer);
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

static constexpr int  MaxRetries             = 20;
static constexpr auto IntervalBetweenRetries = std::chrono::milliseconds(10);

void Session::SendLocalShardedStateRegistryToPeerWithRetries_(cipc::Rank rank, std::string const& peer)
{
    for (int attempt = 1; attempt <= MaxRetries; ++attempt)
    {
        try
        {
            SendLocalShardedStateRegistryToPeer_(rank, peer);
            return;
        } catch (std::exception const& ex)
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][ERROR] Attempt {} SendLocalShardedStateRegistryToPeer_ rank={} peer={} failed sending "
                       "shards: {}\n",
                       *server,
                       sessionId,
                       attempt,
                       rank,
                       peer,
                       ex.what());
            std::this_thread::sleep_for(IntervalBetweenRetries);
        }
    }
}

void Session::SendToPeerWithRetries_(std::string const& peer, std::string const& body, std::string const& target) const
{
    std::string lastError;
    for (int attempt = 1; attempt <= MaxRetries; ++attempt)
    {
        try
        {
            boost::asio::io_context  ioc;
            boost::beast::tcp_stream stream(ioc);
            auto                     req = CreateRequestForPeer(ioc, stream, peer, target);
            req.body().data = const_cast<char*>(body.data());    // NOLINT — beast buffer_body is non-owning; body outlives the write
            req.body().size = body.size();
            req.body().more = false;
            req.prepare_payload();
            boost::beast::http::write(stream, req);

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
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][ERROR] Attempt {} peer={} failed target={} {}\n",
                       *server,
                       sessionId,
                       attempt,
                       peer,
                       target,
                       ex.what());
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

void Session::BroadcastGlobalRegistryToPeers_(cipc::SnapshotId snapshotId, StateTree const& registry) const
{
    if (topo.peers.size() <= 1) { return; }
    std::ostringstream oss;
    StateSerDes::Write(oss, registry, StateNode::Handle::Root);
    std::string const body = oss.str();

    for (size_t p = 1; p < topo.peers.size(); ++p)
    {
        SendToPeerWithRetries_(topo.peers[p], body, fmt::format("/api/peer/globalregistry/{}/{}", sessionId, snapshotId));
    }

    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}][INFO] Broadcast global registry for snapshot {} to {} peer(s)\n",
               *server,
               sessionId,
               snapshotId,
               topo.peers.size() - 1);
}

void Session::WriteGlobalStateTreeToFile_(Snapshot const& snapshot, std::filesystem::path const& ckptPath) const
{
    auto start1 = PerfClock::now();
    std::filesystem::create_directories(ckptPath.parent_path());
    snapshot.globalRegistry->WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::TensorShard)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}#{}][ERROR] DebugWriteMetdataKeyValuePairs: Unexpected TensorShard node in activeGlobalState",
                *server,
                sessionId,
                snapshot.globalRegistry->GetNodeId(node.handle)));
        }
        if (type == StateNode::Type::ShardedTensor)
        {
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            for (auto const& shard : tnsr.shards) { shard.AssertShardIsValid(tnsr.fulldim); }
            if (tnsr.shards.empty())
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][WARN] empty sharded tensor='{}' in metadata output\n",
                           *server,
                           sessionId,
                           snapshot.globalRegistry->GetNodeId(node.handle));
                return;
            }
            snapshot.globalRegistry->AssertShardedTensorHasNoHoles(node.handle);
        }
    });
    {
        std::ofstream sstrm(ckptPath, std::ios::binary);
        if (!sstrm.good())
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:svr][{}#{}][ERROR] Failed to open metadata file for writing: {}", *server, sessionId, ckptPath));
        }
        StateSerDes::Write(sstrm, *snapshot.globalRegistry, StateNode::Handle::Root);
        fmt::print(stderr, "[native_ckpt:svr][{}#{}][INFO] Finished writing metadata to file={} \n", *server, sessionId, ckptPath);
    }
    // Verify if this was written correctly
    // Note because of empty nodes not being serialized this might not be a perfect match
    {
// #define NATIVE_CKPT_VERIFY_METADATA
#ifdef NATIVE_CKPT_VERIFY_METADATA
        std::ifstream sstrm(ckptPath, std::ios::binary);
        if (!sstrm.good())
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}#{}][ERROR] Failed to open metadata file for verification: {}", *server, sessionId, ckptPath));
        }
        auto readBack = StateSerDes::Read(snapshot.globalRegistry->GlobalRank(), snapshot.globalRegistry->GlobalWorldSize(), sstrm);

        // Walk the original tree and compare every node against the read-back tree.
        // Nodes are stored at stable indices, so the same Handle addresses the same node in both trees.
        // StateNode::Variant supports operator== for all alternatives (Primitives64Bit included).
        bool mismatch = false;
        snapshot.globalRegistry->WalkNodes([&](StateNode const& origNode) {
            auto const& readBackNode = readBack->GetNode(origNode.handle);
            if (origNode.data != readBackNode.data)
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][ERROR] Node='{}' mismatch after roundtrip\n\tOriginal={}\n\tReadBack={}\n",
                           *server,
                           sessionId,
                           snapshot.globalRegistry->GetNodeId(origNode.handle),
                           origNode.data,
                           readBackNode.data);
                mismatch = true;
            }
        });

        if (mismatch)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Metadata roundtrip verification failed for {}",
                                                 *server,
                                                 sessionId,
                                                 snapshot.globalRegistry->GlobalRank(),
                                                 ckptPath));
        }
        fmt::print(stderr, "[native_ckpt:svr][{}#{}][INFO] Metadata roundtrip verification passed for: {}\n", *server, sessionId, ckptPath);
#endif
    }
    for (auto const& [localStateRank, localState] : snapshot.localStates)
    {
        DebugDumpLocalStateTreeKVP(workDir / fmt::format("cpp_metadata_{}.txt", localStateRank), *localState->localRegistry);
    }
    DebugDumpLocalStateTreeKVP(workDir / fmt::format("cpp_metadata.txt"), *sessionState.globalRegistry);
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] Accumulated all localState for global State: Dumped states in {}\n",
               *server,
               sessionId,
               (PerfClock::now() - start1));
}

void Session::Activate()
{
    start = PerfClock::now();
    // noop
}

void Session::Handle(cipc::Message::LocalStateRegistry const& hdr, std::span<char const> buf)
{
    BufStream istrm{buf.data(), hdr.messageSize};
    auto      localRegistry  = StateSerDes::Read(istrm);
    Rank      rank           = localRegistry->Topo().rank;
    auto&     localState     = FindOrCreateLocalState_(rank);
    localState.localRegistry = std::move(localRegistry);
    // Reset tensor handles to be offsets in the local tensor storage.
    uint64_t offset = 0;
    localState.localRegistry->WalkNodes([&](auto& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            auto& tnsr  = std::get<StateNode::TensorShard>(node.data);
            tnsr.handle = offset;
            offset += tnsr.ShardSizeBytes();
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            auto&                 tnsr = std::get<StateNode::ShardedTensor>(node.data);
            decltype(tnsr.shards) newShards;
            for (auto shard : tnsr.shards)
            {
                shard.handle = offset;
                newShards.insert(shard);
                offset += shard.ShardSizeBytes(tnsr.dtype);
            }    // will be filled in when tensor data is received
            tnsr.shards = std::move(newShards);
        }
    });
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}][INFO] Received sharded state registry from rank={} data of size={}, Total Tensor Size={}\n",
               *server,
               sessionId,
               rank,
               hdr.messageSize,
               offset);
    {
        // Concurrent handlers push here; an unlocked emplace corrupts sibling futures.
        std::scoped_lock lock(mutex);
        asyncTasks.emplace_back(
            std::async(std::launch::async | std::launch::deferred, &Session::OnRecieveLocalStateRegistryAsync_, this, rank));

        // Coordinator-peer ranks merge into the global registry locally (OnRecieve); only
        // ranks on other peers forward their fragment to the coordinator (peers[0]).
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

void Session::Handle(cipc::Message::LocalStateData const& hdr1, std::span<char const> buf)
{
    void* ptr = nullptr;
    {
        std::scoped_lock lock(mutex);
        auto&            snapshot   = GetSnapshot_(lock, hdr1.snapshotId);
        auto&            localState = *snapshot.localStates.at(hdr1.globalRank);
        auto             endOffset  = hdr1.offset + hdr1.messageSize;
        if (endOffset > localState.tensorStorage.size)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}#{}][ERROR] Received tensor data exceeds allocated size={} for session id={} offset={} size={}",
                *server,
                sessionId,
                localState.tensorStorage.size,
                hdr1.sessionId,
                hdr1.offset,
                hdr1.messageSize));
        }

        localState.writtenTensorSize += hdr1.messageSize;
        localState.writtenTensorRangeBegin = std::min(localState.writtenTensorRangeBegin, static_cast<uint64_t>(hdr1.offset));
        localState.writtenTensorRangeEnd   = std::max(localState.writtenTensorRangeEnd, endOffset);
        if (localState.writtenTensorSize > localState.tensorStorage.size)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Received tensor data size={} (range {}-{}) exceeds "
                                                 "total tensor size={} for session id={}",
                                                 *server,
                                                 sessionId,
                                                 localState.writtenTensorSize,
                                                 localState.writtenTensorRangeBegin,
                                                 localState.writtenTensorRangeEnd,
                                                 localState.tensorStorage.size,
                                                 hdr1.sessionId));
        }

        ptr = localState.tensorStorage[hdr1.offset];
    }
    std::memcpy(ptr, buf.data(), hdr1.messageSize);
}

void Session::Handle(cipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf)
{
    BufStream                bufStream{buf.data(), buf.size()};
    auto                     ckptPath         = std::filesystem::path(serdes::ReadBlobWithSize<uint16_t>(bufStream));
    auto                     numSnapshotBlobs = serdes::ReadTrivial<uint32_t>(bufStream);
    StateTree::SnapshotBlobs snapshotBlobs;
    snapshotBlobs.reserve(numSnapshotBlobs);
    for (uint32_t i = 0; i < numSnapshotBlobs; ++i)
    {
        snapshotBlobs.push_back(
            {.fqn = serdes::ReadBlobWithSize<uint16_t>(bufStream), .data = serdes::ReadBlobWithSize<uint32_t>(bufStream)});
    }
    if (hdr.globalRank != Rank{0} && !snapshotBlobs.empty())
    {
        throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Nonzero rank={} sent snapshot Blob fields for snapshot={}",
                                             *server,
                                             sessionId,
                                             hdr.globalRank,
                                             hdr.snapshotId));
    }
    {
        std::scoped_lock lock(mutex);
        auto             snapshotIt = snapshots.find(hdr.snapshotId);
        if (snapshotIt == snapshots.end())
        {
            // Retained snapshots are useful only until their replacement arrives.  Waiting
            // until the replacement is finalized to evict a completed predecessor makes the
            // daemon hold both snapshots' accumulation blobs at once.  For large models that
            // is hundreds of GiB of avoidable peak RSS on every node and can OOM one server
            // while the other nodes continue training.
            //
            // Free completed predecessors before allocating tensor storage for the new
            // snapshot.  Do the destruction while holding the session lock: concurrent local
            // ranks may activate the same new snapshot, and none of them may allocate ahead of
            // this release.  No checkpoint work can still reference a snapshot whose complete
            // set has reached ExpectedWriteCompletions().
            std::vector<std::unique_ptr<Snapshot>> evicted;
            auto const                             retained = std::max<uint32_t>(numActiveSnapshots, 1);
            while (!snapshots.empty() && snapshots.size() >= retained)
            {
                auto oldest = snapshots.begin();
                if (!IsSnapshotEvictable(*oldest->second)) { break; }
                evicted.push_back(std::move(oldest->second));
                snapshots.erase(oldest);
            }
            if (!evicted.empty())
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][INFO] Evicting {} completed snapshot(s) before activating snapshot={}\n",
                           *server,
                           sessionId,
                           evicted.size(),
                           hdr.snapshotId);
                evicted.clear();
            }

            auto snapshot        = std::make_unique<Snapshot>();
            snapshot->snapshotId = hdr.snapshotId;
            snapshot->name       = fmt::format("snapshot_{}", hdr.snapshotId);
            snapshot->path       = ckptPath / fmt::format("checkpoint_session_{}.snapshot_{}.bin", sessionId, hdr.snapshotId);
            snapshotIt           = snapshots.insert({hdr.snapshotId, std::move(snapshot)}).first;
        }

        if (hdr.globalRank == Rank{0}) { snapshotIt->second->snapshotBlobs = std::move(snapshotBlobs); }

        if (snapshotIt->second->localStates.contains(hdr.globalRank))
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}#{}][ERROR] Snapshot={} for global rank={} already has local state when activating snapshot={}",
                *server,
                sessionId,
                hdr.snapshotId,
                hdr.globalRank,
                hdr.snapshotId));
        }

        auto it = sessionState.localStates.find(hdr.globalRank);
        if (it == sessionState.localStates.end())
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:svr][{}#{}][ERROR] No local state found for global rank={} when activating snapshot={}",
                            *server,
                            sessionId,
                            hdr.globalRank,
                            hdr.snapshotId));
        }
        auto localState           = std::make_unique<LocalState>();
        localState->localRegistry = it->second->localRegistry;

        if (hdr.snapshotSize != it->second->localRegistry->GetTotalTensorStateSizeInBytes())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Snapshot size={} does not match expected total tensor "
                                                 "state size={} from local registry for global rank={} when activating snapshot={}",
                                                 *server,
                                                 sessionId,
                                                 hdr.snapshotSize,
                                                 it->second->localRegistry->GetTotalTensorStateSizeInBytes(),
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
        snapshotIt->second->localStates[hdr.globalRank] = std::move(localState);
    }

    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] Activated snapshot={} for global rank={} with expected tensor data size={} path={}\n",
               *server,
               sessionId,
               hdr.snapshotId,
               hdr.globalRank,
               hdr.snapshotSize,
               ckptPath);
}

void Session::Handle(cipc::Message::FinalizeSnapshot const& hdr, std::span<char const> /* buf */)
{
    {
        std::scoped_lock lock(mutex);

        auto& snapshot   = GetSnapshot_(lock, hdr.snapshotId);
        auto& localState = *snapshot.localStates[hdr.globalRank];

        if (localState.writtenTensorSize != localState.tensorStorage.size)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Incomplete tensor data received. sessionid={}. "
                                                 "rcvd-size={}(range{} - {}) expected={} ",
                                                 *server,
                                                 sessionId,
                                                 hdr.sessionId,
                                                 localState.writtenTensorSize,
                                                 localState.writtenTensorRangeBegin,
                                                 localState.writtenTensorRangeEnd,
                                                 localState.tensorStorage.size));
        }
        if (localState.writtenTensorRangeBegin != 0 || localState.writtenTensorRangeEnd != localState.tensorStorage.size)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Tensor data received has invalid range. sessionid={}. "
                                                 "rcvd-size={}(range{} - {}) expected range=0 - {}",
                                                 *server,
                                                 sessionId,
                                                 hdr.sessionId,
                                                 localState.writtenTensorSize,
                                                 localState.writtenTensorRangeBegin,
                                                 localState.writtenTensorRangeEnd,
                                                 localState.tensorStorage.size));
        }
        if (sessionState.globalRegistry == nullptr)
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][WARN] Global state registry is not initialized when finalizing rank={} snapshot={}\n",
                       *server,
                       sessionId,
                       hdr.globalRank,
                       hdr.snapshotId);
        }
        snapshot.finalizedRanks.insert(hdr.globalRank);
    }
    // TODO_SAVETENSORS : Write out the rank specific checkpoint file
    // WriteTensorsToFile(snapshot, snapshot.path / "tensors.bin");

    {
        std::scoped_lock lock(mutex);    // see Handle(LocalStateRegistry): the task list is shared across concurrent handlers
        asyncTasks.emplace_back(std::async(std::launch::async | std::launch::deferred, &Session::OnFinalizeSnapshotAsync_, this, hdr));
    }
}

void Session::HandleStateRegistryFragment(Rank                                                 recvRank,
                                          std::vector<TensorShardInfo> const&                  tensorShards,
                                          std::vector<std::pair<std::string, uint64_t>> const& blobMetasIn)
{
    std::scoped_lock lock(mutex);
    MergeFragmentToGlobalState_(lock, tensorShards);
    // recvRank is always a non-coordinator rank here (rank 0 lives on the coordinator
    // peer and never streams a fragment), so every Blob it reports is local
    // to that rank. Snapshot-scoped common state is added separately on rank 0.
    for (auto const& [fqn, length] : blobMetasIn) { blobMetas[fqn][recvRank] = {.length = length}; }
    recievedFrom.insert(recvRank);
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] HandleStateRegistryFragment rank={} shards={} Blobs={} recievedFrom=[{}]\n",
               *server,
               sessionId,
               recvRank,
               tensorShards.size(),
               blobMetasIn.size(),
               fmt::join(recievedFrom, ","));
}

void Session::RecordBlobMetasLocked_(Rank rank, StateTree const& localRegistry)
{
    localRegistry.WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
        auto const& blob                                      = std::get<StateNode::Blob>(node.data);
        blobMetas[localRegistry.GetNodeId(node.handle)][rank] = {.length = blob.size()};
    });
}

void Session::PrepareBlobState_(Snapshot& snapshot, LocalState& localState, Rank rank) const
{
    // The snapshot-scoped common state lives only on rank 0 and carries fresh
    // per-save bytes. Exclude those fqns from the registry walk — its Blob nodes hold
    // stale registration bytes — and add rank 0's fresh copies from the snapshot.
    std::unordered_set<std::string> commonFqns;
    for (auto const& blob : snapshot.snapshotBlobs) { commonFqns.insert(blob.fqn); }

    localState.blobs.clear();
    localState.localRegistry->WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
        auto fqn = localState.localRegistry->GetNodeId(node.handle);
        if (commonFqns.contains(fqn)) { return; }
        auto const& blob                 = std::get<StateNode::Blob>(node.data);
        localState.blobs[std::move(fqn)] = std::string_view(blob.data(), blob.size());
    });
    if (rank == Rank{0})
    {
        for (auto const& blob : snapshot.snapshotBlobs) { localState.blobs[blob.fqn] = blob.data; }
    }

    if (rank != Rank{0}) { return; }

    // The readiness barrier guarantees that all registry fragments are
    // present. Copy the session template under the same mutex used by fragment
    // handlers; the snapshot owns all per-save adjustments from this point.
    {
        std::scoped_lock lock(mutex);
        snapshot.blobMetadata = blobMetas;
    }

    // Ranks on the coordinator peer saw rank 0's fresh snapshot Blob list and
    // excluded those stale registry values from their writes. Remote peers do
    // not receive that list and still write their registered copies, so retain
    // those ranks to keep later offsets aligned with their physical files.
    for (auto const& blob : snapshot.snapshotBlobs)
    {
        auto metadataIt = snapshot.blobMetadata.find(blob.fqn);
        if (metadataIt == snapshot.blobMetadata.end()) { continue; }
        for (uint32_t localRank = 0; localRank < topo.ranksPerPeer; ++localRank) { metadataIt->second.erase(Rank{localRank}); }
        if (metadataIt->second.empty()) { snapshot.blobMetadata.erase(metadataIt); }
    }

    for (auto const& [fqn, bytes] : localState.blobs) { snapshot.blobMetadata[fqn][Rank{0}] = {.length = bytes.size()}; }
}

void Session::Handle(cipc::Message::FinalizeSession const& /* hdr */, std::span<char const> /* buf */)
{}

bool Session::ResolveSnapshotId(int64_t iteration, cipc::SnapshotId& out) const
{
    if (iteration >= 0)
    {
        out = static_cast<cipc::SnapshotId>(static_cast<uint32_t>(iteration));
        return true;
    }
    // Negative: 1-based index back from the newest known snapshot.
    std::scoped_lock lock(mutex);
    auto const idx = static_cast<size_t>(-iteration);
    if (idx <= snapshots.size())
    {
        auto it = snapshots.rbegin();
        std::advance(it, static_cast<std::ptrdiff_t>(idx - 1));
        out = it->first;
        return true;
    }
    // Nothing (or not enough) resident: -1 can still name the newest durable-evicted snapshot.
    if (idx == 1 && highestEvictedDurableSnapshot != cipc::SnapshotId::Invalid)
    {
        out = highestEvictedDurableSnapshot;
        return true;
    }
    return false;
}

bool Session::IsSnapshotDurable(cipc::SnapshotId snapshotId) const
{
    std::scoped_lock lock(mutex);
    auto it = snapshots.find(snapshotId);
    if (it != snapshots.end()) { return it->second->writeCkptCompletedRanks.size() >= ExpectedWriteCompletions(); }
    // Not resident: durable iff it completed and was evicted (snapshot ids are monotonic).
    return highestEvictedDurableSnapshot != cipc::SnapshotId::Invalid
           && static_cast<uint32_t>(snapshotId) <= static_cast<uint32_t>(highestEvictedDurableSnapshot);
}

void Server::Start()
{
    std::scoped_lock lock(_mutex);
    _stopRequested = false;
    if (_running) { return; }
    _ioc.restart();

    _handlerPool = std::make_unique<boost::asio::thread_pool>(HandlerThreads);
    StartTcpListener_();
    _running             = true;
    auto       queueSize = cipc::MsgQueue::DetectShmSize(
                               cipc::Message::RequestChannelName(_index, cipc::MaxRanksPerNode + 1), cipc::MaxShmSize, cipc::MinShmSize)
                           / cipc::MaxRanksPerNode;
    std::latch queuesReady{cipc::MaxRanksPerNode};
    // Start server thread
    //_serverThread = std::thread([this]() {
    for (uint8_t i = 0; i < cipc::MaxRanksPerNode; ++i)
    {
        _threads.emplace_back([this, i, queueSize, &queuesReady]() {
            SetThreadName(fmt::format("ncs:svrq:{}", i).c_str());
            // Map queues across host-memory NUMA domains. Pinning around Create
            // places the control page; materializing the data pages in the owner
            // mapping prevents CUDA registration in a restricted client process
            // from faulting every shared page onto one socket.
            auto const numaNodes = DetectHostMemoryNumaNodes();
            int        queueNuma = numaNodes.front();
            if (numaNodes.size() > 1)
            {
                size_t queuesPerDomain = static_cast<size_t>(cipc::MaxRanksPerNode) / numaNodes.size();
                if (queuesPerDomain == 0) { queuesPerDomain = 1; }
                size_t const domain = std::min(numaNodes.size() - 1, static_cast<size_t>(i) / queuesPerDomain);
                queueNuma           = numaNodes[domain];
                BindThreadCpuToNumaNode(queueNuma);
            }
            auto msgQueue = cipc::MsgQueue::Create(cipc::Message::RequestChannelName(_index, i), queueSize);
            if (numaNodes.size() > 1)
            {
                auto region = msgQueue.GetShmDataRegion();
                BindRegionToNumaNode(queueNuma, region.data(), region.size());
                std::fill(region.begin() + static_cast<std::ptrdiff_t>(cipc::MsgQueue::HdrAllocSize), region.end(), char{});
            }
            UnbindThreadCpu();
            // Create publishes the queue geometry. Only after every queue reaches
            // this point may a ping report the service as ready: clients start their
            // fixed shared-memory open window immediately after SvcPing returns.
            queuesReady.count_down();
            {
                std::scoped_lock lock1(_mutex);
                _msgQueue.insert(&msgQueue);
            }
            fmt::print(stderr, "[native_ckpt:svr][{}] Checkpoint server started QueueSize={} Waiting for messages...\n", *this, queueSize);
            std::fflush(stderr);
            while (true) try
                {
                    auto const [stop, hdr, buf] = msgQueue.BeginRead();
                    if (stop)
                    {
                        std::scoped_lock lock1(_mutex);
                        fmt::print(stderr, "[native_ckpt:svr][{}] Stop requested. Shutting down server thread.\n", *this);
                        _msgQueue.erase(&msgQueue);
                        return;
                    }
                    // The slot returns to the writer even on a throw; a skipped EndRead
                    // leaks it permanently and a saturated leak wedges the queue.
                    try
                    {
                        HandleMessage(hdr, buf);
                    } catch (std::exception const& e)
                    {
                        fmt::print(
                            stderr, "[native_ckpt:svr][{}][ERROR] Exception handling message (slot released): {}\n", *this, e.what());
                    }
                    msgQueue.EndRead();

                } catch (std::exception const& e) { fmt::print(stderr, "[native_ckpt:svr][{}][ERROR] Exception: {}\n", *this, e.what()); }
        });
    }
    queuesReady.wait();
    StartTcpAcceptors_();
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
    while (true)
    {
        if (_stopRequested || (((_lastPingTime + AutoShutdownTimeout) < PerfClock::now()) && HasSessionsAndAllSessionsComplete()))
        {
            break;
        }
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
        return;    // TODO : reset the shutdown timer
    }
    case cipc::Message::Type::Shutdown:
    {
        Stop();
    }
    break;
    // break;
    case cipc::Message::Type::ActivateSession:
    {
        std::scoped_lock lock(_mutex);
        _lastPingTime = PerfClock::now();

        auto      hdr1 = std::get<cipc::Message::ActivateSession>(hdr);
        BufStream bufStream{buf.data(), buf.size()};
        auto      workDirStr    = serdes::ReadBlobWithSize<uint16_t>(bufStream);
        auto      workDir       = workDirStr.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(workDirStr);
        auto      createSession = [&]() {
            auto topo                   = StateSerDes::ReadTopology(bufStream);
            auto session                = std::make_unique<Session>(this, hdr1.sessionId, std::move(workDir), std::move(topo));
            session->numActiveSnapshots = std::min(hdr1.numActiveSnapshots, Session::MaxActiveSnapshots);
            session->Activate();
            return session;
        };

        fmt::print(stderr, "[native_ckpt:svr][{}#{}] ActivateSession workDir={}\n", *this, hdr1.sessionId, workDir);
        auto it = _activeSessions.find(hdr1.sessionId);
        it      = it != _activeSessions.end() ? it : _activeSessions.insert({hdr1.sessionId, createSession()}).first;
    }
    break;
    case cipc::Message::Type::FinalizeSession:
    {
        auto const& hdr1 = std::get<cipc::Message::FinalizeSession>(hdr);
        fmt::print(stderr, "[native_ckpt:svr][{}#{}] FinalizeSession    \n", *this, hdr1.sessionId);
        SessionById_(hdr1.sessionId).Handle(hdr1, buf);
    }
    break;
    case cipc::Message::Type::LocalStateRegistry:
    {
        auto const& hdr1 = std::get<cipc::Message::LocalStateRegistry>(hdr);
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
        // fmt::print(stderr, "[native_ckpt:svr] {}\n", _msgQueue->rlatency);
        auto const& hdr1 = std::get<cipc::Message::FinalizeSnapshot>(hdr);
        SessionById_(hdr1.sessionId).Handle(hdr1, buf);
    }
    break;
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
                        fmt::print(stderr,
                                   "[native_ckpt:svr][{}#{}][WARN] stopping with snapshot {} INCOMPLETE: "
                                   "finalized {}/{}, write-completed {}/{}\n",
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
        // wait-for-ckpt-finalize handlers observe _stopRequested and exit within a heartbeat;
        // destroying their futures joins the underlying threads.
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
        if (acceptor.close(ec))
        {
            fmt::print(stderr, "[native_ckpt:svr][{}][TCP][WARN] Failed to close acceptor: {}\n", *this, ec.message());
        }
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
                         *snapshot->globalRegistry,
                         *localState->localRegistry,
                         localState->tensorStorage);
            }
        }
    }
}

Session& Server::SessionById_(cipc::SessionId sessionId)
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

void Server::NotifyFinalizeComplete_(cipc::SessionId sessionId, cipc::Rank fromRank, cipc::SnapshotId snapshotId)
{
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] Finalize complete notification received for snapshot={} from rank={}\n",
               *this,
               sessionId,
               snapshotId,
               fromRank);
    bool emitOnce = false;
    auto start    = PerfClock::now();
    while (true)
    {
        // Release _mutex before the wait below: ActivateSession also needs it to create the
        // session, so holding it across the sleep would block the very creation we wait for
        // (and stall every other handler, including spin-down's idle check).
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            auto                         it = _activeSessions.find(sessionId);
            if (it != _activeSessions.end())
            {
                auto it2 = it->second->snapshots.find(snapshotId);
                if (it2 == it->second->snapshots.end())
                {
                    throw std::runtime_error(
                        fmt::format("Received finalize complete notification for non-existent snapshot={} from rank={}. Ignoring.\n",
                                    snapshotId,
                                    fromRank));
                }
                std::scoped_lock<std::mutex> lock2(it->second->mutex);
                it2->second->finalizedRanks.insert(fromRank);
                if (emitOnce)
                {
                    fmt::print(stderr,
                               "[native_ckpt:svr][{}#{}] Marked snapshot={} finalized for rank={}. Finalized ranks={}/{}\n",
                               *this,
                               sessionId,
                               snapshotId,
                               fromRank,
                               it2->second->finalizedRanks.size(),
                               it->second->topo.worldSize);
                }
                return;
            }
        }

        if (PerfClock::now() - start > 30s)
        {
            throw std::runtime_error(
                fmt::format("Received finalize complete notification for non-existent session={} from rank={}.\n", sessionId, fromRank));
        }
        if (!emitOnce)
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}] Finalize complete notification received for non-existent session={} from rank={}. "
                       "Waiting...\n",
                       *this,
                       sessionId,
                       sessionId,
                       fromRank);
            emitOnce = true;
        }
        std::this_thread::sleep_for(100ms);
    }
}

void Server::NotifyWriteCkptComplete_(cipc::SessionId sessionId, cipc::Rank fromRank, cipc::SnapshotId snapshotId)
{
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] Write checkpoint complete notification received for snapshot={} from rank={}\n",
               *this,
               sessionId,
               snapshotId,
               fromRank);
    auto lock = std::scoped_lock<std::mutex>(_mutex);
    auto it   = _activeSessions.find(sessionId);
    if (it == _activeSessions.end())
    {
        throw std::runtime_error(fmt::format(
            "Received write checkpoint complete notification for non-existent session={} from rank={}. Ignoring.\n", sessionId, fromRank));
    }
    auto it2 = it->second->snapshots.find(snapshotId);
    if (it2 == it->second->snapshots.end())
    {
        throw std::runtime_error(
            fmt::format("Received write checkpoint complete notification for non-existent snapshot={} from rank={}. Ignoring.\n",
                        snapshotId,
                        fromRank));
    }
    auto lock2 = std::scoped_lock<std::mutex>(it->second->mutex);
    it2->second->writeCkptCompletedRanks.insert(fromRank);
}

// Coordinator -> peer: preserve the complete global registry and derive this
// daemon's shared storage plan. The arrival also means finalize is globally
// done.
void Server::ReceiveGlobalRegistry_(cipc::SessionId sessionId, cipc::SnapshotId snapshotId, std::string_view body)
{
    BufStream                  bufStream{body.data(), body.size()};
    std::shared_ptr<StateTree> globalRegistry = StateSerDes::Read(bufStream);
    auto dedupResult = DeduplicateTensorReplicas(*globalRegistry, static_cast<uint32_t>(globalRegistry->GlobalWorldSize()));
    std::vector<std::unique_ptr<Session::Snapshot>> evicted;
    size_t                                          remainingSnapshots = 0;
    uint32_t                                        numActiveSnapshots = 0;
    {
        std::scoped_lock lock(_mutex);
        auto             it = _activeSessions.find(sessionId);
        if (it == _activeSessions.end())
        {
            throw std::runtime_error(fmt::format("Received global registry for non-existent session {}\n", sessionId));
        }
        auto&            session = *it->second;
        std::scoped_lock lock2(session.mutex);
        auto             it2 = session.snapshots.find(snapshotId);
        if (it2 == session.snapshots.end())
        {
            throw std::runtime_error(fmt::format("Received global registry for non-existent snapshot {}\n", snapshotId));
        }
        it2->second->globalRegistry        = std::move(globalRegistry);
        it2->second->dedupedGlobalRegistry = std::move(dedupResult.dedupedRegistry);
        auto const worldSize               = static_cast<uint32_t>(session.topo.worldSize);
        for (uint32_t r = 0; r < worldSize; ++r) { it2->second->finalizedRanks.insert(cipc::Rank{r}); }
        fmt::print(stderr,
                   "[native_ckpt:svr][{}#{}] Received global registry for snapshot {} and selected owners for {} logical tensor "
                   "shards; removed {} replicas; assigned bytes by rank=[{}]\n",
                   *this,
                   sessionId,
                   snapshotId,
                   dedupResult.stats.logicalShards,
                   dedupResult.stats.replicasRemoved,
                   fmt::join(dedupResult.stats.assignedBytesPerRank, ","));

        // Writers wait for dedupedGlobalRegistry, so any completed local writes
        // necessarily used this locally derived storage plan.
        while (!session.snapshots.empty() && session.snapshots.size() > session.numActiveSnapshots)
        {
            auto oldest = session.snapshots.begin();
            if (!session.IsSnapshotEvictable(*oldest->second)) { break; }
            evicted.push_back(std::move(oldest->second));
            session.snapshots.erase(oldest);
        }
        remainingSnapshots = session.snapshots.size();
        numActiveSnapshots = session.numActiveSnapshots;
    }
    if (!evicted.empty())
    {
        fmt::print(stderr,
                   "[native_ckpt:svr][{}#{}][INFO] Evicted {} snapshots after receiving global registry for snapshot={}. "
                   "Remaining snapshots in memory={} numActiveSnapshots={}\n",
                   *this,
                   sessionId,
                   evicted.size(),
                   snapshotId,
                   remainingSnapshots,
                   numActiveSnapshots);
    }
}

void Server::RecieveStateRegistryFragments_(cipc::SessionId const&                                               sessionId,
                                            cipc::Rank                                                           fromRank,
                                            tcp_stream&                                                          stream,
                                            boost::beast::flat_buffer&                                           buffer,
                                            boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser)
{
    fmt::print(stderr, "[native_ckpt:svr][{}#{}] Receiving state registry chunks for rank={}\n", *this, sessionId, fromRank);

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
        if (recordTag == PeerRecordTensor)
        {
            TensorShardInfo info;
            uint32_t        nodeIdLen{};
            readTrivial(info.rank);
            readTrivial(info.tensorInfo);
            readTrivial(nodeIdLen);
            info.nodeId = std::string(nodeIdLen, '\0');
            readBlob(info.nodeId.data(), nodeIdLen);
            info.tensorInfo.AssertShardIsValid();
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
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][INFO] Received state registry fragment for non-existent session. Adding to "
                       "pending data\n",
                       *this,
                       sessionId,
                       fromRank);
            auto pendingData          = std::make_unique<PendingSessionData>();
            pendingData->tensorShards = std::move(tensorShards);
            pendingData->blobMetas    = std::move(blobMetas);
            pendingData->recvFrom     = fromRank;
            pendingData->recievedAt   = PerfClock::now();
            _pendingSessionData[sessionId].push_back(std::move(pendingData));
            return;
        }
        session = it->second.get();
    }
    session->HandleStateRegistryFragment(fromRank, tensorShards, blobMetas);
    fmt::print(stderr,
               "[native_ckpt:svr][{}#{}] Successfully received and processed {} tensor shards from rank={}\n",
               *this,
               sessionId,
               tensorShards.size(),
               fromRank);
}

void Server::SpinDownWaitLoop_(tcp_stream stream)
{
    SetThreadName("ncs:spindown");
    static constexpr auto HeartbeatInterval = 2s;
    // No explicit timeout falls back to a generous default so a wedged server can't hang
    // a caller forever — matched to WaitForShutdown's idle auto-shutdown window.

    auto const deadline = PerfClock::now() + std::chrono::duration_cast<PerfClock::duration>(AutoShutdownTimeout);

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
        fmt::print(stderr, "[native_ckpt:svr][{}][ERROR] spindown timed out before idle; hard-stopping (aborting in-flight work)\n", *this);
        std::scoped_lock lock(_mutex);
        for (auto& session : _activeSessions)
        {
            std::scoped_lock slock(session.second->mutex);
            session.second->stopRequested = true;
        }
    }
    else
    {
        fmt::print(stderr, "[native_ckpt:svr][{}][INFO] spindown: all sessions idle; spinning down\n", *this);
    }

    // Final empty "{}" payload, matching the hard-stop reply, then the terminating chunk.
    static constexpr std::string_view Final = "{}";
    boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(Final)), ec);
    boost::asio::write(stream.socket(), boost::beast::http::make_chunk_last(), ec);
    TryCleanShutdown(stream);

    _stopRequested = true;    // WaitForShutdown observes this and calls Stop()
}

// Hold one client connection open, heartbeating, until the requested snapshot is durable on this
// daemon (or the deadline elapses). Scoped to a single snapshot and, unlike spin-down, never
// triggers shutdown. Runs on a dedicated thread (see _finalizeWaits): completion depends on peer
// writeckptcomplete/finalizecomplete notifications that the handler pool services, so occupying
// that pool would deadlock the very progress it waits on.
void Server::FinalizeWaitLoop_(tcp_stream stream, cipc::SessionId sessionId, int64_t iteration)
{
    SetThreadName("ncs:finwait");
    static constexpr auto HeartbeatInterval = 2s;
    auto const            deadline          = PerfClock::now() + std::chrono::duration_cast<PerfClock::duration>(AutoShutdownTimeout);
    stream.expires_never();

    boost::system::error_code                                    ec;
    boost::beast::http::response<boost::beast::http::empty_body> res{boost::beast::http::status::ok, 11};
    res.set(boost::beast::http::field::server, "NativeCheckpointServer");
    res.set(boost::beast::http::field::content_type, "application/x-ndjson");
    res.chunked(true);
    boost::beast::http::response_serializer<boost::beast::http::empty_body> sr{res};
    boost::beast::http::write_header(stream, sr, ec);

    bool             resolved = false;
    cipc::SnapshotId snapshotId{cipc::SnapshotId::Invalid};
    bool             durable  = false;
    bool             timedOut = false;
    while (!ec)
    {
        if (IsFinalizeComplete_(sessionId, iteration, snapshotId, resolved))
        {
            durable = true;
            break;
        }
        if (_stopRequested) { break; }
        if (PerfClock::now() >= deadline)
        {
            timedOut = true;
            break;
        }
        // Newline-delimited JSON heartbeat keeps the client read alive and leaves room for real
        // progress events later; the client only blocks on the terminating chunk for now.
        auto heartbeat = fmt::format("{{\"event\":\"waiting\",\"iteration\":{}}}\n", iteration);
        boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(heartbeat)), ec);
        if (ec) { break; }    // client hung up
        std::this_thread::sleep_for(HeartbeatInterval);
    }

    std::string finalEvent;
    if (durable)
    {
        finalEvent
            = fmt::format("{{\"event\":\"complete\",\"iteration\":{},\"snapshot\":{}}}\n", iteration, static_cast<uint32_t>(snapshotId));
    }
    else if (timedOut)
    {
        finalEvent = fmt::format("{{\"event\":\"timeout\",\"iteration\":{}}}\n", iteration);
        fmt::print(
            stderr, "[native_ckpt:svr][{}#{}][ERROR] wait-for-ckpt-finalize timed out for iteration={}\n", *this, sessionId, iteration);
    }
    else
    {
        finalEvent = fmt::format("{{\"event\":\"aborted\",\"iteration\":{}}}\n", iteration);
    }
    boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(finalEvent)), ec);
    boost::asio::write(stream.socket(), boost::beast::http::make_chunk_last(), ec);
    TryCleanShutdown(stream);
}

// Wait-loop helper: resolve `iteration` to a snapshot once, then report its durability. Returns
// false (keep waiting) while the session or its snapshot is not yet known.
bool Server::IsFinalizeComplete_(cipc::SessionId sessionId, int64_t iteration, cipc::SnapshotId& snapshotId, bool& resolved)
{
    std::scoped_lock lock(_mutex);
    auto             it = _activeSessions.find(sessionId);
    if (it == _activeSessions.end()) { return false; }
    auto& session = *it->second;
    if (!resolved)
    {
        if (!session.ResolveSnapshotId(iteration, snapshotId)) { return false; }
        resolved = true;
    }
    return session.IsSnapshotDurable(snapshotId);
}

void Server::HandleRequest_(tcp_stream& stream)
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
        RecieveStateRegistryFragments_(sessionId, rank, stream, buffer, bufferParser);
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    // Check if this is the state upload endpoint: POST /api/peer/finalizecomplete/{sessionId}/{rank}/{snapshotId}

    else if (matches({"api", "peer", "finalizecomplete"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId  = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto rank       = cipc::Rank{boost::lexical_cast<uint32_t>(*(it++))};
        auto snapshotId = cipc::SnapshotId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in state registry upload endpoint"); }

        boost::beast::http::request_parser<boost::beast::http::buffer_body> bufferParser{std::move(headerParser)};
        NotifyFinalizeComplete_(sessionId, rank, snapshotId);
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    else if (matches({"api", "peer", "writeckptcomplete"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId  = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto rank       = cipc::Rank{boost::lexical_cast<uint32_t>(*(it++))};
        auto snapshotId = cipc::SnapshotId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in state registry upload endpoint"); }

        boost::beast::http::request_parser<boost::beast::http::buffer_body> bufferParser{std::move(headerParser)};
        NotifyWriteCkptComplete_(sessionId, rank, snapshotId);
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
    }
    // Coordinator broadcast: the merged global registry for a snapshot (body = serialized tree).
    else if (matches({"api", "peer", "globalregistry"}) && headers.method() == boost::beast::http::verb::post && (it != segs.end()))
    {
        auto sessionId  = cipc::SessionId{boost::lexical_cast<uint32_t>(*(it++))};
        auto snapshotId = cipc::SnapshotId{boost::lexical_cast<uint32_t>(*(it++))};
        if (it != segs.end()) { throw std::invalid_argument("Unexpected extra URL segments in globalregistry endpoint"); }

        boost::beast::http::request_parser<boost::beast::http::string_body> bodyParser{std::move(headerParser)};
        bodyParser.body_limit(std::numeric_limits<std::uint64_t>::max());
        boost::beast::http::read(stream, buffer, bodyParser);
        ReceiveGlobalRegistry_(sessionId, snapshotId, bodyParser.get().body());
        boost::beast::http::response<boost::beast::http::string_body> res{boost::beast::http::status::ok, headers.version()};
        res.set(boost::beast::http::field::server, "NativeCheckpointServer");
        res.keep_alive(headers.keep_alive());
        res.body() = "{}";
        res.prepare_payload();
        boost::beast::http::write(stream, res);
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
        // accept loop must stay free to service the peer writeckptcomplete/finalizecomplete
        // notifications that idle-ness depends on, otherwise the wait would deadlock.
        std::scoped_lock lock(_mutex);
        _spinDownThreads.emplace_back([this, s = std::move(stream)]() mutable { SpinDownWaitLoop_(std::move(s)); });
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
        _finalizeWaits.emplace_back(std::async(std::launch::async, [this, s = std::move(stream), sessionId, iteration]() mutable {
            FinalizeWaitLoop_(std::move(s), sessionId, iteration);
        }));
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

#if defined(SO_REUSEPORT) && !defined(PHOENIX_BUILD_FOR_VALGRIND)
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

void Server::StartTcpListener_()
{
    using boost::asio::ip::tcp;
    auto const    address = boost::asio::ip::make_address("0.0.0.0");
    tcp::endpoint endpoint(address, Port + _index);

#if defined(SO_REUSEPORT) && !defined(PHOENIX_BUILD_FOR_VALGRIND)
    // When SO_REUSEPORT is enabled, the TCP port no longer elects a single owner.
    // Losers must throw before touching acceptors or shm rings; Start() spawns the
    // shm consumer threads only after this returns.
    AcquireInstanceLock_();
#endif

    _tcpAcceptors.clear();
    _tcpAcceptors.reserve(AcceptorCount);
    for (size_t i = 0; i < AcceptorCount; ++i)
    {
        auto& acceptor = _tcpAcceptors.emplace_back(_ioc);
        acceptor.open(endpoint.protocol());
        acceptor.set_option(boost::asio::socket_base::reuse_address(true));
#if defined(SO_REUSEPORT) && !defined(PHOENIX_BUILD_FOR_VALGRIND)
        int const enable = 1;
        ::setsockopt(acceptor.native_handle(), SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable));
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
                } catch (std::exception& e) { fmt::print(stderr, "Error in acceptor: {}\n", e.what()); }
        });
    }

    for (size_t i = 0; i < AcceptorCount; i++)
    {
        _threads.emplace_back([this, i]() {
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
        auto socket = co_await acceptor.async_accept();
        boost::asio::post(*_handlerPool, [this, sock = std::move(socket)]() mutable {
            static constexpr auto Timeout = std::chrono::seconds{30};
            tcp_stream            stream(std::move(sock));
            stream.expires_after(Timeout);
            try
            {
                HandleRequest_(stream);
            } catch (boost::system::system_error const& e)
            {
                TryCleanShutdown(stream);
                if (e.code() != boost::beast::http::error::end_of_stream) { fmt::print(stderr, "Error handling request: {}\n", e.what()); }
            } catch (std::exception const& e)
            {
                TryCleanShutdown(stream);
                fmt::print(stderr, "Error: {}\n", e.what());
            }
        });
    }
}

void Session::OnRecieveLocalStateRegistryAsync_(Rank const& rank)
{
    {
        std::scoped_lock lock(mutex);
        auto&            localState = *sessionState.localStates[rank];
        recievedFrom.insert(rank);
        fmt::print(stderr,
                   "[native_ckpt:svr][{}#{}] OnRecieveLocalStateRegistryAsync_  rank={} recievedFrom=[{}]\n",
                   *server,
                   sessionId,
                   rank,
                   fmt::join(recievedFrom, ","));

        if (sessionState.globalRegistry == nullptr)
        {
            sessionState.globalRegistry = localState.localRegistry->Clone();
            sessionState.globalRegistry->ConvertTensorShardsToShardedTensors();
            sessionState.globalRegistry->WalkNodes([&](auto& node) {
                if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
                {
                    throw std::runtime_error(fmt::format("[native_ckpt:svr][{}#{}][ERROR] Unexpected TensorShard node in "
                                                         "activeGlobalState. Should only contain ShardedTensor nodes",
                                                         *server,
                                                         sessionId,
                                                         sessionState.globalRegistry->GetNodeId(node.handle)));
                }
                if (node.data.index() != static_cast<size_t>(StateNode::Type::ShardedTensor)) { return; }
                tensorMap[localState.localRegistry->GetNodeId(node.handle)] = node.handle;
            });
        }
        else
        {
            std::vector<TensorShardInfo> tensorShards;
            ForeachTensorShardInfo(*localState.localRegistry, [&](TensorShardInfo const& info) { tensorShards.push_back(info); });
            MergeFragmentToGlobalState_(lock, tensorShards);
        }

        // Blobs are written per-rank rather than gathered onto the coordinator.
        // Record only lengths here; rank 0's own and snapshot-scoped values are
        // assembled once their fresh per-save bytes are known.
        if (IsCoordinatorPeer() && rank != Rank{0}) { RecordBlobMetasLocked_(rank, *localState.localRegistry); }
#ifdef NATIVE_CKPT_TRACE_NODE
        if (tensorMap.contains(NATIVE_CKPT_TRACE_NODE))
        {
            auto const& shTen
                = std::get<StateNode::ShardedTensor>(sessionState.globalRegistry->GetNode(tensorMap[NATIVE_CKPT_TRACE_NODE]).data);
            fmt::print(stderr,
                       "[native_ckpt:svr][{}#{}][VERBOSE] Rank={} Found in tensorMap NATIVE_CKPT_TRACE_NODE={}  {}\n",
                       *server,
                       sessionId,
                       rank,
                       NATIVE_CKPT_TRACE_NODE,
                       shTen);
        }
#endif
    }
    {
        // TODO_NOPENDING_STRUCT
        auto lock = std::scoped_lock<std::mutex>(server->_mutex);
        auto itp  = server->_pendingSessionData.find(sessionId);
        if (itp != server->_pendingSessionData.end())
        {
            for (auto& p : itp->second)
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}#{}][WARN] Found pending session data. activating session with pending data\n",
                           *server,
                           sessionId);
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
    for (uint16_t i = 0; i < nodes; i++)
    {
        fmt::print(stderr, "[native_ckpt:svr] Starting inproc server instance for node={}\n", i);
        ServerSingletons.push_back(std::make_unique<Server>(i));
        ServerSingletons.back()->Start();
    }
    fmt::print(stderr, "[native_ckpt:svr] Inproc server started.\n");
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
    for (auto& server : ServerSingletons)
    {
        if (server)
        {
            server->Stop();
            server.reset();
        }
    }
}

void WaitForInprocServerWrites()
{
    auto const deadline = PerfClock::now() + AutoShutdownTimeout;
    while (true)
    {
        bool foundServer = false;
        bool allComplete = true;
        for (auto const& server : ServerSingletons)
        {
            if (!server) { continue; }
            foundServer = true;
            allComplete = allComplete && server->HasSessionsAndAllSessionsComplete();
        }
        if (foundServer && allComplete) { return; }
        if (PerfClock::now() >= deadline)
        {
            throw std::runtime_error("Timed out waiting for in-process checkpoint server writes to become durable");
        }
        std::this_thread::sleep_for(50ms);
    }
}

void QueryServerSessions(ServerSessionCallback callback, void* cbData)
{
    for (auto& server : ServerSingletons)
    {
        if (server) { server->QuerySessions(callback, cbData); }
    }
}
