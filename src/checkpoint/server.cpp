#include "CommonMacros.h"
#include "SetThreadName.h"

#include "blob.h"
#include "fmttensor.h"
#include "messages.h"
#include "numa_utils.h"
#include "serdes_utils.h"
#include "server.h"
#include "session.h"
#include "state.h"
#include "stateserdes.h"
#include "tensor.h"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/buffer_body.hpp>
#include <boost/beast/http/empty_body.hpp>
#include <boost/beast/http/field.hpp>
#include <boost/beast/http/file_body.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/url.hpp>
#include <hwloc.h>
#include <openssl/evp.h>

#include <chrono>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace std::chrono_literals;
namespace cipc  = checkpoint::ipc;
using PerfClock = std::chrono::high_resolution_clock;

// uint32_t ranks
// uint32_t iterations
using Request = boost::beast::http::request<boost::beast::http::buffer_body>;

using tcp        = boost::asio::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using tcp_stream = typename boost::beast::tcp_stream::rebind_executor<
    boost::asio::use_awaitable_t<>::executor_with_default<boost::asio::any_io_executor>>::other;

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

static void DebugDumpLocalStateTreeKVP(std::filesystem::path const& fname, StateTree const& state)
{

    std::ofstream ostrm(fname);
    if (!ostrm.good()) { throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Failed to open file for writing: {}", fname)); }

    state.WalkNodes([&](auto& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::Invalid || type >= StateNode::Type::Unknown)
        {
            throw std::runtime_error(fmt::format("Invalid node type: {}", static_cast<uint8_t>(type)));
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
            if (tnsr.shards.size() == 0) throw std::runtime_error(fmt::format("Empty Sharded tensor : {}", name));
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
    default: throw std::runtime_error(fmt::format("Invalid DType: {}", static_cast<int>(dtype)));
    }
}

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
#if defined NATIVE_CKPT_TRACE_NODE
        auto fqn = registry.GetNodeId(node.handle);
        if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}][{}] ForeachTensorShardInfo NATIVE_CKPT_TRACE_NODE = {}. Node data "
                       "index = {}\n",
                       sessionId,
                       rank,
                       fqn,
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
                        StateNode::TensorShard{
                            .sharddim = shard.sharddim, .fulldim = tnsr.fulldim, .dtype = tnsr.dtype, .device = shard.device},
                        registry.GetNodeId(node.handle)});
            }
        }
    });
}

namespace
{
[[nodiscard]] static auto CreateRequestForPeer(boost::asio::io_context& ioc, boost::beast::tcp_stream& stream, std::string const& peer)
{
    auto peerAddr = boost::urls::url(peer);
    if (!peerAddr.has_port()) { peerAddr.set_port(std::to_string(cipc::Message::Port)); }

    // Connect to peer
    boost::asio::ip::tcp::resolver resolver(ioc);
    auto                           results = resolver.resolve(peerAddr.host(), peerAddr.port());
    stream.connect(results);

    Request req;
    // Write HTTP/1.1 request headers with Transfer-Encoding: chunked (no Content-Length)
    req.method(boost::beast::http::verb::post);
    static constexpr int HttpVersion11 = 11;    // HTTP/1.1
    req.version(HttpVersion11);
    req.set(boost::beast::http::field::host, fmt::format("{}:{}", peerAddr.host(), peerAddr.port()));
    req.set(boost::beast::http::field::content_type, "application/octet-stream");
    return req;
}
}    // namespace

Session::Session(Server* serverIn, cipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn) :
    server(serverIn), sessionId(sessionIdIn), topo(std::move(topoIn)), workDir(std::move(workDirIn))
{}

Session::LocalState& Session::GetSessionLocalState_(cipc::Rank rank)
{
    std::lock_guard<std::mutex> lock(mutex);
    return *sessionState.localStates[rank];
}

Session::LocalState& Session::FindOrCreateLocalState_(cipc::Rank rank)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto&                       localState = sessionState.localStates[rank];
    if (localState == nullptr) { localState = std::make_unique<LocalState>(); }
    return *localState;
}

Session::Snapshot& Session::GetSnapshot_(cipc::SnapshotId snapshotId)
{
    std::lock_guard<std::mutex> lock(mutex);
    return *snapshots[snapshotId];
}

// Write {peerIndex}.metadata.info: flat sorted "key=value" pairs for all tensor nodes.
// Field names and value formats are aligned with reference.metadata.info (produced by
// distcp_trace.py via recursive dict flattening) so the two files can be diffed line-by-line.
//
// Fields present in both files (comparable):
//   {fqn}.properties.dtype -- torch.bfloat16 etc.
//   {fqn}.size             -- [d0, d1, ...]
//   {fqn}.chunks.N.offsets -- [o0, o1, ...]
//   {fqn}.chunks.N.sizes   -- [s0, s1, ...]
//   {fqn}.md5sum  (single-chunk) or {fqn}.chunks.N.md5sum (multi-chunk)
//
// Fields present only in native (no distcp equivalent):
//   {fqn}.device           -- cuda:0 etc.

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
        throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Failed to open native metadata file for writing: {}", tmpFile));
    }

    if (snapshot.registry == nullptr)
    {
        throw std::runtime_error(
            fmt::format("[native_ckpt:svr][ERROR] Active global state is null when finalizing snapshot {} for session id {}",
                        snapshot.name,
                        sessionId));
    }

    auto& globalState = *snapshot.registry;

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
                           "[native_ckpt:svr][ERROR] Replicate shard {} md5sums dont match {} != {}\n",
                           fqnShardName,
                           it->second,
                           md5) /*)  d*/;
            }
        }
    };

    for (auto const& [localRank, localState] : snapshot.localStates)
    {
        localState->localRegistry->WalkNodes([&](StateNode const& node) {
            auto const type = static_cast<StateNode::Type>(node.data.index());
            if (type == StateNode::Type::TensorShard)
            {
                auto const& tnsr = std::get<StateNode::TensorShard>(node.data);
                auto const  size = tnsr.ShardSizeBytes();
                auto const* data = static_cast<char const*>(localState->blob.data) + tnsr.handle;    // NOLINT
#if defined NATIVE_CKPT_TRACE_NODE
                auto fqn = localState->localRegistry->GetNodeId(node.handle);
                if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                {
                    fmt::print(stderr,
                               "[native_ckpt:svr][{}] WriteNativeMetadataInfo_ NATIVE_CKPT_TRACE_NODE = {}. Blob Offset = {}\n",
                               sessionId,
                               fqn,
                               tnsr.handle);
                }
#endif
                addmd5(fmt::format("{}[{}]", localState->localRegistry->GetNodeId(node.handle), tnsr.sharddim), ComputeMd5Hex(data, size));
            }
            else if (type == StateNode::Type::ShardedTensor)
            {
                auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
                for (auto const& shard : tnsr.shards)
                {
                    auto const  size = shard.ShardSizeBytes(tnsr.dtype);
                    auto const* data = static_cast<char const*>(localState->blob.data) + shard.handle;    // NOLINT
#if defined NATIVE_CKPT_TRACE_NODE
                    auto fqn = localState->localRegistry->GetNodeId(node.handle);
                    if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                    {
                        fmt::print(stderr,
                                   "[native_ckpt:svr][{}] WriteNativeMetadataInfo_ NATIVE_CKPT_TRACE_NODE = {}. Blob Offset = {}\n",
                                   sessionId,
                                   fqn,
                                   shard.handle);
                    }
#endif
                    addmd5(fmt::format("{}[{}]", localState->localRegistry->GetNodeId(node.handle), shard.sharddim),
                           ComputeMd5Hex(data, size));
                }
            }
        });
    }

    globalState.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::ShardedTensor) { return; }
        auto const  nodeName = globalState.GetNodeId(node.handle);
        std::string name     = nodeName.starts_with("model.") ? nodeName.substr(std::string_view("model.").size()) : nodeName;

        auto const& shardedTnsr = std::get<StateNode::ShardedTensor>(node.data);
        if (shardedTnsr.shards.empty())
        {
            throw std::runtime_error(fmt::format("Sharded tensor {} has no shards in metadata output\n", name));
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
            if (lastShard == rankShard.sharddim) { continue; }
            lastShard         = rankShard.sharddim;
            auto fqnShardName = fmt::format("{}[{}]", globalState.GetNodeId(node.handle), rankShard.sharddim);
            auto md5It        = md5Map.find(fqnShardName);
            if (md5It == md5Map.end())
            {
                throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Missing md5sum for {}\n", fqnShardName));
            }
            emit(fmt::format("{}.chunks.{}.offsets", name, i), FormatShardOffsets(rankShard.sharddim));
            emit(fmt::format("{}.chunks.{}.sizes", name, i), FormatShardSizes(rankShard.sharddim));
            emit(fmt::format("{}.chunks.{}.md5sum", name, i), md5It->second);
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

void Session::OnRecieveLocalStateRegistryAsync_(Rank const& rank)
{
    fmt::print(stderr, "[native_ckpt:svr][{}] OnRecieveLocalStateRegistryAsync_\n", rank);
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto&                       localState = *sessionState.localStates[rank];
        recievedFrom.insert(rank);
        if (sessionState.registry == nullptr)
        {
            sessionState.registry = localState.localRegistry->Clone();
            sessionState.registry->ConvertTensorShardsToShardedTensors();
            sessionState.registry->WalkNodes([&](auto& node) {
                if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
                {
                    throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Unexpected TensorShard node in "
                                                         "activeGlobalState. Should only contain ShardedTensor nodes",
                                                         sessionId,
                                                         sessionState.registry->GetNodeId(node.handle)));
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
#if defined NATIVE_CKPT_TRACE_NODE
        if (tensorMap.contains(NATIVE_CKPT_TRACE_NODE))
        {
            auto const& shTen = std::get<StateNode::ShardedTensor>(sessionState.registry->GetNode(tensorMap[NATIVE_CKPT_TRACE_NODE]).data);
            fmt::print(stderr,
                       "[native_ckpt:svr][{}][{}] Found in tensorMap NATIVE_CKPT_TRACE_NODE = {}  {}\n",
                       sessionId,
                       rank,
                       NATIVE_CKPT_TRACE_NODE,
                       shTen);
        }
#endif
    }

    // if (recievedFrom.size() == rankPeerMap.size())
    //{    // This will only be invoked for 1 Node runs
    //     WriteGlobalStateTreeToFile(*sessionState.registry, workDir / "registry.bin");
    // }
}

void Session::OnFinalizeSnapshotAsync_(cipc::Message::FinalizeSnapshot const& hdr)
{
    // WriteRankShard reads this rank's blob + localRegistry without holding the
    // session mutex. Safe today because Handle(FinalizeSnapshot) validates
    // writtenTensorSize==blob.size under the mutex before dispatching us and
    // each rank's messages run on a dedicated queue thread (server.cpp:986).
    // If the dispatcher ever shares threads across ranks, this collapses.
    Snapshot*   snapshotPtr   = nullptr;
    LocalState* localStatePtr = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex);
        snapshotPtr   = snapshots.at(hdr.snapshotId).get();
        localStatePtr = snapshotPtr->localStates.at(hdr.globalRank).get();
    }
    auto& snapshot   = *snapshotPtr;
    auto& localState = *localStatePtr;
    // Rank 0 writes registry.bin before any per-rank distcp file lands. Wait
    // order matches origin/ankurv/phoenix: recievedFrom, then finalizedRanks.
    // TODO_MULTILOCAL_CKPT : For multinode local checkpointing this needs to happen once on each node so designate one rank
    if (hdr.globalRank == Rank{0})
    {
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (recievedFrom.size() >= static_cast<size_t>(topo.worldSize)) { break; }
                fmt::print(
                    stderr,
                    "[native_ckpt:svr][{}][{}] FinalizeSnapshot waiting for snapshot={} local-state-registry {} of {} remaining...\n",
                    sessionId,
                    hdr.globalRank,
                    hdr.snapshotId,
                    static_cast<size_t>(topo.worldSize) - recievedFrom.size(),
                    static_cast<size_t>(topo.worldSize));
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (snapshot.finalizedRanks.size() >= static_cast<size_t>(topo.worldSize)) { break; }
                fmt::print(stderr,
                           "[native_ckpt:svr][{}][{}] FinalizeSnapshot waiting for finalized ranks {} of {} remaining...\n",
                           sessionId,
                           hdr.globalRank,
                           hdr.snapshotId,
                           static_cast<size_t>(topo.worldSize) - snapshot.finalizedRanks.size(),
                           static_cast<size_t>(topo.worldSize));
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        WriteGlobalStateTreeToFile_(snapshot, snapshot.path.parent_path() / "registry.bin");
    }

    checkpoint::distcp::WriteRankShard(
        snapshot.path.parent_path(), hdr.globalRank, *localState.localRegistry, localState.blob, localState.writeLog);
    {
        std::lock_guard<std::mutex> lock(mutex);
        snapshot.writeCompletedRanks.insert(hdr.globalRank);
    }

    // Rank 0 emits metadata (DebugWriteNativeMetadataInfo_ and the distcp
    // .metadata pickle) once every rank has finished its WriteRankShard.
    if (hdr.globalRank == Rank{0})
    {
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (snapshot.writeCompletedRanks.size() >= static_cast<size_t>(topo.worldSize)) { break; }
                fmt::print(
                    stderr,
                    "[native_ckpt:svr][{}][{}] FinalizeSnapshot waiting for snapshot={} write-completed ranks {} of {} remaining...\n",
                    sessionId,
                    hdr.globalRank,
                    hdr.snapshotId,
                    static_cast<size_t>(topo.worldSize) - snapshot.writeCompletedRanks.size(),
                    static_cast<size_t>(topo.worldSize));
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        DebugWriteNativeMetadataInfo_(snapshot);
        // Assemble the per-rank manifest from snapshot.localStates. We take the
        // mutex briefly to read localStates — same pattern as every other
        // accessor of that map. The busy-wait above already established
        // happens-before with every worker's writeLog population (insert into
        // `writeCompletedRanks` under the mutex on the writer side, read of
        // its size under the mutex here), so the writeLog pointers we borrow
        // refer to fully-populated vectors that no other thread will touch.
        std::vector<checkpoint::distcp::RankManifestRef> manifest;
        {
            std::lock_guard<std::mutex> lock(mutex);
            manifest.reserve(snapshot.localStates.size());
            for (auto const& [r, ls] : snapshot.localStates)
            {
                manifest.push_back(checkpoint::distcp::RankManifestRef{.rank = r, .writeLog = &ls->writeLog});
            }
        }
        // Log on failure so the operator sees what broke. The async worker's
        // future is stored in `asyncTasks` but never `.get()`d (pre-existing
        // gap tracked in plan.md), so a silent rethrow at session teardown
        // would otherwise be the first user-visible signal.
        try
        {
            checkpoint::distcp::WriteMetadata(snapshot.path.parent_path(), *snapshot.registry, std::span{manifest});
        } catch (std::exception const& ex)
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}][{}] WriteMetadata failed for snapshot={}: {}\n",
                       sessionId,
                       hdr.globalRank,
                       hdr.snapshotId,
                       ex.what());
            throw;
        }
    }

    fmt::print(
        stderr, "[native_ckpt:svr][{}][{}] OnFinalizeSnapshotAsync_ for snapshotId : {}\n", sessionId, hdr.globalRank, hdr.snapshotId);
    // SendLocalShardedStateRegistryToPeer(cipc::Rank rank, std::string const& peer)
}

void Session::MergeFragmentToGlobalState_(std::lock_guard<std::mutex> const& /* lock */, std::vector<TensorShardInfo> const& tensorShards)
{
    if (sessionState.registry == nullptr)
    {
        throw std::logic_error("Cannot merge state registry fragments. Global state registry is not initialized");
    }

    for (auto const& tensorShard : tensorShards)
    {
#if defined NATIVE_CKPT_TRACE_NODE
        if (tensorShard.nodeId.contains(NATIVE_CKPT_TRACE_NODE))
        {
            fmt::print(stderr,
                       "[native_ckpt:svr][{}][{}] MergeFragmentToGlobalState_ Found tensor shard with nodeId containing "
                       "NATIVE_CKPT_TRACE_NODE = {}\n",
                       sessionId,
                       tensorShard.rank,
                       NATIVE_CKPT_TRACE_NODE);
        }
#endif
        tensorShard.tensorInfo.AssertShardIsValid();
        // Look up the node by its string NodeId
        auto it = tensorMap.find(tensorShard.nodeId);
        if (it == tensorMap.end())
        {
            auto const& stateName        = tensorShard.nodeId;
            auto [childName, parentNode] = sessionState.registry->FindOrAddNodeByFQN(stateName);
            // Add the final node which contains the handler
            auto& child = sessionState.registry->AddChild(parentNode, childName, true);
            auto  shTen = StateNode::ShardedTensor(tensorShard.tensorInfo.dtype, tensorShard.tensorInfo.fulldim);
            shTen.AddShard(StateNode::ShardedTensor::RankedTensorShard{
                .sharddim = tensorShard.tensorInfo.sharddim, .device = tensorShard.tensorInfo.device, .rank = tensorShard.rank});
            child.data                    = shTen;
            tensorMap[tensorShard.nodeId] = child.handle;
            continue;
        }
        auto  nodeHandle = it->second;
        auto& node       = sessionState.registry->GetNode(nodeHandle);
        if (node.data.index() != static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            throw std::runtime_error(fmt::format("Received tensor shard for non-sharded tensor '{}' expected-type({}) != actual-type({})",
                                                 tensorShard.nodeId,
                                                 static_cast<size_t>(StateNode::Type::ShardedTensor),
                                                 node.data.index()));
        }
        auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
        if (tnsr.shards.empty())
        {
            throw std::runtime_error(fmt::format("Sharded tensor {} has no shards in metadata output\n", tensorShard.nodeId));
        }

        tensorShard.tensorInfo.AssertShardIsValid();
        tnsr.AddShard({.sharddim = tensorShard.tensorInfo.sharddim, .device = tensorShard.tensorInfo.device, .rank = tensorShard.rank});
    }
}

// TODO_MULTILOCAL_CKPT
void Session::SendPeerFinalizeSnapshotComplete_(std::string const& peer, cipc::Message::FinalizeSnapshot const& msg) const
{
    boost::asio::io_context  ioc;
    boost::beast::tcp_stream stream(ioc);
    {
        auto req = CreateRequestForPeer(ioc, stream, peer);
        req.target(fmt::format("/api/peer/finalizecomplete/{}/{}/{}", sessionId, msg.globalRank, msg.snapshotId));

        // req.set(boost::beast::http::field::expect, "100-continue");
        boost::beast::http::request_serializer<boost::beast::http::buffer_body, boost::beast::http::fields> sr{req};
        boost::beast::http::write_header(stream, sr);
    }
    // Read the response
    boost::beast::flat_buffer                                     buffer;
    boost::beast::http::response<boost::beast::http::string_body> res;
    boost::beast::http::read(stream, buffer, res);

    if (res.result() != boost::beast::http::status::ok)
    {
        throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Peer {} returned error: {}", peer, res.result_int()));
    }
    TryCleanShutdown(stream);
}

void Session::SendLocalShardedStateRegistryToPeer_(cipc::Rank rank, std::string const& peer)
{
    boost::asio::io_context  ioc;
    boost::beast::tcp_stream stream(ioc);
    {
        auto req = CreateRequestForPeer(ioc, stream, peer);
        req.target(fmt::format("/api/peer/shardedstateregistry/{}/{}", sessionId, rank));
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
            WriteTensorShardInfo(oss, info);
            boost::asio::write(stream.socket(), boost::beast::http::make_chunk(boost::asio::buffer(oss.str())));
            numTensors++;
        });

        // Write terminal chunk
        boost::asio::write(stream.socket(), boost::beast::http::make_chunk_last());

        fmt::print(stderr, "[native_ckpt:svr][{}] Successfully sent {} tensor shards to {}\n", sessionId, numTensors, peer);
    }
    // Read the response
    boost::beast::flat_buffer                                     buffer;
    boost::beast::http::response<boost::beast::http::string_body> res;
    boost::beast::http::read(stream, buffer, res);

    if (res.result() != boost::beast::http::status::ok)
    {
        throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Peer {} returned error: {}", peer, res.result_int()));
    }
    TryCleanShutdown(stream);
}

void Session::WriteGlobalStateTreeToFile_(Snapshot const& snapshot, std::filesystem::path const& ckptPath) const
{
    auto start1 = PerfClock::now();
    std::filesystem::create_directories(ckptPath.parent_path());
    snapshot.registry->WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::TensorShard)
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:svr][ERROR] DebugWriteMetdataKeyValuePairs: Unexpected TensorShard node in activeGlobalState",
                            snapshot.registry->GetNodeId(node.handle)));
        }
        if (type == StateNode::Type::ShardedTensor)
        {
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            for (auto const& shard : tnsr.shards) { shard.AssertShardIsValid(tnsr.fulldim); }
            if (tnsr.shards.size() == 0)
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}][WARN] empty sharded tensor '{}' in metadata output\n",
                           snapshot.registry->GlobalRank(),
                           snapshot.registry->GetNodeId(node.handle));
                return;
            }
            snapshot.registry->AssertShardedTensorHasNoHoles(node.handle);
        }
    });
    {
        std::ofstream sstrm(ckptPath, std::ios::binary);
        if (!sstrm.good())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Failed to open metadata file for writing: {}",
                                                 snapshot.registry->GlobalRank(),
                                                 ckptPath.string()));
        }
        StateSerDes::Write(sstrm, *snapshot.registry, StateNode::Handle::Root);
        fmt::print(stderr,
                   "[native_ckpt:svr][{}][INFO] Finished writing metadata to file: {} \n",
                   snapshot.registry->GlobalRank(),
                   ckptPath.string());
    }
    // Verify if this was written correctly
    // Note because of empty nodes not being serialized this might not be a perfect match
    {
// #define NATIVE_CKPT_VERIFY_METADATA
#ifdef NATIVE_CKPT_VERIFY_METADATA
        std::ifstream sstrm(ckptPath, std::ios::binary);
        if (!sstrm.good())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Failed to open metadata file for verification: {}",
                                                 snapshot.registry->GlobalRank(),
                                                 ckptPath.string()));
        }
        auto readBack = StateSerDes::Read(snapshot.registry->GlobalRank(), snapshot.registry->GlobalWorldSize(), sstrm);

        // Walk the original tree and compare every node against the read-back tree.
        // Nodes are stored at stable indices, so the same Handle addresses the same node in both trees.
        // StateNode::Variant supports operator== for all alternatives (Primitives64Bit included).
        bool mismatch = false;
        snapshot.registry->WalkNodes([&](StateNode const& origNode) {
            auto const& readBackNode = readBack->GetNode(origNode.handle);
            if (origNode.data != readBackNode.data)
            {
                fmt::print(stderr,
                           "[native_ckpt:svr][{}][ERROR] Node '{}' mismatch after roundtrip\n\tOriginal: {}\n\tReadBack: {}\n",
                           snapshot.registry->GlobalRank(),
                           snapshot.registry->GetNodeId(origNode.handle),
                           origNode.data,
                           readBackNode.data);
                mismatch = true;
            }
        });

        if (mismatch)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Metadata roundtrip verification failed for {}",
                                                 snapshot.registry->GlobalRank(),
                                                 ckptPath.string()));
        }
        fmt::print(stderr,
                   "[native_ckpt:svr][{}][INFO] Metadata roundtrip verification passed for: {}\n",
                   snapshot.registry->GlobalRank(),
                   ckptPath.string());
#endif
    }
    for (auto const& [localStateRank, localState] : snapshot.localStates)
    {
        DebugDumpLocalStateTreeKVP(workDir / fmt::format("cpp_metadata_{}.txt", localStateRank), *localState->localRegistry);
    }
    DebugDumpLocalStateTreeKVP(workDir / fmt::format("cpp_metadata.txt"), *sessionState.registry);
    fmt::print(stderr, "[native_ckpt:svr] Accumulated all localState for global State: Dumped states in {}\n", (PerfClock::now() - start1));
}

void Session::Activate()
{
    start = PerfClock::now();
    // noop
}

void Session::Handle(cipc::Message::LocalStateRegistry const& hdr, std::array<char, cipc::D2HBufferSize> const& buf)
{
    BufStream istrm{buf.data(), hdr.messageSize};
    auto      localRegistry  = StateSerDes::Read(istrm);
    Rank      rank           = localRegistry->Topo().rank;
    auto&     localState     = FindOrCreateLocalState_(rank);
    localState.localRegistry = std::move(localRegistry);
    // Reset tensor handles to be offsets in the localState blob
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
               "[native_ckpt:svr][{}][{}] Received sharded state registry data of size {}, Total Tensor Size: {}\n",
               sessionId,
               rank,
               hdr.messageSize,
               offset);
    asyncTasks.emplace_back(
        std::async(std::launch::async | std::launch::deferred, &Session::OnRecieveLocalStateRegistryAsync_, this, rank));

    if (rank != Rank{0})    // TODO_MULTILOCAL_CKPT
    {
        asyncTasks.push_back(std::async(
            std::launch::async | std::launch::deferred, &Session::SendLocalShardedStateRegistryToPeer_, this, rank, topo.peers[0]));
    }
}

void Session::Handle(cipc::Message::LocalStateData const& hdr1, std::array<char, cipc::D2HBufferSize> const& buf)
{
    void* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto& snapshot   = *snapshots[hdr1.snapshotId];
        auto& localState = *snapshot.localStates.at(hdr1.globalRank);
        auto  endOffset  = hdr1.offset + hdr1.messageSize;
        if (endOffset > localState.blob.size)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][ERROR] Received tensor data exceeds allocated size({}) for session id {} offset = {} size = {}",
                localState.blob.size,
                hdr1.sessionId,
                hdr1.offset,
                hdr1.messageSize));
        }

        localState.writtenTensorSize += hdr1.messageSize;
        localState.writtenTensorRangeBegin = std::min(localState.writtenTensorRangeBegin, static_cast<uint64_t>(hdr1.offset));
        localState.writtenTensorRangeEnd   = std::max(localState.writtenTensorRangeEnd, endOffset);
        if (localState.writtenTensorSize > localState.blob.size)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][ERROR] Received tensor data size {} (range {}-{}) exceeds total tensor size({}) for session id {}",
                localState.writtenTensorSize,
                localState.writtenTensorRangeBegin,
                localState.writtenTensorRangeEnd,
                localState.blob.size,
                hdr1.sessionId));
        }

        ptr = localState.blob[hdr1.offset];
    }
    std::memcpy(ptr, buf.data(), hdr1.messageSize);
}

void Session::Handle(cipc::Message::ActivateSnapshot const& hdr, std::array<char, cipc::D2HBufferSize> const& buf)
{
    BufStream bufStream{buf.data(), buf.size()};
    auto      ckptPath = std::filesystem::path(serdes::ReadBlobWithSize<uint16_t>(bufStream));
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto                        snapshotIt = snapshots.find(hdr.snapshotId);
        if (snapshotIt == snapshots.end())
        {
            auto snapshot        = std::make_unique<Snapshot>();
            snapshot->snapshotId = hdr.snapshotId;
            snapshot->name       = fmt::format("snapshot_{}", hdr.snapshotId);
            snapshot->path       = ckptPath / fmt::format("checkpoint_session_{}.snapshot_{}.bin", sessionId, hdr.snapshotId);
            snapshotIt           = snapshots.insert({hdr.snapshotId, std::move(snapshot)}).first;
        }

        if (snapshotIt->second->localStates.contains(hdr.globalRank))
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}][ERROR] Snapshot {} for global rank {} already has local state when activating snapshot {}",
                hdr.globalRank,
                hdr.snapshotId,
                hdr.globalRank,
                hdr.snapshotId));
        }

        auto it = sessionState.localStates.find(hdr.globalRank);
        if (it == sessionState.localStates.end())
        {
            throw std::runtime_error(
                fmt::format("[native_ckpt:svr][{}][ERROR] No local state found for global rank {} when activating snapshot {}",
                            hdr.globalRank,
                            hdr.globalRank,
                            hdr.snapshotId));
        }
        auto localState           = std::make_unique<LocalState>();
        localState->localRegistry = it->second->localRegistry;

        if (hdr.snapshotSize != it->second->localRegistry->GetTotalTensorStateSizeInBytes())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Snapshot size {} does not match expected total tensor "
                                                 "state size {} from local registry for global rank {} when activating snapshot {}",
                                                 hdr.globalRank,
                                                 hdr.snapshotSize,
                                                 it->second->localRegistry->GetTotalTensorStateSizeInBytes(),
                                                 hdr.globalRank,
                                                 hdr.snapshotId));
        }
        localState->blob                                = Blob(hdr.snapshotSize);
        snapshotIt->second->localStates[hdr.globalRank] = std::move(localState);
    }

    fmt::print(stderr,
               "[native_ckpt:svr][{}] Activated snapshot {} for global rank {} with expected tensor data size {} path = {}\n",
               hdr.sessionId,
               hdr.snapshotId,
               hdr.globalRank,
               hdr.snapshotSize,
               ckptPath.string());
}

void Session::Handle(cipc::Message::FinalizeSnapshot const& hdr, std::array<char, cipc::D2HBufferSize> const& /* buf */)
{
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto& snapshot   = *snapshots[hdr.snapshotId];
        auto& localState = *snapshot.localStates[hdr.globalRank];

        if (localState.writtenTensorSize != localState.blob.size)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][ERROR] Incomplete tensor data received. sessionid={}. rcvd-size={}(range{} - {}) expected={} ",
                hdr.sessionId,
                localState.writtenTensorSize,
                localState.writtenTensorRangeBegin,
                localState.writtenTensorRangeEnd,
                localState.blob.size));
        }
        if (localState.writtenTensorRangeBegin != 0 || localState.writtenTensorRangeEnd != localState.blob.size)
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Tensor data received has invalid range. sessionid={}. "
                                                 "rcvd-size={}(range{} - {}) expected range=0 - {}",
                                                 hdr.sessionId,
                                                 localState.writtenTensorSize,
                                                 localState.writtenTensorRangeBegin,
                                                 localState.writtenTensorRangeEnd,
                                                 localState.blob.size));
        }
        auto it = snapshots.find(hdr.snapshotId);
        if (it == snapshots.end())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Snapshot with id {} does not exist for session id {}",
                                                 hdr.globalRank,
                                                 hdr.snapshotId,
                                                 hdr.sessionId));
        }
        if (sessionState.registry == nullptr)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}][ERROR] Global state registry is not initialized when finalizing snapshot {} for session id {}",
                hdr.globalRank,
                hdr.snapshotId,
                hdr.sessionId));
        }
        if (it->second->registry == nullptr) { it->second->registry = sessionState.registry; }
        snapshot.finalizedRanks.insert(hdr.globalRank);
    }
    // TODO_SAVETENSORS : Write out the rank specific checkpoint file
    // WriteTensorsToFile(snapshot, snapshot.path / "tensors.bin");

    asyncTasks.emplace_back(std::async(std::launch::async | std::launch::deferred, &Session::OnFinalizeSnapshotAsync_, this, hdr));
}

void Session::HandleStateRegistryFragment(Rank recvRank, std::vector<TensorShardInfo> const& tensorShards)
{
    std::lock_guard<std::mutex> lock(mutex);
    MergeFragmentToGlobalState_(lock, tensorShards);
    recievedFrom.insert(recvRank);
}

void Session::Handle(cipc::Message::FinalizeSession const& /* hdr */, std::array<char, cipc::D2HBufferSize> const& /* buf */)
{}

struct Server
{
    static constexpr uint16_t Port = cipc::Message::Port;

    public:
    Server() = default;
    ~Server() { Stop(); }
    CLASS_DELETE_COPY_AND_MOVE(Server);

    void Start()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopRequested = false;
        if (_running) { return; }
        _ioc.restart();

        StartTcpListener_();
        _running = true;

        // Start server thread
        //_serverThread = std::thread([this]() {
        for (uint8_t i = 0; i < cipc::MaxRanksPerNode; ++i)
        {
            _threads.emplace_back([this, i]() {
                SetThreadName(fmt::format("ncs:svrq:{}", i).c_str());
                auto msgQueue = cipc::MsgQueue::Create(cipc::Message::RequestChannelName(i));
                {
                    std::lock_guard<std::mutex> lock1(_mutex);
                    _msgQueue.insert(&msgQueue);
                }
                fmt::print(stderr, "[native_ckpt:svr] Checkpoint server started. Waiting for messages...\n");
                std::fflush(stderr);
                while (true) try
                    {
                        auto const [stop, hdr, buf] = msgQueue.BeginRead();
                        if (stop)
                        {
                            std::lock_guard<std::mutex> lock1(_mutex);
                            fmt::print(stderr, "[native_ckpt:svr] Stop requested. Shutting down server thread.\n");
                            _msgQueue.erase(&msgQueue);
                            return;
                        }
                        HandleMessage(hdr, buf);
                        msgQueue.EndRead();

                    } catch (std::exception const& e)
                    {
                        fmt::print(stderr, "[native_ckpt:svr][ERROR] Exception: {}\n", e.what());
                        // Stopping could cause clients to hang if they are waiting for a response.
                        // _running = false;
                        // _msgQueue.erase(std::remove(_msgQueue.begin(), _msgQueue.end(), &msgQueue), _msgQueue.end());    // NOLINT
                        // throw;
                    }
            });
        }
    }

    void WaitForShutdown()
    {
#ifdef PHOENIX_BUILD_FOR_VALGRIND
        static constexpr auto AutoShutdownTimeout = 600s;
#else
        static constexpr auto AutoShutdownTimeout = 150s;
#endif
        while (true)
        {
            if (_stopRequested || (_lastPingTime + AutoShutdownTimeout) < PerfClock::now()) { break; }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        Stop();
    }

    void Ping()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _lastPingTime = PerfClock::now();
    }

    void HandleMessage(cipc::Message::Header const& hdr, std::array<char, cipc::D2HBufferSize> const& buf)
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
            std::lock_guard<std::mutex> lock(_mutex);
            _lastPingTime = PerfClock::now();

            auto      hdr1 = std::get<cipc::Message::ActivateSession>(hdr);
            BufStream bufStream{buf.data(), buf.size()};
            auto      workDirStr    = serdes::ReadBlobWithSize<uint16_t>(bufStream);
            auto      workDir       = workDirStr.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(workDirStr);
            auto      createSession = [&]() {
                auto topo    = StateSerDes::ReadTopology(bufStream);
                auto session = std::make_unique<Session>(this, hdr1.sessionId, std::move(workDir), std::move(topo));
                session->Activate();
                return session;
            };

            fmt::print(stderr, "[native_ckpt:svr] ActivateSession {} {}\n", hdr1.sessionId, workDir);
            auto it = _activeSessions.find(hdr1.sessionId);
            it      = it != _activeSessions.end() ? it : _activeSessions.insert({hdr1.sessionId, createSession()}).first;
        }
        break;
        case cipc::Message::Type::FinalizeSession:
        {
            auto const& hdr1 = std::get<cipc::Message::FinalizeSession>(hdr);
            fmt::print(stderr, "[native_ckpt:svr] FinalizeSession {}\n", hdr1.sessionId);
            SessionById_(hdr1.sessionId).Handle(hdr1, buf);
        }
        break;
        case cipc::Message::Type::LocalStateRegistry:
        {
            auto const& hdr1    = std::get<cipc::Message::LocalStateRegistry>(hdr);
            auto&       session = SessionById_(hdr1.sessionId);
            session.Handle(hdr1, buf);
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

    void Stop()
    {
        bool wasRunning = false;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            wasRunning     = _running;
            _stopRequested = true;
            for (auto const& msgQueue : _msgQueue)
            {
                if (msgQueue != nullptr) { msgQueue->Stop(); }
            }
        }
        if (!wasRunning) { return; }

        boost::system::error_code ec;
        if (_tcpAcceptor.close(ec)) { fmt::print(stderr, "[native_ckpt:svr][TCP][WARN] Failed to close acceptor: {}\n", ec.message()); }
        _ioc.stop();
        for (auto& t : _threads)
        {
            if (t.joinable()) { t.join(); }
        }

        std::lock_guard<std::mutex> lock(_mutex);
        _running = false;
    }

    void QuerySessions(ServerSessionCallback callback, void* cbData)
    {
        std::lock_guard<std::mutex> lock(_mutex);
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
                             *snapshot->registry,
                             *localState->localRegistry,
                             localState->blob);
                }
            }
        }
    }

    private:
    Session& SessionById_(cipc::SessionId sessionId)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _lastPingTime = PerfClock::now();
        auto it       = _activeSessions.find(sessionId);
        if (it == _activeSessions.end())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Session id {} not found", sessionId));
        }
        return *it->second;
    }

    static void RecieveStateRegistryFragments_(Session&                                                             session,
                                               cipc::Rank                                                           rank,
                                               tcp_stream&                                                          stream,
                                               boost::beast::flat_buffer&                                           buffer,
                                               boost::beast::http::request_parser<boost::beast::http::buffer_body>& parser)
    {
        fmt::print(stderr, "[native_ckpt:svr][{}][{}] Receiving state registry chunks\n", session.sessionId, rank);

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
            std::span<char> dest{reinterpret_cast<char*>(&value), sizeof(value)};
            return readSpan(dest, allowDone) == 0;
        };
        auto readBlob = [&](char* dest, size_t size, bool allowDone = false) -> bool {
            std::span<char> destSpan{dest, size};
            return readSpan(destSpan, allowDone) == 0;
        };

        std::vector<TensorShardInfo> tensorShards;

        // Read body in chunks; Beast transparently decodes chunked transfer encoding
        while (true)
        {
            TensorShardInfo info;
            uint32_t        nodeIdLen{};

            if (readTrivial(info.rank, true))
            {
                break;    // Clean end-of-stream at a record boundary
            }
            readTrivial(info.tensorInfo);
            readTrivial(nodeIdLen);
            info.nodeId = std::string(nodeIdLen, '\0');
            readBlob(info.nodeId.data(), nodeIdLen);
            info.tensorInfo.AssertShardIsValid();
            tensorShards.push_back(std::move(info));
        }

        session.HandleStateRegistryFragment(rank, tensorShards);
        fmt::print(stderr,
                   "[native_ckpt:svr][{}][{}] Successfully received and processed {} tensor shards\n",
                   session.sessionId,
                   rank,
                   tensorShards.size());
    }

    void HandleRequest_(tcp_stream& stream)
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
            RecieveStateRegistryFragments_(SessionById_(sessionId), rank, stream, buffer, bufferParser);
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

    void StartTcpListener_()
    {
        using boost::asio::ip::tcp;
        auto const    address = boost::asio::ip::make_address("0.0.0.0");
        tcp::endpoint endpoint(address, Port);
        // use synchronous acceptor to simplify shutdown logic (no need to cancel async accept)
        _tcpAcceptor.open(endpoint.protocol());
        _tcpAcceptor.set_option(boost::asio::socket_base::reuse_address(true));
        boost::system::error_code ec;
        ec = _tcpAcceptor.bind(endpoint, ec);
        if (ec) { throw cipc::ErrorAlreadyRunning(); }
        _tcpAcceptor.listen(boost::asio::socket_base::max_listen_connections);

        boost::asio::co_spawn(_ioc, DoListen_(), [](std::exception_ptr const& e) {
            if (e) try
                {
                    std::rethrow_exception(e);
                } catch (boost::system::system_error& e)
                {
                    if (e.code() != boost::asio::error::operation_aborted) { throw; }
                } catch (std::exception& e) { fmt::print(stderr, "Error in acceptor: {}\n", e.what()); }
        });

        for (size_t i = 0; i < 4; i++)
        {
            _threads.emplace_back([this, i]() {
                SetThreadName(fmt::format("ncs:ios:{}", i).c_str());
                _ioc.run();
            });
        }
    }

    boost::asio::awaitable<void> DoListen_()
    {
        static constexpr auto Timeout = std::chrono::seconds{30};

        while (true)
        {
            tcp_stream stream(co_await _tcpAcceptor.async_accept());
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
        }
    }

    private:
    std::mutex _mutex;

    PerfClock::time_point               _lastPingTime{PerfClock::now()};
    std::atomic<bool>                   _stopRequested{false};
    std::unordered_set<cipc::MsgQueue*> _msgQueue{nullptr};
    bool                                _running{false};
    std::vector<std::thread>            _threads;

    boost::asio::io_context        _ioc;
    boost::asio::ip::tcp::acceptor _tcpAcceptor{_ioc};

    std::unordered_map<cipc::SessionId, std::unique_ptr<Session>> _activeSessions;
    friend struct Session;
};

SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
static std::unique_ptr<Server> ServerSingleton;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
SUPPRESS_WARNINGS_END

void RunInprocServer()
{
    ServerSingleton = std::make_unique<Server>();
    SetThreadName("ncs:main");
    ServerSingleton->Start();
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
    if (ServerSingleton)
    {
        ServerSingleton->Stop();
        ServerSingleton.reset();
    }
}

void QueryServerSessions(ServerSessionCallback callback, void* cbData)
{
    if (!ServerSingleton) { return; }
    ServerSingleton->QuerySessions(callback, cbData);
}
