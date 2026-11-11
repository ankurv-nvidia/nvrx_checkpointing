#include "session.h"
#include "logging.h"
#include "net_utils.h"
#include "SetThreadName.h"
#include "server.h"
#include "stateserdes.h"
#include "telemetry.h"

#include <boost/asio/io_context.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/counter.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/stream.hpp>
#include <utility>

namespace cipc  = checkpoint::ipc;
namespace tel   = checkpoint::telemetry;
using PerfClock = std::chrono::system_clock;
using BufStream = boost::iostreams::stream<boost::iostreams::array_source>;

template <> struct fmt::formatter<Session> : fmt::formatter<std::string_view>
{
    auto format(Session const& session, fmt::format_context& ctx) const    // NOLINT
    {
        uint32_t ptrVal = reinterpret_cast<uintptr_t>(session.server) % std::numeric_limits<uint32_t>::max();    // NOLINT

        return fmt::format_to(ctx.out(), "{:x}#{}", ptrVal, session.sessionId);
    }
};

static constexpr int              MaxRetries                     = 20;
static constexpr auto             IntervalBetweenRetries         = std::chrono::milliseconds(100);
static constexpr std::string_view RegistryContentEncoding        = "deflate";

template <typename TStream> static void TryCleanShutdown(TStream& stream)
{
    boost::system::error_code ec;
    ec = stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    if (ec && ec != boost::system::errc::not_connected)
    {    //
        ERR("Shutdown Error: {}", ec.what());
    }
}

static void SendToPeerWithRetries(Session const&          session,
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
                WARN("[{}] Attempt {} peer={} failed target={} {}", session, attempt, peer, target, ex.what());
            }
            else
            {
                ERR("[{}] Attempt {} peer={} failed target={} {}", session, attempt, peer, target, ex.what());
            }
            if (attempt < MaxRetries) { std::this_thread::sleep_for(IntervalBetweenRetries); }
        }
    }
    throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Failed to notify peer={} target={} after {} attempts: {}",
                                         session,
                                         peer,
                                         target,
                                         MaxRetries,
                                         lastError));
}

void Session::DebugOutputToFile(std::string_view fileName, std::function<void(std::ostream&)> const& write)
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

static void
WriteGlobalStateTreeToFile(std::ostream& ostrm, Session const& session, Session::Snapshot const& snapshot)
{
    auto start1 = PerfClock::now();
    StateSerDes::Write(ostrm, *snapshot.globalRegistry, StateNode::Handle::Root);
    INFO("[{}] Accumulated all localState for global State: Dumped states in {}", session, (PerfClock::now() - start1));
}

static void ValidateGlobalStateTree(Session const& session, Session::Snapshot const& snapshot)
{
    snapshot.globalRegistry->WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::Tensor)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}][ERROR] DebugWriteMetdataKeyValuePairs: Unexpected TensorShard node in activeGlobalState",
                session,
                snapshot.globalRegistry->GetNodeId(node.handle)));
        }
        if (type == StateNode::Type::DTensor)
        {
            auto const& tnsr = std::get<DTensor>(node.data);
            for (auto const& shard : tnsr.shards) { shard.AssertShardIsValid(tnsr.fulldim); }
            if (tnsr.shards.empty())
            {
                WARN("[{}] empty sharded tensor='{}' in metadata output",
                     session,
                     snapshot.globalRegistry->GetNodeId(node.handle));
                return;
            }
            snapshot.globalRegistry->AssertShardedTensorHasNoHoles(node.handle);
        }
    });
}

// Build a separate storage registry with one deterministic owner for every
// logical tensor shard. The merged source registry remains unchanged. Replica
// identity comes exclusively from normalized server metadata (FQN + dtype +
// full shape + global shard range).
// TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
// This information should be used to correctly identify and deduplicate tensor replicas and identify the tensors needing snapshots
static TensorReplicaDeduplicationResult DeduplicateTensorReplicas(StateTree const& registry, uint32_t worldSize)
{
    if (worldSize == 0) { throw std::invalid_argument("Cannot deduplicate tensor replicas with worldSize=0"); }

    auto dedupedRegistry = registry.Clone();

    struct LogicalShard
    {
        StateNode::Handle           nodeHandle{StateNode::Handle::Invalid};
        std::string                 fqn;
        Tensor::DimRange            sharddim;
        uint64_t                    sizeBytes{};
        std::vector<DTensor::Shard> candidates;
    };

    std::vector<LogicalShard> logicalShards;
    registry.WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::DTensor) { return; }
        auto const& tnsr = std::get<DTensor>(node.data);

        std::map<Tensor::DimRange, std::vector<DTensor::Shard>> candidatesByRange;
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
            logicalShards.push_back({
                .nodeHandle = node.handle,
                .fqn        = fqn,
                .sharddim   = sharddim,
                .sizeBytes  = candidates.front().StorageSizeBytes(),
                .candidates = std::move(candidates),
            });
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
    std::map<StateNode::Handle, std::vector<DTensor::Shard>> selectedByNode;

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
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::DTensor) { return; }
        auto& tnsr = std::get<DTensor>(node.data);
        if (tnsr.shards.empty()) { return; }

        DTensor selected(tnsr.durableDtype, tnsr.fulldim);
        for (auto const& shard : selectedByNode.at(node.handle)) { selected.AddShard(shard); }
        tnsr = std::move(selected);
    });

    return {.dedupedRegistry = std::move(dedupedRegistry), .stats = std::move(stats)};
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

void Session::RelayWriteCkptCompleteToPeers(cipc::SnapshotId        snapshotId,
                                           std::string_view const& failure,
                                           size_t                  startPeerIndex,
                                           size_t                  endPeerIndex) const
{
    if (startPeerIndex >= endPeerIndex) { return; }
    auto        delegateStart = (startPeerIndex + endPeerIndex) / 2;
    auto const& peer          = topo.peers[delegateStart];
    auto const  postTarget
        = fmt::format("/api/peer/writeckptcomplete/{}/0/{}?fwd={}:{}", sessionId, snapshotId, delegateStart, endPeerIndex);
    SendToPeerWithRetries(*this, peer, failure, postTarget, {});
    RelayWriteCkptCompleteToPeers( snapshotId, failure, startPeerIndex, delegateStart);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Session::AsyncWorkFinalizeSnapshot_(cipc::Message::FinalizeSnapshot const& hdr) noexcept
{
    // Ranks on the coordinator peer (peers[0]) record their finalize/write-complete
    // locally; only ranks on other peers forward those notifications to the coordinator.
    bool const onCoordinatorPeer = static_cast<uint32_t>(hdr.globalRank) / topo.ranksPerPeer == 0;
    bool const notifyCoordinator = !onCoordinatorPeer;

    Snapshot*   snapshotPtr   = nullptr;
    LocalState* localStatePtr = nullptr;
    {
        std::scoped_lock lock(mutex);
        snapshotPtr   = snapshots.at(hdr.snapshotId).get();
        localStatePtr = snapshotPtr->localStates.at(hdr.globalRank).get();
    }
    auto& snapshot   = *snapshotPtr;
    auto& localState = *localStatePtr;

    auto const startedAt                   = localState.startedAt;
    auto const numTensors                  = localState.localRegistry->GetTotalNumTensors();
    auto const totalTensorStateSizeInBytes = localState.localRegistry->GetTotalTensorStateSizeInBytes();

    auto const localNodeWorldSize = onCoordinatorPeer ? static_cast<uint32_t>(topo.worldSize) : topo.ranksPerPeer;
    try
    {
        while (true)
        {
            {
                std::scoped_lock lock(mutex);
                if (!snapshot.failures.empty()) { throw std::runtime_error("Aborting Wait for local-state-registry"); }
                if (receivedFrom.size() >= localNodeWorldSize && receivedFrom.contains(hdr.globalRank))
                {
                    if (onCoordinatorPeer && snapshotPtr->globalRegistry == nullptr)
                    {
                        snapshotPtr->globalRegistry = sessionState.globalRegistry;
                    }
                    break;
                }
                INFO(
                    "[{}] FinalizeSnapshot waiting for snapshot={} local-state-registry {}",
                    *this,
                    hdr.snapshotId,
                    WaitingForRanks{receivedFrom,
                                    hdr.globalRank,
                                    static_cast<Rank>(static_cast<uint32_t>(hdr.globalRank) + static_cast<uint32_t>(receivedFrom.size()))});
            }
            if (stopRequested) { throw std::runtime_error("Stop requested while waiting for ranks to send local state registry"); }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (notifyCoordinator)
        {
            SendToPeerWithRetries(*this,
                                  topo.peers[0],
                                  {},
                                  fmt::format("/api/peer/snapshotcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                  {});
        }

        // Rank 0 dumps and validates the merged registry once every rank has finalized.
        // TODO_MULTILOCAL_CKPT : For multinode local checkpointing this needs to happen once on each node so designate one rank
        if (hdr.globalRank == Rank{0})
        {
            while (true)
            {
                {
                    std::scoped_lock lock(mutex);
                    if (!snapshot.failures.empty()) { throw std::runtime_error("Aborting Wait for finalize-snapshots"); }
                    if (snapshot.finalizedRanks.size() >= static_cast<size_t>(topo.worldSize)) { break; }
                    INFO("[{}] FinalizeSnapshot waiting for snapshot={} finalized ranks {} of {} remaining... Finalized ranks=[{}]",
                         *this,
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

            PERF() << tel::span::SvrWriteMetadata::Begin{.id
                                                         = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId}};
            DebugOutputToFile("registry.global.bin",
                         [&](std::ostream& ostrm) { WriteGlobalStateTreeToFile(ostrm, *this, snapshot); });
            ValidateGlobalStateTree(*this, snapshot);
            PERF() << tel::span::SvrWriteMetadata::End{.id
                                                       = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId}};
            BroadcastGlobalRegistryToPeers_(*globalRegistry);

            auto dedupResult = DeduplicateTensorReplicas(*globalRegistry, static_cast<uint32_t>(topo.worldSize));
            INFO("[{}] Selected owners for {} logical tensor shards; removed {} replicas; assigned bytes by rank=[{}]",
                 *this,
                 dedupResult.stats.logicalShards,
                 dedupResult.stats.replicasRemoved,
                 fmt::join(dedupResult.stats.assignedBytesPerRank, ","));
            {
                std::scoped_lock lock(mutex);
                sessionState.dedupedGlobalRegistry = snapshot.dedupedGlobalRegistry = std::move(dedupResult.dedupedRegistry);
            }
        }

        std::shared_ptr<StateTree> dedupedGlobalRegistry;
        while (true)
        {
            SetThreadName("ncs:finwt");
            {
                std::scoped_lock lock(mutex);
                if (!snapshot.failures.empty()) { throw std::runtime_error("Aborting Wait for deduped global registry"); }
                dedupedGlobalRegistry = sessionState.dedupedGlobalRegistry;
                if (dedupedGlobalRegistry != nullptr) { break; }
                INFO("[{}] FinalizeSnapshot waiting for global registry rank={}...", *this, hdr.globalRank);
                snapshot.dedupedGlobalRegistry = sessionState.dedupedGlobalRegistry;
            }
            if (stopRequested) { throw std::runtime_error("Stop requested while waiting for global registry"); }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        localState.blobs.clear();
        localState.localRegistry->WalkNodes([&](StateNode const& node) {
            if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
            auto        fqn                  = localState.localRegistry->GetNodeId(node.handle);
            auto const& blob                 = std::get<StateNode::Blob>(node.data);
            localState.blobs[std::move(fqn)] = std::string_view(blob.data(), blob.size());
        });
        if (hdr.globalRank == Rank{0})
        {
            std::scoped_lock lock(mutex);
            snapshot.blobMetadata = blobMetas;
        }
        checkpoint::distcp::WriteRankShard(
            snapshot.path.parent_path(), hdr.globalRank, *dedupedGlobalRegistry, localState.tensorStorage, localState.blobs);
        if (notifyCoordinator)
        {
            SendToPeerWithRetries(*this,
                                  topo.peers[0],
                                  {},
                                  fmt::format("/api/peer/writeckptcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                  {});
        }
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
                    if (!snapshot.failures.empty()) { throw std::runtime_error("Aborting Wait for snapshot write completion"); }
                    if (snapshot.writeCkptCompletedRanks.size() >= otherRanks) { break; }
                    INFO("[{}] FinalizeSnapshot waiting for snapshot={} write-completed ranks {} of {} remaining...",
                         *this,
                         hdr.snapshotId,
                         otherRanks - snapshot.writeCkptCompletedRanks.size(),
                         otherRanks);
                }
                if (stopRequested) { throw std::runtime_error("Stop requested while waiting for ranks to write checkpoint"); }

                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            try
            {
                checkpoint::distcp::WriteMetadata(snapshot.path.parent_path(), *dedupedGlobalRegistry, snapshot.blobMetadata);
            } catch (std::exception const& ex)
            {
                throw std::runtime_error(fmt::format("[{}] WriteMetadata failed for snapshot={} rank={}: {}",
                                                     *this,
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
                       "[native_ckpt:svr][{}][PERF] Snapshot={} Durable ranks={} | Time(us): activate_to_durable_us={}\n",
                       *this,
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
        ERR("[{}] Snapshot finalization failed for snapshot={} rank={}: {}",
            *this,
            hdr.snapshotId,
            hdr.globalRank,
            ex.what());
        if (hdr.globalRank == Rank{0})
        {
            BroadcastWriteCkptCompleteToPeers_(snapshot);
            std::scoped_lock lock(mutex);
            snapshot.complete = true;
        }
        else if (notifyCoordinator && !snapshot.complete)
        {
            SendToPeerWithRetries(*this,
                                  topo.peers[0],
                                  failure,
                                  fmt::format("/api/peer/writeckptcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                  {});
            // snapshot.complete = true; we set complete when we hear back from the coordinator
        }
    }

    std::string failure;
    {
        std::scoped_lock lock(mutex);
        if (!snapshot.failures.empty()) { failure = snapshot.failures.front(); }
    }
    INFO("[{}] OnFinalizeSnapshotAsync_ for rank={}, snapshotId={} failure={}",
         *this,
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
                    std::ranges::count_if(snapshots, [](auto const& entry) { return !entry.second->localStates.empty(); }));
                for (auto const& [id, candidate] : snapshots)
                {
                    if (remainingSnapshots <= numActiveSnapshots) { break; }
                    if (candidate->localStates.empty()) { continue; }
                    // Registry broadcast and local writes finish before their buffers are released.
                    if (!candidate->failures.empty() || candidate->finalizedRanks.size() < ExpectedWriteCompletions()
                        || candidate->writeCkptCompletedRanks.size() < ExpectedWriteCompletions())
                    {
                        break;
                    }
                    auto& released = *evicted.emplace_back(std::make_unique<Snapshot>());
                    released.localStates.swap(candidate->localStates);
                    released.blobMetadata.swap(candidate->blobMetadata);
                    released.globalRegistry.swap(candidate->globalRegistry);
                    released.dedupedGlobalRegistry.swap(candidate->dedupedGlobalRegistry);
                    --remainingSnapshots;
                }
            }
            INFO("[{}] Released {} snapshot buffers after finalizing snapshot={} rank={}. Remaining snapshots with buffers={} "
                 "numActiveSnapshots={}",
                 *this,
                 evicted.size(),
                 hdr.snapshotId,
                 hdr.globalRank,
                 remainingSnapshots,
                 numActiveSnapshots);
        }
    } catch (std::exception const& ex)
    {
        ERR("[{}] Exception occurred while releasing snapshot buffers: {}", *this, ex.what());
    }
}

void Session::MergeFragmentToGlobalState_(std::scoped_lock<std::mutex> const& /* lock */, std::vector<TensorShardInfo> const& tensorShards)
{
    INFO("[{}] MergeFragmentToGlobalState_ shards={}", *this, tensorShards.size());

    if (sessionState.globalRegistry == nullptr)
    {
        throw std::logic_error("Cannot merge state registry fragments. Global state registry is not initialized");
    }

    for (auto const& tensorShard : tensorShards)
    {
        // tensorShard.tensorInfo.AssertShardIsValid();
        //  Look up the node by its string NodeId
        auto it = tensorMap.find(tensorShard.nodeId);
        if (it == tensorMap.end())
        {
            auto const& stateName        = tensorShard.nodeId;
            auto [childName, parentNode] = sessionState.globalRegistry->FindOrAddNodeByFQN(stateName);
            // Add the final node which contains the handler
            auto& child = sessionState.globalRegistry->AddChild(parentNode, childName, true);
            auto  shTen = DTensor(tensorShard.durableDtype, tensorShard.fulldim);
            shTen.AddShard(tensorShard.tensorInfo);
            child.data                    = shTen;
            tensorMap[tensorShard.nodeId] = child.handle;
            continue;
        }
        auto  nodeHandle = it->second;
        auto& node       = sessionState.globalRegistry->GetNode(nodeHandle);
        if (node.data.index() != static_cast<size_t>(StateNode::Type::DTensor))
        {
            throw std::runtime_error(fmt::format("Received tensor shard for non-sharded tensor='{}' expected-type={} != actual-type={}",
                                                 tensorShard.nodeId,
                                                 static_cast<size_t>(StateNode::Type::DTensor),
                                                 node.data.index()));
        }
        auto& tnsr = std::get<DTensor>(node.data);
        if (tnsr.shards.empty())
        {
            throw std::runtime_error(fmt::format("Sharded tensor={} has no shards in metadata output\n", tensorShard.nodeId));
        }

        if (tnsr.durableDtype != tensorShard.durableDtype || tnsr.fulldim != tensorShard.fulldim)
        {
            throw std::runtime_error(fmt::format("Received incompatible replicas for tensor='{}': expected dtype={} full-shape=[{}], got "
                                                 "dtype={} full-shape=[{}] on rank={}",
                                                 tensorShard.nodeId,
                                                 tnsr.durableDtype,
                                                 tnsr.fulldim,
                                                 tensorShard.durableDtype,
                                                 tensorShard.fulldim,
                                                 tensorShard.tensorInfo.rank));
        }
        tnsr.AddShard(tensorShard.tensorInfo);
    }
}

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
            if (attempt < MaxRetries)
            {
                WARN("[{}] Attempt {} SendLocalShardedStateRegistryToPeer_ rank={} peer={} failed sending shards: {}",
                     *this,
                     attempt,
                     rank,
                     peer,
                     ex.what());
            }
            else
            {
                ERR("[{}] Attempt {} SendLocalShardedStateRegistryToPeer_ rank={} peer={} failed sending shards: {}",
                    *this,
                    attempt,
                    rank,
                    peer,
                    ex.what());
            }
            std::this_thread::sleep_for(IntervalBetweenRetries);
        }
    }
}

void Session::BroadcastWriteCkptCompleteToPeers_(Snapshot const& snapshot) const
{
    if (topo.peers.size() <= 1) { return; }
    std::string failure;
    {
        std::scoped_lock lock(mutex);
        failure = fmt::format("{}", fmt::join(snapshot.failures, ","));
    }
    RelayWriteCkptCompleteToPeers( snapshot.snapshotId, failure, 1, topo.peers.size());

    INFO("[{}] Broadcast global registry to {} peer(s)", *this, topo.peers.size() - 1);
}

void Session::BroadcastGlobalRegistryToPeers_(StateTree const& globalRegistry) const
{
    if (topo.peers.size() <= 1) { return; }
    uint64_t          rawSize = 0;
    std::string const body    = DeflateRegistry(globalRegistry, rawSize);

    INFO("[{}] Global registry {} bytes -> {} deflated ({:.1f}x)",
         *this,
         rawSize,
         body.size(),
         static_cast<double>(rawSize) / static_cast<double>(body.size()));

    RelayGlobalRegistryToPeers(body, 1, topo.peers.size());

    INFO("[{}] Broadcast global registry to {} peer(s)", *this, topo.peers.size() - 1);
}

void Session::RelayGlobalRegistryToPeers(std::string_view const& registry, size_t startPeerIndex, size_t endPeerIndex) const
{
    if (startPeerIndex >= endPeerIndex) { return; }
    auto        delegateStart = (startPeerIndex + endPeerIndex) / 2;
    auto const& peer          = topo.peers[delegateStart];
    auto const  postTarget    = fmt::format("/api/peer/globalregistry/{}?fwd={}:{}", sessionId, delegateStart, endPeerIndex);
    SendToPeerWithRetries(*this, peer, registry, postTarget, RegistryContentEncoding);
    RelayGlobalRegistryToPeers(registry, startPeerIndex, delegateStart);
}

void Session::HandleGlobalRegistry(std::shared_ptr<StateTree> globalRegistry)
{
    auto dedupResult = DeduplicateTensorReplicas(*globalRegistry, static_cast<uint32_t>(globalRegistry->GlobalWorldSize()));
    std::scoped_lock lock(mutex);
    sessionState.globalRegistry = std::move(globalRegistry);
    sessionState.dedupedGlobalRegistry = std::move(dedupResult.dedupedRegistry);
}

Session::Session(Server* serverIn LFTBND, cipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn) :
    server(serverIn), sessionId(sessionIdIn), topo(std::move(topoIn)), workDir(std::move(workDirIn))
{}

std::shared_ptr<StateTree> Session::GetLocalRegistry_(cipc::Rank rank)
{
    std::scoped_lock lock(mutex);
    return sessionState.localRegistryForRank[rank];
}

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
            fmt::format("[native_ckpt:svr][{}][ERROR] GetSnapshot_: no snapshot with id={}", *this, snapshotId));
    }
    return *it->second;
}

void Session::Activate()
{ start = PerfClock::now(); }

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
                "[native_ckpt:svr][{}][ERROR] Received tensor data exceeds allocated size={} for session id={} offset={} size={}",
                *this,
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
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Received tensor data size={} (range {}-{}) exceeds "
                                                 "total tensor size={} for session id={}",
                                                 *this,
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

void Session::Handle(cipc::Message::FinalizeSnapshot const& hdr, std::span<char const> /* buf */)
{
    PERF() << tel::span::SvrFinalize::Begin{.id = {.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId}};
    {
        std::scoped_lock lock(mutex);

        auto& snapshot = GetSnapshot_(lock, hdr.snapshotId);
        try
        {
            auto& localState = *snapshot.localStates.at(hdr.globalRank);

            if (localState.writtenTensorSize != localState.tensorStorage.size)
            {
                throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Incomplete tensor data received. sessionid={}. "
                                                     "rcvd-size={}(range{} - {}) expected={} ",
                                                     *this,
                                                     hdr.sessionId,
                                                     localState.writtenTensorSize,
                                                     localState.writtenTensorRangeBegin,
                                                     localState.writtenTensorRangeEnd,
                                                     localState.tensorStorage.size));
            }
            if (localState.writtenTensorRangeBegin != 0 || localState.writtenTensorRangeEnd != localState.tensorStorage.size)
            {
                throw std::runtime_error(
                    fmt::format("[native_ckpt:svr][{}][ERROR] Tensor data received has invalid range. sessionid={}. "
                                "rcvd-size={}(range{} - {}) expected range=0 - {}",
                                *this,
                                hdr.sessionId,
                                localState.writtenTensorSize,
                                localState.writtenTensorRangeBegin,
                                localState.writtenTensorRangeEnd,
                                localState.tensorStorage.size));
            }
            if (sessionState.globalRegistry == nullptr)
            {
                WARN("[{}] Global state registry is not initialized when finalizing rank={} snapshot={}",
                     *this,
                     hdr.globalRank,
                     hdr.snapshotId);
            }
            snapshot.finalizedRanks.insert(hdr.globalRank);
        } catch (std::exception const& e) { snapshot.failures.push_back(fmt::format("Error finalizing snapshot: {}", e.what())); }
    }
    {
        std::scoped_lock lock(mutex);    // local registry uploads can arrive on concurrent HTTP handlers
        asyncTasks.emplace_back(std::async(std::launch::async | std::launch::deferred, &Session::AsyncWorkFinalizeSnapshot_, this, hdr));
    }
}

void Session::HandleStateRegistryFragment(Rank                                                 recvRank,
                                          std::vector<TensorShardInfo> const&                  tensorShards,
                                          std::vector<std::pair<std::string, uint64_t>> const& blobMetasIn)
{
    std::scoped_lock lock(mutex);
    MergeFragmentToGlobalState_(lock, tensorShards);
    // Blob lengths are fixed by initial registry synchronization.
    for (auto const& [fqn, length] : blobMetasIn) { blobMetas[fqn][recvRank] = {.length = length}; }
    receivedFrom.insert(recvRank);
    INFO("[{}] HandleStateRegistryFragment rank={} shards={} Blobs={} {}",
         *this,
         recvRank,
         tensorShards.size(),
         blobMetasIn.size(),
         WaitingForRanks{receivedFrom, Rank{0}, topo.worldSize});
}

void Session::RecordBlobMetasLocked_(std::scoped_lock<std::mutex> const& /* lock */, Rank rank, StateTree const& localRegistry)
{
    localRegistry.WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
        auto const& blob                                      = std::get<StateNode::Blob>(node.data);
        blobMetas[localRegistry.GetNodeId(node.handle)][rank] = {.length = blob.size()};
    });
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
