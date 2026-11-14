#include "session.h"
#include "logging.h"
#include "net_utils.h"
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
            if (attempt < MaxRetries) { WARN("[{}] Attempt {} peer={} failed target={} {}", session, attempt, peer, target, ex.what()); }
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

static void DebugOutputToFile(std::string_view fileName, std::function<void(std::ostream&)> const& write)
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

static std::string WriteCkptCompleteBody(std::string_view failure, std::unordered_map<std::string, uint64_t> const& dynamicBlobSizes)
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Session::AsyncWorkFinalizeSnapshot_(cipc::Message::FinalizeSnapshot const& hdr) noexcept
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
            SendToPeerWithRetries(*this,
                                  topo.peers[0],
                                  {},
                                  fmt::format("/api/peer/snapshotcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                  {});
        }
        auto const dynamicBlobSizes = checkpoint::distcp::WriteRankShard(
            snapshot.path.parent_path(), hdr.globalRank, *snapshotState.snapshotRegistry, snapshotState.snapshotData);
        if (notifyCoordinator)
        {
            SendToPeerWithRetries(*this,
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
            std::unordered_map<std::string, uint64_t> blobSizes;
            auto const                                otherRanks = static_cast<size_t>(topo.worldSize) - 1;
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
                auto const id = tel::Id{.sessionId = sessionId, .rank = hdr.globalRank, .snapshotId = hdr.snapshotId};
                PERF() << tel::span::SvrWriteMetadata::Begin{.id = id};
                checkpoint::distcp::WriteMetadata(snapshot.path.parent_path(), *sessionState.dedupedGlobalRegistry, blobSizes);
                PERF() << tel::span::SvrWriteMetadata::End{.id = id};
            } catch (std::exception const& ex)
            {
                throw std::runtime_error(
                    fmt::format("[{}] WriteMetadata failed for snapshot={} rank={}: {}", *this, hdr.snapshotId, hdr.globalRank, ex.what()));
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
        ERR("[{}] Snapshot finalization failed for snapshot={} rank={}: {}", *this, hdr.snapshotId, hdr.globalRank, ex.what());
        if (hdr.globalRank == Rank{0})
        {
            try
            {
                BroadcastWriteCkptCompleteToPeers_(snapshot);
            } catch (std::exception const& notifyError)
            {
                ERR("[{}] Failed to broadcast checkpoint failure: {}", *this, notifyError.what());
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
                    SendToPeerWithRetries(*this,
                                          topo.peers[0],
                                          WriteCkptCompleteBody(failure, {}),
                                          fmt::format("/api/peer/writeckptcomplete/{}/{}/{}", sessionId, hdr.globalRank, hdr.snapshotId),
                                          {});
                } catch (std::exception const& notifyError)
                {
                    ERR("[{}] Failed to report checkpoint failure: {}", *this, notifyError.what());
                }
            }
        }
    }

    std::string failure;
    {
        std::scoped_lock lock(mutex);
        if (!snapshot.failures.empty()) { failure = snapshot.failures.front(); }
    }
    INFO("[{}] AsyncWorkFinalizeSnapshot_ for rank={}, snapshotId={} failure={}",
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
            INFO("[{}] Released {} snapshot buffers after finalizing snapshot={} rank={}. Remaining snapshots with buffers={} "
                 "numActiveSnapshots={}",
                 *this,
                 evicted.size(),
                 hdr.snapshotId,
                 hdr.globalRank,
                 remainingSnapshots,
                 numActiveSnapshots);
        }
    } catch (std::exception const& ex) { ERR("[{}] Exception occurred while releasing snapshot buffers: {}", *this, ex.what()); }
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
        StateSerDes::Write(ostrm, registry, StateNode::Handle::Root, true);
        ostrm.strict_sync();
        rawSize = static_cast<uint64_t>(ostrm.component<boost::iostreams::counter>(0)->characters());
    }
    return out;
}

void Session::RelayWriteCkptCompleteToPeers(cipc::SnapshotId        snapshotId,
                                            std::string_view const& msg,
                                            size_t                  startPeerIndex,
                                            size_t                  endPeerIndex) const
{
    if (startPeerIndex >= endPeerIndex) { return; }
    auto        delegateStart = (startPeerIndex + endPeerIndex) / 2;
    auto const& peer          = topo.peers[delegateStart];
    auto const  postTarget
        = fmt::format("/api/peer/writeckptcomplete/{}/0/{}?fwd={}:{}", sessionId, snapshotId, delegateStart, endPeerIndex);
    SendToPeerWithRetries(*this, peer, msg, postTarget, {});
    RelayWriteCkptCompleteToPeers(snapshotId, msg, startPeerIndex, delegateStart);
}

void Session::BroadcastWriteCkptCompleteToPeers_(Snapshot const& snapshot) const
{
    if (topo.peers.size() <= 1) { return; }
    std::string failure;
    {
        std::scoped_lock lock(mutex);
        failure = fmt::format("{}", fmt::join(snapshot.failures, ","));
    }
    RelayWriteCkptCompleteToPeers(snapshot.snapshotId, WriteCkptCompleteBody(failure, {}), 1, topo.peers.size());

    INFO("[{}] Broadcast global registry to {} peer(s)", *this, topo.peers.size() - 1);
}

void Session::BroadcastGlobalRegistryToPeers_(StateTree const& globalRegistry) const
{
    if (topo.peers.size() <= 1) { return; }
    uint64_t           rawSize  = 0;
    std::string const  registry = DeflateRegistry(globalRegistry, rawSize);
    std::ostringstream body;
    serdes::WriteBlobWithSize<uint32_t>(body, {});
    body.write(registry.data(), static_cast<std::streamsize>(registry.size()));

    INFO("[{}] Global registry {} bytes -> {} deflated ({:.1f}x)",
         *this,
         rawSize,
         registry.size(),
         static_cast<double>(rawSize) / static_cast<double>(registry.size()));

    RelayGlobalRegistryToPeers(body.str(), 1, topo.peers.size());

    INFO("[{}] Broadcast global registry to {} peer(s)", *this, topo.peers.size() - 1);
}

void Session::RelayGlobalRegistryToPeers(std::string_view const& registry, size_t startPeerIndex, size_t endPeerIndex) const
{
    if (startPeerIndex >= endPeerIndex) { return; }
    auto        delegateStart = (startPeerIndex + endPeerIndex) / 2;
    auto const& peer          = topo.peers[delegateStart];
    auto const  postTarget    = fmt::format("/api/peer/globalregistry/{}?fwd={}:{}", sessionId, delegateStart, endPeerIndex);
    SendToPeerWithRetries(*this, peer, registry, postTarget, {});
    RelayGlobalRegistryToPeers(registry, startPeerIndex, delegateStart);
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
        throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] GetSnapshot_: no snapshot with id={}", *this, snapshotId));
    }
    return *it->second;
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
    INFO("[{}] Received local registry rank={} metadata bytes={} payload bytes={}", *this, rank, hdr.messageSize, snapshotSize);

    bool registryAccepted = false;
    {
        // Concurrent handlers push here; an unlocked emplace corrupts sibling futures.
        std::scoped_lock lock(mutex);
        auto&            registeredRegistry = sessionState.localRegistryForRank[rank];
        if (registeredRegistry != nullptr)
        {
            WARN("[{}] Received a duplicate local state registry from rank={} for an unknown reason; ignoring it", *this, rank);
            registryTopo = registeredRegistry->Topo();
        }
        else
        {
            registeredRegistry = std::move(localRegistry);
            registryAccepted   = true;
            asyncTasks.emplace_back(
                std::async(std::launch::async | std::launch::deferred, &Session::AsyncWorkReceiveLocalStateRegistry_, this, rank));
        }
    }

    if (registryAccepted)
    {
        auto const digits
            = static_cast<size_t>(std::log10(std::max(static_cast<double>(static_cast<uint32_t>(registryTopo.worldSize)), 1.0))) + 1;
        auto fname = fmt::format("registry.{:0{}}.bin", static_cast<uint32_t>(rank), digits);
        DebugOutputToFile(fname, [&](std::ostream& ostrm) { ostrm.write(buf.data(), static_cast<std::streamsize>(buf.size())); });
    }
}

void Session::Handle(cipc::Message::SnapshotData const& hdr1, std::span<char const> buf)
{
    void* ptr = nullptr;
    {
        std::scoped_lock lock(mutex);
        auto&            snapshot      = GetSnapshot_(lock, hdr1.snapshotId);
        auto&            snapshotState = *snapshot.snapshotStateForRank.at(hdr1.globalRank);
        auto             endOffset     = hdr1.offset + hdr1.messageSize;
        if (endOffset > snapshotState.snapshotData.size)
        {
            throw std::runtime_error(fmt::format(
                "[native_ckpt:svr][{}][ERROR] Received snapshot data exceeds allocated size={} for session id={} offset={} size={}",
                *this,
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
            throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Received snapshot data size={} (range {}-{}) exceeds "
                                                 "total snapshot size={} for session id={}",
                                                 *this,
                                                 snapshotState.writtenPayloadSize,
                                                 snapshotState.writtenPayloadRangeBegin,
                                                 snapshotState.writtenPayloadRangeEnd,
                                                 snapshotState.snapshotData.size,
                                                 hdr1.sessionId));
        }

        ptr = snapshotState.snapshotData[hdr1.offset];
    }
    std::memcpy(ptr, buf.data(), hdr1.messageSize);
}

void Session::Handle(cipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf)
{
    auto addSize = [](uint64_t total, uint64_t size) -> uint64_t {
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
                        offset       = addSize(offset, shard.storageSize);
                        packed.insert(shard);
                    }
                    tensor.shards = std::move(packed);
                }
                else if (type == group && type == StateNode::Type::DynamicSizeBlob)
                {
                    auto& blob  = std::get<StateNode::DynamicSizeBlob>(node.data);
                    blob.handle = offset;
                    offset      = addSize(offset, blob.storageSize);
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
        snapshotState->snapshotData     = ByteStorage(payloadSize);
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
    INFO("[{}] Activated snapshot={} rank={} payload bytes={} path={}", *this, hdr.snapshotId, hdr.globalRank, hdr.snapshotSize, ckptPath);
}

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
                throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Incomplete snapshot data received. sessionid={}. "
                                                     "rcvd-size={}(range{} - {}) expected={} ",
                                                     *this,
                                                     hdr.sessionId,
                                                     snapshotState.writtenPayloadSize,
                                                     snapshotState.writtenPayloadRangeBegin,
                                                     snapshotState.writtenPayloadRangeEnd,
                                                     snapshotState.snapshotData.size));
            }
            if (snapshotState.writtenPayloadRangeBegin != 0 || snapshotState.writtenPayloadRangeEnd != snapshotState.snapshotData.size)
            {
                throw std::runtime_error(fmt::format("[native_ckpt:svr][{}][ERROR] Snapshot data received has invalid range. sessionid={}. "
                                                     "rcvd-size={}(range{} - {}) expected range=0 - {}",
                                                     *this,
                                                     hdr.sessionId,
                                                     snapshotState.writtenPayloadSize,
                                                     snapshotState.writtenPayloadRangeBegin,
                                                     snapshotState.writtenPayloadRangeEnd,
                                                     snapshotState.snapshotData.size));
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

static void FailPlanning(Session& session, std::string failure)
{
    std::scoped_lock lock(session.mutex);
    if (!session.sessionState.planningFailure.empty()) { return; }
    session.sessionState.planningFailure = std::move(failure);
    if (session.stopRequested) { return; }
    ERR("[{}] Snapshot planning failed: {}", session, session.sessionState.planningFailure);
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
                SendToPeerWithRetries(
                    session, session.topo.peers[peer], body, fmt::format("/api/peer/globalregistry/{}", session.sessionId), {});
            } catch (std::exception const& error) { ERR("[{}] Failed to propagate planning error: {}", session, error.what()); }
        }
    }));
}

void Session::HandleGlobalRegistry(std::shared_ptr<StateTree> globalRegistry, std::string failure)
try
{
    if (!failure.empty())
    {
        if (IsCoordinatorPeer()) { FailPlanning(*this, std::move(failure)); }
        else
        {
            std::scoped_lock lock(mutex);
            sessionState.planningFailure = std::move(failure);
        }
        return;
    }

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
    auto const firstRank = static_cast<uint32_t>(PeerIndex()) * topo.ranksPerPeer;
    auto const rankCount = std::min(topo.ranksPerPeer, static_cast<uint32_t>(topo.worldSize) - firstRank);
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
        std::unordered_map<Tensor::DimRange, std::vector<TensorShard const*>> groups;
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

    std::unordered_map<Rank, std::shared_ptr<StateTree>> registries;
    for (uint32_t index = 0; index < rankCount; ++index)
    {
        auto topology = globalRegistry->Topo();
        topology.rank = Rank{firstRank + index};
        auto registry = std::make_shared<StateTree>(globalRegistry->SessionId(), std::move(topology));
        registries.emplace(registry->GlobalRank(), std::move(registry));
    }
    auto dedupedGlobalRegistry = std::make_shared<StateTree>(globalRegistry->SessionId(), topo);
    std::unordered_map<StateNode::Handle, std::vector<Assignment const*>> byNode;
    for (auto const& item : assignments)
    {
        if (IsCoordinatorPeer() || registries.contains(item.owner)) { byNode[item.node].push_back(&item); }
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
        std::unordered_map<Rank, std::vector<Assignment const*>> byRank;
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
    std::scoped_lock lock(mutex);
    sessionState.dedupedGlobalRegistry   = std::move(dedupedGlobalRegistry);
    sessionState.globalRegistry          = std::move(globalRegistry);
    sessionState.snapshotRegistryForRank = std::move(registries);
    sessionState.planningReady           = true;
} catch (std::exception const& error) { FailPlanning(*this, error.what()); }

void Session::HandleLocalStateRegistry(Rank recvRank, std::shared_ptr<StateTree> const& registry)
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
                        auto& added = global.AddChild(parent, name);
                        added.data  = supplied.data;
                        if (type == StateNode::Type::Scalar)
                        {
                            auto& scalar    = std::get<StateNode::Scalar>(added.data);
                            scalar.replicas = {recvRank};
                            Primitives64Bit::UnsignedTraits<uint64_t>::Assign(scalar, uint64_t{0});
                        }
                        else if (type == StateNode::Type::DynamicSizeBlob)
                        {
                            std::get<StateNode::DynamicSizeBlob>(added.data).replicas = {recvRank};
                        }
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
                    target.replicas.insert(recvRank);
                }
                else if (type == StateNode::Type::DynamicSizeBlob)
                {
                    std::get<StateNode::DynamicSizeBlob>(registered.data).replicas.insert(recvRank);
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
                DebugOutputToFile("registry.global.bin", [&](std::ostream& stream) {
                    StateSerDes::Write(stream, *globalRegistry, StateNode::Handle::Root, true);
                });
                HandleGlobalRegistry(globalRegistry, {});
                BroadcastGlobalRegistryToPeers_(*globalRegistry);
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

void Session::AsyncWorkReceiveLocalStateRegistry_(Rank const& rank)
try
{
    std::shared_ptr<StateTree> registry;
    {
        std::scoped_lock lock(mutex);
        if (!IsCoordinatorPeer()) { receivedFrom.insert(rank); }    // Otherwise HandleLocalStateRegistry will do it
        registry = sessionState.localRegistryForRank.at(rank);
    }
    if (IsCoordinatorPeer()) { HandleLocalStateRegistry(rank, std::move(registry)); }
    else
    {
        // Send to coordinator
        std::ostringstream body;
        StateSerDes::Write(body, *registry, StateNode::Handle::Root, true);
        SendToPeerWithRetries(*this, topo.peers[0], body.str(), fmt::format("/api/peer/localstateregistry/{}/{}", sessionId, rank), {});
    }
} catch (std::exception const& error) { FailPlanning(*this, error.what()); }
