#pragma once
#include "byte_storage.h"
#include "distcp.h"
#include "messages.h"
#include "state.h"
#include "tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Server is defined in server.cpp at file scope; Session only holds a pointer to it.
struct Server;

struct TensorShardInfo
{
    Tensor::DType     dtype{};
    Tensor::Dim       fulldim;
    DTensor::Shard    tensorInfo;
    StateTree::NodeId nodeId;
};

struct Session
{
    // State for Devices directly connected to this peer
    struct LocalState
    {
        std::shared_ptr<StateTree>     localRegistry;
        ByteStorage                    tensorStorage;
        checkpoint::distcp::BlobsByFqn blobs;
        std::chrono::nanoseconds       startedAt{};

        uint64_t writtenTensorRangeBegin{0};
        uint64_t writtenTensorRangeEnd{0};
        uint64_t writtenTensorSize{0};
    };

    struct Snapshot
    {
        checkpoint::ipc::SnapshotId snapshotId{checkpoint::ipc::SnapshotId::Invalid};
        std::string                 name;
        std::filesystem::path       path;

        std::chrono::system_clock::time_point activatedAt = std::chrono::system_clock::now();

        checkpoint::distcp::BlobMetadataByFqn blobMetadata;
        std::shared_ptr<StateTree>            globalRegistry;
        std::shared_ptr<StateTree>            dedupedGlobalRegistry;

        std::unordered_map<checkpoint::ipc::Rank, std::unique_ptr<LocalState>> localStates;

        std::unordered_set<checkpoint::ipc::Rank> finalizedRanks;
        std::unordered_set<checkpoint::ipc::Rank> writeCkptCompletedRanks;
    };

    Server* const                         server;
    checkpoint::ipc::SessionId const      sessionId{checkpoint::ipc::SessionId::Invalid};
    RankTopology const                    topo;
    std::filesystem::path const           workDir;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    uint32_t           numActiveSnapshots{0};
    mutable std::mutex mutex;

    // Session State variables
    Snapshot                                                 sessionState;    // The tensor storage on this is empty
    std::unordered_map<StateTree::NodeId, StateNode::Handle> tensorMap;       // Only needed for constructing sessionState
    std::unordered_set<checkpoint::ipc::Rank>                receivedFrom;    // Ranks from which data has been received.
    uint64_t                                                 largestSnapshotSize{0};

    checkpoint::distcp::BlobMetadataByFqn blobMetas;

    std::map<checkpoint::ipc::SnapshotId, std::unique_ptr<Snapshot>> snapshots;

    checkpoint::ipc::SnapshotId highestEvictedDurableSnapshot{checkpoint::ipc::SnapshotId::Invalid};

    std::vector<std::future<void>> asyncTasks;

    Session(Server* serverIn LFTBND, checkpoint::ipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn);
    std::atomic<bool> stopRequested{false};

    void Activate();

    void Handle(checkpoint::ipc::Message::LocalStateRegistry const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::LocalStateData const& hdr1, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::FinalizeSnapshot const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::FinalizeSession const& hdr, std::span<char const> buf);

    void   HandleStateRegistryFragment(checkpoint::ipc::Rank                                recvRank,
                                       std::vector<TensorShardInfo> const&                  tensorShards,
                                       std::vector<std::pair<std::string, uint64_t>> const& blobMetasIn);
    bool   IsCoordinatorPeer() const { return PeerIndex() == 0; }
    size_t PeerIndex() const { return static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer; }

    void SendGlobalRegistryToPeers(std::string_view const& registry, size_t startPeerIndex, size_t endPeerIndex) const;

    size_t ExpectedWriteCompletions() const
    { return IsCoordinatorPeer() ? static_cast<size_t>(topo.worldSize) : static_cast<size_t>(topo.ranksPerPeer); }
    bool IsSnapshotEvictable(Snapshot const& snapshot) const
    {
        return snapshot.writeCkptCompletedRanks.size() >= ExpectedWriteCompletions()
               && snapshot.finalizedRanks.size() >= ExpectedWriteCompletions();
    }
    bool AllSnapshotsComplete() const
    {
        std::scoped_lock lock(mutex);
        return std::ranges::all_of(snapshots, [&](auto const& snapshot) { return IsSnapshotEvictable(*snapshot.second); });
    }

    // Map a wait-for-finalize iteration to a concrete snapshotId. A non-negative iteration is the
    // snapshotId directly. A negative iteration is an index from the newest known snapshot
    // (-1 = latest, -2 = second-latest, ...), resolved against the ordered `snapshots` map (or the
    // evicted-durable highwater when nothing is resident). Returns false if not yet resolvable.
    bool ResolveSnapshotId(int64_t iteration, checkpoint::ipc::SnapshotId& out) const;
    bool IsSnapshotDurable(checkpoint::ipc::SnapshotId snapshotId) const;

    private:
    LocalState& GetSessionLocalState_(checkpoint::ipc::Rank rank) LFTBND;
    Snapshot&   GetSnapshot_(checkpoint::ipc::SnapshotId snapshotId) LFTBND;
    Snapshot&   GetSnapshot_(std::scoped_lock<std::mutex> const& lock, checkpoint::ipc::SnapshotId snapshotId) LFTBND;

#ifdef DEBUG_MD5SUM
    void DebugWriteNativeMetadataInfo_(Snapshot const& snapshot) const;
#endif

    void OnReceiveLocalStateRegistryAsync_(checkpoint::ipc::Rank const& rank);
    void OnFinalizeSnapshotAsync_(checkpoint::ipc::Message::FinalizeSnapshot const& hdr);

    void MergeFragmentToGlobalState_(std::scoped_lock<std::mutex> const& lock, std::vector<TensorShardInfo> const& tensorShards);
    void RecordBlobMetasLocked_(std::scoped_lock<std::mutex> const& lock, checkpoint::ipc::Rank rank, StateTree const& localRegistry);
    void SendToPeerWithRetries_(std::string_view const& peer,
                                std::string_view const& body,
                                std::string_view const& target,
                                std::string_view const& contentEncoding) const;

    void SendPeerFinalizeSnapshotComplete_(std::string_view const& peer, checkpoint::ipc::Message::FinalizeSnapshot const& msg) const;
    void SendPeerWriteCkptComplete_(std::string_view const& peer, checkpoint::ipc::Message::FinalizeSnapshot const& msg) const;

    void SendLocalShardedStateRegistryToPeer_(checkpoint::ipc::Rank rank, std::string const& peer);
    void SendLocalShardedStateRegistryToPeerWithRetries_(checkpoint::ipc::Rank rank, std::string const& peer);

    void BroadcastGlobalRegistryToPeers_(StateTree const& globalRegistry) const;
};
