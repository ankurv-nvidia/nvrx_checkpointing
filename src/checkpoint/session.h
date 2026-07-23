#pragma once
#include "blob.h"
#include "distcp.h"
#include "messages.h"
#include "state.h"
#include "tensor.h"

#include <array>
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
#include <vector>

// Server is defined in server.cpp at file scope; Session only holds a pointer to it.
struct Server;

struct TensorShardInfo
{
    checkpoint::ipc::Rank rank{};
    DS4Tensor             tensorInfo{};
    StateTree::NodeId     nodeId;
};

struct Session
{
#ifdef PHOENIX_BUILD_FOR_VALGRIND
    // Snapshots can be too large to retain in memory for valgrind runs.
    static constexpr uint32_t MaxActiveSnapshots = 0;
#else
    static constexpr uint32_t MaxActiveSnapshots = 4;
#endif

    // State for Devices directly connected to this peer
    struct LocalState
    {
        std::shared_ptr<StateTree> localRegistry;
        Blob                       blob;

        uint64_t writtenTensorRangeBegin{0};
        uint64_t writtenTensorRangeEnd{0};
        uint64_t writtenTensorSize{0};
    };

    struct Snapshot
    {
        checkpoint::ipc::SnapshotId snapshotId{checkpoint::ipc::SnapshotId::Invalid};
        std::string                 name;
        std::filesystem::path       path;

        // Rank 0 attaches updated values for Blob fields whose contents are
        // snapshot-scoped rather than fixed by the session registry.
        StateTree::SnapshotBlobs snapshotBlobs;

        std::shared_ptr<StateTree>                                             registry;
        std::unordered_map<checkpoint::ipc::Rank, std::unique_ptr<LocalState>> localStates;
        // Ranks that have sent FinalizeSnapshot for this snapshot
        std::unordered_set<checkpoint::ipc::Rank> finalizedRanks;
        // On peers, ranks whose shard is durable and whose completion notification
        // the coordinator acknowledged. On the coordinator, rank 0 joins only after
        // metadata is durable, so a world-size set means the checkpoint is loadable.
        std::unordered_set<checkpoint::ipc::Rank> writeCkptCompletedRanks;
    };

    Server* const                                  server;
    checkpoint::ipc::SessionId const               sessionId{checkpoint::ipc::SessionId::Invalid};
    RankTopology const                             topo;
    std::filesystem::path const                    workDir;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    uint32_t           numActiveSnapshots{0};
    mutable std::mutex mutex;

    // Session State variables
    Snapshot                                                 sessionState;    // The blob on this is empty
    std::unordered_map<StateTree::NodeId, StateNode::Handle> tensorMap;       // Only needed for constructing sessionState
    std::unordered_set<checkpoint::ipc::Rank>                recievedFrom;    // Ranks from which data has been recieved.

    std::map<checkpoint::ipc::SnapshotId, std::unique_ptr<Snapshot>> snapshots;

    std::vector<std::future<void>> asyncTasks;

    Session(Server* serverIn, checkpoint::ipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn);
    std::atomic<bool> stopRequested{false};

    void Activate();

    void Handle(checkpoint::ipc::Message::LocalStateRegistry const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::LocalStateData const& hdr1, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::FinalizeSnapshot const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::FinalizeSession const& hdr, std::span<char const> buf);

    void   HandleStateRegistryFragment(checkpoint::ipc::Rank recvRank, std::vector<TensorShardInfo> const& tensorShards);
    bool   IsCoordinatorPeer() const { return static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer == 0; }
    size_t ExpectedWriteCompletions() const
    { return IsCoordinatorPeer() ? static_cast<size_t>(topo.worldSize) : static_cast<size_t>(topo.ranksPerPeer); }
    bool AllSnapshotsComplete() const
    {
        std::scoped_lock lock(mutex);
        for (auto const& [snapshotId, snapshot] : snapshots)
        {
            if (snapshot->writeCkptCompletedRanks.size() < ExpectedWriteCompletions()) { return false; }
        }
        return true;
    }

    private:
    LocalState& GetSessionLocalState_(checkpoint::ipc::Rank rank);
    LocalState& FindOrCreateLocalState_(checkpoint::ipc::Rank rank);
    Snapshot&   GetSnapshot_(checkpoint::ipc::SnapshotId snapshotId);
    Snapshot&   GetSnapshot_(std::scoped_lock<std::mutex> const& lock, checkpoint::ipc::SnapshotId snapshotId);

#ifdef DEBUG_MD5SUM
    void DebugWriteNativeMetadataInfo_(Snapshot const& snapshot) const;
#endif

    void OnRecieveLocalStateRegistryAsync_(checkpoint::ipc::Rank const& rank);
    void OnFinalizeSnapshotAsync_(checkpoint::ipc::Message::FinalizeSnapshot const& hdr);

    void MergeFragmentToGlobalState_(std::scoped_lock<std::mutex> const& lock, std::vector<TensorShardInfo> const& tensorShards);

    void SendToPeerWithRetries_(std::string const& peer, std::string const& body, std::string const& target) const;

    void SendPeerFinalizeSnapshotComplete_(std::string const& peer, checkpoint::ipc::Message::FinalizeSnapshot const& msg) const;
    void SendPeerWriteCkptComplete_(std::string const& peer, checkpoint::ipc::Message::FinalizeSnapshot const& msg) const;

    void SendLocalShardedStateRegistryToPeer_(checkpoint::ipc::Rank rank, std::string const& peer);
    void SendLocalShardedStateRegistryToPeerWithRetries_(checkpoint::ipc::Rank rank, std::string const& peer);

    // Coordinator-only: push the final merged registry to the other nodes before
    // they write their local shards.
    void BroadcastGlobalRegistryToPeers_(checkpoint::ipc::SnapshotId snapshotId, StateTree const& registry) const;

    void WriteGlobalStateTreeToFile_(Snapshot const& snapshot, std::filesystem::path const& ckptPath) const;
};
