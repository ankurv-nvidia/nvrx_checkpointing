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
#include <memory>
#include <mutex>
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
    // State for Devices directly connected to this peer
    struct LocalState
    {
        std::shared_ptr<StateTree> localRegistry;
        Blob                       blob;

        uint64_t writtenTensorRangeBegin{0};
        uint64_t writtenTensorRangeEnd{0};
        uint64_t writtenTensorSize{0};

        std::vector<checkpoint::distcp::WriteRecord> writeLog;
    };

    struct Snapshot
    {
        checkpoint::ipc::SnapshotId snapshotId{checkpoint::ipc::SnapshotId::Invalid};
        std::string                 name;
        std::filesystem::path       path;

        std::shared_ptr<StateTree>                                                    registry;
        std::unordered_map<checkpoint::ipc::Rank, std::unique_ptr<LocalState>>        localStates;
        // Ranks that have sent FinalizeSnapshot for this snapshot
        std::unordered_set<checkpoint::ipc::Rank>                                     finalizedRanks;
        // Ranks whose __<r>_0.distcp file has been written and writeLog populated.
        // Mirrors `finalizedRanks`: rank-0 polls this set to gate `.metadata` write.
        std::unordered_set<checkpoint::ipc::Rank>                                     writeCompletedRanks;
    };

    Server* const                                  server;
    checkpoint::ipc::SessionId const               sessionId{checkpoint::ipc::SessionId::Invalid};
    RankTopology const                             topo;
    std::filesystem::path const                    workDir;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    std::mutex mutex;

    // Session State variables
    Snapshot                                                 sessionState;    // The blob on this is empty
    std::unordered_map<StateTree::NodeId, StateNode::Handle> tensorMap;       // Only needed for constructing sessionState
    std::unordered_set<checkpoint::ipc::Rank>                recievedFrom;    // Ranks from which data has been recieved.

    std::unordered_map<checkpoint::ipc::SnapshotId, std::unique_ptr<Snapshot>> snapshots;

    std::vector<std::future<void>> asyncTasks;

    Session(Server*                    serverIn,
            checkpoint::ipc::SessionId sessionIdIn,
            std::filesystem::path      workDirIn,
            RankTopology               topoIn);

    void Activate();

    void Handle(checkpoint::ipc::Message::LocalStateRegistry const&     hdr,
                std::array<char, checkpoint::ipc::D2HBufferSize> const& buf);

    void Handle(checkpoint::ipc::Message::LocalStateData const&         hdr1,
                std::array<char, checkpoint::ipc::D2HBufferSize> const& buf);

    void Handle(checkpoint::ipc::Message::ActivateSnapshot const&       hdr,
                std::array<char, checkpoint::ipc::D2HBufferSize> const& buf);

    void Handle(checkpoint::ipc::Message::FinalizeSnapshot const&       hdr,
                std::array<char, checkpoint::ipc::D2HBufferSize> const& buf);

    void Handle(checkpoint::ipc::Message::FinalizeSession const&        hdr,
                std::array<char, checkpoint::ipc::D2HBufferSize> const& buf);

    void HandleStateRegistryFragment(checkpoint::ipc::Rank               recvRank,
                                     std::vector<TensorShardInfo> const& tensorShards);

    private:
    LocalState& GetSessionLocalState_(checkpoint::ipc::Rank rank);
    LocalState& FindOrCreateLocalState_(checkpoint::ipc::Rank rank);
    Snapshot&   GetSnapshot_(checkpoint::ipc::SnapshotId snapshotId);

    void DebugWriteNativeMetadataInfo_(Snapshot const& snapshot) const;

    void OnRecieveLocalStateRegistryAsync_(checkpoint::ipc::Rank const& rank);
    void OnFinalizeSnapshotAsync_(checkpoint::ipc::Message::FinalizeSnapshot const& hdr);

    void MergeFragmentToGlobalState_(std::lock_guard<std::mutex> const&  lock,
                                     std::vector<TensorShardInfo> const& tensorShards);

    void SendPeerFinalizeSnapshotComplete_(std::string const&                                peer,
                                           checkpoint::ipc::Message::FinalizeSnapshot const& msg) const;
    void SendLocalShardedStateRegistryToPeer_(checkpoint::ipc::Rank rank, std::string const& peer);

    void WriteGlobalStateTreeToFile_(Snapshot const& snapshot, std::filesystem::path const& ckptPath) const;
};
