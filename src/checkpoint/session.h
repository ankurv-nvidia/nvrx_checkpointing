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
#include <vector>

// Server is defined in server.cpp at file scope; Session only holds a pointer to it.
struct Server;

struct Session
{
    struct SnapshotState
    {
        std::shared_ptr<StateTree> snapshotRegistry;
        ByteStorage                snapshotData;
        std::chrono::nanoseconds   startedAt{};
        int                        numaNode{-1};

        uint64_t writtenPayloadRangeBegin{0};
        uint64_t writtenPayloadRangeEnd{0};
        uint64_t writtenPayloadSize{0};
    };

    struct RegistryState
    {
        std::shared_ptr<StateTree>                                            globalRegistry;
        std::shared_ptr<StateTree>                                            dedupedGlobalRegistry;
        std::unordered_map<checkpoint::ipc::Rank, std::shared_ptr<StateTree>> localRegistryForRank;
        std::unordered_map<checkpoint::ipc::Rank, std::shared_ptr<StateTree>> snapshotRegistryForRank;
        bool                                                                  planningStarted{false};
        bool                                                                  planningReady{false};
        std::string                                                           planningFailure;
    };

    struct Snapshot
    {
        checkpoint::ipc::SnapshotId snapshotId{checkpoint::ipc::SnapshotId::Invalid};
        std::string                 name;
        std::filesystem::path       path;

        std::chrono::system_clock::time_point activatedAt = std::chrono::system_clock::now();

        std::unordered_map<std::string, uint64_t> dynamicBlobSizes;

        std::unordered_map<checkpoint::ipc::Rank, std::unique_ptr<SnapshotState>> snapshotStateForRank;

        std::unordered_set<checkpoint::ipc::Rank> finalizedRanks;
        std::unordered_set<checkpoint::ipc::Rank> writeCkptCompletedRanks;
        bool                                      complete{false};
        std::vector<std::string>                  failures;    // complete && failure.empty() implies success
    };

    Server* const                         server;
    checkpoint::ipc::SessionId const      sessionId{checkpoint::ipc::SessionId::Invalid};
    RankTopology const                    topo;
    std::filesystem::path const           workDir;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    uint32_t           numActiveSnapshots{0};
    mutable std::mutex mutex;

    // Session State variables
    RegistryState                             sessionState;
    std::unordered_set<checkpoint::ipc::Rank> receivedFrom;    // Ranks from which data has been received.
    uint64_t                                  largestSnapshotSize{0};

    std::map<checkpoint::ipc::SnapshotId, std::unique_ptr<Snapshot>> snapshots;

    std::vector<std::future<void>> asyncTasks;

    Session(Server* serverIn LFTBND, checkpoint::ipc::SessionId sessionIdIn, std::filesystem::path workDirIn, RankTopology topoIn);
    std::atomic<bool> stopRequested{false};

    void Activate();

    void Handle(checkpoint::ipc::Message::LocalStateRegistry const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::SnapshotData const& hdr1, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::ActivateSnapshot const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::FinalizeSnapshot const& hdr, std::span<char const> buf);
    void Handle(checkpoint::ipc::Message::FinalizeSession const& hdr, std::span<char const> buf);

    void HandleLocalStateRegistry(checkpoint::ipc::Rank recvRank, std::shared_ptr<StateTree> registry);
    void HandleGlobalRegistry(std::shared_ptr<StateTree> registry, std::string failure);

    bool   IsCoordinatorPeer() const { return PeerIndex() == 0; }
    size_t PeerIndex() const { return static_cast<uint32_t>(topo.rank) / topo.ranksPerPeer; }

    void RelayGlobalRegistryToPeers(std::string_view const& registry, size_t startPeerIndex, size_t endPeerIndex) const;
    void RelayWriteCkptCompleteToPeers(checkpoint::ipc::SnapshotId snapshotId,
                                       std::string_view const&     msg,
                                       size_t                      startPeerIndex,
                                       size_t                      endPeerIndex) const;

    size_t ExpectedWriteCompletions() const
    { return IsCoordinatorPeer() ? static_cast<size_t>(topo.worldSize) : static_cast<size_t>(topo.ranksPerPeer); }

    bool AllSnapshotsComplete() const
    {
        std::scoped_lock lock(mutex);
        return std::ranges::all_of(snapshots, [&](auto const& entry) { return entry.second->complete; });
    }

    // Negative iterations count backward through retained checkpoint IDs.
    bool ResolveSnapshotId(int64_t iteration, checkpoint::ipc::SnapshotId& out) const;

    private:
    Snapshot& GetSnapshot_(checkpoint::ipc::SnapshotId snapshotId) LFTBND;
    Snapshot& GetSnapshot_(std::scoped_lock<std::mutex> const& lock, checkpoint::ipc::SnapshotId snapshotId) LFTBND;

    void AsyncWorkReceiveLocalStateRegistry_(checkpoint::ipc::Rank const& rank);
    void AsyncWorkFinalizeSnapshot_(checkpoint::ipc::Message::FinalizeSnapshot const& hdr) noexcept;

    void SendLocalStateRegistryToCoordinator_(checkpoint::ipc::Rank rank, std::string const& peer);

    void BroadcastGlobalRegistryToPeers_(StateTree const& globalRegistry) const;
    void BroadcastWriteCkptCompleteToPeers_(Snapshot const& snapshot) const;

    friend struct Server;
};
