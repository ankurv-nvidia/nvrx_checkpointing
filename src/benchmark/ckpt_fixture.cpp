#include "ckpt_fixture.h"

#include "checkpoint.h"
#include "messages.h"
#include "server.h"
#include "state.h"
#include "stateserdes.h"
#include "tensor.h"

#include <fmt/format.h>
#include <fmt/std.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cipc = checkpoint::ipc;

Rank InferWorldSize(StateTree const& tree)
{
    uint maxRankPlus1 = 1;
    tree.WalkNodes([&](StateNode const& node) {
        if (node.data.index() != static_cast<size_t>(StateNode::Type::ShardedTensor)) { return; }
        auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
        for (auto const& rankShard : tnsr.shards) { maxRankPlus1 = std::max(maxRankPlus1, static_cast<uint32_t>(rankShard.rank) + 1u); }
        tree.AssertShardedTensorHasNoHoles(node.handle);
    });
    return Rank{maxRankPlus1};
}

std::unique_ptr<StateTree> BuildPerRankStateTree(CheckpointTestContext& ctx, RankTopology topo, std::vector<char>& buffer)
{
    std::unique_ptr<StateTree> tree = std::make_unique<StateTree>(std::move(topo));
    ctx.globalRegistry->CloneInto(*tree);
    std::unordered_map<DeviceId, std::unique_ptr<DeviceContext>> devCtxMap;

    size_t totalAllocSize = 0;

    auto calculateSize
        = [&](DS4Tensor& tnsr, std::string_view /* shardFqn */) { totalAllocSize += static_cast<size_t>(tnsr.ShardSizeBytes()); };

    size_t currentOffset = 0;

    auto allocAndCopy = [&](DS4Tensor& tnsr, [[maybe_unused]] std::string_view shardFqn) {
        tnsr.handle = 0;
        if (tnsr.ShardSizeBytes() == 0) { return; }
        auto it = devCtxMap.find(tnsr.device);
        if (it == devCtxMap.end())
        {
            it = devCtxMap.insert({tnsr.device, DeviceContext::CreateDeviceContext(tnsr.device, 0)}).first;
            it->second->StreamAcquire();
        }

        auto& devCtx  = *it->second;
        auto  bufSize = static_cast<size_t>(tnsr.ShardSizeBytes());
        auto  devAddr = devCtx.Allocate(bufSize);
        auto* dataptr = buffer.data() + currentOffset;
#ifdef NDEBUG
        auto patLen = shardFqn.size();

        std::memcpy(dataptr, shardFqn.data(), std::min(shardFqn.size(), bufSize));
        for (size_t filled = std::min(patLen, bufSize); filled < bufSize;)
        {
            auto chunk = std::min(filled, bufSize - filled);
            std::memcpy(dataptr + filled, dataptr, chunk);
            filled += chunk;
        }
#endif
        devCtx.AddH2D(dataptr, devAddr, bufSize);
        currentOffset += bufSize;
        tnsr.handle = devAddr;
    };

    auto pass = [&](auto execFn) {
        tree->WalkNodes([&](StateNode& node) {
            if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
            {
                execFn(std::get<StateNode::TensorShard>(node.data), tree->GetNodeId(node.handle));
            }
            else if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
            {
                auto&                    tnsr = std::get<StateNode::ShardedTensor>(node.data);
                StateNode::ShardedTensor newTnsr{tnsr.dtype, tnsr.fulldim};
                for (auto rshard : tnsr.shards)
                {
                    if (tree->Topo().rank != rshard.rank) { continue; }
                    DS4Tensor shardTnsr
                        = {.sharddim = rshard.sharddim, .fulldim = tnsr.fulldim, .dtype = tnsr.dtype, .device = rshard.device};
                    execFn(shardTnsr, fmt::format("{}[{}]", tree->GetNodeId(node.handle), rshard.sharddim));
                    rshard.handle = shardTnsr.handle;
                    newTnsr.AddShard(rshard);
                }
                if (newTnsr.shards.empty()) { tree->RemoveNode(node.handle); }
                else
                {
                    node.data = std::move(newTnsr);
                }
            }
        });
    };

    pass(calculateSize);
    if (totalAllocSize > buffer.size()) { buffer.resize(totalAllocSize); }
    pass(allocAndCopy);

    for (auto& [device, devCtx] : devCtxMap) { devCtx->Flush(); }

    tree->WalkNodes([&](StateNode& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            auto& shTen = std::get<StateNode::ShardedTensor>(node.data);
            if (shTen.shards.size() == 1)
            {
                auto const& rshard    = *shTen.shards.begin();
                DS4Tensor   shardTnsr = {.sharddim = rshard.sharddim,
                                         .fulldim  = shTen.fulldim,
                                         .dtype    = shTen.dtype,
                                         .handle   = rshard.handle,
                                         .device   = rshard.device};
                node.data             = shardTnsr;
            }
        }
    });
    fmt::print(stderr, "BuildPerRankStateTree: rank {} allocated {} bytes of device memory\n", tree->Topo().rank, totalAllocSize);
    return tree;
}

void LoadStateRegistryFromPath(CheckpointTestContext& ctx, std::filesystem::path const& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { throw std::runtime_error(fmt::format("Failed to open state registry: {}", path.string())); }
    ctx.globalRegistry = std::make_unique<StateTree>(
        RankTopology{.rank = Rank{0}, .worldSize = Rank{1}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {"localhost"}});
    auto registry = StateSerDes::Read(ifs);

    // Adjust tensors sizes till we are below the limit.
    std::priority_queue<std::pair<uint64_t, StateNode::Handle>> tensorSizes;
    std::priority_queue<std::pair<uint64_t, StateNode::Handle>> blobSizes;
    uint64_t                                                    totalBlobSize   = 0;
    uint64_t                                                    totalTensorSize = 0;

    registry->WalkNodes([&](StateNode& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            throw std::runtime_error("LoadStateRegistryFromPath: unexpected TensorShard in global registry");
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            auto  size = tnsr.FullDimSizeBytes();
            totalTensorSize += size;
            tensorSizes.emplace(size, node.handle);
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::Blob))
        {
            auto& blob = std::get<StateNode::Blob>(node.data);
            auto  size = blob.size();
            totalBlobSize += size;
            blobSizes.emplace(size, node.handle);
        }
    });

    // uint32_t ranks = ranksPerNode * nodes;
    //
    // uint64_t maxAvailableRam             = 64ull * 1024 * 1024 * 1024;
    // uint64_t targetProcessRamUtilization = maxAvailableRam * 3 / 4;
    uint64_t snapshots       = 4;
    auto     maxSnapshotSize = ctx.memLimit / snapshots;
    auto     maxBlobSize     = maxSnapshotSize / 32;
    auto     maxTensorSize   = maxSnapshotSize - maxBlobSize;

    // Keep halving the largest dimension of the largest tensor until we are below the limit.
    while (totalTensorSize > maxTensorSize && !tensorSizes.empty())
    {
        auto [size, handle] = tensorSizes.top();
        tensorSizes.pop();
        registry->AssertShardedTensorHasNoHoles(handle);
        auto& tnsr     = std::get<StateNode::ShardedTensor>(registry->GetNode(handle).data);
        auto  olddim   = tnsr.fulldim;
        auto* maxDimIt = std::ranges::max_element(tnsr.fulldim);
        auto  index    = static_cast<size_t>(std::distance(tnsr.fulldim.begin(), maxDimIt));
        if (*maxDimIt <= 1) { throw std::runtime_error("LoadStateRegistryFromPath: ShardedTensor has no dimension > 1 to halve"); }
        *maxDimIt /= 2;
        std::set<StateNode::ShardedTensor::RankedTensorShard> oldShards{};
        std::swap(tnsr.shards, oldShards);
        for (auto const& shard : oldShards)
        {
            auto newShard = shard;
            newShard.sharddim[index].end /= 2;
            newShard.sharddim[index].begin /= 2;
            tnsr.AddShard(newShard);
        }
        registry->AssertShardedTensorHasNoHoles(handle);
        auto newSize    = tnsr.FullDimSizeBytes();
        totalTensorSize = totalTensorSize - size + newSize;
        fmt::print(stderr, "fqn={}: dim={}->{} size={}->{}\n", registry->GetNodeId(handle), olddim, tnsr.fulldim, size, newSize);
        tensorSizes.emplace(newSize, handle);
    }

    while (totalBlobSize > maxBlobSize && !blobSizes.empty())
    {
        auto [size, handle] = blobSizes.top();
        blobSizes.pop();
        auto& blob = std::get<StateNode::Blob>(registry->GetNode(handle).data);
        blob.resize(size / 2);
        totalBlobSize = totalBlobSize - size + blob.size();
        fmt::print(stderr, "fqn={}: blob size={}->{}\n", registry->GetNodeId(handle), size, blob.size());
        blobSizes.emplace(blob.size(), handle);
    }

    registry->CloneInto(*ctx.globalRegistry);
    if (!ifs.eof())
    {
        auto currentPos = ifs.tellg();
        ifs.seekg(0, std::ios::end);
        auto extraBytes = ifs.tellg() - currentPos;
        if (extraBytes > 0)
        {
            throw std::runtime_error(fmt::format("Failed to read entire state registry file (extra data at end: {} bytes)", extraBytes));
        }
    }
    ctx.worldSize = InferWorldSize(*ctx.globalRegistry);

    if (ctx.worldSize == Rank{0} || static_cast<uint32_t>(ctx.worldSize) % ctx.ranksPerNode != 0)
    {
        throw std::runtime_error(fmt::format("LoadStateRegistryFromPath: worldSize={} is not a multiple of ranksPerNode={}",
                                             static_cast<uint32_t>(ctx.worldSize),
                                             ctx.ranksPerNode));
    }

    auto const requestedNodes = static_cast<uint16_t>(static_cast<uint32_t>(ctx.worldSize) / ctx.ranksPerNode);

    if (!ctx.outofproc) { RunInprocServer(requestedNodes); }
    std::vector<std::string> peers;
    peers.reserve(requestedNodes);
    for (uint16_t i = 0; i < requestedNodes; ++i) { peers.emplace_back(fmt::format("localhost:{}", cipc::Message::Port + i)); }
    std::vector<char> buffer;
    for (uint32_t r = 0; r < static_cast<uint32_t>(ctx.worldSize); ++r)
    {
        RankTopology topo{
            .rank = Rank{r}, .worldSize = ctx.worldSize, .ranksPerPeer = ctx.ranksPerNode, .peersPerNode = requestedNodes, .peers = peers};

        ctx.localRegistries.insert(BuildPerRankStateTree(ctx, std::move(topo), buffer));
    }
}

void ValidateRoundTrip(StateTree const& localState, ByteStorage const& tensorStorage)
{
    auto localRank = localState.Topo().rank;
    localState.WalkNodes([&](auto& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            auto& tnsr          = std::get<StateNode::TensorShard>(node.data);
            auto* data          = &tensorStorage.At<char>(tnsr.handle);    // NOLINT
            auto  size          = tnsr.ShardSizeBytes();
            auto  shardFqn      = localState.GetNodeId(node.handle);
            auto  actualPattern = std::string_view(static_cast<char const*>(data), std::min(size, static_cast<uint64_t>(shardFqn.size())));
            if (actualPattern != shardFqn)
            {
                throw std::runtime_error(fmt::format("[ERROR] Validation failed for shard {} at offset {} rank = {}. Got: {}\n",
                                                     shardFqn,
                                                     tnsr.handle,
                                                     localRank,
                                                     actualPattern));
            }
        }
        else if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                auto* data          = &tensorStorage.At<char>(shard.handle);    // NOLINT
                auto  shardFqn      = fmt::format("{}[{}]", localState.GetNodeId(node.handle), shard.sharddim);
                auto  size          = std::min(shard.ShardSizeBytes(tnsr.dtype), static_cast<uint64_t>(shardFqn.size()));
                auto  actualPattern = std::string_view(static_cast<char const*>(data), size);
                if (actualPattern != std::string_view(shardFqn.data(), size))
                {
                    throw std::runtime_error(fmt::format("[ERROR] Validation failed for shard {} at offset {} rank = {}. Got: {}\n",
                                                         shardFqn,
                                                         shard.handle,
                                                         localRank,
                                                         actualPattern));
                }
            }
        }
    });
}

void SaveCheckpointOnly(StateTree& tree, std::filesystem::path const& ckptDir)
{
    static constexpr uint32_t NumSnapshots = 5;
    for (uint32_t i = 1; i <= NumSnapshots; i++)
    {
        SaveCheckpoint(tree, cipc::SnapshotId{i}, ckptDir);
        WaitForCheckpointSave(tree.Topo());
    }
}

static void RoundTripCallback(void*                             cbData,
                              cipc::SessionId                   sessionId,
                              cipc::SnapshotId                  snapshotId,
                              std::string_view const&           name,
                              std::filesystem::path const&      path,
                              [[maybe_unused]] StateTree const& globalRegistry,
                              StateTree const&                  localState,
                              ByteStorage const&                tensorStorage)
{
    auto localRank = localState.Topo().rank;
    if (localRank != *static_cast<Rank*>(cbData)) { return; }
    fmt::print(
        stderr, "Validating sessionId={} snapshotId={} name={} path={} rank={}\n", sessionId, snapshotId, name, path.string(), localRank);
    ValidateRoundTrip(localState, tensorStorage);
}

void SaveCheckpointAndValidate(StateTree& tree, std::filesystem::path const& ckptDir)
{
    SaveCheckpointOnly(tree, ckptDir);
    auto rank = tree.Topo().rank;
    QueryServerSessions(&RoundTripCallback, &rank);
}

void RunCheckpointClient(CheckpointTestContext& ctx, StateTree& tree)
{
    FinalizeStateRegistry(tree);
    while (!ctx.stopRequested.load(std::memory_order_acquire))
    {
        FixtureFunc           func{nullptr};
        std::filesystem::path ckptDir;
        {
            std::unique_lock<std::mutex> lock(ctx.mutex);
            if (ctx.fixtureFunc == nullptr)
            {
                ctx.cv.wait(lock);
                continue;
            }
            func    = ctx.fixtureFunc;
            ckptDir = ctx.ckptDir;
        }
        {
            std::exception_ptr workerExc;
            try
            {
                func(tree, ckptDir);
            } catch (...) { workerExc = std::current_exception(); }
            {
                std::unique_lock<std::mutex> lock(ctx.mutex);
                if (workerExc && !ctx.workerException) { ctx.workerException = workerExc; }
                ctx.completedRanks++;
                if (ctx.completedRanks == static_cast<uint32_t>(ctx.worldSize))
                {
                    ctx.completedRanks = 0;
                    ctx.fixtureFunc    = nullptr;
                    ctx.cv.notify_all();
                }
            }
            while (true)
            {
                std::unique_lock<std::mutex> lock(ctx.mutex);
                if (ctx.fixtureFunc == nullptr) { break; }
                ctx.cv.wait(lock);
            }
        }
    }
}

std::vector<std::thread> StartClientThreads(CheckpointTestContext& ctx)
{
    std::vector<std::thread> tasks;
    tasks.reserve(ctx.localRegistries.size());
    for (auto const& tree : ctx.localRegistries)
    {
        tasks.emplace_back([&ctx, treeptr = tree.get()]() { RunCheckpointClient(ctx, *treeptr); });
    }
    return tasks;
}

void JoinClientThreads(CheckpointTestContext& ctx, std::vector<std::thread>& tasks)
{
    {
        std::unique_lock<std::mutex> lock(ctx.mutex);
        ctx.stopRequested.store(true, std::memory_order_release);
        ctx.cv.notify_all();
    }
    for (auto& task : tasks)
    {
        if (task.joinable()) { task.join(); }
    }
}

void RunOnAllRanks(CheckpointTestContext& ctx, FixtureFunc func, std::filesystem::path const& ckptDir)
{
    std::exception_ptr capturedExc;
    {
        std::unique_lock<std::mutex> lock(ctx.mutex);
        ctx.workerException = nullptr;
        ctx.completedRanks  = 0;
        ctx.fixtureFunc     = func;
        ctx.ckptDir         = ckptDir;
        ctx.cv.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(ctx.mutex);
        ctx.cv.wait(lock, [&] { return ctx.fixtureFunc == nullptr; });
        capturedExc         = ctx.workerException;
        ctx.workerException = nullptr;
    }
    if (capturedExc) { std::rethrow_exception(capturedExc); }
}
