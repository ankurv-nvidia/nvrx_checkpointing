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
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cipc = checkpoint::ipc;

static uint64_t NativeShardSize(Tensor::DType dtype, Tensor::DimRange const& sharddim)
{
    if (sharddim.empty()) { return 0; }
    uint64_t size = Tensor::GetDTypeSize(dtype);
    for (auto const& range : sharddim)
    {
        if (range.begin >= range.end) { return 0; }
        size *= range.end - range.begin;
    }
    return size;
}

static uint64_t TensorStorageSize(DTensor const& tensor)
{
    return std::accumulate(tensor.shards.begin(), tensor.shards.end(), uint64_t{0}, [](uint64_t size, TensorShard const& shard) {
        return size + shard.storageSize;
    });
}

static bool CoversFullTensor(DTensor const& tensor, TensorShard const& shard)
{
    if (tensor.fulldim.size() != shard.sharddim.size()) { return false; }
    for (size_t dim = 0; dim < tensor.fulldim.size(); ++dim)
    {
        if (shard.sharddim[dim].begin != 0 || shard.sharddim[dim].end != tensor.fulldim[dim]) { return false; }
    }
    return true;
}

Rank InferWorldSize(StateTree const& tree)
{
    uint maxRankPlus1 = 1;
    tree.WalkNodes([&](StateNode const& node) {
        if (node.data.index() != static_cast<size_t>(StateNode::Type::DTensor)) { return; }
        auto const& tnsr = std::get<DTensor>(node.data);
        for (auto const& rankShard : tnsr.shards) { maxRankPlus1 = std::max(maxRankPlus1, static_cast<uint32_t>(rankShard.rank) + 1u); }
        tree.AssertShardedTensorHasNoHoles(node.handle);
    });
    return Rank{maxRankPlus1};
}

static std::unique_ptr<StateTree> BuildPerRankStateTree(CheckpointTestContext& ctx, RankTopology topo, std::vector<char>& buffer)
{
    std::unique_ptr<StateTree> tree = std::make_unique<StateTree>(checkpoint::ipc::SessionId::Default, std::move(topo));
    ctx.globalRegistry->CloneInto(*tree);
    std::unordered_map<DeviceId, std::unique_ptr<DeviceContext>> devCtxMap;

    size_t totalAllocSize = 0;

    auto calculateSize = [&](DTensor::Shard const& tnsr, std::string_view /* shardFqn */) {
        totalAllocSize += static_cast<size_t>(tnsr.StorageSizeBytes());
    };

    size_t currentOffset = 0;
    auto   allocAndCopy  = [&](DTensor::Shard& tnsr, [[maybe_unused]] std::string_view shardFqn) {
        tnsr.handle = 0;
        if (tnsr.StorageSizeBytes() == 0) { return; }
        auto it = devCtxMap.find(tnsr.device);
        if (it == devCtxMap.end())
        {
            it = devCtxMap.insert({tnsr.device, DeviceContext::CreateDeviceContext(tnsr.device, 0)}).first;
            it->second->StreamAcquire();
        }

        auto& devCtx  = *it->second;
        auto  bufSize = static_cast<size_t>(tnsr.StorageSizeBytes());
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
            if (node.data.index() == static_cast<size_t>(StateNode::Type::Tensor))
            {
                auto& tnsr  = std::get<Tensor>(node.data);
                auto  shard = DTensor::Shard{
                    .rank          = topo.rank,
                    .snapshotDtype = tnsr.snapshotDtype,
                    .handle        = tnsr.handle,
                    .storageSize   = tnsr.storageSize,
                    .sharddim      = {},
                    .device        = tnsr.device,
                };
                execFn(shard, tree->GetNodeId(node.handle));
                tnsr.handle = shard.handle;
            }
            else if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
            {
                auto&   tnsr = std::get<DTensor>(node.data);
                DTensor newTnsr{tnsr.durableDtype, tnsr.fulldim};
                for (auto rshard : tnsr.shards)
                {
                    if (tree->Topo().rank != rshard.rank) { continue; }
                    execFn(rshard, fmt::format("{}[{}]", tree->GetNodeId(node.handle), rshard.sharddim));
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

    for (auto& [deviceId, devCtx] : devCtxMap) { devCtx->Flush(); }

    tree->WalkNodes([&](StateNode& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
        {
            auto& shTen = std::get<DTensor>(node.data);
            if (shTen.shards.size() == 1 && CoversFullTensor(shTen, *shTen.shards.begin()))
            {
                auto const& rshard    = *shTen.shards.begin();
                Tensor      shardTnsr = {
                    .durableDtype  = shTen.durableDtype,
                    .snapshotDtype = rshard.snapshotDtype,
                    .handle        = rshard.handle,
                    .storageSize   = rshard.storageSize,
                    .fulldim       = shTen.fulldim,
                    .device        = rshard.device,
                };
                node.data = shardTnsr;
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
        checkpoint::ipc::SessionId::Default,
        RankTopology{.rank = Rank{0}, .worldSize = Rank{1}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {"localhost"}});
    auto registry = StateSerDes::Read(ifs);

    // Adjust tensors sizes till we are below the limit.
    std::priority_queue<std::pair<uint64_t, StateNode::Handle>> tensorSizes;
    std::priority_queue<std::pair<uint64_t, StateNode::Handle>> blobSizes;
    uint64_t                                                    totalBlobSize   = 0;
    uint64_t                                                    totalTensorSize = 0;

    registry->WalkNodes([&](StateNode& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::Tensor))
        {
            throw std::runtime_error("LoadStateRegistryFromPath: unexpected TensorShard in global registry");
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
        {
            auto& tnsr = std::get<DTensor>(node.data);
            auto  size = TensorStorageSize(tnsr);
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
        auto& tnsr     = std::get<DTensor>(registry->GetNode(handle).data);
        auto  olddim   = tnsr.fulldim;
        auto* maxDimIt = std::ranges::max_element(tnsr.fulldim);
        auto  index    = static_cast<size_t>(std::distance(tnsr.fulldim.begin(), maxDimIt));
        if (*maxDimIt <= 1) { throw std::runtime_error("LoadStateRegistryFromPath: DTensor has no dimension > 1 to halve"); }
        for (auto const& shard : tnsr.shards)
        {
            if (shard.storageSize != NativeShardSize(tnsr.durableDtype, shard.sharddim))
            {
                throw std::runtime_error(
                    fmt::format("LoadStateRegistryFromPath: cannot resize tensor {} with opaque storage", registry->GetNodeId(handle)));
            }
        }
        *maxDimIt /= 2;
        std::set<DTensor::Shard> oldShards{};
        std::swap(tnsr.shards, oldShards);
        for (auto const& shard : oldShards)
        {
            auto newShard = shard;
            newShard.sharddim[index].end /= 2;
            newShard.sharddim[index].begin /= 2;
            newShard.storageSize = NativeShardSize(tnsr.durableDtype, newShard.sharddim);
            if (newShard.storageSize == 0) { continue; }
            tnsr.AddShard(newShard);
        }
        registry->AssertShardedTensorHasNoHoles(handle);
        auto newSize    = TensorStorageSize(tnsr);
        totalTensorSize = totalTensorSize - size + newSize;
        fmt::print(stderr, "fqn={}: dim={}->{} size={}->{}\n", registry->GetNodeId(handle), olddim, tnsr.fulldim, size, newSize);
        tensorSizes.emplace(newSize, handle);
    }

    while (totalBlobSize > maxBlobSize && !blobSizes.empty())
    {
        auto [size, handle] = blobSizes.top();
        blobSizes.pop();
        auto&      blob    = std::get<StateNode::Blob>(registry->GetNode(handle).data);
        auto const newSize = size / 2;
        blob.resize(newSize);
        totalBlobSize = totalBlobSize - size + newSize;
        fmt::print(stderr, "fqn={}: blob size={}->{}\n", registry->GetNodeId(handle), size, newSize);
        blobSizes.emplace(newSize, handle);
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
            .rank         = Rank{r},
            .worldSize    = ctx.worldSize,
            .ranksPerPeer = ctx.ranksPerNode,
            .peersPerNode = requestedNodes,
            .peers        = peers,
        };

        ctx.localRegistries.insert(BuildPerRankStateTree(ctx, std::move(topo), buffer));
    }
}

static void ValidateRoundTrip(StateTree const& localState, ByteStorage const& tensorStorage)
{
    auto localRank = localState.Topo().rank;
    localState.WalkNodes([&](auto& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::Tensor))
        {
            auto& tnsr          = std::get<Tensor>(node.data);
            auto* data          = &tensorStorage.At<char>(tnsr.handle);    // NOLINT
            auto  size          = tnsr.StorageSizeBytes();
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
        else if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
        {
            auto& tnsr = std::get<DTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                auto* data          = &tensorStorage.At<char>(shard.handle);    // NOLINT
                auto  shardFqn      = fmt::format("{}[{}]", localState.GetNodeId(node.handle), shard.sharddim);
                auto  size          = std::min(shard.StorageSizeBytes(), static_cast<uint64_t>(shardFqn.size()));
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
        WaitForCheckpointSave(tree);
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
        tasks.emplace_back([&ctx, treeptr = tree.get()] { RunCheckpointClient(ctx, *treeptr); });
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
