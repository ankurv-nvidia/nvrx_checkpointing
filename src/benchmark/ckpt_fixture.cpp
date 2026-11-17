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

Rank InferWorldSize(StateTree const& tree)
{
    uint maxRankPlus1 = 1;
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::Scalar)
        {
            for (auto rank : std::get<StateNode::Scalar>(node.data).replicas)
            {
                maxRankPlus1 = std::max(maxRankPlus1, static_cast<uint32_t>(rank) + 1u);
            }
            return;
        }
        if (type == StateNode::Type::DynamicSizeBlob)
        {
            for (auto rank : std::get<StateNode::DynamicSizeBlob>(node.data).replicas)
            {
                maxRankPlus1 = std::max(maxRankPlus1, static_cast<uint32_t>(rank) + 1u);
            }
            return;
        }
        if (type != StateNode::Type::DTensor && type != StateNode::Type::FixedSizeBlob) { return; }
        auto const& tnsr = type == StateNode::Type::DTensor ? std::get<DTensor>(node.data)
                                                            : static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data));
        for (auto const& rankShard : tnsr.shards) { maxRankPlus1 = std::max(maxRankPlus1, static_cast<uint32_t>(rankShard.rank) + 1u); }
        if (!tnsr.holes.empty()) { throw std::runtime_error("State registry contains incomplete shard coverage"); }
    });
    return Rank{maxRankPlus1};
}

static std::unique_ptr<StateTree> BuildPerRankStateTree(CheckpointTestContext& ctx, RankTopology topo)
{
    std::unique_ptr<StateTree> tree = std::make_unique<StateTree>(checkpoint::ipc::SessionId::Default, std::move(topo));
    ctx.globalRegistry->CloneInto(*tree);
    tree->WalkNodes([&](StateNode& node) {
        auto const type = node.GetType();
        if (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob)
        {
            auto&   tensor = type == StateNode::Type::FixedSizeBlob ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                                                    : std::get<DTensor>(node.data);
            DTensor local{tensor.durableDtype, tensor.fulldim};
            for (auto shard : tensor.shards)
            {
                if (tree->Topo().rank != shard.rank) { continue; }
                shard.handle = 0;
                local.AddShard(shard);
            }
            if (local.shards.empty()) { tree->RemoveNode(node.handle); }
            else
            {
                tensor = std::move(local);
            }
        }
        else if (type == StateNode::Type::DynamicSizeBlob)
        {
            auto& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            if (!blob.replicas.contains(tree->Topo().rank)) { tree->RemoveNode(node.handle); }
            else
            {
                blob.replicas    = {tree->Topo().rank};
                blob.handle      = 0;
                blob.storageSize = 0;
            }
        }
        else if (type == StateNode::Type::Scalar)
        {
            auto& scalar = std::get<StateNode::Scalar>(node.data);
            if (!scalar.replicas.contains(tree->Topo().rank)) { tree->RemoveNode(node.handle); }
            else
            {
                scalar.replicas = {tree->Topo().rank};
            }
        }
    });
    return tree;
}

static void FillSnapshotRegistry(StateTree& tree, StateTree const& globalRegistry, std::vector<std::vector<char>>& blobStorage)
{
    std::unordered_map<DeviceId, std::unique_ptr<DeviceContext>> devCtxMap;

    size_t totalAllocSize = 0;
    std::as_const(tree).WalkNodes([&](StateNode const& node) {
        if (node.GetType() == StateNode::Type::DTensor) { totalAllocSize += TensorStorageSize(std::get<DTensor>(node.data)); }
    });
    std::vector<char> buffer(totalAllocSize);

    size_t currentOffset = 0;
    auto   allocAndCopy  = [&](DeviceId device, size_t bufSize, [[maybe_unused]] std::string_view shardFqn) -> uint64_t {
        if (bufSize == 0) { return 0; }
        auto it = devCtxMap.find(device);
        if (it == devCtxMap.end())
        {
            it = devCtxMap.insert({device, DeviceContext::CreateDeviceContext(device, 0)}).first;
            it->second->StreamAcquire();
        }

        auto& devCtx  = *it->second;
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
        return devAddr;
    };
    std::as_const(tree).WalkNodes([&](StateNode const& stored) {
        auto&      node = tree.GetNode(stored.handle);
        auto const type = node.GetType();
        auto const fqn  = tree.GetNodeId(node.handle);
        if (type == StateNode::Type::DTensor || type == StateNode::Type::FixedSizeBlob)
        {
            auto& tensor = type == StateNode::Type::FixedSizeBlob ? static_cast<DTensor&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                                                  : std::get<DTensor>(node.data);
            for (auto it = tensor.shards.begin(); it != tensor.shards.end();)
            {
                auto  entry    = tensor.shards.extract(it++);
                auto& shard    = entry.value();
                auto  identity = fmt::format("{}[{}]", fqn, shard.sharddim);
                if (type == StateNode::Type::FixedSizeBlob)
                {
                    auto& bytes = blobStorage.emplace_back(shard.storageSize);
                    for (size_t index = 0; index < bytes.size(); ++index) { bytes[index] = identity[index % identity.size()]; }
                    shard.handle = reinterpret_cast<uint64_t>(bytes.data());
                    shard.device = DeviceId{"cpu"};
                }
                else
                {
                    shard.handle = allocAndCopy(shard.device, shard.storageSize, identity);
                }
                tensor.shards.insert(std::move(entry));
            }
        }
        else if (type == StateNode::Type::DynamicSizeBlob)
        {
            auto&       blob     = std::get<StateNode::DynamicSizeBlob>(node.data);
            auto const& metadata = std::get<StateNode::DynamicSizeBlob>(globalRegistry.GetNode(globalRegistry.FindNodeByFQN(fqn)).data);
            auto&       bytes    = blobStorage.emplace_back(metadata.storageSize);
            for (size_t index = 0; index < bytes.size(); ++index) { bytes[index] = fqn[index % fqn.size()]; }
            blob.handle      = reinterpret_cast<uint64_t>(bytes.data());
            blob.storageSize = bytes.size();
        }
    });
    for (auto& [deviceId, devCtx] : devCtxMap) { devCtx->Flush(); }
    fmt::print(stderr, "FillSnapshotRegistry: rank {} allocated {} bytes of device memory\n", tree.Topo().rank, totalAllocSize);
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
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
        {
            auto& tnsr = std::get<DTensor>(node.data);
            auto  size = TensorStorageSize(tnsr);
            totalTensorSize += size;
            tensorSizes.emplace(size, node.handle);
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DynamicSizeBlob))
        {
            auto& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            auto  size = blob.storageSize;
            totalBlobSize += size;
            blobSizes.emplace(size, node.handle);
        }
        if (node.data.index() == static_cast<size_t>(StateNode::Type::FixedSizeBlob))
        {
            auto size = TensorStorageSize(std::get<StateNode::FixedSizeBlob>(node.data));
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
        auto const newSize = size / 2;
        auto&      node    = registry->GetNode(handle);
        uint64_t   updatedSize{};
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DynamicSizeBlob))
        {
            auto& blob       = std::get<StateNode::DynamicSizeBlob>(node.data);
            blob.storageSize = newSize;
            updatedSize      = newSize;
        }
        else
        {
            auto&                 blob = std::get<StateNode::FixedSizeBlob>(node.data);
            std::set<TensorShard> reduced;
            for (auto shard : blob.shards)
            {
                shard.storageSize /= 2;
                updatedSize += shard.storageSize;
                reduced.insert(shard);
            }
            blob.shards = std::move(reduced);
        }
        totalBlobSize = totalBlobSize - size + updatedSize;
        fmt::print(stderr, "fqn={}: blob size={}->{}\n", registry->GetNodeId(handle), size, updatedSize);
        blobSizes.emplace(updatedSize, handle);
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
    for (uint32_t r = 0; r < static_cast<uint32_t>(ctx.worldSize); ++r)
    {
        RankTopology topo{
            .rank         = Rank{r},
            .worldSize    = ctx.worldSize,
            .ranksPerPeer = ctx.ranksPerNode,
            .peersPerNode = requestedNodes,
            .peers        = peers,
        };

        ctx.localRegistries.insert(BuildPerRankStateTree(ctx, std::move(topo)));
    }
}

static void ValidateRoundTrip(StateTree const& localState, ByteStorage const& tensorStorage)
{
    auto localRank = localState.Topo().rank;
    localState.WalkNodes([&](auto& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
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
        else if (node.GetType() == StateNode::Type::DynamicSizeBlob)
        {
            auto const& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            if (blob.storageSize == 0) { return; }
            auto const fqn    = localState.GetNodeId(node.handle);
            auto const size   = std::min(blob.storageSize, static_cast<uint64_t>(fqn.size()));
            auto const actual = std::string_view(&tensorStorage.At<char>(blob.handle), size);
            if (actual != std::string_view(fqn.data(), size))
            {
                throw std::runtime_error(fmt::format("Validation failed for blob '{}' on rank {}", fqn, localRank));
            }
        }
    });
}

void SaveCheckpointOnly(StateTree& tree, std::filesystem::path const& ckptDir)
{
    for (uint32_t i = 1; i <= NumSnapshots; i++)
    {
        SaveCheckpoint(tree, cipc::SnapshotId{i}, ckptDir);
        WaitForCheckpointSave(tree);
    }
}

static void RoundTripCallback(void*                        cbData,
                              cipc::SessionId              sessionId,
                              cipc::SnapshotId             snapshotId,
                              std::string_view const&      name,
                              std::filesystem::path const& path,
                              StateTree const&,
                              StateTree const&,
                              StateTree const& snapshotRegistry,
                              StateTree const&,
                              ByteStorage const& snapshotData)
{
    auto localRank = snapshotRegistry.Topo().rank;
    if (localRank != *static_cast<Rank*>(cbData)) { return; }
    fmt::print(
        stderr, "Validating sessionId={} snapshotId={} name={} path={} rank={}\n", sessionId, snapshotId, name, path.string(), localRank);
    ValidateRoundTrip(snapshotRegistry, snapshotData);
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
    auto                           snapshotRegistry = GetSnapshotRegistry(tree);
    std::vector<std::vector<char>> blobStorage;
    FillSnapshotRegistry(*snapshotRegistry, *ctx.globalRegistry, blobStorage);
    while (!ctx.stopRequested.load(std::memory_order_acquire))
    {
        FixtureFunc           func{nullptr};
        std::filesystem::path ckptDir;
        {
            std::unique_lock<std::mutex> lock(ctx.mutex);
            if (ctx.fixtureFunc == nullptr)
            {
                ctx.cv.wait(lock, [&] { return ctx.stopRequested.load(std::memory_order_acquire) || ctx.fixtureFunc != nullptr; });
                continue;
            }
            func    = ctx.fixtureFunc;
            ckptDir = ctx.ckptDir;
        }
        {
            std::exception_ptr workerExc;
            try
            {
                func(*snapshotRegistry, ckptDir);
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
