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

std::unique_ptr<StateTree> BuildPerRankStateTree(CheckpointTestContext& ctx, RankTopology topo)
{
    std::unique_ptr<StateTree> tree = std::make_unique<StateTree>(std::move(topo));
    ctx.globalRegistry->CloneInto(*tree);
    std::unordered_map<DeviceId, std::unique_ptr<DeviceContext>> devCtxMap;

    auto allocAndCopy = [&](DS4Tensor& tnsr, std::string_view shardFqn) {
        tnsr.handle = 0;
        if (tnsr.ShardSizeBytes() == 0) { return; }
        auto it = devCtxMap.find(tnsr.device);
        if (it == devCtxMap.end()) { it = devCtxMap.insert({tnsr.device, DeviceContext::CreateDeviceContext(tnsr.device, 0)}).first; }
        auto&             devCtx  = *it->second;
        auto              devAddr = devCtx.Allocate(tnsr.ShardSizeBytes());
        auto              bufSize = static_cast<size_t>(tnsr.ShardSizeBytes());
        auto              patLen  = shardFqn.size();
        std::vector<char> pattern(bufSize);
        std::memcpy(pattern.data(), shardFqn.data(), std::min(shardFqn.size(), bufSize));
        for (size_t filled = std::min(patLen, bufSize); filled < bufSize;)
        {
            auto chunk = std::min(filled, bufSize - filled);
            std::memcpy(&pattern[filled], pattern.data(), chunk);
            filled += chunk;
        }
        devCtx.StreamAcquire();
        devCtx.AddH2D(pattern.data(), devAddr, pattern.size());
        devCtx.Flush();
        tnsr.handle = devAddr;
    };

    tree->WalkNodes([&](StateNode& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            allocAndCopy(std::get<StateNode::TensorShard>(node.data), tree->GetNodeId(node.handle));
        }
        else if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
        {
            auto& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            tree->AssertShardedTensorHasNoHoles(node.handle);
            StateNode::ShardedTensor newTnsr{tnsr.dtype, tnsr.fulldim};
            for (auto rshard : tnsr.shards)
            {
                if (tree->Topo().rank != rshard.rank) { continue; }
                DS4Tensor shardTnsr = {.sharddim = rshard.sharddim, .fulldim = tnsr.fulldim, .dtype = tnsr.dtype, .device = rshard.device};
                allocAndCopy(shardTnsr, fmt::format("{}[{}]", tree->GetNodeId(node.handle), rshard.sharddim));
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
    return tree;
}

void LoadStateRegistryFromPath(CheckpointTestContext& ctx, std::filesystem::path const& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { throw std::runtime_error(fmt::format("Failed to open state registry: {}", path.string())); }
    ctx.globalRegistry = std::make_unique<StateTree>(
        RankTopology{.rank = Rank{0}, .worldSize = Rank{1}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {"localhost"}});
    auto registry = StateSerDes::Read(ifs);
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

    ctx.nodes = std::min(ctx.nodes, static_cast<uint16_t>(ctx.worldSize));

    if (!ctx.outofproc) { RunInprocServer(ctx.nodes); }
    std::vector<std::string> peers;
    peers.reserve(ctx.nodes);
    for (uint16_t i = 0; i < ctx.nodes; ++i) { peers.emplace_back(fmt::format("localhost:{}", cipc::Message::Port + i)); }

    for (uint32_t r = 0; r < static_cast<uint32_t>(ctx.worldSize); ++r)
    {
        RankTopology topo{.rank         = Rank{r},
                          .worldSize    = ctx.worldSize,
                          .ranksPerPeer = static_cast<uint32_t>(ctx.worldSize) / ctx.nodes,
                          .peersPerNode = ctx.nodes,
                          .peers        = peers};
        ctx.localRegistries.insert(BuildPerRankStateTree(ctx, std::move(topo)));
    }
}

void ValidateRoundTrip(StateTree const& localState, Blob const& blob)
{
    auto localRank = localState.Topo().rank;
    localState.WalkNodes([&](auto& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            auto& tnsr          = std::get<StateNode::TensorShard>(node.data);
            auto* data          = &blob.At<char>(tnsr.handle);    // NOLINT
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
                auto* data          = &blob.At<char>(shard.handle);    // NOLINT
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
    SaveCheckpoint(tree, cipc::SnapshotId{1}, ckptDir);
    WaitForCheckpointSave(tree.Topo());
}

namespace
{
void RoundTripCallback(void*                             cbData,
                       cipc::SessionId                   sessionId,
                       cipc::SnapshotId                  snapshotId,
                       std::string_view const&           name,
                       std::filesystem::path const&      path,
                       [[maybe_unused]] StateTree const& globalRegistry,
                       StateTree const&                  localState,
                       Blob const&                       blob)
{
    auto localRank = localState.Topo().rank;
    if (localRank != *static_cast<Rank*>(cbData)) { return; }
    fmt::print(
        stderr, "Validating sessionId={} snapshotId={} name={} path={} rank={}\n", sessionId, snapshotId, name, path.string(), localRank);
    ValidateRoundTrip(localState, blob);
}
}    // namespace

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
