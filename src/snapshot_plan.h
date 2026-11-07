#pragma once

#include "state.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace checkpoint::planning
{
using NodeType   = StateNode::Type;
using LocalPlans = std::map<Rank, std::shared_ptr<StateTree>>;

inline NodeType TypeOf(StateNode const& node)
{ return static_cast<NodeType>(node.data.index()); }

inline bool IsDistributed(NodeType type)
{ return type == NodeType::DTensor || type == NodeType::FixedSizeBlob; }

inline DTensor const& Distributed(StateNode const& node LFTBND)
{
    if (TypeOf(node) == NodeType::FixedSizeBlob) { return std::get<StateNode::FixedSizeBlob>(node.data); }
    return std::get<DTensor>(node.data);
}

inline DTensor& Distributed(StateNode& node LFTBND)
{
    if (TypeOf(node) == NodeType::FixedSizeBlob) { return std::get<StateNode::FixedSizeBlob>(node.data); }
    return std::get<DTensor>(node.data);
}

inline uint64_t AddSize(uint64_t total, uint64_t size)
{
    if (size > std::numeric_limits<uint64_t>::max() - total) { throw std::overflow_error("Snapshot payload size overflow"); }
    return total + size;
}

struct Payload
{
    StateNode::Handle node{StateNode::Handle::Invalid};
    Rank              rank{Rank::Invalid};
    Tensor::DType     snapshotDtype{Tensor::DType::Invalid};
    DeviceId          device;
    uint64_t          handle{};
    uint64_t          storageSize{};
    Tensor::DimRange  sharddim;
    bool              immediate{};
};

template <typename TCallback> void ForEachPayload(StateTree const& tree, TCallback const& callback)
{
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = TypeOf(node);
        if (type == NodeType::Tensor)
        {
            auto const&      tensor = std::get<Tensor>(node.data);
            Tensor::DimRange ranges;
            for (auto const dim : tensor.fulldim) { ranges.push_back({.begin = 0, .end = dim}); }
            callback(Payload{.node          = node.handle,
                             .rank          = tree.GlobalRank(),
                             .snapshotDtype = tensor.snapshotDtype,
                             .device        = tensor.device,
                             .handle        = tensor.handle,
                             .storageSize   = tensor.storageSize,
                             .sharddim      = ranges});
        }
        else if (IsDistributed(type))
        {
            for (auto const& shard : Distributed(node).shards)
            {
                auto const immediate = type == NodeType::FixedSizeBlob;
                callback(Payload{.node          = node.handle,
                                 .rank          = shard.rank,
                                 .snapshotDtype = immediate ? Tensor::DType::UInt8 : shard.snapshotDtype,
                                 .device        = immediate ? DeviceId("cpu") : shard.device,
                                 .handle        = shard.handle,
                                 .storageSize   = shard.storageSize,
                                 .sharddim      = shard.sharddim,
                                 .immediate     = immediate});
            }
        }
        else if (type == NodeType::DynamicSizeBlob)
        {
            auto const& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            if (blob.replicas.size() > 1) { throw std::runtime_error("A local blob payload has multiple replica ranks"); }
            auto const rank = blob.replicas.empty() ? tree.GlobalRank() : *blob.replicas.begin();
            callback(Payload{.node          = node.handle,
                             .rank          = rank,
                             .snapshotDtype = Tensor::DType::UInt8,
                             .device        = DeviceId("cpu"),
                             .handle        = blob.handle,
                             .storageSize   = blob.storageSize,
                             .sharddim      = {},
                             .immediate     = true});
        }
    });
}

inline uint64_t AssignPayloadOffsets(StateTree& tree)
{
    uint64_t offset = 0;
    tree.WalkNodes([&](StateNode& node) {
        auto const type = TypeOf(node);
        if (type == NodeType::Tensor)
        {
            auto& tensor  = std::get<Tensor>(node.data);
            tensor.handle = offset;
            offset        = AddSize(offset, tensor.storageSize);
        }
        else if (IsDistributed(type))
        {
            auto&                 tensor = Distributed(node);
            std::set<TensorShard> packed;
            for (auto shard : tensor.shards)
            {
                shard.handle = offset;
                offset       = AddSize(offset, shard.storageSize);
                packed.insert(shard);
            }
            tensor.shards = std::move(packed);
        }
        else if (type == NodeType::DynamicSizeBlob)
        {
            auto& blob  = std::get<StateNode::DynamicSizeBlob>(node.data);
            blob.handle = offset;
            offset      = AddSize(offset, blob.storageSize);
        }
    });
    return offset;
}

inline void NormalizeLocal(StateTree& tree)
{
    tree.WalkNodes([&](StateNode& node) {
        auto const type = TypeOf(node);
        if (type == NodeType::Scalar)
        {
            auto& scalar = std::get<StateNode::Scalar>(node.data);
            if (scalar.replicas.empty()) { scalar.replicas.insert(tree.GlobalRank()); }
        }
        else if (type == NodeType::DynamicSizeBlob)
        {
            auto& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            if (blob.replicas.empty()) { blob.replicas.insert(tree.GlobalRank()); }
        }
        else if (type == NodeType::Tensor)
        {
            auto const       tensor = std::get<Tensor>(node.data);
            DTensor          distributed(tensor.durableDtype, tensor.fulldim);
            Tensor::DimRange ranges;
            for (auto const dim : tensor.fulldim) { ranges.push_back({.begin = 0, .end = dim}); }
            distributed.AddShard(TensorShard{.rank          = tree.GlobalRank(),
                                             .snapshotDtype = tensor.snapshotDtype,
                                             .handle        = tensor.handle,
                                             .storageSize   = tensor.storageSize,
                                             .sharddim      = ranges,
                                             .device        = tensor.device});
            node.data = std::move(distributed);
        }
    });
}

inline void ClearValues(StateTree& tree)
{
    tree.WalkNodes([](StateNode& node) {
        auto const type = TypeOf(node);
        if (type == NodeType::Scalar) { std::get<StateNode::Scalar>(node.data).ClearValue(); }
        else if (type == NodeType::DynamicSizeBlob)
        {
            auto& blob       = std::get<StateNode::DynamicSizeBlob>(node.data);
            blob.handle      = 0;
            blob.storageSize = 0;
        }
        else if (type == NodeType::Tensor) { std::get<Tensor>(node.data).handle = 0; }
        else if (IsDistributed(type))
        {
            std::set<TensorShard> cleared;
            for (auto shard : Distributed(node).shards)
            {
                shard.handle = 0;
                cleared.insert(shard);
            }
            Distributed(node).shards = std::move(cleared);
        }
    });
}

inline std::shared_ptr<StateTree> MetadataClone(StateTree const& tree)
{
    auto result = tree.Clone();
    NormalizeLocal(*result);
    ClearValues(*result);
    return result;
}

inline void ValidateRanks(std::set<Rank> const& replicas, Rank worldSize, std::string_view fqn)
{
    if (replicas.empty()) { throw std::runtime_error(fmt::format("State '{}' has no replica ranks", fqn)); }
    for (auto const rank : replicas)
    {
        if (rank >= worldSize) { throw std::runtime_error(fmt::format("State '{}' has rank {} outside the world", fqn, rank)); }
    }
}

inline void ValidateDistributed(DTensor const& tensor, bool blob, Rank worldSize, std::string_view fqn)
{
    if (tensor.shards.empty()) { throw std::runtime_error(fmt::format("State '{}' has no registered shards", fqn)); }
    std::map<Tensor::DimRange, uint64_t> fixedSizes;
    for (auto const& shard : tensor.shards)
    {
        if (shard.rank >= worldSize || shard.sharddim.size() != tensor.fulldim.size())
        {
            throw std::runtime_error(fmt::format("State '{}' has incompatible shard metadata", fqn));
        }
        if (!blob && shard.storageSize == 0) { throw std::runtime_error(fmt::format("Tensor '{}' has empty storage", fqn)); }
        for (size_t dim = 0; dim < tensor.fulldim.size(); ++dim)
        {
            auto const range = shard.sharddim[dim];
            if (range.begin >= range.end || range.end > tensor.fulldim[dim])
            {
                throw std::runtime_error(fmt::format("State '{}' has an invalid shard range", fqn));
            }
            if (blob && range.end - range.begin != 1)
            {
                throw std::runtime_error(fmt::format("Fixed blob '{}' shards contain one logical object", fqn));
            }
        }
        if (blob)
        {
            auto const [entry, inserted] = fixedSizes.emplace(shard.sharddim, shard.storageSize);
            if (!inserted && entry->second != shard.storageSize)
            {
                throw std::runtime_error(fmt::format("Fixed blob '{}' replicas have inconsistent sizes", fqn));
            }
        }
    }
}

inline void ValidateLocalRegistry(StateTree const& tree)
{
    std::set<StateTree::NodeId> names;
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = TypeOf(node);
        if (type == NodeType::Invalid || type == NodeType::Dict) { return; }
        auto const fqn = tree.GetNodeId(node.handle);
        if (!names.insert(fqn).second) { throw std::runtime_error(fmt::format("Local state '{}' is registered more than once", fqn)); }
        if (type == NodeType::Scalar || type == NodeType::DynamicSizeBlob)
        {
            auto const& replicas = type == NodeType::Scalar ? std::get<StateNode::Scalar>(node.data).replicas
                                                            : std::get<StateNode::DynamicSizeBlob>(node.data).replicas;
            if (replicas != std::set<Rank>{tree.GlobalRank()})
            {
                throw std::runtime_error(fmt::format("Local state '{}' has incompatible replica ranks", fqn));
            }
        }
        else if (IsDistributed(type))
        {
            auto const& tensor = Distributed(node);
            ValidateDistributed(tensor, type == NodeType::FixedSizeBlob, tree.GlobalWorldSize(), fqn);
            for (auto const& shard : tensor.shards)
            {
                if (shard.rank != tree.GlobalRank())
                {
                    throw std::runtime_error(fmt::format("Local state '{}' contains another rank's shard", fqn));
                }
            }
        }
        else if (type != NodeType::Tensor) { throw std::runtime_error(fmt::format("Local state '{}' has an unsupported type", fqn)); }
    });
}

inline void MergeRegistryImpl(StateTree& global, StateTree const& incoming, bool metadataOnly)
{
    if (global.SessionId() != incoming.SessionId() || global.GlobalWorldSize() != incoming.GlobalWorldSize()
        || global.Topo().peers != incoming.Topo().peers || global.Topo().ranksPerPeer != incoming.Topo().ranksPerPeer
        || global.Topo().peersPerNode != incoming.Topo().peersPerNode)
    {
        throw std::runtime_error("Cannot merge state registries from incompatible sessions or topologies");
    }
    auto source = incoming.Clone();
    NormalizeLocal(*source);
    ValidateLocalRegistry(*source);
    if (metadataOnly) { ClearValues(*source); }
    source->WalkNodes([&](StateNode const& supplied) {
        auto const type = TypeOf(supplied);
        if (type == NodeType::Invalid || supplied.handle == StateNode::Handle::Root) { return; }
        auto const fqn      = source->GetNodeId(supplied.handle);
        auto const previous = global.FindNodeByFQN(fqn);
        if (previous == StateNode::Handle::Invalid)
        {
            auto const [name, parent] = global.FindOrAddNodeByFQN(fqn);
            if (type == NodeType::Dict) { global.FindOrAddDictChild(parent, name); }
            else
            {
                global.AddChild(parent, name).data = supplied.data;
            }
            return;
        }
        auto& registered = global.GetNode(previous);
        if (TypeOf(registered) != type) { throw std::runtime_error(fmt::format("State '{}' has conflicting registered types", fqn)); }
        if (type == NodeType::Dict) { return; }
        if (type == NodeType::Scalar)
        {
            auto&       target = std::get<StateNode::Scalar>(registered.data);
            auto const& value  = std::get<StateNode::Scalar>(supplied.data);
            if (target.GetType() != value.GetType()) { throw std::runtime_error(fmt::format("Scalar '{}' has conflicting types", fqn)); }
            if (!metadataOnly && target.replicas != value.replicas)
            {
                throw std::runtime_error(fmt::format("Scalar '{}' has multiple snapshot owners", fqn));
            }
            target.replicas.insert(value.replicas.begin(), value.replicas.end());
        }
        else if (type == NodeType::DynamicSizeBlob)
        {
            auto&       target = std::get<StateNode::DynamicSizeBlob>(registered.data);
            auto const& value  = std::get<StateNode::DynamicSizeBlob>(supplied.data);
            if (!metadataOnly && target.replicas != value.replicas)
            {
                throw std::runtime_error(fmt::format("Dynamic blob '{}' has multiple snapshot owners", fqn));
            }
            target.replicas.insert(value.replicas.begin(), value.replicas.end());
        }
        else if (IsDistributed(type))
        {
            auto&       target = Distributed(registered);
            auto const& value  = Distributed(supplied);
            if (target.durableDtype != value.durableDtype || target.fulldim != value.fulldim)
            {
                throw std::runtime_error(fmt::format("State '{}' has incompatible dtype or global shape", fqn));
            }
            for (auto const& shard : value.shards)
            {
                auto const existing = target.shards.find(shard);
                if (existing != target.shards.end())
                {
                    if (*existing != shard) { throw std::runtime_error(fmt::format("State '{}' has conflicting shard metadata", fqn)); }
                    continue;
                }
                target.AddShard(shard, type == NodeType::FixedSizeBlob);
            }
            ValidateDistributed(target, type == NodeType::FixedSizeBlob, global.GlobalWorldSize(), fqn);
        }
    });
}

inline void MergeRegistry(StateTree& global, StateTree const& incoming)
{ MergeRegistryImpl(global, incoming, true); }
inline void MergeSnapshot(StateTree& global, StateTree const& snapshot)
{ MergeRegistryImpl(global, snapshot, false); }

struct Assignment
{
    StateNode::Handle node{StateNode::Handle::Invalid};
    NodeType          type{NodeType::Invalid};
    std::string       fqn;
    Tensor::DimRange  ranges;
    uint64_t          size{};
    std::vector<Rank> candidates;
    Rank              owner{Rank::Invalid};
};

inline std::vector<Assignment> ComputeAssignments(StateTree const& global)
{
    auto const worldSize = static_cast<uint32_t>(global.GlobalWorldSize());
    if (worldSize == 0) { throw std::runtime_error("Cannot plan a zero-rank state registry"); }
    std::vector<Assignment> assignments;
    global.WalkNodes([&](StateNode const& node) {
        auto const type = TypeOf(node);
        if (type == NodeType::Dict || type == NodeType::Invalid) { return; }
        auto const fqn = global.GetNodeId(node.handle);
        if (type == NodeType::Scalar || type == NodeType::DynamicSizeBlob)
        {
            auto const& replicas = type == NodeType::Scalar ? std::get<StateNode::Scalar>(node.data).replicas
                                                            : std::get<StateNode::DynamicSizeBlob>(node.data).replicas;
            ValidateRanks(replicas, global.GlobalWorldSize(), fqn);
            assignments.push_back(Assignment{.node       = node.handle,
                                             .type       = type,
                                             .fqn        = fqn,
                                             .ranges     = {},
                                             .size       = type == NodeType::Scalar ? sizeof(Primitives64Bit) : 0,
                                             .candidates = {replicas.begin(), replicas.end()}});
            return;
        }
        if (!IsDistributed(type)) { throw std::runtime_error(fmt::format("State '{}' is not normalized for planning", fqn)); }
        auto const& tensor = Distributed(node);
        ValidateDistributed(tensor, type == NodeType::FixedSizeBlob, global.GlobalWorldSize(), fqn);
        if (!tensor.holes.empty()) { throw std::runtime_error(fmt::format("State '{}' has incomplete global shard coverage", fqn)); }
        std::map<Tensor::DimRange, std::vector<TensorShard const*>> groups;
        for (auto const& shard : tensor.shards) { groups[shard.sharddim].push_back(&shard); }
        for (auto const& [ranges, copies] : groups)
        {
            Assignment entry{
                .node = node.handle, .type = type, .fqn = fqn, .ranges = ranges, .size = copies.front()->storageSize, .candidates = {}};
            for (auto const* copy : copies) { entry.candidates.push_back(copy->rank); }
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
        item.owner           = selected;
        auto const index     = static_cast<uint32_t>(selected);
        assignedBytes[index] = AddSize(assignedBytes[index], item.size);
    }
    return assignments;
}

inline LocalPlans BuildLocalPlans(StateTree const& source, uint32_t firstRank, uint32_t rankCount)
{
    auto       global    = MetadataClone(source);
    auto const worldSize = static_cast<uint32_t>(global->GlobalWorldSize());
    if (firstRank > worldSize || rankCount > worldSize - firstRank)
    {
        throw std::runtime_error("Snapshot plan rank range exceeds the world");
    }
    auto const assignments = ComputeAssignments(*global);
    LocalPlans plans;
    for (uint32_t index = 0; index < rankCount; ++index)
    {
        auto topology = global->Topo();
        topology.rank = Rank{firstRank + index};
        plans.emplace(topology.rank, std::make_shared<StateTree>(global->SessionId(), std::move(topology)));
    }
    std::map<StateNode::Handle, std::vector<Assignment const*>> byNode;
    for (auto const& item : assignments)
    {
        if (plans.contains(item.owner)) { byNode[item.node].push_back(&item); }
    }
    global->WalkNodes([&](StateNode const& node) {
        auto const entries = byNode.find(node.handle);
        if (entries == byNode.end()) { return; }
        std::map<Rank, std::vector<Assignment const*>> byRank;
        for (auto const* entry : entries->second) { byRank[entry->owner].push_back(entry); }
        for (auto const& [rank, selected] : byRank)
        {
            auto&      plan           = *plans.at(rank);
            auto const fqn            = global->GetNodeId(node.handle);
            auto const [name, parent] = plan.FindOrAddNodeByFQN(fqn);
            auto& output              = plan.AddChild(parent, name);
            output.data               = node.data;
            if (TypeOf(node) == NodeType::Scalar) { std::get<StateNode::Scalar>(output.data).replicas = {rank}; }
            else if (TypeOf(node) == NodeType::DynamicSizeBlob) { std::get<StateNode::DynamicSizeBlob>(output.data).replicas = {rank}; }
            else
            {
                auto const& metadata = Distributed(node);
                DTensor     chosen(metadata.durableDtype, metadata.fulldim);
                for (auto const* entry : selected)
                {
                    auto const sourceShard = std::ranges::find_if(
                        metadata.shards, [&](TensorShard const& shard) { return shard.rank == rank && shard.sharddim == entry->ranges; });
                    chosen.AddShard(*sourceShard, TypeOf(node) == NodeType::FixedSizeBlob);
                }
                if (TypeOf(node) == NodeType::FixedSizeBlob) { output.data = StateNode::FixedSizeBlob(chosen); }
                else
                {
                    output.data = std::move(chosen);
                }
            }
        }
    });
    for (auto const& [rank, plan] : plans)
    {
        static_cast<void>(rank);
        plan->FinalizeRegistry();
    }
    return plans;
}

inline void ValidateLayout(StateNode const& planned, StateNode const& supplied, std::string_view fqn)
{
    auto const type = TypeOf(planned);
    if (type != TypeOf(supplied)) { throw std::runtime_error(fmt::format("Snapshot state '{}' changed type", fqn)); }
    if (type == NodeType::Scalar)
    {
        auto const& expected = std::get<StateNode::Scalar>(planned.data);
        auto const& actual   = std::get<StateNode::Scalar>(supplied.data);
        if (expected.GetType() != actual.GetType() || expected.replicas != actual.replicas)
        {
            throw std::runtime_error(fmt::format("Snapshot scalar '{}' changed type or owner", fqn));
        }
    }
    else if (type == NodeType::DynamicSizeBlob)
    {
        if (std::get<StateNode::DynamicSizeBlob>(planned.data).replicas != std::get<StateNode::DynamicSizeBlob>(supplied.data).replicas)
        {
            throw std::runtime_error(fmt::format("Snapshot blob '{}' changed owner", fqn));
        }
    }
    else if (IsDistributed(type))
    {
        auto const& expected = Distributed(planned);
        auto const& actual   = Distributed(supplied);
        if (expected.durableDtype != actual.durableDtype || expected.fulldim != actual.fulldim
            || expected.shards.size() != actual.shards.size())
        {
            throw std::runtime_error(fmt::format("Snapshot state '{}' changed tensor metadata", fqn));
        }
        for (auto const& shard : expected.shards)
        {
            auto const found = actual.shards.find(shard);
            if (found == actual.shards.end() || found->storageSize != shard.storageSize || found->snapshotDtype != shard.snapshotDtype)
            {
                throw std::runtime_error(fmt::format("Snapshot state '{}' changed shard metadata", fqn));
            }
        }
    }
}

inline std::shared_ptr<StateTree> BindSnapshot(StateTree const& plan, StateTree const& currentLocal)
{
    if (plan.SessionId() != currentLocal.SessionId() || plan.GlobalRank() != currentLocal.GlobalRank()
        || plan.GlobalWorldSize() != currentLocal.GlobalWorldSize())
    {
        throw std::runtime_error("Snapshot plan and local registry identify different sessions or ranks");
    }
    auto source = currentLocal.Clone();
    NormalizeLocal(*source);
    ValidateLocalRegistry(*source);
    auto snapshot       = plan.Clone();
    snapshot->startedAt = currentLocal.startedAt;
    snapshot->WalkNodes([&](StateNode& target) {
        auto const type = TypeOf(target);
        if (type == NodeType::Dict || type == NodeType::Invalid) { return; }
        auto const fqn            = snapshot->GetNodeId(target.handle);
        auto const suppliedHandle = source->FindNodeByFQN(fqn);
        if (suppliedHandle == StateNode::Handle::Invalid)
        {
            throw std::runtime_error(fmt::format("Snapshot state '{}' is missing locally", fqn));
        }
        auto const& supplied = source->GetNode(suppliedHandle);
        if (TypeOf(supplied) != type) { throw std::runtime_error(fmt::format("Snapshot state '{}' changed type", fqn)); }
        if (type == NodeType::Scalar)
        {
            auto const expected = std::get<StateNode::Scalar>(target.data);
            auto       value    = std::get<StateNode::Scalar>(supplied.data);
            value.replicas      = expected.replicas;
            target.data         = std::move(value);
        }
        else if (type == NodeType::DynamicSizeBlob)
        {
            auto const ranks = std::get<StateNode::DynamicSizeBlob>(target.data).replicas;
            auto       value = std::get<StateNode::DynamicSizeBlob>(supplied.data);
            value.replicas   = ranks;
            target.data      = std::move(value);
        }
        else if (IsDistributed(type))
        {
            auto const& expected = Distributed(target);
            auto const& actual   = Distributed(supplied);
            if (expected.durableDtype != actual.durableDtype || expected.fulldim != actual.fulldim)
            {
                throw std::runtime_error(fmt::format("Snapshot state '{}' changed global metadata", fqn));
            }
            DTensor selected(expected.durableDtype, expected.fulldim);
            for (auto const& shard : expected.shards)
            {
                auto const found = actual.shards.find(shard);
                if (found == actual.shards.end() || found->storageSize != shard.storageSize || found->snapshotDtype != shard.snapshotDtype)
                {
                    throw std::runtime_error(fmt::format("Snapshot state '{}' changed shard metadata", fqn));
                }
                selected.AddShard(*found, type == NodeType::FixedSizeBlob);
            }
            if (type == NodeType::FixedSizeBlob) { target.data = StateNode::FixedSizeBlob(selected); }
            else
            {
                target.data = std::move(selected);
            }
        }
    });
    plan.WalkNodes([&](StateNode const& node) {
        auto const fqn    = plan.GetNodeId(node.handle);
        auto const handle = snapshot->FindNodeByFQN(fqn);
        ValidateLayout(node, snapshot->GetNode(handle), fqn);
    });
    ForEachPayload(*snapshot, [&](Payload const& payload) {
        if (payload.storageSize != 0 && payload.handle == 0)
        {
            throw std::runtime_error(fmt::format("Snapshot state '{}' has no current source data", snapshot->GetNodeId(payload.node)));
        }
    });
    return snapshot;
}

inline void ValidateSnapshot(StateTree const& plan, StateTree const& snapshot)
{
    if (plan.SessionId() != snapshot.SessionId() || plan.GlobalRank() != snapshot.GlobalRank()
        || plan.GlobalWorldSize() != snapshot.GlobalWorldSize())
    {
        throw std::runtime_error("Snapshot metadata identifies an incompatible session or rank");
    }
    ValidateLocalRegistry(plan);
    ValidateLocalRegistry(snapshot);
    size_t expectedCount = 0;
    plan.WalkNodes([&](StateNode const& node) {
        if (TypeOf(node) == NodeType::Dict || TypeOf(node) == NodeType::Invalid) { return; }
        ++expectedCount;
        auto const fqn    = plan.GetNodeId(node.handle);
        auto const handle = snapshot.FindNodeByFQN(fqn);
        if (handle == StateNode::Handle::Invalid) { throw std::runtime_error(fmt::format("Snapshot state '{}' is missing", fqn)); }
        ValidateLayout(node, snapshot.GetNode(handle), fqn);
    });
    size_t actualCount = 0;
    snapshot.WalkNodes([&](StateNode const& node) {
        if (TypeOf(node) != NodeType::Dict && TypeOf(node) != NodeType::Invalid) { ++actualCount; }
    });
    if (expectedCount != actualCount) { throw std::runtime_error("Snapshot contains unplanned state values"); }
    uint64_t offset = 0;
    ForEachPayload(snapshot, [&](Payload const& payload) {
        if (payload.handle != offset) { throw std::runtime_error("Snapshot payload offsets do not match the planned packing order"); }
        offset = AddSize(offset, payload.storageSize);
    });
}

inline std::string BlobShardKey(std::string_view key, Tensor::Dim const& fullShape, Tensor::DimRange const& ranges)
{
    std::string result(key);
    result += "/shard_";
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        if (i != 0) { result += '.'; }
        result += std::to_string(ranges[i].begin);
    }
    result += '_';
    for (size_t i = 0; i < fullShape.size(); ++i)
    {
        if (i != 0) { result += '.'; }
        result += std::to_string(fullShape[i]);
    }
    return result;
}
}    // namespace checkpoint::planning
