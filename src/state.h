#pragma once
#include "CommonMacros.h"

#include "fmttensor.h"    // IWYU pragma: keep
#include "messages.h"
#include "primitives64bit.h"
#include "tensor.h"

#include <fmt/std.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

// #define NATIVE_CKPT_TRACE_NODE "chained_0.optimizer.distributed.dp_group_idx_0"
// #define NATIVE_CKPT_WARN_FQN

struct CheckpointWriteItem
{
    std::string fqn;
    Tensor::Dim offsets{};
    uint64_t    sizeBytes{};
    bool        isTensor{false};

    bool operator==(CheckpointWriteItem const&) const = default;
};

struct StateNode
{
    enum class Handle : uint32_t
    {
        Root    = 0,
        Invalid = std::numeric_limits<uint32_t>::max(),
    };

    using Dict = std::vector<Handle>;

    struct Scalar : Primitives64Bit
    {
        using Primitives64Bit::Primitives64Bit;
        Scalar() = default;
        explicit Scalar(Primitives64Bit const& value) : Primitives64Bit(value) {}

        std::set<Rank> replicas;

        bool operator==(Scalar const& other) const = default;
    };

    struct DynamicSizeBlob
    {
        std::set<Rank> replicas;
        uint64_t       handle{};
        uint64_t       storageSize{};

        bool operator==(DynamicSizeBlob const& other) const = default;
    };

    struct FixedSizeBlob : DTensor
    {
        using DTensor::DTensor;
        FixedSizeBlob() = default;
        explicit FixedSizeBlob(DTensor const& metadata) : DTensor(metadata) {}
    };

    static_assert(sizeof(FixedSizeBlob) == sizeof(DTensor));

    enum class Type : uint8_t
    {
        Invalid         = 0,
        Dict            = 1,    // Data comprises of nested State
        Scalar          = 2,    // 64bit data
        DynamicSizeBlob = 3,
        DTensor         = 4,
        FixedSizeBlob   = 5,
        Unknown         = 6,
        Count           = Unknown,
    };

    using Variant = std::variant<std::monostate, Dict, Scalar, DynamicSizeBlob, DTensor, FixedSizeBlob>;
    static_assert(std::is_same_v<std::monostate, std::variant_alternative_t<static_cast<size_t>(Type::Invalid), Variant>>);
    static_assert(std::is_same_v<Dict, std::variant_alternative_t<static_cast<size_t>(Type::Dict), Variant>>);
    static_assert(std::is_same_v<Scalar, std::variant_alternative_t<static_cast<size_t>(Type::Scalar), Variant>>);
    static_assert(std::is_same_v<DynamicSizeBlob, std::variant_alternative_t<static_cast<size_t>(Type::DynamicSizeBlob), Variant>>);
    static_assert(std::is_same_v<DTensor, std::variant_alternative_t<static_cast<size_t>(Type::DTensor), Variant>>);
    static_assert(std::is_same_v<FixedSizeBlob, std::variant_alternative_t<static_cast<size_t>(Type::FixedSizeBlob), Variant>>);

    [[nodiscard]] Type GetType() const { return static_cast<Type>(data.index()); }

    StateNode()  = default;
    ~StateNode() = default;
    StateNode(Handle handleIn, Handle parentIn, std::string_view nameIn, Variant&& dataIn) :
        handle(handleIn), parent(parentIn), name(nameIn), data(std::move(dataIn))
    {}

    CLASS_DELETE_COPY_DEFAULT_MOVE(StateNode);

    Handle handle{};    // root handle is always 0; This should be non-zero
    Handle parent{};

    std::string name;
    Variant     data;
};

template <> struct fmt::formatter<StateNode::Handle> : fmt::formatter<std::string_view>
{
    auto format(StateNode::Handle const& handle, fmt::format_context& ctx) const    // NOLINT
    {
        if (handle == StateNode::Handle::Invalid) { return fmt::format_to(ctx.out(), "?handle?"); }
        return fmt::format_to(ctx.out(), "{}", static_cast<uint32_t>(handle));
    }
};

struct StateTree
{
    using NodeId = std::string;

    explicit StateTree(checkpoint::ipc::SessionId sessionId, RankTopology topo) : _sessionId(sessionId), _topo(std::move(topo))
    {
#ifndef NDEBUG
        _topo.AssertValid();
#endif
        _nodes.emplace_back(StateNode::Handle::Root, StateNode::Handle::Root, "", StateNode::Dict{});    //
    }
    ~StateTree() = default;
    CLASS_DELETE_COPY_AND_MOVE(StateTree);

    auto& FindOrAddDictChild(StateNode::Handle node, std::string_view const& stateName) LFTBND
    {
        AssertMutable_();
        auto& parent = _nodes[static_cast<size_t>(node)].data;
        if (parent.index() != static_cast<size_t>(StateNode::Type::Dict)) { throw std::runtime_error("Node data is not of type Dict"); }
        for (auto childHandle : std::get<StateNode::Dict>(parent))
        {
            auto& child = _nodes[static_cast<size_t>(childHandle)];
            if (child.name == stateName) { return child; }
        }
        auto newHandle = static_cast<StateNode::Handle>(_nodes.size());
        std::get<StateNode::Dict>(parent).push_back(newHandle);
        // WARNING: push to nodes can invalidate all the other variables here due to a memmove
        _nodes.emplace_back(newHandle, node, std::string(stateName), StateNode::Dict{});
        // Do not reuse any above variables
        return _nodes.back();
    }

    auto& AddChild(StateNode::Handle node, std::string_view const& stateName, [[maybe_unused]] bool allowDuplicates = true) LFTBND
    {
        AssertMutable_();
        auto& parent = _nodes[static_cast<size_t>(node)].data;
        if (parent.index() != static_cast<size_t>(StateNode::Type::Dict)) { throw std::runtime_error("Node data is not of type Dict"); }
#ifndef NDEBUG
        if (!allowDuplicates)
        {
            for (auto childHandle : std::get<StateNode::Dict>(parent))
            {
                auto const& child = _nodes[static_cast<size_t>(childHandle)];
                if (child.name == stateName)
                {
                    fmt::print(stderr, "Duplicate state name {} found in root registry. Parent Node: {} \n", stateName, GetNodeId(node));
                    throw std::runtime_error(fmt::format("Duplicate state name {} found in root registry", stateName));
                }
            }
        }
#endif
        auto newHandle = static_cast<StateNode::Handle>(_nodes.size());
        std::get<StateNode::Dict>(parent).push_back(newHandle);
        // WARNING: push to nodes can invalidate all the other variables here due to a memmove
        _nodes.push_back({newHandle, node, stateName, {}});
        // Do not reuse any above variables
        return _nodes.back();
    }

    auto FindOrAddNodeByFQN(std::string_view const& fqn)
    {
#ifdef NATIVE_CKPT_TRACE_NODE
        if (fqn == NATIVE_CKPT_TRACE_NODE)
        {
            fmt::print(stderr, "FindOrAddNodeByFQN: Found trace node NATIVE_CKPT_TRACE_NODE: {}\n", NATIVE_CKPT_TRACE_NODE);
        }
#endif
        auto const* itb  = fqn.begin();
        auto const* ite  = std::find(itb, fqn.end(), '.');
        auto        node = StateNode::Handle::Root;    // start from root
        while (ite != fqn.end())
        {
            auto  childName = std::string_view(itb, ite);
            auto& child     = FindOrAddDictChild(node, childName);
            node            = child.handle;
            itb             = ite + 1;    // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            ite             = std::find(itb, fqn.end(), '.');
        }
        return std::make_pair(std::string_view(itb, ite), node);
    }

    auto FindParentDictNodeByFQN(std::string_view const& fqn)
    {
        auto const separator = fqn.rfind('.');
        if (separator == std::string_view::npos) { return std::make_pair(fqn, StateNode::Handle::Root); }

        auto const parentPath   = fqn.substr(0, separator);
        auto const parentHandle = FindNodeByFQN(parentPath);
        if (parentHandle == StateNode::Handle::Invalid)
        {
            throw std::runtime_error(fmt::format("State '{}' is not registered", parentPath));
        }
        if (GetNode(parentHandle).data.index() != static_cast<size_t>(StateNode::Type::Dict))
        {
            throw std::runtime_error(fmt::format("State '{}' is not a Dict", parentPath));
        }
        return std::make_pair(fqn.substr(separator + 1), parentHandle);
    }

    auto FindChildNodeIfExists(StateNode::Handle parent, std::string_view const& name)
    {
        auto& parentData = GetNode(parent).data;
        if (parentData.index() != static_cast<size_t>(StateNode::Type::Dict)) { throw std::runtime_error("Node data is not of type Dict"); }
        auto const& children = std::get<StateNode::Dict>(parentData);
        auto const child = std::ranges::find_if(children, [&](StateNode::Handle childHandle) { return GetNode(childHandle).name == name; });
        if (child == children.end()) { return std::make_pair(false, StateNode::Handle::Invalid); }
        return std::make_pair(true, *child);
    }

    StateNode& FindChildNode(StateNode::Handle parent, std::string_view const& name) LFTBND
    {
        auto& parentData = GetNode(parent).data;
        if (parentData.index() != static_cast<size_t>(StateNode::Type::Dict)) { throw std::runtime_error("Node data is not of type Dict"); }
        auto const& children = std::get<StateNode::Dict>(parentData);
        auto const child = std::ranges::find_if(children, [&](StateNode::Handle childHandle) { return GetNode(childHandle).name == name; });
        if (child == children.end())
        {
            auto const parentPath = GetNodeId(parent);
            auto const nodePath   = parentPath.empty() ? std::string(name) : fmt::format("{}.{}", parentPath, name);
            throw std::runtime_error(fmt::format("State '{}' is not registered", nodePath));
        }
        return GetNode(*child);
    }

    [[nodiscard]] StateNode::Handle FindNodeByFQN(std::string_view const& fqn) const
    {
        if (fqn.empty()) { return StateNode::Handle::Root; }

        auto        node = StateNode::Handle::Root;
        auto const* itb  = fqn.begin();
        while (itb != fqn.end())
        {
            auto const* const ite       = std::find(itb, fqn.end(), '.');
            auto const        childName = std::string_view(itb, ite);
            auto const&       parent    = GetNode(node).data;
            if (parent.index() != static_cast<size_t>(StateNode::Type::Dict)) { return StateNode::Handle::Invalid; }

            auto const& children = std::get<StateNode::Dict>(parent);
            auto const  childIt
                = std::ranges::find_if(children, [&](StateNode::Handle childHandle) { return GetNode(childHandle).name == childName; });
            if (childIt == children.end()) { return StateNode::Handle::Invalid; }
            node = *childIt;
            itb  = ite == fqn.end() ? ite : ite + 1;    // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
        return node;
    }

    void SetBlobSource(std::string_view const& fqn, uint64_t address, uint64_t size)
    {
        AssertMutable_();
        auto const handle = FindNodeByFQN(fqn);
        if (handle == StateNode::Handle::Invalid) { throw std::runtime_error(fmt::format("State '{}' is not registered", fqn)); }
        auto& node = GetNode(handle);
        if (node.data.index() != static_cast<size_t>(StateNode::Type::DynamicSizeBlob))
        {
            throw std::runtime_error(fmt::format("State '{}' is not a Blob", fqn));
        }
        auto& blob       = std::get<StateNode::DynamicSizeBlob>(node.data);
        blob.handle      = address;
        blob.storageSize = size;
    }

    void SetScalar(std::string_view const& fqn, StateNode::Scalar const& data)
    {
        AssertMutable_();
        auto const handle = FindNodeByFQN(fqn);
        if (handle == StateNode::Handle::Invalid) { throw std::runtime_error(fmt::format("State '{}' is not registered", fqn)); }
        auto& node = GetNode(handle);
        if (node.data.index() != static_cast<size_t>(StateNode::Type::Scalar))
        {
            throw std::runtime_error(fmt::format("State '{}' is not a Scalar", fqn));
        }
        node.data = data;
    }

    StateNode& GetNode(StateNode::Handle handle) LFTBND { return _nodes[static_cast<size_t>(handle)]; }

    [[nodiscard]] auto SessionId() const { return _sessionId; }

    [[nodiscard]] StateNode const& GetNode(StateNode::Handle handle) const LFTBND { return _nodes[static_cast<size_t>(handle)]; }
    [[nodiscard]] auto             GlobalRank() const { return _topo.rank; }
    [[nodiscard]] auto             GlobalWorldSize() const { return _topo.worldSize; }

    [[nodiscard]] std::string GetNodeId(StateNode::Handle nodeHandle) const
    {
        if (nodeHandle == StateNode::Handle::Root) { return ""; }
        auto const& node = GetNode(nodeHandle);
        if (node.parent == StateNode::Handle::Root) { return node.name; }
        return GetNodeId(node.parent) + "." + node.name;
    }

    template <typename TCallback> void ForEachPayload(TCallback const& callback) const
    {
        WalkNodes([&](StateNode const& node) {
            if (node.GetType() != StateNode::Type::DTensor) { return; }
            for (auto const& shard : std::get<DTensor>(node.data).shards)
            {
                callback(node.handle, shard.snapshotDtype, shard.device, shard.handle, shard.storageSize);
            }
        });
        WalkNodes([&](StateNode const& node) {
            if (node.GetType() != StateNode::Type::FixedSizeBlob) { return; }
            for (auto const& shard : std::get<StateNode::FixedSizeBlob>(node.data).shards)
            {
                callback(node.handle, Tensor::DType::UInt8, DeviceId("cpu"), shard.handle, shard.storageSize);
            }
        });
        WalkNodes([&](StateNode const& node) {
            if (node.GetType() != StateNode::Type::DynamicSizeBlob) { return; }
            auto const& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            if (blob.replicas.size() > 1) { throw std::runtime_error("A local blob payload has multiple replica ranks"); }
            callback(node.handle, Tensor::DType::UInt8, DeviceId("cpu"), blob.handle, blob.storageSize);
        });
    }

    [[nodiscard]] uint64_t GetTotalTensorStateSizeInBytes() const
    {
        uint64_t totalSize = 0;
        for (auto const& node : _nodes)
        {
            if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
            {
                auto const& tensor = std::get<DTensor>(node.data);
                for (auto const& shard : tensor.shards) { totalSize += shard.StorageSizeBytes(); }
            }
        }
        return totalSize;
    }

    [[nodiscard]] uint64_t GetTotalSnapshotStateSizeInBytes() const
    {
        auto totalSize = GetTotalTensorStateSizeInBytes();
        for (auto const& node : _nodes)
        {
            if (node.data.index() == static_cast<size_t>(StateNode::Type::DynamicSizeBlob))
            {
                auto const blobSize = std::get<StateNode::DynamicSizeBlob>(node.data).storageSize;
                if (blobSize > std::numeric_limits<uint64_t>::max() - totalSize)
                {
                    throw std::overflow_error("Snapshot state size overflow");
                }
                totalSize += blobSize;
            }
            else if (node.data.index() == static_cast<size_t>(StateNode::Type::FixedSizeBlob))
            {
                for (auto const& shard : std::get<StateNode::FixedSizeBlob>(node.data).shards)
                {
                    if (shard.storageSize > std::numeric_limits<uint64_t>::max() - totalSize)
                    {
                        throw std::overflow_error("Snapshot state size overflow");
                    }
                    totalSize += shard.storageSize;
                }
            }
        }
        return totalSize;
    }

    [[nodiscard]] uint64_t GetTotalPayloadSizeInBytes() const { return GetTotalSnapshotStateSizeInBytes(); }

    [[nodiscard]] size_t GetTotalNumTensors() const
    {
        size_t count = 0;
        for (auto const& node : _nodes)
        {
            if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor)) { count += std::get<DTensor>(node.data).shards.size(); }
        }
        return count;
    }

    [[nodiscard]] std::string FindTensorStateByHandle(uint64_t handle) const;

    void PrintShardedTensor(StateNode::Handle nodeHandle) const
    {
        auto        fqn  = GetNodeId(nodeHandle);
        auto const& tnsr = std::get<DTensor>(GetNode(nodeHandle).data);
        fmt::print(stderr,
                   "[native_ckpt:cli][{}][WARN] sharded tensor {}[{}] holes={}\n\t{}\n",
                   GlobalRank(),
                   tnsr.fulldim,
                   fqn,
                   fmt::join(tnsr.holes, ","),
                   fmt::join(tnsr.shards, "\n\t"));
    }

    void AssertShardedTensorHasNoHoles(StateNode::Handle nodeHandle) const
    {
        auto const& node = GetNode(nodeHandle);
        auto const& tnsr = std::get<DTensor>(node.data);
        if (tnsr.shards.empty())
        {
            throw std::runtime_error(fmt::format("[native_ckpt:cli][{}] sharded tensor {}[{}] has no shards in metadata output\n",
                                                 GlobalRank(),
                                                 GetNodeId(nodeHandle),
                                                 tnsr.fulldim));
        }
        if (!tnsr.holes.empty())
        {
            PrintShardedTensor(nodeHandle);
            throw std::runtime_error(fmt::format("[native_ckpt:cli][{}] sharded tensor {}[{}] has holes in metadata output\n",
                                                 GlobalRank(),
                                                 GetNodeId(nodeHandle),
                                                 tnsr.fulldim));
        }
    }

    template <typename TLambda> void WalkNodes(TLambda const& lambda)
    {
        AssertMutable_();
        for (auto& node : _nodes)
        {
            // TODO_PURGE_INVALID
            if (node.handle == StateNode::Handle::Invalid) { continue; }
            lambda(node);
        }
    }

    template <typename TLambda> void WalkNodes(TLambda const& lambda) const
    {
        for (auto const& node : _nodes)
        {
            // TODO_PURGE_INVALID
            if (node.handle == StateNode::Handle::Invalid) { continue; }
            lambda(node);
        }
    }

    void FinalizeRegistry()
    {
        if (_finalized) { return; }
        auto& root = _nodes[0];
        try
        {
            SortDictionaryNode_(root);
        } catch (std::exception const& ex)
        {
            fmt::print(stderr, "Error finalizing registry: {}\n", ex.what());
            std::unordered_set<StateNode::Handle> visited;
            DebugPrintRecr_(0, root.handle, visited);
            for (auto const& node : _nodes)
            {
                if (visited.contains(node.handle)) { continue; }
                fmt::print(stderr,
                           "Unvisited node : [{}]{} Parent Node: [{}]{}\n",
                           node.handle,
                           node.name,
                           static_cast<uint32_t>(node.parent),
                           GetNodeId(node.parent));
            }
            throw;
        }
        _finalized = true;
    }

    [[nodiscard]] bool IsFinalized() const { return _finalized; }

    void CloneInto(StateTree& clone) const
    {
        clone.AssertMutable_();
        clone._nodes.reserve(_nodes.size());
        clone._nodes.clear();
        for (auto const& node : _nodes) { clone._nodes.emplace_back(node.handle, node.parent, node.name, StateNode::Variant{node.data}); }
        clone.startedAt            = startedAt;
        clone.commonStateKey       = commonStateKey;
        clone.checkpointWriteItems = checkpointWriteItems;
        clone._finalized           = false;
    }

    [[nodiscard]] auto Clone() const
    {
        auto clonedTree = std::make_shared<StateTree>(_sessionId, _topo);
        CloneInto(*clonedTree);
        clonedTree->_finalized = _finalized;
        return clonedTree;
    }

    void RemoveNode(StateNode::Handle nodeHandle)
    {
        AssertMutable_();
        if (nodeHandle == StateNode::Handle::Root) { throw std::runtime_error("Cannot remove root node"); }
        auto& node       = GetNode(nodeHandle);
        auto  parent     = node.parent;
        auto& parentNode = GetNode(parent);
        if (parentNode.data.index() != static_cast<size_t>(StateNode::Type::Dict))
        {
            throw std::runtime_error(fmt::format("Parent node is not of type dict. Node : [{}]{} Parent Node: [{}]{} \n",
                                                 node.handle,
                                                 node.name,
                                                 parentNode.handle,
                                                 parentNode.name));
        }
        auto& parentChildren = std::get<StateNode::Dict>(parentNode.data);
        auto  e              = std::ranges::remove(parentChildren, node.handle);
        parentChildren.erase(e.begin(), e.end());
        node.parent = StateNode::Handle::Invalid;
        node.handle = StateNode::Handle::Invalid;
        node.data   = std::monostate{};
        node.name   = {};
    }

    [[nodiscard]] auto const& Topo() const LFTBND { return _topo; }

    private:
    void AssertMutable_() const
    {
        if (_finalized) { throw std::logic_error("Finalized state trees are immutable"); }
    }

    void DebugPrintRecr_(size_t depth, StateNode::Handle nodeHandle, std::unordered_set<StateNode::Handle>& visited) const
    {
        if (visited.contains(nodeHandle))
        {
            fmt::print(stderr,
                       "Duplicate child : [{}]{} Parent Node: [{}]{}\n",
                       nodeHandle,
                       GetNode(nodeHandle).name,
                       GetNode(nodeHandle).parent,
                       GetNodeId(GetNode(nodeHandle).parent));
        }
        visited.insert(nodeHandle);
        auto const& node = GetNode(nodeHandle);
        if (node.handle != StateNode::Handle::Root)
        {
            auto const& parent = GetNode(node.parent);
            if (parent.data.index() != static_cast<size_t>(StateNode::Type::Dict))
            {
                fmt::print(stderr,
                           "Parent node is not of type dict. Node : [{}]{} Parent Node: [{}]{} \n",
                           node.handle,
                           node.name,
                           parent.handle,
                           parent.name);
            }
            auto parentChildren = std::get<StateNode::Dict>(parent.data);
            if (!std::ranges::any_of(parentChildren, [&](auto childHandle) { return childHandle == node.handle; }))
            {
                fmt::print(stderr,
                           "Node : [{}]{} is not a child of its parent Node: [{}]{} \n",
                           node.handle,
                           node.name,
                           parent.handle,
                           parent.name);
            }
        }

        fmt::print(stderr, "{}- {} (handle: {}, type: {})\n", std::string(depth * 2, ' '), node.name, node.handle, node.data.index());
        if (node.data.index() == static_cast<size_t>(StateNode::Type::Dict))
        {
            auto const& children = std::get<StateNode::Dict>(node.data);
            for (auto child : children) { DebugPrintRecr_(depth + 1, child, visited); }
        }
    }

    // TODO_PURGE_INVALID
    [[maybe_unused]] void PurgeInvalidNodes_()
    {
        for (auto it = _nodes.rbegin(); it != _nodes.rend();)
        {
            if (it->handle == StateNode::Handle::Invalid)
            {
                it = decltype(it){_nodes.erase(std::next(it).base())};
                for (auto it1 = _nodes.rbegin(); it1 != _nodes.rend(); ++it1)
                {
                    it->handle = static_cast<StateNode::Handle>(std::distance(it1, _nodes.rend() - 1));
                    if (it->parent > it->handle) { it->parent = static_cast<StateNode::Handle>(static_cast<uint32_t>(it->parent) - 1); }
                    if (it->data.index() == static_cast<size_t>(StateNode::Type::Dict))
                    {
                        auto& children = std::get<StateNode::Dict>(it->data);
                        for (auto& child : children)
                        {
                            if (child > it->handle) { child = static_cast<StateNode::Handle>(static_cast<uint32_t>(child) - 1); }
                        }
                    }
                }
            }
            else
            {
                ++it;
            }
        }
    }

    [[maybe_unused]] void SortDictionaryNode_(StateNode& node)    // NOLINT
    {
        assert(node.data.index() == static_cast<size_t>(StateNode::Type::Dict));
        auto& children = std::get<StateNode::Dict>(node.data);
        for (size_t i = 0; i < children.size(); ++i)
        {
            auto& childI = GetNode(children[i]);
            assert(GetNode(childI.parent).handle == node.handle);
            if (childI.data.index() == static_cast<size_t>(StateNode::Type::Invalid)) { continue; }
            for (size_t j = i + 1; j < children.size(); ++j)
            {
                auto& childJ = GetNode(children[j]);
                auto  cmp    = childI.name.compare(childJ.name);
                if (childJ.data.index() == static_cast<size_t>(StateNode::Type::Invalid)) { continue; }
                if (cmp == 0)
                {
                    fmt::print(
                        stderr, "Duplicate state name {} found in root registry. Parent Node: {} \n", childI.name, GetNodeId(node.handle));
                    throw std::runtime_error(fmt::format("Duplicate state name {} found in root registry", childI.name));
                }
                if (cmp > 0)
                {
                    // Do not swap the handles. We swap the node contents
                    // That way we make sure the node positions in flattened node list are sorted as well
                    // std::swap(children[i], children[j]);
                    std::swap(childI, childJ);
                    childI.handle = children[i];
                    childJ.handle = children[j];
                    // Adjust the parent index in case of swap
                    if (childI.data.index() == static_cast<size_t>(StateNode::Type::Dict))
                    {
                        auto& dict = std::get<StateNode::Dict>(childI.data);
                        for (auto& handle : dict) { GetNode(handle).parent = childI.handle; }
                        SortDictionaryNode_(childI);
                    }
                    if (childJ.data.index() == static_cast<size_t>(StateNode::Type::Dict))
                    {
                        auto& dict = std::get<StateNode::Dict>(childJ.data);
                        for (auto& handle : dict) { GetNode(handle).parent = childJ.handle; }
                        SortDictionaryNode_(childJ);
                    }
                }
            }
        }
    }

    std::vector<StateNode> _nodes;

    bool _finalized{false};

    checkpoint::ipc::SessionId _sessionId{};
    RankTopology const         _topo{};

    friend struct StateSerDes;

    public:
    std::string                                      commonStateKey;
    std::map<Rank, std::vector<CheckpointWriteItem>> checkpointWriteItems;

    // TODO: Better time tracking.
    std::chrono::nanoseconds startedAt /*NOLINT*/
        = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
};

template <> struct fmt::formatter<StateNode::Variant> : fmt::formatter<std::string>
{
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    auto format(StateNode::Variant const& var, fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        switch (var.index())
        {
        case static_cast<size_t>(StateNode::Type::Invalid):
        {
            return fmt::format_to(ctx.out(), "node:invalid");
        }
        case static_cast<size_t>(StateNode::Type::Dict):
        {
            return fmt::format_to(ctx.out(), "node:dict children={}", std::get<StateNode::Dict>(var).size());
        }
        case static_cast<size_t>(StateNode::Type::Scalar):
        {
            auto scalar = std::get<StateNode::Scalar>(var);
            auto t      = scalar.GetType();
            if (StateNode::Scalar::Type::IsFloat(t)) { return fmt::format_to(ctx.out(), "node:scalar {}", scalar.Cast<double>()); }
            if (StateNode::Scalar::Type::IsSigned(t)) { return fmt::format_to(ctx.out(), "node:scalar {}", scalar.Cast<int64_t>()); }
            if (StateNode::Scalar::Type::IsUnsigned(t)) { return fmt::format_to(ctx.out(), "node:scalar {}", scalar.Cast<uint64_t>()); }
            return fmt::format_to(ctx.out(), "node:scalar {}", scalar.Cast<bool>());
        }
        case static_cast<size_t>(StateNode::Type::DynamicSizeBlob): return fmt::format_to(ctx.out(), "node:dynamicblob");
        case static_cast<size_t>(StateNode::Type::FixedSizeBlob):
            return fmt::format_to(ctx.out(), "node:fixedblob {}", static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(var)));
        case static_cast<size_t>(StateNode::Type::DTensor):
            return fmt::format_to(ctx.out(), "node:shardedtensor {}", std::get<DTensor>(var));
        default: break;
        }
        throw std::runtime_error(fmt::format("Unknown state node type: {}", var.index()));
    }
};
