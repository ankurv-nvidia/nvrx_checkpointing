#pragma once
#include "CommonMacros.h"
#include "fmttensor.h"
#include "primitives64bit.h"
#include "tensor.h"

#include <chrono>
#include <fmt/std.h>

#include <algorithm>
#include <cassert>
#include <future>
#include <mutex>
#include <unordered_set>
#include <variant>
#include <vector>
// #define NATIVE_CKPT_TRACE_NODE "chained_0.optimizer.distributed.dp_group_idx_0"
// #define NATIVE_CKPT_WARN_FQN

struct StateNode
{
    enum class Handle : uint32_t
    {
        Root    = 0,
        Invalid = std::numeric_limits<uint32_t>::max(),
    };

    using Dict        = std::vector<Handle>;
    using Blob        = std::vector<char>;
    using Scalar      = Primitives64Bit;
    using TensorShard = DS4Tensor;

    using ShardedTensor = DS4TensorShardMap;

    enum class Type : uint8_t
    {
        Invalid       = 0,
        Dict          = 1,    // Data comprises of nested State
        Scalar        = 2,    // 64bit data
        Blob          = 3,    // data stored inline. if its a large blob, use tensor
        TensorShard   = 4,    // data is decoupled from metadata and not stored inline
        ShardedTensor = 5,
        Unknown       = 6,
        Count         = Unknown
    };

    using Variant = std::variant<std::monostate, Dict, Scalar, Blob, TensorShard, ShardedTensor>;
    static_assert(std::is_same_v<std::monostate, std::variant_alternative_t<static_cast<size_t>(Type::Invalid), Variant>>);
    static_assert(std::is_same_v<Dict, std::variant_alternative_t<static_cast<size_t>(Type::Dict), Variant>>);
    static_assert(std::is_same_v<Scalar, std::variant_alternative_t<static_cast<size_t>(Type::Scalar), Variant>>);
    static_assert(std::is_same_v<Blob, std::variant_alternative_t<static_cast<size_t>(Type::Blob), Variant>>);
    static_assert(std::is_same_v<TensorShard, std::variant_alternative_t<static_cast<size_t>(Type::TensorShard), Variant>>);
    static_assert(std::is_same_v<ShardedTensor, std::variant_alternative_t<static_cast<size_t>(Type::ShardedTensor), Variant>>);

    StateNode()  = default;
    ~StateNode() = default;
    StateNode(Handle handleIn, Handle parentIn, std::string_view nameIn, Variant&& dataIn) :
        handle(handleIn), parent(parentIn), name(nameIn), data(std::move(dataIn))
    {}

    CLASS_DELETE_COPY_DEFAULT_MOVE(StateNode);

    struct IHandler
    {
        IHandler(std::string_view const& nameIn) : name(nameIn) {}

        virtual ~IHandler() = default;
        CLASS_DELETE_COPY_AND_MOVE(IHandler);
        virtual Type GetType() = 0;

        virtual void               Update() = 0;
        [[nodiscard]] virtual bool IsSnapshotScoped() const { return false; }

        virtual Scalar    GetScalarState() = 0;
        virtual Blob      GetBlob()        = 0;
        virtual DS4Tensor GetTensorState() = 0;

        virtual void SetScalarState(Scalar const&)    = 0;
        virtual void SetBlob(std::string_view const&) = 0;
        virtual void SetTensorState(DS4Tensor const&) = 0;

        std::string name;
    };

    std::unique_ptr<IHandler> handler{nullptr};
    Handle                    handle{};    // root handle is always 0; This should be non-zero
    Handle                    parent{};

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
    struct SnapshotBlob
    {
        std::string fqn;
        std::string data;
    };

    using NodeId        = std::string;
    using SnapshotBlobs = std::vector<SnapshotBlob>;

    StateTree(RankTopology topo) : _topo(std::move(topo))
    {
#ifndef NDEBUG
        _topo.AssertValid();
#endif
        _nodes.emplace_back(StateNode::Handle::Root, StateNode::Handle::Root, "", StateNode::Dict{});    //
    }

    ~StateTree() { WaitForAsyncCalls(); }

    CLASS_DELETE_COPY_AND_MOVE(StateTree);

    using StateType     = StateNode::Type;
    using Scalar        = StateNode::Scalar;
    using IStateHandler = StateNode::IHandler;

    template <StateType T> struct StateHandler;
    // template <> struct StateHandler<StateType::Scalar>;
    // template <> struct StateHandler<StateType::Scalar>;
    // template <> struct StateHandler<StateType::Scalar>;

    auto& FindOrAddDictChild(StateNode::Handle node, std::string_view const& stateName)
    {
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

    auto& AddChild(StateNode::Handle node, std::string_view const& stateName, [[maybe_unused]] bool allowDuplicates = true)
    {
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
        _nodes.push_back({newHandle, node, std::string(stateName), {}});
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

    StateNode::Handle FindNodeByFQN(std::string_view const& fqn) const
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

    void SetBlob(std::string_view const& fqn, std::string_view const& data)
    {
        auto const handle = FindNodeByFQN(fqn);
        if (handle == StateNode::Handle::Invalid)
        {
            // The coordinator gathers only tensor shards across ranks, so a
            // snapshot-scoped common-state fqn may not exist in the merged registry
            // yet (it lives only on rank 0). Materialize it on first set.
            auto [childName, parent] = FindOrAddNodeByFQN(fqn);
            auto& child              = AddChild(parent, childName);
            child.data               = StateNode::Blob(data.begin(), data.end());
            return;
        }
        auto& node = GetNode(handle);
        if (node.data.index() != static_cast<size_t>(StateNode::Type::Blob))
        {
            throw std::runtime_error(fmt::format("State '{}' is not a Blob", fqn));
        }
        node.data = StateNode::Blob(data.begin(), data.end());
    }

    SnapshotBlobs CaptureSnapshotBlobs() const
    {
        SnapshotBlobs result;
        WalkNodes([&](StateNode const& node) {
            if (node.data.index() != static_cast<size_t>(StateNode::Type::Blob) || node.handler == nullptr
                || !node.handler->IsSnapshotScoped())
            {
                return;
            }
            auto const blob = node.handler->GetBlob();
            result.push_back({.fqn = GetNodeId(node.handle), .data = std::string(blob.begin(), blob.end())});
        });
        return result;
    }

    void RegisterStateHandler(StateNode::Handle node, std::unique_ptr<IStateHandler> handler)
    {
        auto const& stateName = handler->name;
        if (stateName.contains('.'))
        {
#ifdef NATIVE_CKPT_WARN_FQN
            auto parentFQN = GetNodeId(node);
            fmt::print(stderr,
                       "[native_ckpt:cli][WARN] StateName contains dots. Treating it as a fqn. Will be reparented to root of state-dict"
                       "This can cause unexpected behavior.\n"
                       "\tparent = {}\n"
                       "\tname   = {}\n",
                       parentFQN,
                       stateName);
#endif
            if (handler->GetType() != StateNode::Type::TensorShard)
            {
                throw std::runtime_error(
                    fmt::format("StateName with dots only allowed for TensorShard type. Invalid stateName: {}", stateName));
            }

            auto [childName, parentNode] = FindOrAddNodeByFQN(stateName);
            // Add the final node which contains the handler
            auto& child   = AddChild(parentNode, childName, true);
            child.handler = std::move(handler);
            child.data    = child.handler->GetTensorState();
            return;
        }

        auto& child   = AddChild(node, stateName, false);
        auto* ptr     = handler.get();
        child.handler = std::move(handler);
        switch (ptr->GetType())
        {
        case StateType::Blob: child.data = ptr->GetBlob(); break;
        case StateType::Scalar: child.data = ptr->GetScalarState(); break;
        case StateType::TensorShard:
        {
            auto tnsr = ptr->GetTensorState();
            //            tnsr.deviceGlobalRank = _topo.rank;
            child.data = tnsr;
            break;
        }
        case StateType::ShardedTensor: [[fallthrough]];    // Not expected to have a handler
        case StateType::Dict: [[fallthrough]];             // child.data = StateNode::Dict{}; break;
        case StateType::Invalid: [[fallthrough]];
        case StateType::Unknown: [[fallthrough]];
        default: throw std::logic_error("Unknown state type");
        }
    }

    StateNode const& GetNode(StateNode::Handle handle) const { return _nodes[static_cast<size_t>(handle)]; }
    StateNode&       GetNode(StateNode::Handle handle) { return _nodes[static_cast<size_t>(handle)]; }

    auto GlobalRank() const { return _topo.rank; }
    auto GlobalWorldSize() const { return _topo.worldSize; }

    std::string GetNodeId(StateNode::Handle nodeHandle) const
    {
        if (nodeHandle == StateNode::Handle::Root) { return ""; }
        auto const& node = GetNode(nodeHandle);
        if (node.parent == StateNode::Handle::Root) { return node.name; }
        return GetNodeId(node.parent) + "." + node.name;
    }

    template <typename TLambda> void ChainAsyncCall(TLambda&& lambda)
    {
        std::scoped_lock lock(_mutex);
        if (_async.valid())
        {
            _async = std::async(std::launch::async | std::launch::deferred,
                                [prev = std::move(_async), lambda1 = std::forward<TLambda>(lambda)]() mutable {
                                    prev.get();
                                    lambda1();
                                });
        }
        else
        {
            _async = std::async(std::launch::async | std::launch::deferred, std::forward<TLambda>(lambda));
        }
    }

    uint64_t GetTotalTensorStateSizeInBytes() const
    {
        uint64_t totalSize = 0;
        for (auto const& node : _nodes)
        {
            if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
            {
                auto const& tensor = std::get<StateNode::TensorShard>(node.data);
                totalSize += tensor.ShardSizeBytes();
            }
            if (node.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
            {
                auto const& tensor = std::get<StateNode::ShardedTensor>(node.data);
                for (auto const& shard : tensor.shards) { totalSize += shard.ShardSizeBytes(tensor.dtype); }
            }
        }
        return totalSize;
    }

    size_t GetTotalNumTensors() const
    {
        return static_cast<size_t>(std::ranges::count_if(
            _nodes, [](StateNode const& node) { return node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard); }));
    }

    std::string FindTensorStateByHandle(uint64_t handle) const;

    void PrintShardedTensor(StateNode::Handle nodeHandle) const
    {
        auto        fqn  = GetNodeId(nodeHandle);
        auto const& tnsr = std::get<StateNode::ShardedTensor>(GetNode(nodeHandle).data);
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
        auto const& tnsr = std::get<StateNode::ShardedTensor>(GetNode(nodeHandle).data);
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

    void ConvertTensorShardsToShardedTensors()
    {
        for (auto& node : _nodes)
        {
#ifdef NATIVE_CKPT_TRACE_NODE
            auto fqn = GetNodeId(node.handle);
            if (fqn == NATIVE_CKPT_TRACE_NODE)
            {
                fmt::print(
                    stderr, "ConvertTensorShardsToShardedTensors: Found trace node NATIVE_CKPT_TRACE_NODE: {}\n", NATIVE_CKPT_TRACE_NODE);
            }
#endif
            if (node.data.index() != static_cast<size_t>(StateNode::Type::TensorShard)) { continue; }
            bool       deduped = false;
            auto const tnsr    = std::get<StateNode::TensorShard>(node.data);
            tnsr.AssertShardIsValid();

            for (auto& node1 : _nodes)
            {
                if (node1.parent == node.parent && node1.handle != node.handle && node1.name == node.name
                    && node1.data.index() == static_cast<size_t>(StateNode::Type::ShardedTensor))
                {
                    auto& shardedTensor = std::get<StateNode::ShardedTensor>(node1.data);
                    shardedTensor.AddShard(StateNode::ShardedTensor::RankedTensorShard{
                        .sharddim = tnsr.sharddim, .device = tnsr.device, .handle = tnsr.handle, .rank = _topo.rank});
                    deduped = true;
                    break;
                }
            }
            if (deduped)
            {
                RemoveNode(node.handle);
                continue;
            }
            // Do not do a ref. We will be overwriting the mem location shortly
            node.data           = StateNode::ShardedTensor(tnsr.dtype, tnsr.fulldim);
            auto& shardedTensor = std::get<StateNode::ShardedTensor>(node.data);
            shardedTensor.AddShard(StateNode::ShardedTensor::RankedTensorShard{
                .sharddim = tnsr.sharddim, .device = tnsr.device, .handle = tnsr.handle, .rank = _topo.rank});
        }
    }

    template <typename TLambda> void WalkNodes(TLambda const& lambda)
    {
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

    void WaitForAsyncCalls()
    {
        std::scoped_lock lock(_mutex);
        if (_async.valid())
        {
            _async.get();
            _async = std::future<void>{};
        }
    }

    void FinalizeRegistry()
    {
        std::scoped_lock lock(_mutex);
        auto&            root = _nodes[0];
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
    }

    void CloneInto(StateTree& clone) const
    {
        clone._nodes.reserve(_nodes.size());
        clone._nodes.clear();
        for (auto const& node : _nodes) { clone._nodes.emplace_back(node.handle, node.parent, node.name, StateNode::Variant{node.data}); }
    }

    auto Clone() const
    {
        auto clonedTree = std::make_shared<StateTree>(_topo);
        CloneInto(*clonedTree);
        return clonedTree;
    }

    void RemoveNode(StateNode::Handle nodeHandle)
    {
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

    auto const& Topo() const { return _topo; }

    void Update()
    {
        WalkNodes([](StateNode& node) {
            if (node.handler) { node.handler->Update(); }
            auto type = static_cast<StateNode::Type>(node.data.index());
            if (type == StateNode::Type::TensorShard) { node.data = node.handler->GetTensorState(); }
            else if (type == StateNode::Type::ShardedTensor)
            {
                auto                                                  shTen = std::get<StateNode::ShardedTensor>(node.data);
                std::set<StateNode::ShardedTensor::RankedTensorShard> updatedShards;
                for (auto shard : shTen.shards)
                {
                    shard.handle = node.handler->GetTensorState().handle;
                    updatedShards.insert(shard);
                }
                std::swap(shTen.shards, updatedShards);
            }
        });
    }

    private:
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

    [[maybe_unused]] void SortDictionaryNode_(StateNode& node)
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
                    auto const tensorShardType = static_cast<size_t>(StateNode::Type::TensorShard);
                    if (childI.data.index() != tensorShardType || childJ.data.index() != tensorShardType)
                    {
                        fmt::print(stderr,
                                   "Duplicate state name {} found in root registry. Parent Node: {} \n",
                                   childI.name,
                                   GetNodeId(node.handle));
                        throw std::runtime_error(fmt::format("Duplicate state name {} found in root registry", childI.name));
                    }
                    auto const& shardI = std::get<StateNode::TensorShard>(childI.data);
                    auto const& shardJ = std::get<StateNode::TensorShard>(childJ.data);
                    if (shardI.dtype != shardJ.dtype || shardI.fulldim != shardJ.fulldim || shardI.sharddim == shardJ.sharddim)
                    {
                        throw std::runtime_error(fmt::format("Duplicate tensor shard {} found in root registry", childI.name));
                    }
                    // Megatron emits stacked layers and experts as distinct local shards under one FQN.
                    cmp = shardJ.sharddim < shardI.sharddim ? 1 : -1;
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

    RankTopology const _topo{};

    mutable std::mutex _mutex;
    std::future<void>  _async;
    friend struct StateSerDes;

    public:
    // TODO: Better time tracking.
    std::chrono::nanoseconds startedAt /*NOLINT*/
        = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
};

template <> struct StateTree::StateHandler<StateTree::StateType::Scalar> : StateTree::IStateHandler
{
    StateHandler(std::string_view const& nameIn) : IStateHandler(nameIn) {}

    [[nodiscard]] StateType      GetType() override { return StateType::Scalar; }
    [[noreturn]] StateNode::Blob GetBlob() override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] DS4Tensor       GetTensorState() override { throw std::runtime_error("Invalid operation"); }

    [[noreturn]] void SetBlob(std::string_view const& /* data */) override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] void SetTensorState(DS4Tensor const& /* data */) override { throw std::runtime_error("Invalid operation"); }
};

template <> struct StateTree::StateHandler<StateTree::StateType::Blob> : StateTree::IStateHandler
{
    StateHandler(std::string_view const& nameIn) : IStateHandler(nameIn) {}

    [[nodiscard]] StateType GetType() override { return StateType::Blob; }
    [[noreturn]] Scalar     GetScalarState() override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] DS4Tensor  GetTensorState() override { throw std::runtime_error("Invalid operation"); }

    [[noreturn]] void SetScalarState(Scalar const& /* data */) override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] void SetTensorState(DS4Tensor const& /* data */) override { throw std::runtime_error("Invalid operation"); }
};

template <> struct StateTree::StateHandler<StateTree::StateType::TensorShard> : StateTree::IStateHandler
{
    StateHandler(std::string_view const& nameIn) : IStateHandler(nameIn) {}
    [[nodiscard]] StateType GetType() override { return StateType::TensorShard; }

    [[noreturn]] Scalar          GetScalarState() override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] StateNode::Blob GetBlob() override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] DS4Tensor       GetTensorState() override { throw std::runtime_error("Invalid operation"); }

    [[noreturn]] void SetScalarState(Scalar const& /* data */) override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] void SetBlob(std::string_view const& /* data */) override { throw std::runtime_error("Invalid operation"); }
    [[noreturn]] void SetTensorState(DS4Tensor const& /* data */) override { throw std::runtime_error("Invalid operation"); }
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
        case static_cast<size_t>(StateNode::Type::Blob): return fmt::format_to(ctx.out(), "node:blob");
        case static_cast<size_t>(StateNode::Type::TensorShard):
            return fmt::format_to(ctx.out(), "node:tensorshard {}", std::get<StateNode::TensorShard>(var));
        case static_cast<size_t>(StateNode::Type::ShardedTensor):
            return fmt::format_to(ctx.out(), "node:shardedtensor {}", std::get<StateNode::ShardedTensor>(var));
        default: break;
        }
        throw std::runtime_error(fmt::format("Unknown state node type: {}", var.index()));
    }
};
