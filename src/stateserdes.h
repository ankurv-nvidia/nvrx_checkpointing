#pragma once
#include "messages.h"
#include "primitives64bit.h"
#include "serdes_utils.h"
#include "state.h"
#include "tensor.h"

struct StateSerDes
{
    using Type = StateNode::Type;

    struct DictHeader
    { uint32_t numChild{}; };

    struct BlobHeader
    { uint32_t dataSize{}; };

    struct ShardedTensorHeader
    { uint32_t numShards{}; };

// NOLINTBEGIN
#pragma pack(push, 1)
    struct Header
    {
        Type              type{};
        uint8_t           namelen{};
        uint16_t          padding{};
        StateNode::Handle handle{};
        StateNode::Handle parent{};
    };
    static_assert(sizeof(Header) == 12, "Header struct must be 12 bytes to ensure packed serialization without padding");
#pragma pack(pop)
    // NOLINTEND

    private:
    public:
    static std::ostream& WriteTopology(std::ostream& ostrm LFTBND, RankTopology const& topo)
    {
        serdes::WriteTrivial(ostrm, topo.rank);
        serdes::WriteTrivial(ostrm, topo.worldSize);
        serdes::WriteTrivial(ostrm, topo.ranksPerPeer);
        serdes::WriteTrivial(ostrm, topo.peersPerNode);
        auto numPeers = (static_cast<uint32_t>(topo.worldSize) + topo.ranksPerPeer - 1) / topo.ranksPerPeer;
        if (numPeers != topo.peers.size())
        {
            throw std::runtime_error(fmt::format("Number of peers {} does not match world size {} and ranks per peer {}",
                                                 topo.peers.size(),
                                                 topo.worldSize,
                                                 topo.ranksPerPeer));
        }
        for (auto const& peer : topo.peers) { serdes::WriteBlobWithSize<uint32_t>(ostrm, peer); }
        return ostrm;
    }

    static checkpoint::ipc::SessionId ReadSessionId(std::istream& istrm) { return serdes::ReadTrivial<checkpoint::ipc::SessionId>(istrm); }

    static RankTopology ReadTopology(std::istream& istrm)
    {
        RankTopology topo;
        topo.rank         = serdes::ReadTrivial<Rank>(istrm);
        topo.worldSize    = serdes::ReadTrivial<Rank>(istrm);
        topo.ranksPerPeer = serdes::ReadTrivial<uint32_t>(istrm);
        topo.peersPerNode = serdes::ReadTrivial<uint32_t>(istrm);
        auto numPeers     = (static_cast<uint32_t>(topo.worldSize) + topo.ranksPerPeer - 1) / topo.ranksPerPeer;
        topo.peers.reserve(numPeers);
        for (uint32_t i = 0; i < numPeers; ++i) { topo.peers.push_back(serdes::ReadBlobWithSize<uint32_t>(istrm)); }
        return topo;
    }

    static std::ostream& Write(std::ostream& ostrm LFTBND, StateTree const& tree, StateNode::Handle nodeHandle)
    {
        serdes::WriteTrivial(ostrm, tree.SessionId());
        WriteTopology(ostrm, tree._topo);

        serdes::WriteTrivial(ostrm, static_cast<uint32_t>(tree._nodes.size()));
        for (auto const& node : tree._nodes)
        {
            Header hdr{
                .type    = static_cast<Type>(node.data.index()),
                .namelen = static_cast<uint8_t>(node.name.size()),
                .handle  = node.handle,
                .parent  = node.parent,
            };
            serdes::WriteTrivial(ostrm, hdr);
            serdes::WriteBlob(ostrm, std::span(node.name));
            switch (hdr.type)
            {
            case Type::Blob:
            {
                auto       data = std::get<StateNode::Blob>(node.data);
                BlobHeader typehdr{.dataSize = static_cast<uint32_t>(data.size())};
                serdes::WriteTrivial(ostrm, typehdr);
                serdes::WriteBlob(ostrm, std::span(data));
            }
            break;
            case Type::Scalar:
            {
                serdes::WriteTrivial(ostrm, std::get<StateNode::Scalar>(node.data));
            }
            break;
            case Type::Tensor:
            {
#ifdef NATIVE_CKPT_TRACE_NODE
                auto fqn = tree.GetNodeId(nodeHandle);
                if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                {
                    fmt::print(stderr, "StateSerDes::Write!Tensor: Found trace node NATIVE_CKPT_TRACE_NODE = {}\n", NATIVE_CKPT_TRACE_NODE);
                }
#endif
                auto const& shard = std::get<Tensor>(node.data);
                serdes::WriteTrivial(ostrm, shard);
            }
            break;
            case Type::DTensor:
            {
#ifdef NATIVE_CKPT_TRACE_NODE
                auto fqn = tree.GetNodeId(nodeHandle);
                if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                {
                    fmt::print(
                        stderr, "StateSerDes::Write!DTensor: Found trace node NATIVE_CKPT_TRACE_NODE = {}\n", NATIVE_CKPT_TRACE_NODE);
                }
#endif
                auto const&         shTen = std::get<DTensor>(node.data);
                ShardedTensorHeader typehdr{.numShards = static_cast<uint32_t>(shTen.shards.size())};
                if (shTen.shards.empty())
                {
                    throw std::runtime_error(
                        fmt::format("Sharded tensor {} has no shards in metadata output\n", tree.GetNodeId(nodeHandle)));
                }
                serdes::WriteTrivial(ostrm, typehdr);
                serdes::WriteTrivial(ostrm, shTen.durableDtype);
                serdes::WriteTrivial(ostrm, shTen.fulldim);

                for (auto const& rankShard : shTen.shards)
                {
                    serdes::WriteTrivial(ostrm, rankShard);
                    rankShard.AssertShardIsValid(shTen.fulldim);
                }
            }

            break;
            case Type::Dict:
            {
                auto const& children = std::get<StateNode::Dict>(node.data);
                DictHeader  typehdr{.numChild = static_cast<uint32_t>(children.size())};
                serdes::WriteTrivial(ostrm, typehdr);
                for (auto child : children) { serdes::WriteTrivial(ostrm, child); }
            }
            break;
            case Type::Invalid:
            {
                // TODO_PURGE_INVALID:
                break;
            }
            case Type::Unknown: [[fallthrough]];
            default: throw std::logic_error("Unknown state type encountered in _Finalize");
            }
        }
        return ostrm;
    }

    static StateTree& Read(std::istream& istrm, StateTree& tree LFTBND)
    {
        auto numChild = serdes::ReadTrivial<uint32_t>(istrm);
        tree._nodes.clear();
        tree._nodes.reserve(numChild);
        for (uint32_t i = 0; i < numChild; ++i)
        {
            StateNode node;
            auto      hdr = serdes::ReadTrivial<Header>(istrm);
            node.handle   = hdr.handle;
            node.parent   = hdr.parent;
            node.name     = serdes::ReadBlob(istrm, hdr.namelen);
            if (node.name.empty() && node.handle != StateNode::Handle::Root
                && hdr.type != Type::Invalid)    // Only root node can have empty name
            {
                throw std::runtime_error(fmt::format(
                    "Node name cannot be empty. Node handle: {} Parent Handle: {}", node.handle, static_cast<uint32_t>(node.parent)));
            }
            // auto& dictNode = tree.GetNode(dictNodeHandle);
            switch (hdr.type)
            {
            case Type::Dict:
            {
                auto dictNode = StateNode::Dict{};
                auto typehdr  = serdes::ReadTrivial<DictHeader>(istrm);
                for (uint32_t j = 0; j < typehdr.numChild; ++j)
                {
                    auto childHandle = serdes::ReadTrivial<StateNode::Handle>(istrm);
                    dictNode.push_back(childHandle);
                }
                node.data = std::move(dictNode);
            }
            break;
            case Type::Blob:
            {
                auto typehdr = serdes::ReadTrivial<BlobHeader>(istrm);
                auto blob    = serdes::ReadBlob(istrm, typehdr.dataSize);
                node.data    = StateNode::Blob{blob.begin(), blob.end()};
            }
            break;
            case Type::Scalar:
            {
                node.data = serdes::ReadTrivial<Primitives64Bit>(istrm);
            }
            break;
            case Type::Tensor:
            {
#ifdef NATIVE_CKPT_TRACE_NODE
                auto fqn = tree.GetNodeId(node.handle);
                if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                {
                    fmt::print(stderr, "StateSerDes::Read!Tensor: Found trace node NATIVE_CKPT_TRACE_NODE = {}\n", NATIVE_CKPT_TRACE_NODE);
                }
#endif
                node.data = serdes::ReadTrivial<Tensor>(istrm);
            }
            break;
            case Type::DTensor:
            {
#ifdef NATIVE_CKPT_TRACE_NODE
                auto fqn = tree.GetNodeId(node.handle);
                if (fqn.contains(NATIVE_CKPT_TRACE_NODE))
                {
                    fmt::print(stderr, "StateSerDes::Read!DTensor: Found trace node NATIVE_CKPT_TRACE_NODE = {}\n", NATIVE_CKPT_TRACE_NODE);
                }
#endif
                auto typehdr      = serdes::ReadTrivial<ShardedTensorHeader>(istrm);
                auto durableDtype = serdes::ReadTrivial<Tensor::DType>(istrm);
                auto fulldim      = serdes::ReadTrivial<Tensor::Dim>(istrm);

                DTensor shTen{durableDtype, fulldim};
                if (typehdr.numShards == 0)
                {
                    throw std::runtime_error(fmt::format("Sharded tensor {} has no shards in metadata output\n", node.name));
                }
                for (uint32_t j = 0; j < typehdr.numShards; ++j)
                {
                    auto rankshard = serdes::ReadTrivial<TensorShard>(istrm);
                    rankshard.AssertShardIsValid(shTen.fulldim);
                    shTen.AddShard(rankshard);
                }
                node.data = std::move(shTen);
            }
            break;
            case Type::Invalid:    //
                node.data = std::monostate{};
                break;
            case Type::Unknown:
            default:
                throw std::runtime_error(
                    fmt::format("Unknown state type encountered at offset :{} Node-index = {}", static_cast<size_t>(istrm.tellg()), i));
            }
            tree._nodes.push_back(std::move(node));
        }

        return tree;
    }

    static std::unique_ptr<StateTree> Read(std::istream& istrm)
    {
        auto sessionId = ReadSessionId(istrm);
        auto topo      = ReadTopology(istrm);
        auto state     = std::make_unique<StateTree>(sessionId, topo);
        Read(istrm, *state);
        return state;
    }

    static void DebugDumpLocalStateTreeKVP(std::filesystem::path const& fname, StateTree const& state)
    {
        std::filesystem::create_directories(fname.parent_path());
        std::ofstream ostrm(fname);
        if (!ostrm.good()) { throw std::runtime_error(fmt::format("[native_ckpt:svr][ERROR] Failed to open file for writing: {}", fname)); }
        fmt::print(stderr, "[native_ckpt] DebugDumpLocalStateTreeKVP for file {}. Dumping local metadata key-value pairs\n", fname);
        state.WalkNodes([&](auto& node) {
            auto const type = static_cast<StateNode::Type>(node.data.index());
            if (type == StateNode::Type::Invalid || type >= StateNode::Type::Unknown)
            {
                throw std::runtime_error(fmt::format("Invalid node type: {}", static_cast<uint8_t>(type)));
            }
            if (type == StateNode::Type::Dict) { return; }
            auto const name = state.GetNodeId(node.handle);

            if (type == StateNode::Type::Scalar)
            {
                auto const& scalar = std::get<StateNode::Scalar>(node.data);
                auto const& t      = scalar.GetType();
                if (StateNode::Scalar::Type::IsFloat(t)) { fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<double>()); }
                else if (StateNode::Scalar::Type::IsSigned(t)) { fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<int64_t>()); }
                else if (StateNode::Scalar::Type::IsUnsigned(t)) { fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<uint64_t>()); }
                else
                {
                    fmt::print(ostrm, "{}: {}\n", name, scalar.template Cast<bool>());
                }
            }
            else if (type == StateNode::Type::Blob) { fmt::print(ostrm, "{}: [blob]\n", name); }
            else if (type == StateNode::Type::Tensor)
            {
                auto const& tnsr = std::get<Tensor>(node.data);
                fmt::print(ostrm, "{}.fulldim: [{}]\n", name, tnsr.fulldim);
                fmt::print(ostrm, "{}.durableDtype: {}\n", name, tnsr.durableDtype);
                fmt::print(ostrm, "{}.snapshotDtype: {}\n", name, tnsr.snapshotDtype);
                fmt::print(ostrm, "{}.handle: {}\n", name, tnsr.handle);
            }
            else if (type == StateNode::Type::DTensor)
            {
                auto const& tnsr = std::get<DTensor>(node.data);
                if (tnsr.shards.size() == 0) throw std::runtime_error(fmt::format("Empty Sharded tensor : {}", name));
                fmt::print(ostrm, "{}.fulldim: [{}]\n", name, tnsr.fulldim);
                fmt::print(ostrm, "{}.durableDtype: {}\n", name, tnsr.durableDtype);

                for (auto const& rshard : tnsr.shards)
                {
                    fmt::print(ostrm, "{}.sharddim@{}.shards: [{}]\n", name, rshard.rank, rshard.sharddim);
                    fmt::print(ostrm, "{}.sharddim@{}.handle: {}\n", name, rshard.rank, rshard.handle);
                }
            }
        });
    }

};    // namespace StateSerDes
