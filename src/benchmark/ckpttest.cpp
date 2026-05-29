#include "CommonMacros.h"

#include "blob.h"
#include "ckpt_fixture.h"
#include "distcp.h"
#include "distcp_internal.h"
#include "server.h"
#include "state.h"
#include "tensor.h"

#include <fmt/format.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include <sys/statvfs.h>
#include <unistd.h>

// Populated from --registry on the command line in main(). Tests that need
// it (the [round-trip] case) assert non-empty before use; ctest wires the
// path via catch_discover_tests(... EXTRA_ARGS --registry <path>) so the
// binary itself stays portable.
namespace
{
SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
std::filesystem::path GRegistryPath;
SUPPRESS_WARNINGS_END

std::filesystem::path const& TestRegistryPath()
{ return GRegistryPath; }

// Stem-based subdirectory under the test temp root. Not strictly unique across
// concurrent ctest workers, but every call site `remove_all`s before use, so
// single-process runs are safe and `ctest -jN` only collides if two distinct
// jobs share the same stem (we use distinct stems per TEST_CASE).
std::filesystem::path CkptDir(std::string_view stem)
{
    auto base = std::filesystem::temp_directory_path() / "phoenix_ckpttest";
    std::filesystem::create_directories(base);
    return base / stem;
}

RankTopology SingleRankTopo(uint32_t rank = 0, uint32_t worldSize = 1)
{
    return RankTopology{
        .rank = Rank{rank}, .worldSize = Rank{worldSize}, .ranksPerPeer = worldSize, .peersPerNode = 1, .peers = {"localhost"}};
}

struct TestShardEntry
{
    char const*         srcPtr;
    uint64_t            length;
    StateTree::NodeId   fqn;
    DS4Tensor::DType    dtype;
    DS4Tensor::DimRange sharddim;
    DS4Tensor::Dim      shape;
};

std::vector<TestShardEntry> CollectShardsForTest(StateTree const& tree, Rank rank, Blob const& blob)
{
    std::vector<TestShardEntry> out;
    auto const*                 base = static_cast<char const*>(blob.data);
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        auto const fqn  = tree.GetNodeId(node.handle);
        if (type == StateNode::Type::TensorShard)
        {
            // Per-rank TensorShards belong to the tree's rank — no per-shard
            // rank field to filter on. Tests build single-rank or per-rank
            // trees, so emit unconditionally.
            auto const& tnsr = std::get<StateNode::TensorShard>(node.data);
            auto const  size = tnsr.ShardSizeBytes();
            if (size == 0) { return; }
            DS4Tensor::Dim shape{};
            for (auto const& r : tnsr.sharddim) { shape.push_back(r.end - r.begin); }
            out.push_back(TestShardEntry{
                .srcPtr   = base + tnsr.handle,
                .length   = size,
                .fqn      = fqn,
                .dtype    = tnsr.dtype,
                .sharddim = tnsr.sharddim,
                .shape    = shape,
            });
        }
        else if (type == StateNode::Type::ShardedTensor)
        {
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                if (shard.rank != rank) { continue; }
                auto const size = shard.ShardSizeBytes(tnsr.dtype);
                if (size == 0) { continue; }
                DS4Tensor::Dim shape{};
                for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
                out.push_back(TestShardEntry{
                    .srcPtr   = base + shard.handle,
                    .length   = size,
                    .fqn      = fqn,
                    .dtype    = tnsr.dtype,
                    .sharddim = shard.sharddim,
                    .shape    = shape,
                });
            }
        }
    });
    return out;
}

// Attaches a fully-occupied (sharddim == fulldim) TensorShard leaf at the
// given FQN. `handle` is the byte offset into the surrounding Blob.
void AddTensorShardLeaf(StateTree& tree, std::string_view fqn, DS4Tensor::DType dtype, std::vector<size_t> const& shape, uint64_t handle)
{
    auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
    auto&     leaf          = tree.AddChild(parent, leafName);
    DS4Tensor tnsr;
    tnsr.dtype  = dtype;
    tnsr.handle = handle;
    tnsr.device = DeviceId{"cpu"};
    for (auto d : shape)
    {
        tnsr.fulldim.push_back(d);
        tnsr.sharddim.push_back({.begin = 0, .end = d});
    }
    leaf.data = StateNode::TensorShard{tnsr};
}

void AddShardedTensorLeaf(StateTree&                 tree,
                          std::string_view           fqn,
                          DS4Tensor::DType           dtype,
                          std::vector<size_t> const& shape,
                          Rank                       rank,
                          uint64_t                   handle)
{
    auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
    auto&          leaf     = tree.AddChild(parent, leafName);
    DS4Tensor::Dim fulldim;
    for (auto d : shape) { fulldim.push_back(d); }
    StateNode::ShardedTensor st(dtype, fulldim);
    DS4Tensor::DimRange      sharddim;
    for (auto d : shape) { sharddim.push_back({.begin = 0, .end = d}); }
    st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = handle, .rank = rank});
    leaf.data = std::move(st);
}
}    // namespace

TEST_CASE("scaffolding", "[scaffolding]")
{ REQUIRE(true); }

TEST_CASE("distcp shard walk visits TensorShard leaves in insertion order", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo());
    Blob      blob(64);

    AddTensorShardLeaf(tree, "a", DS4Tensor::DType::Float32, {2, 2}, 0);    // 16 B
    AddTensorShardLeaf(tree, "b", DS4Tensor::DType::Int8, {4}, 16);         //  4 B
    AddTensorShardLeaf(tree, "c", DS4Tensor::DType::Float64, {2}, 32);      // 16 B

    auto const entries = CollectShardsForTest(tree, Rank{0}, blob);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "a");
    REQUIRE(entries[1].fqn == "b");
    REQUIRE(entries[2].fqn == "c");

    REQUIRE(entries[0].dtype == DS4Tensor::DType::Float32);
    REQUIRE(entries[1].dtype == DS4Tensor::DType::Int8);
    REQUIRE(entries[2].dtype == DS4Tensor::DType::Float64);

    REQUIRE(entries[0].length == 16);
    REQUIRE(entries[1].length == 4);
    REQUIRE(entries[2].length == 16);

    auto const* base = static_cast<char const*>(blob.data);
    REQUIRE(entries[0].srcPtr == base + 0);
    REQUIRE(entries[1].srcPtr == base + 16);
    REQUIRE(entries[2].srcPtr == base + 32);
}

TEST_CASE("distcp shard walk derives shape from sharddim for partial slices", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo());
    Blob      blob(128);

    // Fully-occupied 1-D tensor — shape matches fulldim.
    AddTensorShardLeaf(tree, "full", DS4Tensor::DType::Float32, {4}, 0);

    // Non-full 2-D slice: fulldim is 8x4 but this shard covers rows [2,6) and cols [1,3).
    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("slice");
        auto&     leaf          = tree.AddChild(parent, leafName);
        DS4Tensor tnsr;
        tnsr.dtype  = DS4Tensor::DType::Int32;
        tnsr.handle = 16;
        tnsr.device = DeviceId{"cpu"};
        tnsr.fulldim.push_back(8);
        tnsr.fulldim.push_back(4);
        tnsr.sharddim.push_back({.begin = 2, .end = 6});
        tnsr.sharddim.push_back({.begin = 1, .end = 3});
        leaf.data = StateNode::TensorShard{tnsr};
    }

    auto const entries = CollectShardsForTest(tree, Rank{0}, blob);

    REQUIRE(entries.size() == 2);

    REQUIRE(entries[0].shape.size() == 1);
    REQUIRE(entries[0].shape[0] == 4);

    REQUIRE(entries[1].shape.size() == 2);
    REQUIRE(entries[1].shape[0] == 4);
    REQUIRE(entries[1].shape[1] == 2);
    REQUIRE(entries[1].length == 4u * 2u * 4u);    // 8 int32 elements
}

TEST_CASE("distcp shard walk expands ShardedTensor into per-rank entries", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo(0, 2));
    Blob      blob(64);

    auto [leafName, parent] = tree.FindOrAddNodeByFQN("shrd");
    auto&          leaf     = tree.AddChild(parent, leafName);
    DS4Tensor::Dim fulldim;
    fulldim.push_back(8);
    StateNode::ShardedTensor tnsr(DS4Tensor::DType::Float32, fulldim);

    DS4Tensor::DimRange r0;
    r0.push_back({.begin = 0, .end = 4});
    DS4Tensor::DimRange r1;
    r1.push_back({.begin = 4, .end = 8});
    tnsr.AddShard({.sharddim = r0, .device = DeviceId{"cpu"}, .handle = 0, .rank = Rank{0}});
    tnsr.AddShard({.sharddim = r1, .device = DeviceId{"cpu"}, .handle = 16, .rank = Rank{1}});
    leaf.data = std::move(tnsr);

    auto const rank0 = CollectShardsForTest(tree, Rank{0}, blob);
    REQUIRE(rank0.size() == 1);
    REQUIRE(rank0[0].fqn == "shrd");
    REQUIRE(rank0[0].sharddim[0].begin == 0);
    REQUIRE(rank0[0].length == 16);
    REQUIRE(rank0[0].srcPtr == static_cast<char const*>(blob.data) + 0);

    auto const rank1 = CollectShardsForTest(tree, Rank{1}, blob);
    REQUIRE(rank1.size() == 1);
    REQUIRE(rank1[0].sharddim[0].begin == 4);
    REQUIRE(rank1[0].srcPtr == static_cast<char const*>(blob.data) + 16);
}

TEST_CASE("distcp shard walk orders same-rank shards by sharddim", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo(0, 2));
    Blob      blob(64);

    auto [leafName, parent] = tree.FindOrAddNodeByFQN("shrd");
    auto&          leaf     = tree.AddChild(parent, leafName);
    DS4Tensor::Dim fulldim;
    fulldim.push_back(8);
    StateNode::ShardedTensor tnsr(DS4Tensor::DType::Float32, fulldim);

    DS4Tensor::DimRange r0low;
    r0low.push_back({.begin = 0, .end = 4});
    DS4Tensor::DimRange r0high;
    r0high.push_back({.begin = 4, .end = 8});
    // Insertion order intentionally inverted vs. the std::set order so the
    // test only passes if `tnsr.shards` is iterated in (rank, sharddim) order.
    tnsr.AddShard({.sharddim = r0high, .device = DeviceId{"cpu"}, .handle = 32, .rank = Rank{0}});
    tnsr.AddShard({.sharddim = r0low, .device = DeviceId{"cpu"}, .handle = 0, .rank = Rank{0}});

    leaf.data = std::move(tnsr);

    auto const entries = CollectShardsForTest(tree, Rank{0}, blob);

    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].sharddim[0].begin == 0);
    REQUIRE(entries[1].sharddim[0].begin == 4);
}

TEST_CASE("distcp shard walk handles mixed TensorShard and ShardedTensor nodes", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo(0, 2));
    Blob      blob(128);

    AddTensorShardLeaf(tree, "before", DS4Tensor::DType::Float32, {4}, 0);

    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("middle");
        auto&          leaf     = tree.AddChild(parent, leafName);
        DS4Tensor::Dim fulldim;
        fulldim.push_back(8);
        StateNode::ShardedTensor tnsr(DS4Tensor::DType::Float32, fulldim);
        DS4Tensor::DimRange      r;
        r.push_back({.begin = 0, .end = 8});
        tnsr.AddShard({.sharddim = r, .device = DeviceId{"cpu"}, .handle = 16, .rank = Rank{0}});
        leaf.data = std::move(tnsr);
    }

    AddTensorShardLeaf(tree, "after", DS4Tensor::DType::Float32, {2}, 48);

    auto const entries = CollectShardsForTest(tree, Rank{0}, blob);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "before");
    REQUIRE(entries[1].fqn == "middle");
    REQUIRE(entries[2].fqn == "after");
}

TEST_CASE("distcp shard walk joins nested dict names with dots", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo());
    Blob      blob(64);

    AddTensorShardLeaf(tree, "model.layer0.weight", DS4Tensor::DType::Float32, {2}, 0);
    AddTensorShardLeaf(tree, "model.layer0.bias", DS4Tensor::DType::Float32, {2}, 8);
    AddTensorShardLeaf(tree, "model.layer1.weight", DS4Tensor::DType::Float32, {2}, 16);

    auto const entries = CollectShardsForTest(tree, Rank{0}, blob);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "model.layer0.weight");
    REQUIRE(entries[1].fqn == "model.layer0.bias");
    REQUIRE(entries[2].fqn == "model.layer1.weight");
}

TEST_CASE("distcp shard walk skips zero-length shards", "[distcp][tier1]")
{
    StateTree tree(SingleRankTopo());
    Blob      blob(16);

    AddTensorShardLeaf(tree, "non_empty", DS4Tensor::DType::Float32, {2}, 0);

    // Empty sharddim — ShardSizeBytes() returns 0.
    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("empty");
        auto&     leaf          = tree.AddChild(parent, leafName);
        DS4Tensor tnsr;
        tnsr.dtype  = DS4Tensor::DType::Float32;
        tnsr.handle = 8;
        tnsr.device = DeviceId{"cpu"};
        leaf.data   = StateNode::TensorShard{tnsr};
    }

    auto const entries = CollectShardsForTest(tree, Rank{0}, blob);

    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].fqn == "non_empty");
}

namespace
{
constexpr char KZipLocalFileMagic[4] = {'P', 'K', '\x03', '\x04'};

std::vector<char> ReadFile(std::filesystem::path const& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error(fmt::format("[ckpttest] cannot open {}", path.string())); }
    std::vector<char> out((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return out;
}
}    // namespace

namespace
{
struct PredictedRecord
{
    std::string fqn;
    uint64_t    fileOffset{0};
    uint64_t    length{0};
};

std::vector<PredictedRecord> PredictRankLayout(StateTree const& tree)
{
    std::vector<PredictedRecord> out;
    uint64_t                     cursor = 0;
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::TensorShard)
        {
            auto const& tnsr    = std::get<StateNode::TensorShard>(node.data);
            auto const  shardSz = tnsr.ShardSizeBytes();
            if (shardSz == 0) { return; }
            DS4Tensor::Dim shape;
            for (auto const& r : tnsr.sharddim) { shape.push_back(r.end - r.begin); }
            uint64_t const archiveLen = checkpoint::distcp::ComputeShardArchiveSize(tnsr.dtype, shape, shardSz);
            out.push_back({.fqn = tree.GetNodeId(node.handle), .fileOffset = cursor, .length = archiveLen});
            cursor += archiveLen;
        }
        else if (type == StateNode::Type::ShardedTensor)
        {
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                auto const shardSz = shard.ShardSizeBytes(tnsr.dtype);
                if (shardSz == 0) { continue; }
                DS4Tensor::Dim shape;
                for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
                uint64_t const archiveLen = checkpoint::distcp::ComputeShardArchiveSize(tnsr.dtype, shape, shardSz);
                out.push_back({.fqn = tree.GetNodeId(node.handle), .fileOffset = cursor, .length = archiveLen});
                cursor += archiveLen;
            }
        }
    });
    return out;
}
}    // namespace

TEST_CASE("distcp::WriteRankShard produces a contiguous file of torch.save ZIPs", "[distcp][tier2]")
{
    StateTree tree(SingleRankTopo());
    Blob      blob(64);
    // Deterministic source bytes — keep them simple so any future stride/dtype
    // bug shows up as a torch.load mismatch on inspection.
    auto* bytes = static_cast<unsigned char*>(blob.data);
    for (size_t i = 0; i < blob.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }

    AddShardedTensorLeaf(tree, "a", DS4Tensor::DType::Float32, {2, 2}, Rank{0}, 0);    // 16 B
    AddShardedTensorLeaf(tree, "b", DS4Tensor::DType::Int8, {4}, Rank{0}, 16);         //  4 B
    AddShardedTensorLeaf(tree, "c", DS4Tensor::DType::Float64, {2}, Rank{0}, 32);      // 16 B

    auto outDir = CkptDir("tier2_layout");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, blob);

    auto const predicted = PredictRankLayout(tree);
    REQUIRE(predicted.size() == 3);
    REQUIRE(predicted[0].fqn == "a");
    REQUIRE(predicted[1].fqn == "b");
    REQUIRE(predicted[2].fqn == "c");

    REQUIRE(predicted[0].fileOffset == 0);
    REQUIRE(predicted[0].fileOffset + predicted[0].length == predicted[1].fileOffset);
    REQUIRE(predicted[1].fileOffset + predicted[1].length == predicted[2].fileOffset);

    auto const path = outDir / "__0_0.distcp";
    REQUIRE(std::filesystem::exists(path));
    auto const fileBytes = ReadFile(path);
    REQUIRE(fileBytes.size() == predicted.back().fileOffset + predicted.back().length);

    for (auto const& rec : predicted)
    {
        REQUIRE(rec.length >= sizeof(KZipLocalFileMagic));
        for (size_t i = 0; i < sizeof(KZipLocalFileMagic); ++i) { REQUIRE(fileBytes[rec.fileOffset + i] == KZipLocalFileMagic[i]); }
    }
}

TEST_CASE("distcp::WriteRankShard runs concurrently across ranks", "[distcp][tier6]")
{
    constexpr uint32_t kWorldSize = 4;

    struct LocalState
    {
        std::unique_ptr<StateTree> tree;
        std::unique_ptr<Blob>      blob;
    };

    struct Snapshot
    {
        std::mutex               mutex;
        std::unordered_set<Rank> writeCkptCompletedRanks;
        std::vector<LocalState>  states;
    };

    Snapshot snapshot;
    snapshot.states.resize(kWorldSize);
    for (uint32_t r = 0; r < kWorldSize; ++r)
    {
        auto& ls = snapshot.states[r];
        ls.blob  = std::make_unique<Blob>(64);
        // Rank-specific deterministic bytes so a cross-rank write would surface
        // as bogus payload at a recorded (offset, length).
        auto* bytes = static_cast<unsigned char*>(ls.blob->data);
        for (size_t i = 0; i < ls.blob->size; ++i) { bytes[i] = static_cast<unsigned char>(((r * 17u) + i) & 0xffu); }
        ls.tree = std::make_unique<StateTree>(SingleRankTopo(r, kWorldSize));
        AddShardedTensorLeaf(*ls.tree, "a", DS4Tensor::DType::Float32, {2, 2}, Rank{r}, 0);    // 16 B
        AddShardedTensorLeaf(*ls.tree, "b", DS4Tensor::DType::Int8, {4}, Rank{r}, 16);         //  4 B
        AddShardedTensorLeaf(*ls.tree, "c", DS4Tensor::DType::Float64, {2}, Rank{r}, 32);      // 16 B
    }

    auto outDir = CkptDir("tier6_concurrent");
    std::filesystem::remove_all(outDir);

    std::vector<std::thread> workers;
    workers.reserve(kWorldSize);
    for (uint32_t r = 0; r < kWorldSize; ++r)
    {
        workers.emplace_back([&snapshot, &outDir, r] {
            auto& ls = snapshot.states[r];
            checkpoint::distcp::WriteRankShard(outDir, Rank{r}, *ls.tree, *ls.blob);
            std::scoped_lock lock(snapshot.mutex);
            snapshot.writeCkptCompletedRanks.insert(Rank{r});
        });
    }
    for (auto& t : workers) { t.join(); }

    REQUIRE(snapshot.writeCkptCompletedRanks.size() == kWorldSize);
    for (uint32_t r = 0; r < kWorldSize; ++r)
    {
        REQUIRE(snapshot.writeCkptCompletedRanks.contains(Rank{r}));

        auto const& ls        = snapshot.states[r];
        auto const  predicted = PredictRankLayout(*ls.tree);
        REQUIRE(predicted.size() == 3);
        REQUIRE(predicted[0].fqn == "a");
        REQUIRE(predicted[1].fqn == "b");
        REQUIRE(predicted[2].fqn == "c");

        REQUIRE(predicted[0].fileOffset == 0);
        REQUIRE(predicted[0].fileOffset + predicted[0].length == predicted[1].fileOffset);
        REQUIRE(predicted[1].fileOffset + predicted[1].length == predicted[2].fileOffset);

        auto const path = outDir / fmt::format("__{}_0.distcp", r);
        REQUIRE(std::filesystem::exists(path));
        auto const fileBytes = ReadFile(path);
        REQUIRE(fileBytes.size() == predicted.back().fileOffset + predicted.back().length);

        for (auto const& rec : predicted)
        {
            REQUIRE(rec.length >= sizeof(KZipLocalFileMagic));
            for (size_t i = 0; i < sizeof(KZipLocalFileMagic); ++i) { REQUIRE(fileBytes[rec.fileOffset + i] == KZipLocalFileMagic[i]); }
        }

        // The 16 bytes of leaf "a" embed a rank-specific signature. They are
        // stored at some aligned offset inside `archive/data/0` — exact offset
        // depends on the ZIP entry headers — so we search the rank's file for
        // the signature instead of slicing at a fixed offset. A cross-rank
        // write would surface here as a missed match.
        std::array<unsigned char, 16> signature{};
        for (size_t i = 0; i < signature.size(); ++i) { signature[i] = static_cast<unsigned char>(((r * 17u) + i) & 0xffu); }
        auto const* needle = reinterpret_cast<char const*>(signature.data());
        auto const  found  = std::search(fileBytes.begin(), fileBytes.end(), needle, needle + signature.size());
        REQUIRE(found != fileBytes.end());
    }
}

// ---------------------------------------------------------------------------
// `ComputeShardArchiveSize` self-consistency. Builds a real `__0_0.distcp`
// for a small per-rank tree, then asserts the predictor's per-shard sums
// match the actual ZIP boundaries on disk:
//   - last predicted offset + last predicted length == file size
//   - every predicted offset lands on the four-byte PK\x03\x04 LFH magic
// If the predictor ever drifts from `WriteShardArchive`'s layout, this is
// what catches it. The shape-coverage list is deliberately the same set the
// tier-2 / tier-6 tests touch (so a regression surfaces at the lowest tier
// first) plus a couple of off-alignment shapes that exercise the FB padding.
// ---------------------------------------------------------------------------

TEST_CASE("ComputeShardArchiveSize matches WriteRankShard's on-disk layout", "[distcp][tier1][predict]")
{
    StateTree tree(SingleRankTopo());
    Blob      blob(256);
    auto*     bytes = static_cast<unsigned char*>(blob.data);
    for (size_t i = 0; i < blob.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }

    // Mix of dtypes that produce different `data.pkl` sizes and shard byte
    // counts. FB padding is offset-dependent, so any shape arrangement gets
    // exercised once the prior shards' cumulative size shifts the cursor.
    AddShardedTensorLeaf(tree, "a", DS4Tensor::DType::Float32, {2, 2}, Rank{0}, 0);    // 16 B
    AddShardedTensorLeaf(tree, "b", DS4Tensor::DType::Int8, {4}, Rank{0}, 16);         //  4 B
    AddShardedTensorLeaf(tree, "c", DS4Tensor::DType::Float64, {2}, Rank{0}, 32);      // 16 B
    AddShardedTensorLeaf(tree, "d", DS4Tensor::DType::Int32, {4}, Rank{0}, 64);        // 16 B
    AddShardedTensorLeaf(tree, "e", DS4Tensor::DType::Float32, {8}, Rank{0}, 128);     // 32 B

    auto outDir = CkptDir("predict_consistency");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, blob);

    auto const predicted = PredictRankLayout(tree);
    REQUIRE(predicted.size() == 5);

    auto const path = outDir / "__0_0.distcp";
    REQUIRE(std::filesystem::exists(path));
    auto const fileBytes = ReadFile(path);

    // File size matches the predicted total. This catches under-/over-
    // counting of any of the per-entry components (LFH, FB extra, CD record,
    // ZIP64 EOCD record / locator, EOCD).
    REQUIRE(fileBytes.size() == predicted.back().fileOffset + predicted.back().length);

    // Each predicted offset lands exactly on a PK\x03\x04 LFH magic. Misaligned
    // offsets would land in the middle of a previous archive's CRC field or
    // similar — almost certainly not on PK\x03\x04, so this is a strong check.
    for (auto const& rec : predicted)
    {
        REQUIRE(rec.fileOffset + sizeof(KZipLocalFileMagic) <= fileBytes.size());
        for (size_t i = 0; i < sizeof(KZipLocalFileMagic); ++i)
        {
            REQUIRE(fileBytes.at(rec.fileOffset + i)
                    == KZipLocalFileMagic[i]);    // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }
}

TEST_CASE("distcp::WriteMetadata emits .metadata for shards rank 0 never wrote", "[distcp][tier1][metadata]")
{
    // Multi-node mock: a global registry with shards belonging to two ranks,
    // but only the parts of the on-disk world rank 0 would actually produce
    // — i.e. its own __0_0.distcp. Rank 1's local registry is intentionally
    // never constructed: WriteMetadata has to emit a valid `.metadata` pickle
    // without it, by analytically reconstructing rank 1's layout from the
    // global ShardedTensor entries. The empty manifest skips per-shard
    // verification (the multi-node case where rank 0 has no view into other
    // ranks' writeLogs).
    StateTree globalTree(SingleRankTopo(0, 2));
    auto      addShard = [&](std::string_view                                                            fqn,
                             DS4Tensor::DType                                                            dtype,
                             std::vector<size_t> const&                                                  fullShape,
                             std::vector<std::pair<Rank, std::vector<std::pair<size_t, size_t>>>> const& shards) {
        DS4Tensor::Dim fulldim;
        for (auto d : fullShape) { fulldim.push_back(d); }
        auto [leafName, parent]       = globalTree.FindOrAddNodeByFQN(fqn);
        auto&                    leaf = globalTree.AddChild(parent, leafName);
        StateNode::ShardedTensor st(dtype, fulldim);
        for (auto const& [r, ranges] : shards)
        {
            DS4Tensor::DimRange sharddim;
            for (auto const& [b, e] : ranges) { sharddim.push_back({b, e}); }
            st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = 0, .rank = r});
        }
        leaf.data = std::move(st);
    };
    // Tensor of full shape (8,) split into two halves across ranks 0 and 1.
    addShard("weight", DS4Tensor::DType::Float32, {8}, {{Rank{0}, {{0, 4}}}, {Rank{1}, {{4, 8}}}});
    addShard("bias", DS4Tensor::DType::Int64, {4}, {{Rank{0}, {{0, 2}}}, {Rank{1}, {{2, 4}}}});

    auto outDir = CkptDir("metadata_multi_rank_inference");
    std::filesystem::remove_all(outDir);

    REQUIRE_NOTHROW(checkpoint::distcp::WriteMetadata(outDir, globalTree));
    auto const metaPath = outDir / ".metadata";
    REQUIRE(std::filesystem::exists(metaPath));
    REQUIRE(std::filesystem::file_size(metaPath) > 0);
}

#ifdef PHOENIX_VENV_PYTHON
TEST_CASE("distcp::WriteMetadata emits an unpickle-able .metadata", "[distcp][tier1][metadata]")
{
    // Single-rank case: localTree and globalTree are the same tree of
    // single-shard ShardedTensors. WriteRankShard requires the global-
    // registry shape (asserted internally), and WriteMetadata consumes the
    // resulting writeLog as a verification side-channel.
    StateTree tree(SingleRankTopo());
    Blob      blob(64);
    auto*     bytes = static_cast<unsigned char*>(blob.data);
    for (size_t i = 0; i < blob.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }
    AddShardedTensorLeaf(tree, "weight", DS4Tensor::DType::Float32, {2, 2}, Rank{0}, 0);
    AddShardedTensorLeaf(tree, "block.weight", DS4Tensor::DType::Int64, {3}, Rank{0}, 16);

    auto outDir = CkptDir("metadata_unpickle");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, blob);
    checkpoint::distcp::WriteMetadata(outDir, tree);

    auto const metaPath = outDir / ".metadata";
    REQUIRE(std::filesystem::exists(metaPath));
    REQUIRE(std::filesystem::file_size(metaPath) > 0);

    // Spawn the venv python (which has torch on its sys.path) and have it
    // unpickle the file. The unpickler walks every GLOBAL, runs
    // TensorProperties.__setstate__, etc. — a non-zero exit signals a real
    // structural problem in our emit. std::system is plenty for a test;
    // metaPath comes from `std::filesystem::temp_directory_path()` and has
    // no shell metacharacters. Stdout is silenced (the script prints nothing
    // on success); stderr is left intact so Catch2 captures the traceback
    // when the load actually breaks.
    auto const cmd
        = fmt::format("{} -c \"import pickle; pickle.load(open(r'{}','rb'))\" >/dev/null", PHOENIX_VENV_PYTHON, metaPath.string());
    int const rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);
}
#endif

// Drives the inproc server + registry round-trip end-to-end. Preconditions:
// (a) `/dev/shm` is large enough for the IPC ring buffers (~1 GiB per queue,
// several queues per rank), (b) the host has enough physical RAM to back
// that tmpfs allocation plus the checkpoint working set, and (c) a registry
// file readable by `StateSerDes::ReadTopology` is passed via `--registry`.
// When any precondition fails the test SKIPs cleanly; ctest reports skips
// separately from passes/failures.
TEST_CASE("save_then_round_trip_validate", "[round-trip]")
{
    auto const& registryPath = TestRegistryPath();
    if (registryPath.empty())
    {
        SKIP("no --registry path provided (pass `--registry <path>` or use ctest, which wires it via EXTRA_ARGS)");
    }
    if (!std::filesystem::exists(registryPath)) { SKIP(fmt::format("registry path does not exist: {}", registryPath.string())); }

    // Conservative RAM threshold. The IPC ring buffers consume ~1 GiB per queue
    // (D2HBufferSize × D2HBufferCount = 32 MiB × 32) — empirically 8 of them
    // for a worldSize=2 registry, ~8 GiB resident in `/dev/shm` (tmpfs) and
    // therefore in RAM. Add slack for the test process and the checkpoint
    // working set. Tune up if a larger registry comes along.
    constexpr uint64_t kMinPhysRamBytes = 16ull * 1024 * 1024 * 1024;
    long const         pages            = ::sysconf(_SC_PHYS_PAGES);
    long const         pageSize         = ::sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || pageSize <= 0) { SKIP("sysconf(_SC_PHYS_PAGES / _SC_PAGE_SIZE) returned <= 0; cannot verify RAM precondition"); }
    auto const physRamBytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(pageSize);
    if (physRamBytes < kMinPhysRamBytes)
    {
        SKIP(fmt::format("host has {:.1f} GiB RAM; round-trip needs >= {:.1f} GiB to back the IPC ring buffers without OOM",
                         static_cast<double>(physRamBytes) / (1024.0 * 1024.0 * 1024.0),
                         static_cast<double>(kMinPhysRamBytes) / (1024.0 * 1024.0 * 1024.0)));
    }

    // `/dev/shm` capacity check. Even with enough physical RAM, the tmpfs
    // mount has its own size cap (the Docker default is 64 MiB; some sandboxes
    // ship smaller). Check both the *total* size (catches the undersized-mount
    // case) and the *available* size (catches stale `/dev/shm/checkpoint_*`
    // files left behind by a SIGKILLed prior run).
    constexpr uint64_t kMinShmBytes = 10ull * 1024 * 1024 * 1024;
    struct statvfs     vfs{};
    if (::statvfs("/dev/shm", &vfs) != 0) { SKIP("statvfs(/dev/shm) failed; cannot verify shm precondition"); }
    auto const shmTotalBytes = static_cast<uint64_t>(vfs.f_blocks) * static_cast<uint64_t>(vfs.f_frsize);
    if (shmTotalBytes < kMinShmBytes)
    {
        SKIP(fmt::format("/dev/shm is {:.1f} GiB total; round-trip needs >= {:.1f} GiB (remount via "
                         "`sudo mount -o remount,size=16G /dev/shm`)",
                         static_cast<double>(shmTotalBytes) / (1024.0 * 1024.0 * 1024.0),
                         static_cast<double>(kMinShmBytes) / (1024.0 * 1024.0 * 1024.0)));
    }
    auto const shmAvailBytes = static_cast<uint64_t>(vfs.f_bavail) * static_cast<uint64_t>(vfs.f_bsize);
    if (shmAvailBytes < kMinShmBytes)
    {
        SKIP(fmt::format("/dev/shm has only {:.1f} GiB free of {:.1f} GiB; round-trip needs >= {:.1f} GiB free "
                         "(check for stale shm files: `ls /dev/shm/checkpoint_*` and `sudo rm -f /dev/shm/checkpoint_*`)",
                         static_cast<double>(shmAvailBytes) / (1024.0 * 1024.0 * 1024.0),
                         static_cast<double>(shmTotalBytes) / (1024.0 * 1024.0 * 1024.0),
                         static_cast<double>(kMinShmBytes) / (1024.0 * 1024.0 * 1024.0)));
    }

    CheckpointTestContext    ctx;
    std::vector<std::thread> tasks;
    bool                     serverRunning = false;

    auto cleanup = [&] {
        if (!tasks.empty()) { JoinClientThreads(ctx, tasks); }
        if (serverRunning) { StopInprocServer(); }
    };

    try
    {
        RunInprocServer();
        serverRunning = true;

        LoadStateRegistryFromPath(ctx, registryPath);
        tasks = StartClientThreads(ctx);
        REQUIRE_NOTHROW(RunOnAllRanks(ctx, &SaveCheckpointAndValidate, CkptDir("save_then_round_trip_validate")));
    } catch (...)
    {
        // Real failures (stale shm files, IPC bugs, etc.) propagate — the
        // pre-checks above already SKIP the known "this sandbox can't run
        // it" cases. Anything past that point is a regression worth seeing.
        cleanup();
        throw;
    }
    cleanup();
}

// Custom main so we can register `--registry` on the Catch2 CLI. The default
// `Catch2WithMain` errors on unknown args, which would break the
// `EXTRA_ARGS --registry <path>` we feed in from CMake.
int main(int argc, char* argv[])
{
    Catch::Session session;
    std::string    registry;
    using namespace Catch::Clara;
    auto cli = session.cli() | Opt(registry, "path")["--registry"]("path to a serialized state registry");
    session.cli(cli);

    int const cliRc = session.applyCommandLine(argc, argv);
    if (cliRc != 0) { return cliRc; }

    GRegistryPath = registry;
    return session.run();
}
