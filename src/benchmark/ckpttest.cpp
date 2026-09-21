#include "CommonMacros.h"

#include "byte_storage.h"
#include "ckpt_fixture.h"
#include "config.h"
#include "distcp.h"
#include "distcp_internal.h"
#include "logging.h"
#include "process_memory.h"
#include "server.h"
#include "session.h"
#include "state.h"
#include "telemetry.h"
#include "tensor.h"

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/iostreams/close.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <fmt/format.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <opentelemetry/proto/collector/logs/v1/logs_service.pb.h>
#include <opentelemetry/proto/collector/metrics/v1/metrics_service.pb.h>
#include <opentelemetry/proto/collector/trace/v1/trace_service.pb.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

// NOLINTBEGIN(readability-magic-numbers)

// Populated from --registry on the command line in main(). Tests that need
// it (the [round-trip] case) assert non-empty before use; ctest wires the
// path via catch_discover_tests(... EXTRA_ARGS --registry <path>) so the
// binary itself stays portable.

SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
static std::filesystem::path GRegistryPath;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
SUPPRESS_WARNINGS_END

static std::filesystem::path const& TestRegistryPath()
{ return GRegistryPath; }

TEST_CASE("shared-memory configuration parses sizes and positive counts", "[shmem][config][tier1]")
{
    using checkpoint::config::types::ByteSize;
    using checkpoint::config::types::Integer;
    using checkpoint::config::types::NonNegativeSize;
    using checkpoint::config::types::PositiveSize;
    using checkpoint::config::types::PositiveUint8;
    CHECK(ByteSize::Parse("48GiB") == 48ull * (1ull << 30));
    CHECK(ByteSize::Parse("5GB") == 5'000'000'000ull);
    CHECK_THROWS_AS(ByteSize::Parse("5gib"), std::invalid_argument);
    CHECK_THROWS_AS(ByteSize::Parse("-1"), std::invalid_argument);
    CHECK(PositiveSize::Parse("32") == 32);
    CHECK_THROWS_AS(PositiveSize::Parse("0"), std::invalid_argument);
    CHECK_THROWS_AS(PositiveSize::Parse("4x"), std::invalid_argument);
    CHECK(PositiveUint8::Parse("255") == 255);
    CHECK_THROWS_AS(PositiveUint8::Parse("256"), std::overflow_error);
    CHECK(NonNegativeSize::Parse("0") == 0);
    CHECK_THROWS_AS(NonNegativeSize::Parse("-1"), std::invalid_argument);
    CHECK(Integer<int8_t>::Parse("-128") == -128);
    CHECK(Integer<int8_t>::Parse("+127") == 127);
    CHECK_THROWS_AS(Integer<int8_t>::Parse("128"), std::overflow_error);
    CHECK_THROWS_AS(Integer<int8_t>::Parse("-129"), std::overflow_error);
}

TEST_CASE("shared-memory ring geometry supports runtime slot counts", "[shmem][config][tier1]")
{
    using Queue = checkpoint::ipc::MsgQueue;
    CHECK(Queue::HeaderAllocationSize(4) >= sizeof(Queue::Control) + (4 * sizeof(checkpoint::ipc::Message::Header)));
    CHECK(Queue::HeaderAllocationSize(32) >= sizeof(Queue::Control) + (32 * sizeof(checkpoint::ipc::Message::Header)));
    CHECK(Queue::MinimumShmSize(32) == Queue::HeaderAllocationSize(32) + (32 * Queue::AlignBoundary));
    CHECK_THROWS_AS(Queue::HeaderAllocationSize(0), std::invalid_argument);

    auto const name = fmt::format("nvrx_ckpttest_runtime_slots_{}", ::getpid());
    boost::interprocess::shared_memory_object::remove(name.c_str());
    auto created = Queue::Create(name, Queue::MinimumShmSize(4), 4);
    auto opened  = Queue::Open(name);
    CHECK(created.SlotCount() == 4);
    CHECK(opened.SlotCount() == 4);
    CHECK(opened.SlotSize() == Queue::AlignBoundary);
}

// Stem-based subdirectory under the test temp root. Not strictly unique across
// concurrent ctest workers, but every call site `remove_all`s before use, so
// single-process runs are safe and `ctest -jN` only collides if two distinct
// jobs share the same stem (we use distinct stems per TEST_CASE).
static std::filesystem::path CkptDir(std::string_view stem)
{
    auto base = std::filesystem::temp_directory_path() / "phoenix_ckpttest";
    std::filesystem::create_directories(base);
    return base / stem;
}

static RankTopology SingleRankTopo(uint32_t rank = 0, uint32_t worldSize = 1)
{
    return RankTopology{
        .rank         = Rank{rank},
        .worldSize    = Rank{worldSize},
        .ranksPerPeer = worldSize,
        .peersPerNode = 1,
        .peers        = {"localhost"},
    };
}
namespace
{
struct TestShardEntry
{
    char const*         srcPtr;
    uint64_t            length;
    StateTree::NodeId   fqn;
    DS4Tensor::DType    dtype;
    DS4Tensor::DimRange sharddim;
    DS4Tensor::Dim      shape;
};
}    // namespace
static std::vector<TestShardEntry> CollectShardsForTest(StateTree const& tree, Rank rank, ByteStorage const& tensorStorage)
{
    std::vector<TestShardEntry> out;
    auto const*                 base = static_cast<char const*>(tensorStorage.data);
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
// given FQN. `handle` is the byte offset into the surrounding ByteStorage.
static void
AddTensorShardLeaf(StateTree& tree, std::string_view fqn, DS4Tensor::DType dtype, std::vector<size_t> const& shape, uint64_t handle)
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

static void AddShardedTensorLeaf(StateTree&                 tree,
                                 std::string_view           fqn,
                                 DS4Tensor::DType           dtype,
                                 std::vector<size_t> const& shape,
                                 Rank                       rank,
                                 uint64_t                   handle)
{
    DS4Tensor::Dim fulldim;
    for (auto d : shape) { fulldim.push_back(d); }

    auto nodeHandle = tree.FindNodeByFQN(fqn);
    if (nodeHandle == StateNode::Handle::Invalid)
    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
        auto& leaf              = tree.AddChild(parent, leafName);
        leaf.data               = StateNode::ShardedTensor(dtype, fulldim);
        nodeHandle              = leaf.handle;
    }
    auto& st = std::get<StateNode::ShardedTensor>(tree.GetNode(nodeHandle).data);
    if (st.dtype != dtype || st.fulldim != fulldim) { throw std::runtime_error("Incompatible test tensor replica"); }

    DS4Tensor::DimRange sharddim;
    for (auto d : shape) { sharddim.push_back({.begin = 0, .end = d}); }
    st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = handle, .rank = rank});
}

TEST_CASE("scaffolding", "[scaffolding]")
{ REQUIRE(true); }

TEST_CASE("state registry permits distinct local shards under one tensor key", "[state][tier1]")
{
    StateTree tree(SingleRankTopo());

    auto addShard = [&](size_t layer, size_t expert) {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("weight");
        auto&     leaf          = tree.AddChild(parent, leafName);
        DS4Tensor tensor;
        tensor.dtype  = DS4Tensor::DType::Float32;
        tensor.device = DeviceId{"cpu"};
        for (auto dim : {size_t{2}, size_t{2}, size_t{4}}) { tensor.fulldim.push_back(dim); }
        tensor.sharddim.push_back({.begin = layer, .end = layer + 1});
        tensor.sharddim.push_back({.begin = expert, .end = expert + 1});
        tensor.sharddim.push_back({.begin = 0, .end = 4});
        leaf.data = tensor;
    };

    addShard(1, 1);
    addShard(0, 1);
    addShard(1, 0);
    addShard(0, 0);
    REQUIRE_NOTHROW(tree.FinalizeRegistry());

    std::vector<DS4Tensor::DimRange> ranges;
    tree.WalkNodes([&](StateNode const& node) {
        if (tree.GetNodeId(node.handle) == "weight") { ranges.push_back(std::get<StateNode::TensorShard>(node.data).sharddim); }
    });
    REQUIRE(ranges.size() == 4);
    REQUIRE(ranges[0][0].begin == 0);
    REQUIRE(ranges[0][1].begin == 0);
    REQUIRE(ranges[1][0].begin == 0);
    REQUIRE(ranges[1][1].begin == 1);
    REQUIRE(ranges[2][0].begin == 1);
    REQUIRE(ranges[2][1].begin == 0);
    REQUIRE(ranges[3][0].begin == 1);
    REQUIRE(ranges[3][1].begin == 1);
}

TEST_CASE("checkpoint telemetry preserves the performance record schema", "[telemetry][tier1]")
{
    auto const outDir     = CkptDir("telemetry_schema");
    auto const stderrPath = outDir / "stderr.log";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    int const savedStderr = ::dup(STDERR_FILENO);
    int const captureFd   = ::creat(stderrPath.c_str(), S_IRUSR | S_IWUSR);
    REQUIRE(savedStderr >= 0);
    REQUIRE(captureFd >= 0);
    std::fflush(stderr);
    REQUIRE(::dup2(captureFd, STDERR_FILENO) >= 0);
    ::close(captureFd);

    PERF() << checkpoint::telemetry::span::CkptSave::Begin{
        .id        = {.sessionId = checkpoint::ipc::SessionId{3}, .rank = Rank{1}, .snapshotId = checkpoint::ipc::SnapshotId{2}},
        .startedAt = std::chrono::nanoseconds{123456},
    };
    PERF() << checkpoint::telemetry::span::CkptSave::End{
        .id                = {.sessionId = checkpoint::ipc::SessionId{3}, .rank = Rank{1}, .snapshotId = checkpoint::ipc::SnapshotId{2}},
        .d2hDuration       = std::chrono::seconds{1},
        .snapshotDuration  = std::chrono::microseconds{15},
        .totalDuration     = std::chrono::microseconds{20},
        .numTensors        = 2,
        .d2hRequestCount   = 4,
        .totalTensorBytes  = 1024ull * 1024,
        .totalWrittenBytes = 2048,
    };
    std::fflush(stderr);
    int const restoreResult = ::dup2(savedStderr, STDERR_FILENO);
    ::close(savedStderr);
    REQUIRE(restoreResult >= 0);

    std::ifstream recordsFile(stderrPath);
    REQUIRE(recordsFile.good());
    std::string const records((std::istreambuf_iterator<char>(recordsFile)), std::istreambuf_iterator<char>());
    std::string const beginRecord = "[native_ckpt:svr][3][PERF] Rank=1 Begin span=ckpt.save started_at_ns=123456 snapshot=2\n";
    REQUIRE(records.starts_with(beginRecord));
    auto const durationSuffix = std::string{R"((?:u|µ)s)"};
    auto const endPattern     = std::regex{
        R"(\[native_ckpt:svr\]\[3\]\[PERF\] Rank=1 Finished bw_mbps=1\.00 \| Time\(us\):  wr_tnsr_us=1000000)" + durationSuffix
        + R"( snapshot_us=15)" + durationSuffix + R"( total_us=20)" + durationSuffix
        + R"( \| Count: num_tensors=2 total_d2h=4 \| Bytes: total_written_bytes=2048 avg_d2h_rq=262144 avg_tnsr=524288 snapshot=2\n)"};
    REQUIRE(std::regex_match(records.substr(beginRecord.size()), endPattern));
    std::filesystem::remove_all(outDir);
}

TEST_CASE("snapshot completion is global on the coordinator and local on peers", "[server][completion]")
{
    auto makeSession = [](uint32_t rank) {
        return std::make_unique<Session>(
            nullptr,
            checkpoint::ipc::SessionId{1},
            std::filesystem::path{},
            RankTopology{.rank = Rank{rank}, .worldSize = Rank{4}, .ranksPerPeer = 2, .peersPerNode = 1, .peers = {"coordinator", "peer"}});
    };
    auto addSnapshot = [](Session& session) -> Session::Snapshot& {
        auto snapshot        = std::make_unique<Session::Snapshot>();
        snapshot->snapshotId = checkpoint::ipc::SnapshotId{1};
        return *session.snapshots.emplace(snapshot->snapshotId, std::move(snapshot)).first->second;
    };

    auto  coordinator         = makeSession(0);
    auto& coordinatorSnapshot = addSnapshot(*coordinator);
    for (uint32_t rank = 1; rank < 4; ++rank) { coordinatorSnapshot.writeCkptCompletedRanks.insert(Rank{rank}); }
    REQUIRE_FALSE(coordinator->AllSnapshotsComplete());
    coordinatorSnapshot.writeCkptCompletedRanks.insert(Rank{0});
    REQUIRE_FALSE(coordinator->AllSnapshotsComplete());
    REQUIRE_FALSE(coordinator->IsSnapshotEvictable(coordinatorSnapshot));
    for (uint32_t rank = 0; rank < 4; ++rank) { coordinatorSnapshot.finalizedRanks.insert(Rank{rank}); }
    REQUIRE(coordinator->AllSnapshotsComplete());
    REQUIRE(coordinator->IsSnapshotEvictable(coordinatorSnapshot));

    auto  peer         = makeSession(2);
    auto& peerSnapshot = addSnapshot(*peer);
    peerSnapshot.writeCkptCompletedRanks.insert(Rank{2});
    REQUIRE_FALSE(peer->AllSnapshotsComplete());
    peerSnapshot.writeCkptCompletedRanks.insert(Rank{3});
    REQUIRE_FALSE(peer->AllSnapshotsComplete());
    REQUIRE_FALSE(peer->IsSnapshotEvictable(peerSnapshot));
    for (uint32_t rank = 0; rank < 4; ++rank) { peerSnapshot.finalizedRanks.insert(Rank{rank}); }
    REQUIRE(peer->AllSnapshotsComplete());
    REQUIRE(peer->IsSnapshotEvictable(peerSnapshot));
}

TEST_CASE("distcp shard walk visits TensorShard leaves in insertion order", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(64);

    AddTensorShardLeaf(tree, "a", DS4Tensor::DType::Float32, {2, 2}, 0);    // 16 B
    AddTensorShardLeaf(tree, "b", DS4Tensor::DType::Int8, {4}, 16);         //  4 B
    AddTensorShardLeaf(tree, "c", DS4Tensor::DType::Float64, {2}, 32);      // 16 B

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

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

    auto const* base = static_cast<char const*>(tensorStorage.data);
    REQUIRE(entries[0].srcPtr == base + 0);
    REQUIRE(entries[1].srcPtr == base + 16);
    REQUIRE(entries[2].srcPtr == base + 32);
}

TEST_CASE("distcp shard walk derives shape from sharddim for partial slices", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(128);

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

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 2);

    REQUIRE(entries[0].shape.size() == 1);
    REQUIRE(entries[0].shape[0] == 4);

    REQUIRE(entries[1].shape.size() == 2);
    REQUIRE(entries[1].shape[0] == 4);
    REQUIRE(entries[1].shape[1] == 2);
    REQUIRE(entries[1].length == 4ull * 2u * 4u);    // 8 int32 elements
}

TEST_CASE("distcp shard walk expands ShardedTensor into per-rank entries", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo(0, 2));
    ByteStorage tensorStorage(64);

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

    auto const rank0 = CollectShardsForTest(tree, Rank{0}, tensorStorage);
    REQUIRE(rank0.size() == 1);
    REQUIRE(rank0[0].fqn == "shrd");
    REQUIRE(rank0[0].sharddim[0].begin == 0);
    REQUIRE(rank0[0].length == 16);
    REQUIRE(rank0[0].srcPtr == static_cast<char const*>(tensorStorage.data) + 0);

    auto const rank1 = CollectShardsForTest(tree, Rank{1}, tensorStorage);
    REQUIRE(rank1.size() == 1);
    REQUIRE(rank1[0].sharddim[0].begin == 4);
    REQUIRE(rank1[0].srcPtr == static_cast<char const*>(tensorStorage.data) + 16);
}

TEST_CASE("distcp shard walk orders same-rank shards by sharddim", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo(0, 2));
    ByteStorage tensorStorage(64);

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

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].sharddim[0].begin == 0);
    REQUIRE(entries[1].sharddim[0].begin == 4);
}

TEST_CASE("distcp shard walk handles mixed TensorShard and ShardedTensor nodes", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo(0, 2));
    ByteStorage tensorStorage(128);

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

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "before");
    REQUIRE(entries[1].fqn == "middle");
    REQUIRE(entries[2].fqn == "after");
}

TEST_CASE("distcp shard walk joins nested dict names with dots", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(64);

    AddTensorShardLeaf(tree, "model.layer0.weight", DS4Tensor::DType::Float32, {2}, 0);
    AddTensorShardLeaf(tree, "model.layer0.bias", DS4Tensor::DType::Float32, {2}, 8);
    AddTensorShardLeaf(tree, "model.layer1.weight", DS4Tensor::DType::Float32, {2}, 16);

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "model.layer0.weight");
    REQUIRE(entries[1].fqn == "model.layer0.bias");
    REQUIRE(entries[2].fqn == "model.layer1.weight");
}

TEST_CASE("distcp shard walk skips zero-length shards", "[distcp][tier1]")
{
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(16);

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

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].fqn == "non_empty");
}

static constexpr char KZipLocalFileMagic[4] = {'P', 'K', '\x03', '\x04'};    // NOLINT

static std::vector<char> ReadFile(std::filesystem::path const& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error(fmt::format("[ckpttest] cannot open {}", path.string())); }
    std::vector<char> out((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return out;
}

TEST_CASE("OpenTelemetry exports checkpoint spans and opt-in metrics", "[telemetry][tier1]")
{
    auto const outDir      = CkptDir("otel_sdk_export");
    auto const spansPath   = outDir / "telemetry.jsonl";
    auto const metricsPath = std::filesystem::path{spansPath.string() + ".metrics"};
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    ::setenv("NATIVE_CKPT_OTEL_EXPORT", "file", 1);             // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_FILE", spansPath.c_str(), 1);    // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_METRICS", "1", 1);               // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_LOGS", "0", 1);                  // NOLINT(concurrency-mt-unsafe)
    ::setenv("MEGATRON_OTEL_RUN_ID", "test-run", 1);            // NOLINT(concurrency-mt-unsafe)
    checkpoint::telemetry::Init();

    auto const startedAt = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
                           - std::chrono::microseconds{20};
    PERF() << checkpoint::telemetry::span::CkptSave::Begin{
        .id        = {.sessionId = checkpoint::ipc::SessionId{3}, .rank = Rank{1}, .snapshotId = checkpoint::ipc::SnapshotId{2}},
        .startedAt = startedAt,
    };
    PERF() << checkpoint::telemetry::span::CkptSave::End{
        .id                = {.sessionId = checkpoint::ipc::SessionId{3}, .rank = Rank{1}, .snapshotId = checkpoint::ipc::SnapshotId{2}},
        .d2hDuration       = std::chrono::microseconds{10},
        .snapshotDuration  = std::chrono::microseconds{15},
        .totalDuration     = std::chrono::microseconds{20},
        .numTensors        = 2,
        .d2hRequestCount   = 4,
        .totalTensorBytes  = 1024,
        .totalWrittenBytes = 2048,
    };
    auto const durableStartedAt = std::chrono::system_clock::now() - std::chrono::microseconds{30};
    auto const durableId        = checkpoint::telemetry::Id{
        .sessionId  = checkpoint::ipc::SessionId{3},
        .rank       = Rank{0},
        .snapshotId = checkpoint::ipc::SnapshotId{2},
    };
    PERF() << checkpoint::telemetry::span::SvrDurable::Begin{.id = durableId, .startedAt = durableStartedAt};
    PERF() << checkpoint::telemetry::span::SvrDurable::End{.id = durableId};
    checkpoint::telemetry::Shutdown();
    ::unsetenv("NATIVE_CKPT_OTEL_EXPORT");     // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_FILE");       // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_METRICS");    // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_LOGS");       // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("MEGATRON_OTEL_RUN_ID");        // NOLINT(concurrency-mt-unsafe)

    REQUIRE(std::filesystem::exists(spansPath));
    REQUIRE(std::filesystem::exists(metricsPath));
    auto const spanBytes   = ReadFile(spansPath);
    auto const metricBytes = ReadFile(metricsPath);
    auto const spanJson    = std::string(spanBytes.begin(), spanBytes.end());
    auto const metricJson  = std::string(metricBytes.begin(), metricBytes.end());
    REQUIRE(spanJson.contains("ckpt.save"));
    REQUIRE(spanJson.contains(checkpoint::telemetry::span::SvrDurable::Name));
    REQUIRE(spanJson.contains("nvrx.session_id"));
    REQUIRE(spanJson.contains("nvrx.snapshot_id"));
    REQUIRE(spanJson.contains("test-run"));
    REQUIRE(spanJson.contains(fmt::format("\"startTimeUnixNano\":\"{}\"", startedAt.count())));
    REQUIRE(spanJson.contains(fmt::format("\"endTimeUnixNano\":\"{}\"", (startedAt + std::chrono::microseconds{20}).count())));
    REQUIRE(metricJson.contains("nvrx.ckpt.save.bw_mbps"));
    REQUIRE(metricJson.contains("nvrx.ckpt.save.wr_tnsr_us"));
    REQUIRE(metricJson.contains("nvrx.ckpt.save.total_written_bytes"));
    std::filesystem::remove_all(outDir);
}

TEST_CASE("Process memory samples export through OpenTelemetry", "[telemetry][memory][tier1]")
{
    auto const outDir   = CkptDir("otel_process_memory");
    auto const outPath  = outDir / "telemetry.jsonl";
    auto const logsPath = std::filesystem::path{outPath.string() + ".logs"};
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    ::setenv("NATIVE_CKPT_OTEL_EXPORT", "file", 1);           // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_FILE", outPath.c_str(), 1);    // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_LOGS", "1", 1);                // NOLINT(concurrency-mt-unsafe)
    checkpoint::telemetry::Init();

    auto const id = checkpoint::telemetry::Id{
        .sessionId  = checkpoint::ipc::SessionId{3},
        .rank       = Rank{1},
        .snapshotId = checkpoint::ipc::SnapshotId{2},
    };
    PERF() << checkpoint::telemetry::span::SvrFinalize::Begin{.id = id};
    PERF() << checkpoint::telemetry::span::SvrFinalize::End{.id = id};
    checkpoint::telemetry::Shutdown();
    ::unsetenv("NATIVE_CKPT_OTEL_EXPORT");    // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_FILE");      // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_LOGS");      // NOLINT(concurrency-mt-unsafe)

    REQUIRE(std::filesystem::exists(logsPath));
    auto const bytes = ReadFile(logsPath);
    auto const json  = std::string(bytes.begin(), bytes.end());
    REQUIRE(json.contains(checkpoint::telemetry::event::ProcessMemory::Name));
    REQUIRE(json.contains("nvrx-checkpoint-server"));
    REQUIRE(json.contains("nvrx.memory.rss_bytes"));
    REQUIRE(json.contains("nvrx.memory.peak_rss_bytes"));
    REQUIRE(json.contains("nvrx.memory.trigger"));
    REQUIRE(json.contains(checkpoint::telemetry::span::SvrFinalize::Name));
    REQUIRE(json.contains("\"key\":\"nvrx.session_id\",\"value\":{\"intValue\":\"3\"}"));
    REQUIRE(json.contains("\"key\":\"nvrx.rank\",\"value\":{\"intValue\":\"1\"}"));
    REQUIRE(json.contains("\"key\":\"nvrx.snapshot_id\",\"value\":{\"intValue\":\"2\"}"));
    REQUIRE_FALSE(json.contains("nvrx.process.role"));
    std::filesystem::remove_all(outDir);
}

TEST_CASE("Logging writes stderr records to OpenTelemetry", "[telemetry][tier1]")
{
    auto const outDir     = CkptDir("otel_log_export");
    auto const outPath    = outDir / "telemetry.jsonl";
    auto const logsPath   = std::filesystem::path{outPath.string() + ".logs"};
    auto const stderrPath = outDir / "stderr.log";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    ::setenv("NATIVE_CKPT_OTEL_EXPORT", "file", 1);           // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_FILE", outPath.c_str(), 1);    // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_METRICS", "0", 1);             // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_LOGS", "1", 1);                // NOLINT(concurrency-mt-unsafe)
    checkpoint::telemetry::Init();
    checkpoint::logging::Init();

    int const savedStderr = ::dup(STDERR_FILENO);
    int const captureFd   = ::creat(stderrPath.c_str(), S_IRUSR | S_IWUSR);
    REQUIRE(savedStderr >= 0);
    REQUIRE(captureFd >= 0);
    std::fflush(stderr);
    REQUIRE(::dup2(captureFd, STDERR_FILENO) >= 0);
    ::close(captureFd);

    static constexpr int TestValue = 42;
    checkpoint::logging::Info("combined sink record {}", TestValue);
    checkpoint::logging::Warn("second sink record");
    checkpoint::logging::Shutdown();
    std::fflush(stderr);
    int const restoreResult = ::dup2(savedStderr, STDERR_FILENO);
    ::close(savedStderr);

    checkpoint::telemetry::Shutdown();
    ::unsetenv("NATIVE_CKPT_OTEL_EXPORT");     // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_FILE");       // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_METRICS");    // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_LOGS");       // NOLINT(concurrency-mt-unsafe)

    REQUIRE(restoreResult >= 0);
    auto const stderrBytes = ReadFile(stderrPath);
    auto const stderrText  = std::string(stderrBytes.begin(), stderrBytes.end());
    REQUIRE(stderrText == "[native_ckpt:svr][INFO] combined sink record 42\n[native_ckpt:svr][WARN] second sink record\n");

    REQUIRE(std::filesystem::exists(logsPath));
    auto const logBytes = ReadFile(logsPath);
    auto const logJson  = std::string(logBytes.begin(), logBytes.end());
    REQUIRE(logJson.contains("combined sink record 42"));
    REQUIRE(logJson.contains("second sink record"));
    REQUIRE_FALSE(logJson.contains("combined sink record 42\\n"));
    REQUIRE_FALSE(logJson.contains("second sink record\\n"));
    REQUIRE_FALSE(logJson.contains("[native_ckpt:svr][INFO] combined sink record 42"));
    REQUIRE_FALSE(logJson.contains("nvrx.memory.rss_bytes"));
    std::filesystem::remove_all(outDir);
}

TEST_CASE("OpenTelemetry daemon mode routes through its file collector", "[telemetry][tier1]")
{
    auto const outDir  = CkptDir("otel_daemon_export");
    auto const outPath = outDir / "telemetry.jsonl";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    ::setenv("NATIVE_CKPT_OTEL_EXPORT", "daemon", 1);         // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_FILE", outPath.c_str(), 1);    // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_LOGS", "0", 1);                // NOLINT(concurrency-mt-unsafe)
    checkpoint::telemetry::Init();
    checkpoint::telemetry::Collect(R"({"discarded":true})");
    checkpoint::telemetry::Shutdown();

    checkpoint::telemetry::Init();
    checkpoint::telemetry::InitLocalCollector(outPath);
    checkpoint::telemetry::InitLocalCollector(outPath);
    checkpoint::telemetry::Collect(R"({"written_once":true})");
    checkpoint::telemetry::Collect(R"({"second_local_line":true})");
    checkpoint::telemetry::Collect(R"({"received_remotely":true})");
    checkpoint::telemetry::Collect(R"({"second_remote_line":true})");
    auto const spanId = checkpoint::telemetry::Id{
        .sessionId  = checkpoint::ipc::SessionId::Default,
        .rank       = Rank{0},
        .snapshotId = checkpoint::ipc::SnapshotId::Default,
    };
    PERF() << checkpoint::telemetry::span::SvrWriteMetadata::Begin{.id = spanId};
    PERF() << checkpoint::telemetry::span::SvrWriteMetadata::End{.id = spanId};
    checkpoint::telemetry::Shutdown();
    checkpoint::telemetry::Shutdown();
    ::unsetenv("NATIVE_CKPT_OTEL_EXPORT");    // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_FILE");      // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_LOGS");      // NOLINT(concurrency-mt-unsafe)

    REQUIRE(std::filesystem::exists(outPath));
    auto const bytes = ReadFile(outPath);
    auto const json  = std::string(bytes.begin(), bytes.end());
    REQUIRE(json.contains(checkpoint::telemetry::span::SvrWriteMetadata::Name));
    REQUIRE(json.contains("{\"received_remotely\":true}\n{\"second_remote_line\":true}\n"));
    REQUIRE(json.contains("{\"written_once\":true}\n{\"second_local_line\":true}\n"));
    REQUIRE_FALSE(json.contains(R"({"discarded":true})"));
    auto const writtenOnce = json.find(R"({"written_once":true})");
    REQUIRE(writtenOnce != std::string::npos);
    REQUIRE(writtenOnce == json.rfind(R"({"written_once":true})"));
    std::filesystem::remove_all(outDir);
}

TEST_CASE("OpenTelemetry both mode selects file and daemon routes", "[telemetry][tier1]")
{
    checkpoint::config::OtelConfig config;

    config.mode = checkpoint::config::OtelExportMode::File;
    REQUIRE(config.WritesProcessFile());
    REQUIRE_FALSE(config.RoutesThroughDaemon());

    config.mode = checkpoint::config::OtelExportMode::Daemon;
    REQUIRE_FALSE(config.WritesProcessFile());
    REQUIRE(config.RoutesThroughDaemon());

    config.mode = checkpoint::config::OtelExportMode::Both;
    REQUIRE(config.WritesProcessFile());
    REQUIRE(config.RoutesThroughDaemon());
}

TEST_CASE("OpenTelemetry file presence enables telemetry relative to the server log", "[telemetry][tier1]")
{
    auto const outDir    = CkptDir("otel_relative_file");
    auto const serverLog = outDir / "server.%h.%p.log";
    ::unsetenv("NATIVE_CKPT_OTEL_FILE");    // NOLINT(concurrency-mt-unsafe)
    REQUIRE_FALSE(checkpoint::config::OtelConfig{}.enabled);

    ::setenv("NATIVE_CKPT_SERVER_LOG", serverLog.c_str(), 1);         // NOLINT(concurrency-mt-unsafe)
    ::setenv("NATIVE_CKPT_OTEL_FILE", "telemetry.%h.%p.jsonl", 1);    // NOLINT(concurrency-mt-unsafe)
    checkpoint::config::OtelConfig const config;

    REQUIRE(config.enabled);
    REQUIRE(config.ExportFile().parent_path() == outDir);
    REQUIRE(config.ExportFile().filename() == checkpoint::config::Substitute("telemetry.%h.%p.jsonl"));
    REQUIRE(config.CollectorFile() == config.ExportFile());

    ::unsetenv("NATIVE_CKPT_SERVER_LOG");    // NOLINT(concurrency-mt-unsafe)
    ::unsetenv("NATIVE_CKPT_OTEL_FILE");     // NOLINT(concurrency-mt-unsafe)
}

TEST_CASE("OTLP HTTP protobuf signals sink as canonical JSONL", "[telemetry][tier1]")
{
    namespace proto_logs    = opentelemetry::proto::collector::logs::v1;
    namespace proto_metrics = opentelemetry::proto::collector::metrics::v1;
    namespace proto_trace   = opentelemetry::proto::collector::trace::v1;

    auto const outDir  = CkptDir("otlp_http_receiver");
    auto const outPath = outDir / "telemetry.jsonl";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);
    checkpoint::telemetry::Shutdown();

    proto_trace::ExportTraceServiceRequest bufferedRequest;
    bufferedRequest.add_resource_spans()->add_scope_spans()->add_spans()->set_name("received.before_topology");
    auto const bufferedResponse
        = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/x-protobuf", {}, bufferedRequest.SerializeAsString());
    checkpoint::telemetry::InitLocalCollector(outPath);

    proto_trace::ExportTraceServiceRequest traceRequest;
    auto*                                  span = traceRequest.add_resource_spans()->add_scope_spans()->add_spans();
    span->set_name("received.trace");
    span->set_trace_id("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16);
    span->set_span_id("\x10\x11\x12\x13\x14\x15\x16\x17", 8);
    span->set_start_time_unix_nano(10);
    span->set_end_time_unix_nano(20);
    auto const traceResponse
        = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/x-protobuf", {}, traceRequest.SerializeAsString());

    proto_metrics::ExportMetricsServiceRequest metricsRequest;
    auto*                                      metric = metricsRequest.add_resource_metrics()->add_scope_metrics()->add_metrics();
    metric->set_name("received.metric");
    metric->mutable_gauge()->add_data_points()->set_as_double(42.0);
    std::string compressedMetrics;
    {
        boost::iostreams::filtering_ostream output;
        output.push(boost::iostreams::gzip_compressor{});
        output.push(boost::iostreams::back_inserter(compressedMetrics));
        auto const bytes = metricsRequest.SerializeAsString();
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        boost::iostreams::close(output);
    }
    auto const metricsResponse = checkpoint::telemetry::ReceiveOtlp("/v1/metrics", "application/x-protobuf", "gzip", compressedMetrics);

    proto_logs::ExportLogsServiceRequest logsRequest;
    auto*                                logRecord = logsRequest.add_resource_logs()->add_scope_logs()->add_log_records();
    logRecord->mutable_body()->set_string_value("received log");
    auto const logsResponse = checkpoint::telemetry::ReceiveOtlp("/v1/logs", "application/x-protobuf", {}, logsRequest.SerializeAsString());

    auto const invalidResponse
        = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/x-protobuf", {}, std::string_view{"\xff", 1});
    auto const unsupportedResponse
        = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/json", {}, traceRequest.SerializeAsString());
    auto const invalidGzipResponse = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/x-protobuf", "gzip", "not gzip");
    auto const emptyResponse       = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/x-protobuf", {}, {});
    checkpoint::telemetry::Shutdown();

    REQUIRE(bufferedResponse.status == 200);
    REQUIRE(bufferedResponse.body.empty());
    REQUIRE(traceResponse.status == 200);
    REQUIRE(traceResponse.body.empty());
    REQUIRE(metricsResponse.status == 200);
    REQUIRE(metricsResponse.body.empty());
    REQUIRE(logsResponse.status == 200);
    REQUIRE(logsResponse.body.empty());
    REQUIRE(invalidResponse.status == 400);
    REQUIRE_FALSE(invalidResponse.body.empty());
    REQUIRE(unsupportedResponse.status == 415);
    REQUIRE_FALSE(unsupportedResponse.body.empty());
    REQUIRE(invalidGzipResponse.status == 400);
    REQUIRE_FALSE(invalidGzipResponse.body.empty());
    REQUIRE(emptyResponse.status == 200);
    REQUIRE(emptyResponse.body.empty());

    std::ifstream            file(outPath);
    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);) { lines.push_back(std::move(line)); }
    REQUIRE(lines.size() == 4);
    auto const json = fmt::format("{}\n{}\n{}\n{}", lines[0], lines[1], lines[2], lines[3]);
    REQUIRE(json.contains("received.before_topology"));
    REQUIRE(json.contains("received.trace"));
    REQUIRE(json.contains(R"("traceId":"000102030405060708090a0b0c0d0e0f")"));
    REQUIRE(json.contains(R"("spanId":"1011121314151617")"));
    REQUIRE(json.contains("received.metric"));
    REQUIRE(json.contains("received log"));
    std::filesystem::remove_all(outDir);
}

TEST_CASE("OTLP relay forwards the encoded request unchanged", "[telemetry][tier1]")
{
    boost::asio::io_context        ioc;
    boost::asio::ip::tcp::acceptor acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    std::string                    target;
    std::string                    contentType;
    std::string                    contentEncoding;
    std::string                    body;

    opentelemetry::proto::collector::trace::v1::ExportTraceServiceResponse downstreamResponse;
    downstreamResponse.mutable_partial_success()->set_error_message("forwarded response");
    auto const   downstreamBody = downstreamResponse.SerializeAsString();
    std::jthread receiver([&] {
        boost::asio::ip::tcp::socket socket{ioc};
        acceptor.accept(socket);
        boost::beast::flat_buffer                                    buffer;
        boost::beast::http::request<boost::beast::http::string_body> request;
        boost::beast::http::read(socket, buffer, request);
        target          = request.target();
        contentType     = request[boost::beast::http::field::content_type];
        contentEncoding = request[boost::beast::http::field::content_encoding];
        body            = request.body();

        boost::beast::http::response<boost::beast::http::string_body> response{boost::beast::http::status::ok, request.version()};
        response.set(boost::beast::http::field::content_type, "application/x-protobuf");
        response.keep_alive(false);
        response.body() = downstreamBody;
        response.prepare_payload();
        boost::beast::http::write(socket, response);
    });

    checkpoint::telemetry::InitRelayCollector(fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port()));
    std::string const opaqueBody{"\x00\xff\x01\x02", 4};
    auto const        response = checkpoint::telemetry::ReceiveOtlp("/v1/traces", "application/x-protobuf", "gzip", opaqueBody);
    receiver.join();
    checkpoint::telemetry::Shutdown();

    REQUIRE(target == "/v1/traces");
    REQUIRE(contentType == "application/x-protobuf");
    REQUIRE(contentEncoding == "gzip");
    REQUIRE(body == opaqueBody);
    REQUIRE(response.status == 200);
    REQUIRE(response.body == downstreamBody);
}

TEST_CASE("Telemetry relay retries one opaque export batch", "[telemetry][tier1]")
{
    boost::asio::io_context        ioc;
    boost::asio::ip::tcp::acceptor acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    std::vector<std::string>       bodies;
    std::vector<bool>              chunked;
    std::vector<std::string>       contentTypes;
    std::jthread                   receiver([&] {
        for (size_t remaining = 3; remaining != 0; --remaining)
        {
            boost::asio::ip::tcp::socket socket{ioc};
            acceptor.accept(socket);
            boost::beast::flat_buffer                                    buffer;
            boost::beast::http::request<boost::beast::http::string_body> request;
            boost::beast::http::read(socket, buffer, request);
            bodies.push_back(request.body());
            chunked.push_back(request.chunked());
            contentTypes.emplace_back(request[boost::beast::http::field::content_type]);

            boost::beast::http::response<boost::beast::http::string_body> response{boost::beast::http::status::service_unavailable,
                                                                                   request.version()};
            response.keep_alive(false);
            response.body() = "{}";
            response.prepare_payload();
            boost::beast::http::write(socket, response);
        }
    });

    checkpoint::telemetry::InitRelayCollector(fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port()));
    REQUIRE_THROWS(checkpoint::telemetry::Collect(R"({"resourceSpans":[]})"));
    receiver.join();

    REQUIRE(bodies == std::vector<std::string>{R"({"resourceSpans":[]})", R"({"resourceSpans":[]})", R"({"resourceSpans":[]})"});
    REQUIRE(chunked == std::vector<bool>{false, false, false});
    REQUIRE(contentTypes == std::vector<std::string>{"application/json", "application/json", "application/json"});
    REQUIRE_NOTHROW(checkpoint::telemetry::Shutdown());
}

namespace
{
struct PredictedRecord
{
    std::string fqn;
    uint64_t    fileOffset{0};
    uint64_t    length{0};
};
}    // namespace

static std::vector<PredictedRecord> PredictRankLayout(StateTree const& tree)
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

TEST_CASE("distcp::WriteRankShard produces a contiguous file of torch.save ZIPs", "[distcp][tier2]")
{
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(64);
    // Deterministic source bytes — keep them simple so any future stride/dtype
    // bug shows up as a torch.load mismatch on inspection.
    auto* bytes = static_cast<unsigned char*>(tensorStorage.data);
    for (size_t i = 0; i < tensorStorage.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }

    AddShardedTensorLeaf(tree, "a", DS4Tensor::DType::Float32, {2, 2}, Rank{0}, 0);    // 16 B
    AddShardedTensorLeaf(tree, "b", DS4Tensor::DType::Int8, {4}, Rank{0}, 16);         //  4 B
    AddShardedTensorLeaf(tree, "c", DS4Tensor::DType::Float64, {2}, Rank{0}, 32);      // 16 B

    auto outDir = CkptDir("tier2_layout");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, tensorStorage, {});

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

TEST_CASE("distcp::WriteRankShard omits ranks with no local payload", "[distcp][tier1][dedup]")
{
    StateTree globalTree(SingleRankTopo(0, 2));
    AddShardedTensorLeaf(globalTree, "weight", DS4Tensor::DType::Float32, {2, 2}, Rank{0}, 0);

    auto outDir = CkptDir("empty_rank_omitted");
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    // A retry must remove a stale/partial file left by an earlier attempt.
    auto const emptyRankPath = outDir / "__1_0.distcp";
    {
        std::ofstream stale(emptyRankPath, std::ios::binary);
        stale << "stale";
    }
    REQUIRE(std::filesystem::exists(emptyRankPath));

    ByteStorage emptyTensorStorage(0);
    checkpoint::distcp::WriteRankShard(outDir, Rank{1}, globalTree, emptyTensorStorage, {});

    REQUIRE_FALSE(std::filesystem::exists(emptyRankPath));
}

TEST_CASE("distcp::WriteRankShard runs concurrently across ranks", "[distcp][tier6]")
{
    constexpr uint32_t kWorldSize = 4;

    struct LocalState
    {
        std::unique_ptr<StateTree>   tree;
        std::unique_ptr<ByteStorage> tensorStorage;
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
        auto& ls         = snapshot.states[r];
        ls.tensorStorage = std::make_unique<ByteStorage>(64);
        // Rank-specific deterministic bytes so a cross-rank write would surface
        // as bogus payload at a recorded (offset, length).
        auto* bytes = static_cast<unsigned char*>(ls.tensorStorage->data);
        for (size_t i = 0; i < ls.tensorStorage->size; ++i) { bytes[i] = static_cast<unsigned char>(((size_t{r} * 17u) + i) & 0xffu); }
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
            checkpoint::distcp::WriteRankShard(outDir, Rank{r}, *ls.tree, *ls.tensorStorage, {});
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
        for (size_t i = 0; i < signature.size(); ++i) { signature[i] = static_cast<unsigned char>(((size_t{r} * 17u) + i) & 0xffu); }
        auto const* needle = reinterpret_cast<char const*>(signature.data());    // NOLINT
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
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(256);
    auto*       bytes = static_cast<unsigned char*>(tensorStorage.data);
    for (size_t i = 0; i < tensorStorage.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }

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

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, tensorStorage, {});

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
            for (auto const& [b, e] : ranges) { sharddim.push_back({.begin = b, .end = e}); }
            st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = 0, .rank = r});
        }
        leaf.data = std::move(st);
    };
    // Tensor of full shape (8,) split into two halves across ranks 0 and 1.
    addShard("weight", DS4Tensor::DType::Float32, {8}, {{Rank{0}, {{0, 4}}}, {Rank{1}, {{4, 8}}}});
    addShard("bias", DS4Tensor::DType::Int64, {4}, {{Rank{0}, {{0, 2}}}, {Rank{1}, {{2, 4}}}});

    auto outDir = CkptDir("metadata_multi_rank_inference");
    std::filesystem::remove_all(outDir);

    REQUIRE_NOTHROW(checkpoint::distcp::WriteMetadata(outDir, globalTree, {}));
    auto const metaPath = outDir / ".metadata";
    REQUIRE(std::filesystem::exists(metaPath));
    REQUIRE(std::filesystem::file_size(metaPath) > 0);
}

TEST_CASE("distcp Blobs live in their rank's shard file", "[distcp][tier1][metadata][blob]")
{
    // Each rank appends its Blobs in fqn order after its tensor region.
    // The coordinator's FQN-grouped metadata view points at those local files.
    StateTree globalTree(SingleRankTopo(0, 2));

    std::vector<char> const alphaBytes{'a', 'l', 'p', 'h', 'a'};
    std::vector<char> const gammaBytes{'g'};
    std::vector<char> const betaBytes{'b', 'e', 't', 'a'};
    std::vector<char> const deltaBytes{'d', 'e', 'l', 't', 'a'};

    checkpoint::distcp::BlobsByFqn rank0Writes{
        {"alpha", std::string_view(alphaBytes.data(), alphaBytes.size())},
        {"gamma", std::string_view(gammaBytes.data(), gammaBytes.size())},
    };
    checkpoint::distcp::BlobsByFqn rank1Writes{
        {"alpha", std::string_view(alphaBytes.data(), alphaBytes.size())},
        {"beta", std::string_view(betaBytes.data(), betaBytes.size())},
        {"delta", std::string_view(deltaBytes.data(), deltaBytes.size())},
    };
    checkpoint::distcp::BlobMetadataByFqn view{
        {"alpha", {{Rank{0}, {.length = alphaBytes.size()}}, {Rank{1}, {.length = alphaBytes.size()}}}},
        {"beta", {{Rank{1}, {.length = betaBytes.size()}}}},
        {"delta", {{Rank{1}, {.length = deltaBytes.size()}}}},
        {"gamma", {{Rank{0}, {.length = gammaBytes.size()}}}},
    };

    auto outDir = CkptDir("blobs_per_rank");
    std::filesystem::remove_all(outDir);

    ByteStorage emptyTensorStorage(0);
    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, globalTree, emptyTensorStorage, rank0Writes);
    checkpoint::distcp::WriteRankShard(outDir, Rank{1}, globalTree, emptyTensorStorage, rank1Writes);
    REQUIRE_NOTHROW(checkpoint::distcp::WriteMetadata(outDir, globalTree, view));

    auto readFile = [](std::filesystem::path const& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    REQUIRE(readFile(outDir / "__0_0.distcp") == "alphag");
    REQUIRE(readFile(outDir / "__1_0.distcp") == "alphabetadelta");

    auto const metadata = readFile(outDir / ".metadata");
    REQUIRE(metadata.contains("__0_0.distcp"));
    REQUIRE(metadata.contains("__1_0.distcp"));

#ifdef PHOENIX_VENV_PYTHON
    // The lowest-rank alpha replica owns the metadata entry. Rank 1 still wrote
    // its replica, so beta's offset must skip those five unreferenced bytes.
    auto const metaPath = outDir / ".metadata";
    auto const cmd      = fmt::format("{} -c \"import pickle; m=pickle.load(open(r'{}','rb')); "
                                      "s={{i.fqn:(v.relative_path,v.offset,v.length) for i,v in m.storage_data.items()}}; "
                                      "assert s['alpha']==('__0_0.distcp',0,5); assert s['gamma']==('__0_0.distcp',5,1); "
                                      "assert s['beta']==('__1_0.distcp',5,4); assert s['delta']==('__1_0.distcp',9,5)\"",
                                      PHOENIX_VENV_PYTHON,
                                      metaPath.string());
    REQUIRE(std::system(cmd.c_str()) == 0);    // NOLINT
#endif

    auto inconsistentView = view;
    inconsistentView["alpha"][Rank{1}].length--;
    REQUIRE_THROWS(checkpoint::distcp::WriteMetadata(outDir, globalTree, inconsistentView));
}

#ifdef PHOENIX_VENV_PYTHON
TEST_CASE("distcp::WriteMetadata emits an unpickle-able .metadata", "[distcp][tier1][metadata]")
{
    // Single-rank case: localTree and globalTree are the same tree of
    // single-shard ShardedTensors. WriteRankShard requires the global-
    // registry shape (asserted internally), and WriteMetadata consumes the
    // resulting writeLog as a verification side-channel.
    StateTree   tree(SingleRankTopo());
    ByteStorage tensorStorage(64);
    auto*       bytes = static_cast<unsigned char*>(tensorStorage.data);
    for (size_t i = 0; i < tensorStorage.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }
    AddShardedTensorLeaf(tree, "weight", DS4Tensor::DType::Float32, {2, 2}, Rank{0}, 0);
    AddShardedTensorLeaf(tree, "block.weight", DS4Tensor::DType::Int64, {3}, Rank{0}, 16);

    auto outDir = CkptDir("metadata_unpickle");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, tensorStorage, {});
    checkpoint::distcp::WriteMetadata(outDir, tree, {});

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
    int const rc = std::system(cmd.c_str());    // NOLINT
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

    // Conservative RAM threshold. The daemon requests the configured size for
    // every IPC ring and halves the node-wide total toward the minimum viable
    // geometry when `/dev/shm` (tmpfs) is short on space; only touched slots
    // become resident. A worldSize=2 registry uses
    // empirically 8 queues. Add slack for the test process and the checkpoint
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

    try
    {
        // LoadStateRegistryFromPath stands up ctx.nodes in-process servers.
        ctx.ranksPerNode = 4;
        serverRunning    = true;

        LoadStateRegistryFromPath(ctx, registryPath);
        tasks = StartClientThreads(ctx);
        REQUIRE_NOTHROW(RunOnAllRanks(ctx, &SaveCheckpointAndValidate, CkptDir("save_then_round_trip_validate")));
    } catch (...)
    {
        // Real failures (stale shm files, IPC bugs, etc.) propagate — the
        // pre-checks above already SKIP the known "this sandbox can't run
        // it" cases. Anything past that point is a regression worth seeing.
        if (!tasks.empty()) { JoinClientThreads(ctx, tasks); }
        if (serverRunning) { StopInprocServer(); }
        throw;
    }
    if (!tasks.empty()) { JoinClientThreads(ctx, tasks); }
    if (serverRunning) { StopInprocServer(); }
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

// NOLINTEND(readability-magic-numbers)
