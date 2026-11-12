#include "CommonMacros.h"

#include "byte_storage.h"
#include "checkpoint.h"
#include "ckpt_fixture.h"
#include "config.h"
#include "distcp.h"
#include "distcp_internal.h"
#include "host_dequant.h"
#include "logging.h"
#include "quant.h"
#include "server.h"
#include "service_client.h"
#include "session.h"
#include "shm_ring_buffer.h"
#include "state.h"
#include "stateserdes.h"
#include "telemetry.h"
#include "tensor.h"

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/iostreams/close.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <fmt/format.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <opentelemetry/proto/collector/logs/v1/logs_service.pb.h>
#include <opentelemetry/proto/collector/metrics/v1/metrics_service.pb.h>
#include <opentelemetry/proto/collector/trace/v1/trace_service.pb.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <csignal>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

// NOLINTBEGIN(readability-magic-numbers)

// Set when the optional --registry argument is supplied.

SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wglobal-constructors")
SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
static std::filesystem::path GRegistryPath;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
SUPPRESS_WARNINGS_END

static constexpr char const* ShmTestServerArg = "--shmem-test-server";

static std::filesystem::path const& TestRegistryPath()
{ return GRegistryPath; }

namespace
{
class TestServerProcess
{
    public:
    TestServerProcess() = default;
    ~TestServerProcess()
    {
        if (_pid > 0) { Stop(); }
    }
    CLASS_DELETE_COPY_AND_MOVE(TestServerProcess);

    void Start(std::filesystem::path const& stderrLog = {})
    {
        // The wheel smoke job runs pytest first, which can leave the shared test daemon running.
        if (TryConnectToCheckpointService())
        {
            SvcStop();
            static constexpr size_t MaxStopAttempts = 50;
            using namespace std::chrono_literals;
            for (size_t attempt = 0; attempt < MaxStopAttempts; ++attempt)
            {
                if (!TryConnectToCheckpointService()) { break; }
                if (attempt + 1 == MaxStopAttempts) { throw std::runtime_error("existing checkpoint test server did not stop"); }
                std::this_thread::sleep_for(100ms);
            }
        }

        auto const executable = std::filesystem::read_symlink("/proc/self/exe");
        _pid                  = ::fork();
        if (_pid < 0) { throw std::runtime_error("failed to fork checkpoint test server"); }
        if (_pid == 0)
        {
            if (!stderrLog.empty())
            {
                int const descriptor = ::open(stderrLog.c_str(), O_WRONLY | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
                if (descriptor < 0) { ::_exit(EXIT_FAILURE); }
                if (::dup2(descriptor, STDERR_FILENO) < 0)
                {
                    ::close(descriptor);
                    ::_exit(EXIT_FAILURE);
                }
                if (descriptor != STDERR_FILENO) { ::close(descriptor); }
            }
            auto                 executableString = executable.string();
            auto                 serverArg        = std::string{ShmTestServerArg};
            std::array<char*, 3> args{executableString.data(), serverArg.data(), nullptr};
            ::execv(executableString.c_str(), args.data());
            ::_exit(EXIT_FAILURE);
        }

        static constexpr size_t MaxAttempts = 50;
        using namespace std::chrono_literals;
        for (size_t attempt = 0; attempt < MaxAttempts; ++attempt)
        {
            if (TryConnectToCheckpointService()) { return; }
            std::this_thread::sleep_for(100ms);
        }
        Stop();
        throw std::runtime_error("checkpoint test server did not become ready");
    }

    bool Stop()
    {
        SvcStop();
        return WaitForExit();
    }

    bool WaitForExit()
    {
        static constexpr size_t MaxAttempts = 50;
        using namespace std::chrono_literals;
        int status = 0;
        for (size_t attempt = 0; attempt < MaxAttempts; ++attempt)
        {
            auto const waitResult = ::waitpid(_pid, &status, WNOHANG);
            if (waitResult == _pid)
            {
                _pid = -1;
                return WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS;
            }
            if (waitResult < 0)
            {
                _pid = -1;
                return false;
            }
            std::this_thread::sleep_for(100ms);
        }

        ::kill(_pid, SIGTERM);
        auto const waitResult = ::waitpid(_pid, &status, 0);
        _pid                  = -1;
        return waitResult >= 0 && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS;
    }

    private:
    pid_t _pid = -1;
};
}    // namespace

TEST_CASE("shared-memory configuration parses sizes and automatic sentinels", "[shmem][config][tier1]")
{
    using checkpoint::config::types::ByteSize;
    using checkpoint::config::types::Integer;
    using UInt32Integer1 = checkpoint::config::types::Integer<uint32_t, 1>;
    using UInt8Integer   = checkpoint::config::types::Integer<uint8_t>;
    CHECK(ByteSize::Parse("48GiB") == 48ull * (1ull << 30u));
    CHECK(ByteSize::Parse("5GB") == 5'000'000'000ull);
    CHECK_THROWS_AS(ByteSize::Parse("5gib"), std::invalid_argument);
    CHECK_THROWS_AS(ByteSize::Parse("-1"), std::invalid_argument);
    CHECK(UInt32Integer1::Parse("32") == 32);
    CHECK_THROWS_AS(UInt32Integer1::Parse("0"), std::invalid_argument);
    CHECK_THROWS_AS(UInt32Integer1::Parse("4x"), std::invalid_argument);
    CHECK(UInt8Integer::Parse("0") == 0);
    CHECK(UInt8Integer::Parse("255") == 255);
    CHECK_THROWS_AS(UInt8Integer::Parse("256"), std::overflow_error);
    CHECK(Integer<uint32_t>::Parse("0") == 0);
    CHECK_THROWS_AS(Integer<uint32_t>::Parse("-1"), std::invalid_argument);
    CHECK(checkpoint::config::ShmRingBufferSize::Default == "0");
    CHECK(checkpoint::config::ShmRingBufferSlots::Default == "0");
    CHECK(checkpoint::config::ShmRingBufferNumQueues::Default == "0");
    CHECK(Integer<int8_t>::Parse("-128") == -128);
    CHECK(Integer<int8_t>::Parse("+127") == 127);
    CHECK_THROWS_AS(Integer<int8_t>::Parse("128"), std::overflow_error);
    CHECK_THROWS_AS(Integer<int8_t>::Parse("-129"), std::overflow_error);
}

TEST_CASE("server auto-shutdown configuration accepts a disabled sentinel", "[server][config][tier1]")
{
    using Config           = checkpoint::config::ServerAutoShutdownTimeoutSeconds;
    constexpr auto envName = checkpoint::config::detail::EnvName<Config>();
    CHECK(std::string_view{envName.data()} == "NATIVE_CKPT_SERVER_AUTO_SHUTDOWN_TIMEOUT_SECONDS");
    CHECK(Config::Default == "900");
    CHECK(Config::Type::Parse("0") == 0);
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

static std::vector<char> ReadWholeFile(std::filesystem::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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
    char const*       srcPtr;
    uint64_t          length;
    StateTree::NodeId fqn;
    Tensor::DType     dtype;
    Tensor::DimRange  sharddim;
    Tensor::Dim       shape;
};
}    // namespace
static std::vector<TestShardEntry> CollectShardsForTest(StateTree const& tree, Rank rank, ByteStorage const& tensorStorage)
{
    std::vector<TestShardEntry> out;
    auto const*                 base = static_cast<char const*>(tensorStorage.data);
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        auto const fqn  = tree.GetNodeId(node.handle);
        if (type == StateNode::Type::DTensor)
        {
            auto const& tnsr = std::get<DTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                if (shard.rank != rank) { continue; }
                auto const size = shard.StorageSizeBytes();
                if (size == 0) { continue; }
                Tensor::Dim shape{};
                for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
                out.push_back(TestShardEntry{
                    .srcPtr   = base + shard.handle,
                    .length   = size,
                    .fqn      = fqn,
                    .dtype    = tnsr.durableDtype,
                    .sharddim = shard.sharddim,
                    .shape    = shape,
                });
            }
        }
    });
    return out;
}

// Attaches a tensor leaf whose handle is an offset into the surrounding storage.
static void
AddTensorShardLeaf(StateTree& tree, std::string_view fqn, Tensor::DType dtype, std::vector<size_t> const& shape, uint64_t handle)
{
    auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
    auto&            leaf   = tree.AddChild(parent, leafName);
    Tensor::Dim      fulldim;
    Tensor::DimRange sharddim;
    uint64_t         storageSize = Tensor::GetDTypeSize(dtype);
    for (auto d : shape)
    {
        fulldim.push_back(d);
        sharddim.push_back({.begin = 0, .end = d});
        storageSize *= d;
    }
    DTensor tnsr(dtype, fulldim);
    tnsr.AddShard({.rank          = tree.GlobalRank(),
                   .snapshotDtype = dtype,
                   .handle        = handle,
                   .storageSize   = storageSize,
                   .sharddim      = sharddim,
                   .device        = DeviceId{"cpu"}});
    leaf.data = std::move(tnsr);
}

static void AddShardedTensorLeaf(StateTree&                 tree,
                                 std::string_view           fqn,
                                 Tensor::DType              dtype,
                                 std::vector<size_t> const& shape,
                                 Rank                       rank,
                                 uint64_t                   handle)
{
    Tensor::Dim fulldim;
    for (auto d : shape) { fulldim.push_back(d); }

    auto nodeHandle = tree.FindNodeByFQN(fqn);
    if (nodeHandle == StateNode::Handle::Invalid)
    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
        auto& leaf              = tree.AddChild(parent, leafName);
        leaf.data               = DTensor(dtype, fulldim);
        nodeHandle              = leaf.handle;
    }
    auto& st = std::get<DTensor>(tree.GetNode(nodeHandle).data);
    if (st.durableDtype != dtype || st.fulldim != fulldim) { throw std::runtime_error("Incompatible test tensor replica"); }

    Tensor::DimRange sharddim;
    for (auto d : shape) { sharddim.push_back({.begin = 0, .end = d}); }
    uint64_t storageSize = Tensor::GetDTypeSize(dtype);
    for (auto d : shape) { storageSize *= d; }
    st.AddShard({.rank          = rank,
                 .snapshotDtype = dtype,
                 .handle        = handle,
                 .storageSize   = storageSize,
                 .sharddim      = sharddim,
                 .device        = DeviceId{"cpu"}});
}

TEST_CASE("scaffolding", "[scaffolding]")
{ REQUIRE(true); }

TEST_CASE("MXFP8 scalar dequantization matches BF16 golden values", "[mxfp8][tier1]")
{
    std::array<char, 8> nativeBytes{};
    REQUIRE_FALSE(checkpoint::host_dequant::HasStorageHeader(nativeBytes));

    struct GoldenValue
    {
        uint8_t  value;
        uint8_t  scale;
        uint16_t expected;
    };
    static constexpr std::array GoldenValues{
        GoldenValue{0x00, 0x7f, 0x0000},
        GoldenValue{0x80, 0x7f, 0x8000},
        GoldenValue{0x38, 0x7f, 0x3f80},
        GoldenValue{0x3c, 0x7f, 0x3fc0},
        GoldenValue{0xb8, 0x7f, 0xbf80},
        GoldenValue{0x38, 0x80, 0x4000},
        GoldenValue{0x38, 0x7e, 0x3f00},
        GoldenValue{0x01, 0x7f, 0x3b00},
        GoldenValue{0x38, 0x00, 0x0040},
        GoldenValue{0x7e, 0x7f, 0x43e0},
        GoldenValue{0x7f, 0x7f, 0x7fff},
        GoldenValue{0x38, 0xff, 0x7fff},
    };
    for (auto const& golden : GoldenValues)
    {
        CHECK(checkpoint::host_dequant::MXFP8E4M3ToBFloat16(golden.value, golden.scale) == golden.expected);
    }
}

TEST_CASE("FP8 E4M3 scalar dequantization matches BF16 golden values", "[fp8][tier1]")
{
    struct GoldenValue
    {
        uint8_t  value;
        float    scaleInv;
        uint16_t expected;
    };
    static constexpr std::array GoldenValues{
        GoldenValue{0x00, 1.0F, 0x0000},
        GoldenValue{0x80, 1.0F, 0x8000},
        GoldenValue{0x38, 1.0F, 0x3f80},
        GoldenValue{0x3c, 1.0F, 0x3fc0},
        GoldenValue{0xb8, 1.0F, 0xbf80},
        GoldenValue{0x38, 2.0F, 0x4000},
        GoldenValue{0x38, 0.5F, 0x3f00},
        GoldenValue{0x01, 1.0F, 0x3b00},
        GoldenValue{0x7e, 1.0F, 0x43e0},
        GoldenValue{0x7f, 1.0F, 0x7fff},
        GoldenValue{0x38, 0.1F, 0x3dcd},
        GoldenValue{0x38, 3.1415927F, 0x4049},
    };
    for (auto const& golden : GoldenValues)
    {
        CHECK(checkpoint::host_dequant::FP8E4M3ToBFloat16(golden.value, golden.scaleInv) == golden.expected);
    }
}

TEST_CASE("NVFP4 scalar dequantization matches BF16 golden values", "[nvfp4][tier1]")
{
    using E4M3Mode = checkpoint::quant::NVFP4E4M3Mode;
    struct GoldenValue
    {
        uint8_t  value;
        uint8_t  scale;
        float    amax;
        E4M3Mode mode;
        uint16_t expected;
    };
    static constexpr std::array GoldenValues{
        GoldenValue{0x0, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0x0000},
        GoldenValue{0x8, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0x8000},
        GoldenValue{0x2, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0x3f80},
        GoldenValue{0x1, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0x3f00},
        GoldenValue{0x3, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0x3fc0},
        GoldenValue{0x7, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0x40c0},
        GoldenValue{0xf, 0x38, 2688.0F, E4M3Mode::ScaleBound448, 0xc0c0},
        GoldenValue{0x2, 0x40, 1344.0F, E4M3Mode::ScaleBound448, 0x3f80},
        GoldenValue{0x2, 0x38, 1536.0F, E4M3Mode::ScaleBound256, 0x3f80},
        GoldenValue{0x2, 0x38, 0.0F, E4M3Mode::ScaleBound448, 0x0000},
        GoldenValue{0x2, 0x7f, 1.0F, E4M3Mode::ScaleBound448, 0x7fff},
    };
    for (auto const& golden : GoldenValues)
    {
        CHECK(checkpoint::host_dequant::NVFP4E2M1ToBFloat16(golden.value, golden.scale, golden.amax, golden.mode) == golden.expected);
    }
}

TEST_CASE("quantized snapshot segments preserve capture handles", "[quantized][tier1]")
{
    std::array<uint8_t, 8> data{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<uint8_t, 2> scale{120, 121};

    checkpoint::quant::Header source{
        .totalSizeBytes  = sizeof(checkpoint::quant::Header) + data.size() + scale.size(),
        .logicalRows     = 1,
        .logicalCols     = 8,
        .encoding        = checkpoint::quant::Encoding::MXFP8E4M3,
        .scaleLayout     = checkpoint::quant::ScaleLayout::Compact,
        .device          = DeviceId{"cpu:3"},
        .rowwiseData     = {.handle    = reinterpret_cast<uint64_t>(data.data()),    // NOLINT
                            .sizeBytes = data.size(),
                            .rows      = 1,
                            .cols      = data.size()},
        .rowwiseScaleInv = {.handle    = reinterpret_cast<uint64_t>(scale.data()),    // NOLINT
                            .sizeBytes = scale.size(),
                            .rows      = 1,
                            .cols      = scale.size()},
    };
    auto const original = source;

    auto snapshot = [&](uint64_t offset, uint64_t size) {
        std::vector<char> result;
        checkpoint::quant::SerializeSnapshotSegments(reinterpret_cast<uint64_t>(&source),    // NOLINT
                                                     offset,
                                                     size,
                                                     [&](DeviceId, uint64_t handle, uint64_t bytes) {
                                                         auto const* begin = reinterpret_cast<char const*>(handle);    // NOLINT
                                                         result.insert(result.end(), begin, begin + bytes);            // NOLINT
                                                     });
        return result;
    };

    auto const first  = snapshot(0, source.totalSizeBytes);
    auto const second = snapshot(0, source.totalSizeBytes);
    REQUIRE(first == second);
    REQUIRE(source == original);

    checkpoint::quant::Header serialized;
    std::memcpy(&serialized, first.data(), sizeof(serialized));
    CHECK(serialized.device == source.device);
    CHECK(serialized.rowwiseData.handle == source.rowwiseData.handle);
    CHECK(serialized.rowwiseScaleInv.handle == source.rowwiseScaleInv.handle);
    CHECK(std::equal(data.begin(), data.end(), first.begin() + static_cast<std::ptrdiff_t>(sizeof(serialized))));
    CHECK(std::equal(scale.begin(), scale.end(), first.end() - static_cast<std::ptrdiff_t>(scale.size())));

    std::vector<char> partitioned;
    for (uint64_t offset = 0; offset < source.totalSizeBytes; offset += 37)
    {
        auto const size  = std::min<uint64_t>(37, source.totalSizeBytes - offset);
        auto const bytes = snapshot(offset, size);
        partitioned.insert(partitioned.end(), bytes.begin(), bytes.end());
    }
    CHECK(partitioned == first);

    std::vector<DeviceId> devices;
    source.device = DeviceId{"cuda:3"};
    checkpoint::quant::SerializeSnapshotSegments(reinterpret_cast<uint64_t>(&source),    // NOLINT
                                                 0,
                                                 source.totalSizeBytes,
                                                 [&](DeviceId device, uint64_t, uint64_t) { devices.push_back(device); });
    REQUIRE(devices.size() == 3);
    CHECK(devices[0] == DeviceId{"cpu"});
    CHECK(devices[1] == DeviceId{"cuda:3"});
    CHECK(devices[2] == DeviceId{"cuda:3"});

    source.magic = {};
    CHECK_THROWS(checkpoint::quant::SerializeSnapshotSegments(reinterpret_cast<uint64_t>(&source),    // NOLINT
                                                              0,
                                                              source.totalSizeBytes,
                                                              [](DeviceId, uint64_t, uint64_t) {}));
    source.magic = checkpoint::quant::Magic;

    auto otherDevice   = source;
    otherDevice.device = DeviceId{"cuda:4"};
    checkpoint::quant::HeaderMap headers{source, source, otherDevice};
    checkpoint::quant::FinalizeHeaders(headers);
    REQUIRE(headers.size() == 2);
    CHECK(checkpoint::quant::FindHeader(headers, source.device, reinterpret_cast<uint64_t>(data.data())) == source);              // NOLINT
    CHECK(checkpoint::quant::FindHeader(headers, otherDevice.device, reinterpret_cast<uint64_t>(data.data())) == otherDevice);    // NOLINT
}

TEST_CASE("FP8 E4M3 snapshots retain packed storage and write identical BF16 DCP", "[fp8][distcp]")
{
    constexpr uint64_t rows          = 4;
    constexpr uint64_t cols          = 256;
    constexpr uint64_t elements      = rows * cols;
    constexpr uint64_t dataOffset    = sizeof(checkpoint::quant::Header);
    constexpr uint64_t dataSize      = elements;
    constexpr uint64_t scaleOffset   = dataOffset + dataSize;
    constexpr uint64_t scaleSize     = sizeof(float);
    constexpr uint64_t quantizedSize = scaleOffset + scaleSize;

    checkpoint::quant::Header header{
        .totalSizeBytes  = quantizedSize,
        .logicalRows     = rows,
        .logicalCols     = cols,
        .encoding        = checkpoint::quant::Encoding::FP8E4M3,
        .scaleLayout     = checkpoint::quant::ScaleLayout::Compact,
        .device          = "cpu",
        .rowwiseData     = {.handle = dataOffset, .sizeBytes = dataSize, .rows = rows, .cols = cols},
        .rowwiseScaleInv = {.handle = scaleOffset, .sizeBytes = scaleSize, .rows = 1, .cols = 1},
    };

    ByteStorage quantizedStorage(quantizedSize);
    auto*       quantized = static_cast<uint8_t*>(quantizedStorage.data);
    std::memcpy(quantized, &header, sizeof(header));
    for (uint64_t i = 0; i < dataSize; ++i) { quantized[dataOffset + i] = static_cast<uint8_t>((i * 29U) & 0xffU); }
    float const scaleInv = 0.1F;
    std::memcpy(quantized + scaleOffset, &scaleInv, sizeof(scaleInv));
    auto snapshot = quantizedStorage.AsSpan<char const>(0, quantizedStorage.size);

    ByteStorage bf16Storage(elements * sizeof(uint16_t));
    auto        bf16 = std::span<uint16_t>(static_cast<uint16_t*>(bf16Storage.data), elements);
    checkpoint::host_dequant::FP8E4M3ToBFloat16Range(snapshot, 0, bf16);

    std::vector<uint16_t> chunked(elements);
    for (uint64_t offset = 0; offset < elements;)
    {
        auto const count = static_cast<size_t>(std::min<uint64_t>(37, elements - offset));
        checkpoint::host_dequant::DequantizeToBFloat16Range(snapshot, offset, std::span(chunked).subspan(offset, count));
        offset += count;
    }
    REQUIRE(std::ranges::equal(chunked, bf16));

    StateTree quantizedTree(checkpoint::ipc::SessionId{0}, SingleRankTopo());
    StateTree bf16Tree(checkpoint::ipc::SessionId{0}, SingleRankTopo());
    AddShardedTensorLeaf(quantizedTree, "weight", Tensor::DType::BFloat16, {rows, cols}, Rank{0}, 0);
    AddShardedTensorLeaf(bf16Tree, "weight", Tensor::DType::BFloat16, {rows, cols}, Rank{0}, 0);
    auto& quantizedTensor = std::get<DTensor>(quantizedTree.GetNode(quantizedTree.FindNodeByFQN("weight")).data);
    auto  quantizedShard  = *quantizedTensor.shards.begin();
    quantizedTensor.shards.clear();
    quantizedShard.snapshotDtype = Tensor::DType::FP8E4M3;
    quantizedShard.storageSize   = quantizedSize;
    quantizedTensor.AddShard(quantizedShard);

    auto quantizedDir = CkptDir("fp8_quantized");
    auto bf16Dir      = CkptDir("fp8_bf16");
    std::filesystem::remove_all(quantizedDir);
    std::filesystem::remove_all(bf16Dir);
    checkpoint::distcp::WriteRankShard(quantizedDir, Rank{0}, quantizedTree, quantizedStorage);
    checkpoint::distcp::WriteMetadata(quantizedDir, quantizedTree);
    checkpoint::distcp::WriteRankShard(bf16Dir, Rank{0}, bf16Tree, bf16Storage);
    checkpoint::distcp::WriteMetadata(bf16Dir, bf16Tree);

    REQUIRE(ReadWholeFile(quantizedDir / "__0_0.distcp") == ReadWholeFile(bf16Dir / "__0_0.distcp"));
    REQUIRE(ReadWholeFile(quantizedDir / ".metadata") == ReadWholeFile(bf16Dir / ".metadata"));
}

TEST_CASE("MXFP8 snapshots retain packed storage and write identical BF16 DCP", "[mxfp8][distcp]")
{
    constexpr uint64_t rows           = 128;
    constexpr uint64_t cols           = 512;
    constexpr uint64_t elements       = rows * cols;
    constexpr uint64_t rowDataOffset  = sizeof(checkpoint::quant::Header);
    constexpr uint64_t rowDataSize    = elements;
    constexpr uint64_t rowScaleOffset = rowDataOffset + rowDataSize;
    constexpr uint64_t rowScaleSize   = 128 * 16;
    constexpr uint64_t quantizedSize  = rowScaleOffset + rowScaleSize;

    checkpoint::quant::Header header{
        .totalSizeBytes  = quantizedSize,
        .logicalRows     = rows,
        .logicalCols     = cols,
        .encoding        = checkpoint::quant::Encoding::MXFP8E4M3,
        .scaleLayout     = checkpoint::quant::ScaleLayout::Compact,
        .device          = "cpu",
        .rowwiseData     = {.handle = rowDataOffset, .sizeBytes = rowDataSize, .rows = rows, .cols = cols},
        .rowwiseScaleInv = {.handle = rowScaleOffset, .sizeBytes = rowScaleSize, .rows = 128, .cols = 16},
    };

    ByteStorage quantizedStorage(quantizedSize);
    auto*       quantized = static_cast<uint8_t*>(quantizedStorage.data);
    std::memcpy(quantized, &header, sizeof(header));
    auto snapshot = quantizedStorage.AsSpan<char const>(0, quantizedStorage.size);
    for (uint64_t i = 0; i < rowDataSize; ++i) { quantized[rowDataOffset + i] = static_cast<uint8_t>((i * 29U) & 0xffU); }
    for (uint64_t i = 0; i < rowScaleSize; ++i) { quantized[rowScaleOffset + i] = static_cast<uint8_t>(120U + (i % 16U)); }
    REQUIRE(checkpoint::host_dequant::HasStorageHeader(quantizedStorage.AsSpan<char const>(0, quantizedStorage.size)));
    // auto const storage = checkpoint::host_dequant::ParseStorage(quantizedStorage.AsSpan<char const>(0, quantizedStorage.size));

    ByteStorage bf16Storage(elements * sizeof(uint16_t));
    auto        bf16 = std::span<uint16_t>(static_cast<uint16_t*>(bf16Storage.data), elements);
    checkpoint::host_dequant::MXFP8E4M3ToBFloat16(snapshot, bf16);

    std::vector<uint16_t> chunked(elements);
    for (uint64_t offset = 0; offset < elements;)
    {
        auto const count = static_cast<size_t>(std::min<uint64_t>(37, elements - offset));
        checkpoint::host_dequant::MXFP8E4M3ToBFloat16Range(snapshot, offset, std::span(chunked).subspan(offset, count));
        offset += count;
    }
    REQUIRE(std::ranges::equal(chunked, bf16));

    StateTree quantizedTree(checkpoint::ipc::SessionId{0}, SingleRankTopo());
    StateTree bf16Tree(checkpoint::ipc::SessionId{0}, SingleRankTopo());
    AddShardedTensorLeaf(quantizedTree, "weight", Tensor::DType::BFloat16, {rows, cols}, Rank{0}, 0);
    AddShardedTensorLeaf(bf16Tree, "weight", Tensor::DType::BFloat16, {rows, cols}, Rank{0}, 0);
    auto& quantizedTensor = std::get<DTensor>(quantizedTree.GetNode(quantizedTree.FindNodeByFQN("weight")).data);
    auto  quantizedShard  = *quantizedTensor.shards.begin();
    quantizedTensor.shards.clear();
    quantizedShard.snapshotDtype = Tensor::DType::MXFP8;
    quantizedShard.storageSize   = quantizedSize;
    quantizedTensor.AddShard(quantizedShard);

    REQUIRE(quantizedTree.GetTotalTensorStateSizeInBytes() == quantizedStorage.size);
    std::stringstream serializedRegistry;
    StateSerDes::Write(serializedRegistry, quantizedTree, StateNode::Handle::Root);
    auto const  restoredTree   = StateSerDes::Read(serializedRegistry);
    auto const& restoredTensor = std::get<DTensor>(restoredTree->GetNode(restoredTree->FindNodeByFQN("weight")).data);
    REQUIRE(restoredTensor.shards.begin()->storageSize == quantizedSize);
    REQUIRE(restoredTree->GetTotalTensorStateSizeInBytes() == quantizedStorage.size);

    std::vector<char> invalid(quantizedStorage.AsSpan<char const>(0, quantizedStorage.size).begin(),
                              quantizedStorage.AsSpan<char const>(0, quantizedStorage.size).end());
    auto              invalidHeader      = header;
    invalidHeader.rowwiseScaleInv.handle = rowDataOffset;
    std::memcpy(invalid.data(), &invalidHeader, sizeof(invalidHeader));
    // REQUIRE_THROWS(checkpoint::host_dequant::ParseStorage(invalid));
    invalidHeader             = header;
    invalidHeader.scaleLayout = checkpoint::quant::ScaleLayout::GemmSwizzled;
    std::memcpy(invalid.data(), &invalidHeader, sizeof(invalidHeader));
    // REQUIRE_THROWS(checkpoint::host_dequant::MXFP8E4M3ToBFloat16(snapshot, bf16));
    invalidHeader             = header;
    invalidHeader.scaleLayout = static_cast<checkpoint::quant::ScaleLayout>(0xff);
    std::memcpy(invalid.data(), &invalidHeader, sizeof(invalidHeader));
    // REQUIRE_THROWS(checkpoint::host_dequant::ParseStorage(invalid));

    auto quantizedDir = CkptDir("mxfp8_quantized");
    auto bf16Dir      = CkptDir("mxfp8_bf16");
    std::filesystem::remove_all(quantizedDir);
    std::filesystem::remove_all(bf16Dir);
    checkpoint::distcp::WriteRankShard(quantizedDir, Rank{0}, quantizedTree, quantizedStorage);
    checkpoint::distcp::WriteMetadata(quantizedDir, quantizedTree);
    checkpoint::distcp::WriteRankShard(bf16Dir, Rank{0}, bf16Tree, bf16Storage);
    checkpoint::distcp::WriteMetadata(bf16Dir, bf16Tree);

    REQUIRE(ReadWholeFile(quantizedDir / "__0_0.distcp") == ReadWholeFile(bf16Dir / "__0_0.distcp"));
    REQUIRE(ReadWholeFile(quantizedDir / ".metadata") == ReadWholeFile(bf16Dir / ".metadata"));
    REQUIRE(quantized[rowDataOffset + 17] == static_cast<uint8_t>((17U * 29U) & 0xffU));
}

TEST_CASE("NVFP4 snapshots retain packed storage and write identical BF16 DCP", "[nvfp4][distcp]")
{
    constexpr uint64_t rows           = 128;
    constexpr uint64_t cols           = 512;
    constexpr uint64_t elements       = rows * cols;
    constexpr uint64_t rowDataOffset  = sizeof(checkpoint::quant::Header);
    constexpr uint64_t rowDataSize    = elements / 2;
    constexpr uint64_t rowScaleOffset = rowDataOffset + rowDataSize;
    constexpr uint64_t rowScaleRows   = 128;
    constexpr uint64_t rowScaleCols   = 32;
    constexpr uint64_t rowScaleSize   = rowScaleRows * rowScaleCols;
    constexpr uint64_t rowAmaxOffset  = rowScaleOffset + rowScaleSize;
    constexpr uint64_t rowAmaxSize    = sizeof(float);
    constexpr uint64_t quantizedSize  = rowAmaxOffset + rowAmaxSize;

    checkpoint::quant::Header header{
        .totalSizeBytes  = quantizedSize,
        .logicalRows     = rows,
        .logicalCols     = cols,
        .encoding        = checkpoint::quant::Encoding::NVFP4E2M1,
        .scaleLayout     = checkpoint::quant::ScaleLayout::Compact,
        .rowScaledNVFP4  = false,
        .nvfp4E4M3Mode   = checkpoint::quant::NVFP4E4M3Mode::ScaleBound448,
        .device          = "cpu",
        .rowwiseData     = {.handle = rowDataOffset, .sizeBytes = rowDataSize, .rows = rows, .cols = cols / 2},
        .rowwiseScaleInv = {.handle = rowScaleOffset, .sizeBytes = rowScaleSize, .rows = rowScaleRows, .cols = rowScaleCols},
        .rowwiseAmax     = {.handle = rowAmaxOffset, .sizeBytes = rowAmaxSize, .rows = 1, .cols = 1},
    };

    ByteStorage quantizedStorage(quantizedSize);
    auto*       quantized = static_cast<uint8_t*>(quantizedStorage.data);
    auto        snapshot  = quantizedStorage.AsSpan<char const>(0, quantizedStorage.size);

    std::memcpy(quantized, &header, sizeof(header));
    for (uint64_t i = 0; i < rowDataSize; ++i) { quantized[rowDataOffset + i] = static_cast<uint8_t>((i * 29U) & 0xffU); }
    for (uint64_t i = 0; i < rowScaleSize; ++i) { quantized[rowScaleOffset + i] = static_cast<uint8_t>(96U + (i % 48U)); }
    float const rowAmax = 123.25F;
    std::memcpy(quantized + rowAmaxOffset, &rowAmax, sizeof(rowAmax));

    // auto const storage = checkpoint::host_dequant::ParseStorage(quantizedStorage.AsSpan<char const>(0, quantizedStorage.size));

    ByteStorage bf16Storage(elements * sizeof(uint16_t));
    auto        bf16 = std::span<uint16_t>(static_cast<uint16_t*>(bf16Storage.data), elements);
    checkpoint::host_dequant::NVFP4E2M1ToBFloat16Range(snapshot, 0, bf16);

    std::vector<uint16_t> chunked(elements);
    for (uint64_t offset = 0; offset < elements;)
    {
        auto const count = static_cast<size_t>(std::min<uint64_t>(37, elements - offset));
        checkpoint::host_dequant::DequantizeToBFloat16Range(snapshot, offset, std::span(chunked).subspan(offset, count));
        offset += count;
    }
    REQUIRE(std::ranges::equal(chunked, bf16));

    ByteStorage swizzledStorage(quantizedSize);
    auto*       swizzled = static_cast<uint8_t*>(swizzledStorage.data);
    std::memcpy(swizzled, quantized, swizzledStorage.size);
    std::memset(swizzled + rowScaleOffset, 0, rowScaleSize);
    auto               swizzledSnapshot = swizzledStorage.AsSpan<char const>(0, swizzledStorage.size);
    constexpr uint64_t tileSize         = 4 * 128;
    constexpr uint64_t numTilesX        = rowScaleCols / 4;
    for (uint64_t row = 0; row < rows; ++row)
    {
        for (uint64_t col = 0; col < rowScaleCols; ++col)
        {
            uint64_t const tileX             = col / 4;
            uint64_t const tileY             = row / 128;
            uint64_t const inTileX           = col % 4;
            uint64_t const inTileY           = row % 128;
            uint64_t const index             = (tileY * numTilesX + tileX) * tileSize + (inTileY % 32) * 16 + (inTileY / 32) * 4 + inTileX;
            swizzled[rowScaleOffset + index] = quantized[rowScaleOffset + row * rowScaleCols + col];
        }
    }
    auto swizzledHeader        = header;
    swizzledHeader.scaleLayout = checkpoint::quant::ScaleLayout::GemmSwizzled;
    std::memcpy(swizzled, &swizzledHeader, sizeof(swizzledHeader));
    std::vector<uint16_t> swizzledResult(elements);
    checkpoint::host_dequant::NVFP4E2M1ToBFloat16Range(swizzledSnapshot, 0, swizzledResult);
    REQUIRE(std::ranges::equal(swizzledResult, bf16));

    StateTree quantizedTree(checkpoint::ipc::SessionId{0}, SingleRankTopo());
    StateTree bf16Tree(checkpoint::ipc::SessionId{0}, SingleRankTopo());
    AddShardedTensorLeaf(quantizedTree, "weight", Tensor::DType::BFloat16, {rows, cols}, Rank{0}, 0);
    AddShardedTensorLeaf(bf16Tree, "weight", Tensor::DType::BFloat16, {rows, cols}, Rank{0}, 0);
    auto& quantizedTensor = std::get<DTensor>(quantizedTree.GetNode(quantizedTree.FindNodeByFQN("weight")).data);
    auto  quantizedShard  = *quantizedTensor.shards.begin();
    quantizedTensor.shards.clear();
    quantizedShard.snapshotDtype = Tensor::DType::NVFP4;
    quantizedShard.storageSize   = quantizedSize;
    quantizedTensor.AddShard(quantizedShard);

    REQUIRE(quantizedTree.GetTotalTensorStateSizeInBytes() == quantizedStorage.size);
    std::stringstream serializedRegistry;
    StateSerDes::Write(serializedRegistry, quantizedTree, StateNode::Handle::Root);
    auto const  restoredTree   = StateSerDes::Read(serializedRegistry);
    auto const& restoredTensor = std::get<DTensor>(restoredTree->GetNode(restoredTree->FindNodeByFQN("weight")).data);
    REQUIRE(restoredTensor.shards.begin()->storageSize == quantizedSize);
    REQUIRE(restoredTree->GetTotalTensorStateSizeInBytes() == quantizedStorage.size);

    auto quantizedDir = CkptDir("nvfp4_quantized");
    auto bf16Dir      = CkptDir("nvfp4_bf16");
    std::filesystem::remove_all(quantizedDir);
    std::filesystem::remove_all(bf16Dir);
    checkpoint::distcp::WriteRankShard(quantizedDir, Rank{0}, quantizedTree, quantizedStorage);
    checkpoint::distcp::WriteMetadata(quantizedDir, quantizedTree);
    checkpoint::distcp::WriteRankShard(bf16Dir, Rank{0}, bf16Tree, bf16Storage);
    checkpoint::distcp::WriteMetadata(bf16Dir, bf16Tree);

    REQUIRE(ReadWholeFile(quantizedDir / "__0_0.distcp") == ReadWholeFile(bf16Dir / "__0_0.distcp"));
    REQUIRE(ReadWholeFile(quantizedDir / ".metadata") == ReadWholeFile(bf16Dir / ".metadata"));
    REQUIRE(quantized[rowDataOffset + 17] == static_cast<uint8_t>((17U * 29U) & 0xffU));
}

TEST_CASE("state serialization preserves opaque tensor storage sizes", "[state][tier1]")
{
    StateTree tree(checkpoint::ipc::SessionId{17}, SingleRankTopo());

    Tensor::Dim fullShape;
    fullShape.push_back(8);
    DTensor          tensor(Tensor::DType::Float32, fullShape);
    Tensor::DimRange fullRange;
    fullRange.push_back({.begin = 0, .end = 8});
    tensor.AddShard({.rank          = Rank{0},
                     .snapshotDtype = Tensor::DType::MXFP8,
                     .handle        = 23,
                     .storageSize   = 13,
                     .sharddim      = fullRange,
                     .device        = DeviceId{"cpu"}});
    tree.AddChild(StateNode::Handle::Root, "tensor").data = tensor;

    Tensor::Dim fulldim;
    fulldim.push_back(16);
    DTensor          dtensor(Tensor::DType::BFloat16, fulldim);
    Tensor::DimRange sharddim;
    sharddim.push_back({.begin = 4, .end = 8});
    dtensor.AddShard({
        .rank          = Rank{0},
        .snapshotDtype = Tensor::DType::NVFP4,
        .handle        = 41,
        .storageSize   = 7,
        .sharddim      = sharddim,
        .device        = DeviceId{"cpu"},
    });
    tree.AddChild(StateNode::Handle::Root, "dtensor").data = dtensor;

    CHECK(tree.GetTotalTensorStateSizeInBytes() == 20);
    CHECK(tree.GetTotalNumTensors() == 2);

    std::stringstream stream;
    StateSerDes::Write(stream, tree, StateNode::Handle::Root);
    auto restored = StateSerDes::Read(stream);

    CHECK(restored->SessionId() == checkpoint::ipc::SessionId{17});
    CHECK(std::get<DTensor>(restored->GetNode(restored->FindNodeByFQN("tensor")).data) == tensor);
    CHECK(std::get<DTensor>(restored->GetNode(restored->FindNodeByFQN("dtensor")).data) == dtensor);
    CHECK(restored->GetTotalTensorStateSizeInBytes() == 20);
    CHECK(restored->GetTotalNumTensors() == 2);

    std::istringstream header(stream.str());
    static_cast<void>(StateSerDes::ReadSessionId(header));
    static_cast<void>(StateSerDes::ReadTopology(header));
    REQUIRE(serdes::ReadTrivial<uint32_t>(header) == StateSerDes::WireMagic);
    auto const     versionOffset = static_cast<size_t>(header.tellg());
    auto           legacyBytes   = stream.str();
    uint32_t const legacyVersion = 2;
    std::memcpy(legacyBytes.data() + versionOffset, &legacyVersion, sizeof(legacyVersion));
    std::istringstream legacy(legacyBytes);
    CHECK_THROWS_AS(StateSerDes::Read(legacy), std::runtime_error);
}

TEST_CASE("registry fixture resizes native tensor storage", "[state][fixture][tier1]")
{
    auto const registryPath = CkptDir("resize_registry") / "registry.bin";
    std::filesystem::remove_all(registryPath.parent_path());
    std::filesystem::create_directories(registryPath.parent_path());

    StateTree source(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    AddShardedTensorLeaf(source, "weight", Tensor::DType::BFloat16, {8}, Rank{0}, 0);
    {
        std::ofstream stream(registryPath, std::ios::binary);
        REQUIRE(stream);
        StateSerDes::Write(stream, source, StateNode::Handle::Root);
    }

    CheckpointTestContext ctx;
    ctx.outofproc    = true;
    ctx.ranksPerNode = 1;
    ctx.memLimit     = 40;
    LoadStateRegistryFromPath(ctx, registryPath);

    auto const& tensor = std::get<DTensor>(std::as_const(*ctx.globalRegistry).GetNode(ctx.globalRegistry->FindNodeByFQN("weight")).data);
    REQUIRE(tensor.fulldim.size() == 1);
    CHECK(tensor.fulldim[0] == 4);
    REQUIRE(tensor.shards.size() == 1);
    auto const& shard = *tensor.shards.begin();
    CHECK(shard.sharddim[0] == Tensor::Range{.begin = 0, .end = 4});
    CHECK(shard.storageSize == 8);

    REQUIRE(ctx.localRegistries.size() == 1);
    auto const& local       = **ctx.localRegistries.begin();
    auto const& localTensor = std::get<DTensor>(local.GetNode(local.FindNodeByFQN("weight")).data);
    REQUIRE(localTensor.shards.size() == 1);
    auto const& localShard = *localTensor.shards.begin();
    CHECK(localShard.handle == 0);
    CHECK(localShard.storageSize == 8);
    CHECK(localShard.device == DeviceId{"cpu"});
}

TEST_CASE("tensor metadata initializes its complete representation", "[state][tier1]")
{
    auto representation = []<typename T>(uint8_t poison, auto const& initialize) {
        alignas(T) std::array<uint8_t, sizeof(T)> storage{};
        storage.fill(poison);
        auto* value = ::new (static_cast<void*>(storage.data())) T;    // NOLINT(cppcoreguidelines-owning-memory)
        initialize(*value);
        std::array<uint8_t, sizeof(T)> result{};
        std::memcpy(result.data(), value, sizeof(T));
        std::destroy_at(value);
        return result;
    };

    auto initializeDim = [](Tensor::Dim& dim) { dim.push_back(8); };
    CHECK(representation.template operator()<Tensor::Dim>(0x5a, initializeDim)
          == representation.template operator()<Tensor::Dim>(0xa5, initializeDim));

    auto initializeDimRange = [](Tensor::DimRange& dim) { dim.push_back({.begin = 4, .end = 8}); };
    CHECK(representation.template operator()<Tensor::DimRange>(0x5a, initializeDimRange)
          == representation.template operator()<Tensor::DimRange>(0xa5, initializeDimRange));

    auto initializeTensor = [](Tensor& tensor) {
        tensor.durableDtype  = Tensor::DType::Float32;
        tensor.snapshotDtype = Tensor::DType::Float32;
        tensor.handle        = 23;
        tensor.storageSize   = 13;
        tensor.fulldim.push_back(8);
        tensor.device = DeviceId{"cpu"};
    };
    CHECK(representation.template operator()<Tensor>(0x5a, initializeTensor)
          == representation.template operator()<Tensor>(0xa5, initializeTensor));

    auto initializeShard = [](TensorShard& shard) {
        shard.rank          = Rank{0};
        shard.snapshotDtype = Tensor::DType::NVFP4;
        shard.handle        = 41;
        shard.storageSize   = 7;
        shard.sharddim.push_back({.begin = 4, .end = 8});
        shard.device = DeviceId{"cpu"};
    };
    CHECK(representation.template operator()<TensorShard>(0x5a, initializeShard)
          == representation.template operator()<TensorShard>(0xa5, initializeShard));

    std::array<uint8_t, sizeof(Tensor::Dim)> invalidRepresentation{};
    invalidRepresentation[sizeof(uint64_t) * Tensor::MaxDims] = static_cast<uint8_t>(Tensor::MaxDims + 1);
    Tensor::Dim invalidDim;
    std::memcpy(&invalidDim, invalidRepresentation.data(), sizeof(invalidDim));
    CHECK_THROWS_AS(invalidDim.size(), std::length_error);

    Tensor::Dim protectedDim;
    protectedDim.push_back(8);
    CHECK_THROWS_AS(protectedDim[1], std::out_of_range);
}

TEST_CASE("full-range DTensor preserves a physical shard", "[state][tier1]")
{
    StateTree   tree(checkpoint::ipc::SessionId{18}, SingleRankTopo());
    Tensor::Dim shape;
    shape.push_back(3);
    shape.push_back(7);
    DTensor          tensor(Tensor::DType::BFloat16, shape);
    Tensor::DimRange ranges;
    ranges.push_back({.begin = 0, .end = 3});
    ranges.push_back({.begin = 0, .end = 7});
    TensorShard const source{.rank          = Rank{0},
                             .snapshotDtype = Tensor::DType::MXFP8,
                             .handle        = 29,
                             .storageSize   = 11,
                             .sharddim      = ranges,
                             .device        = DeviceId{"cuda:2"}};
    tensor.AddShard(source);
    tree.AddChild(StateNode::Handle::Root, "weight").data = tensor;

    auto const& dtensor = std::get<DTensor>(tree.GetNode(tree.FindNodeByFQN("weight")).data);
    REQUIRE(dtensor.shards.size() == 1);
    auto const& shard = *dtensor.shards.begin();
    CHECK(dtensor.durableDtype == tensor.durableDtype);
    CHECK(dtensor.fulldim == tensor.fulldim);
    CHECK(shard == source);
    REQUIRE(shard.sharddim.size() == 2);
    CHECK(shard.sharddim[0] == (Tensor::Range{.begin = 0, .end = 3}));
    CHECK(shard.sharddim[1] == (Tensor::Range{.begin = 0, .end = 7}));
    CHECK(tree.GetTotalTensorStateSizeInBytes() == source.storageSize);
    CHECK(tree.GetTotalNumTensors() == 1);
}

TEST_CASE("registries reject duplicate tensor names", "[state][tier1]")
{
    StateTree tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    AddTensorShardLeaf(tree, "weight", Tensor::DType::Float32, {2}, 0);
    AddTensorShardLeaf(tree, "weight", Tensor::DType::Float32, {2}, 8);

    CHECK_THROWS_AS(tree.FinalizeRegistry(), std::runtime_error);
    CHECK(tree.GetTotalNumTensors() == 2);
}

static std::string SerializedStateRegistry(StateTree const& tree)
{
    std::ostringstream serialized;
    StateSerDes::Write(serialized, tree, StateNode::Handle::Root);
    return serialized.str();
}

TEST_CASE("tensor and fixed blob shards require nonempty storage", "[state][planning][tier1]")
{
    Tensor::Dim shape;
    shape.push_back(2);
    DTensor                  tensor(Tensor::DType::UInt8, shape);
    StateNode::FixedSizeBlob blob(Tensor::DType::UInt8, shape);
    CHECK_THROWS_AS(tensor.AddShard(TensorShard{}), std::runtime_error);
    CHECK_THROWS_AS(blob.AddShard(TensorShard{}), std::runtime_error);
}

TEST_CASE("payload traversal groups blobs and tensors independently of node order", "[state][planning][tier1]")
{
    StateTree tree(checkpoint::ipc::SessionId{55}, SingleRankTopo());
    tree.AddChild(StateNode::Handle::Root, "a_dynamic").data
        = StateNode::DynamicSizeBlob{.replicas = {Rank{0}}, .handle = 9, .storageSize = 4};
    tree.AddChild(StateNode::Handle::Root, "z_empty").data
        = StateNode::DynamicSizeBlob{.replicas = {Rank{0}}, .handle = 13, .storageSize = 0};
    AddShardedTensorLeaf(tree, "a_tensor", Tensor::DType::UInt8, {2}, Rank{0}, 0);
    AddTensorShardLeaf(tree, "m_tensor", Tensor::DType::UInt8, {3}, 2);
    StateNode::Scalar scalar(int64_t{71});
    scalar.replicas                                     = {Rank{0}};
    tree.AddChild(StateNode::Handle::Root, "step").data = scalar;
    for (auto const* name : {"z_fixed", "b_fixed"})
    {
        Tensor::Dim shape;
        shape.push_back(1);
        StateNode::FixedSizeBlob blob(Tensor::DType::UInt8, shape);
        Tensor::DimRange         ranges;
        ranges.push_back({.begin = 0, .end = 1});
        blob.AddShard({.rank          = Rank{0},
                       .snapshotDtype = Tensor::DType::UInt8,
                       .handle        = std::string_view(name) == "b_fixed" ? uint64_t{5} : uint64_t{7},
                       .storageSize   = 2,
                       .sharddim      = ranges,
                       .device        = DeviceId{"cpu"}});
        tree.AddChild(StateNode::Handle::Root, name).data = std::move(blob);
    }

    tree.FinalizeRegistry();
    CHECK(tree.IsFinalized());
    CHECK(tree.GetTotalPayloadSizeInBytes() == 13);
    std::vector<std::string> names;
    std::vector<uint64_t>    offsets;
    tree.ForEachPayload([&](StateNode::Handle node, Tensor::DType, DeviceId, uint64_t handle, uint64_t) {
        names.push_back(tree.GetNodeId(node));
        offsets.push_back(handle);
    });
    CHECK(names == std::vector<std::string>{"a_tensor", "m_tensor", "b_fixed", "z_fixed", "a_dynamic", "z_empty"});
    CHECK(offsets == std::vector<uint64_t>{0, 2, 5, 7, 9, 13});
    CHECK(std::get<StateNode::Scalar>(std::as_const(tree).GetNode(tree.FindNodeByFQN("step")).data) == scalar);

    std::string const packed = "EFGHIABCDTAIL";
    ByteStorage       storage(packed.size());
    std::memcpy(storage.data, packed.data(), packed.size());
    auto const  bytes   = std::string_view(static_cast<char const*>(storage.data), storage.size);
    auto const& sharded = std::get<DTensor>(std::as_const(tree).GetNode(tree.FindNodeByFQN("a_tensor")).data);
    auto const& full    = std::get<DTensor>(std::as_const(tree).GetNode(tree.FindNodeByFQN("m_tensor")).data);
    CHECK(bytes.substr(sharded.shards.begin()->handle, sharded.shards.begin()->storageSize) == "EF");
    CHECK(bytes.substr(full.shards.begin()->handle, full.shards.begin()->storageSize) == "GHI");

    StateTree snapshot(checkpoint::ipc::SessionId{55}, SingleRankTopo());
    tree.CloneInto(snapshot);
    auto const outDir = CkptDir("payload_traversal");
    std::filesystem::remove_all(outDir);
    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, snapshot, storage);
    checkpoint::distcp::WriteMetadata(outDir, snapshot);
    auto const tensorBytes = checkpoint::distcp::ComputeShardArchiveSize(Tensor::DType::UInt8, sharded.fulldim, 2)
                             + checkpoint::distcp::ComputeShardArchiveSize(Tensor::DType::UInt8, full.fulldim, 3);
    auto const written     = ReadWholeFile(outDir / "__0_0.distcp");
    REQUIRE(written.size() == tensorBytes + 8);
    auto const writtenBytes = std::string_view(written.data(), written.size());
    CHECK(writtenBytes.substr(tensorBytes, 4) == "TAIL");
    CHECK(writtenBytes.substr(tensorBytes + 4, 2) == "AB");
    CHECK(writtenBytes.substr(tensorBytes + 6, 2) == "CD");
    auto const metadata      = ReadWholeFile(outDir / ".metadata");
    auto const metadataBytes = std::string_view(metadata.data(), metadata.size());
    CHECK(metadataBytes.contains("a_dynamic"));
    CHECK(metadataBytes.contains("b_fixed/shard_0_1"));
    CHECK(metadataBytes.contains("z_empty"));
    CHECK(metadataBytes.contains("z_fixed/shard_0_1"));
#ifdef PHOENIX_VENV_PYTHON
    auto const cmd = fmt::format("{} -c \"import pickle; m=pickle.load(open(r'{}','rb')); "
                                 "s={{i.fqn:(v.offset,v.length) for i,v in m.storage_data.items()}}; "
                                 "assert len(s)==6; assert s['a_dynamic']==({},4); "
                                 "assert s['b_fixed/shard_0_1']==({},2); assert s['z_empty']==({},0); "
                                 "assert s['z_fixed/shard_0_1']==({},2)\"",
                                 PHOENIX_VENV_PYTHON,
                                 (outDir / ".metadata").string(),
                                 tensorBytes,
                                 tensorBytes + 4,
                                 tensorBytes + 6,
                                 tensorBytes + 6);
    REQUIRE(std::system(cmd.c_str()) == 0);    // NOLINT
#endif

    auto const frozen = SerializedStateRegistry(tree);
    auto       clone  = tree.Clone();
    CHECK(clone->IsFinalized());
    CHECK(SerializedStateRegistry(*clone) == frozen);
    CHECK_NOTHROW(tree.FinalizeRegistry());
    CHECK(SerializedStateRegistry(tree) == frozen);
}

TEST_CASE("finalized registries reject general mutations", "[state][planning][tier1]")
{
    StateTree tree(checkpoint::ipc::SessionId{56}, SingleRankTopo());
    tree.AddChild(StateNode::Handle::Root, "scalar").data = StateNode::Scalar(int64_t{5});
    tree.AddChild(StateNode::Handle::Root, "blob").data = StateNode::DynamicSizeBlob{.replicas = {Rank{0}}, .handle = 0, .storageSize = 0};
    tree.FinalizeRegistry();
    auto const frozen = SerializedStateRegistry(tree);
    CHECK_THROWS(tree.AddChild(StateNode::Handle::Root, "extra"));
    CHECK_THROWS(tree.SetScalar("scalar", StateNode::Scalar(int64_t{6})));
    CHECK_THROWS(tree.SetBlobSource("blob", 0, 1));
    CHECK_THROWS(tree.RemoveNode(tree.FindNodeByFQN("blob")));
    CHECK_THROWS(tree.WalkNodes([](StateNode&) {}));
    std::istringstream serialized(frozen);
    static_cast<void>(StateSerDes::ReadSessionId(serialized));
    static_cast<void>(StateSerDes::ReadTopology(serialized));
    CHECK_THROWS(StateSerDes::Read(serialized, tree));
    CHECK(SerializedStateRegistry(tree) == frozen);
}

TEST_CASE("snapshot shared memory keeps bounded growth room", "[shmem][tier1]")
{
    using checkpoint::ipc::RoundSnapshotShmSize;
    static constexpr size_t GiB = 1ull << 30u;
    CHECK(RoundSnapshotShmSize(1) == checkpoint::ipc::MinSnapshotShmSize);
    CHECK(RoundSnapshotShmSize(3 * GiB) == 10 * GiB);
    CHECK(RoundSnapshotShmSize(5 * GiB) == 10 * GiB);
    CHECK(RoundSnapshotShmSize(33 * GiB) == 48 * GiB);
    CHECK(RoundSnapshotShmSize(25'842'932'288ull) == 32 * GiB);
    CHECK(RoundSnapshotShmSize(28'322'409'024ull) == 40 * GiB);
}

TEST_CASE("shared memory queues retain large transfer slots as their budgets grow", "[shmem][tier1]")
{
    using Queue                 = checkpoint::ipc::MsgQueue;
    static constexpr size_t GiB = 1ull << 30u;

    auto const headerSize = Queue::HeaderAllocationSize(Queue::DefaultSlotCount);
    CHECK(Queue::SlotCountForShmSize(headerSize + (128ull * 1024u), Queue::DefaultSlotCount, 1) == 6);
    CHECK(Queue::SlotCountForShmSize(headerSize + (256ull * 1024u), Queue::DefaultSlotCount, 2) == 12);
    CHECK(Queue::SlotCountForShmSize(headerSize + (4 * GiB), Queue::DefaultSlotCount, 1) == 2);
    CHECK(Queue::SlotCountForShmSize(headerSize + (8 * GiB), Queue::DefaultSlotCount, 1) == 4);
    CHECK(Queue::SlotCountForShmSize(headerSize + (12 * GiB), Queue::DefaultSlotCount, 2) == 6);
    CHECK(Queue::SlotCountForShmSize(headerSize + (16 * GiB), Queue::DefaultSlotCount, 2) == 8);
    CHECK(Queue::SlotCountForShmSize(headerSize + (32 * GiB), Queue::DefaultSlotCount, 1) == 16);
    CHECK(Queue::SlotCountForShmSize(headerSize + (64 * GiB), Queue::DefaultSlotCount, 1) == 32);
    CHECK(Queue::SlotCountForShmSize(headerSize + (128 * GiB), Queue::DefaultSlotCount, 1) == 32);
}

static boost::beast::http::status PostToTestServer(std::string const& target, std::string body, std::string_view encoding = {})
{
    boost::asio::io_context        ioc;
    boost::beast::tcp_stream       stream(ioc);
    boost::asio::ip::tcp::resolver resolver(ioc);
    auto const                     endpoints = resolver.resolve("localhost", std::to_string(checkpoint::ipc::Message::Port));
    stream.connect(endpoints);

    boost::beast::http::request<boost::beast::http::string_body> request{boost::beast::http::verb::post, target, 11};
    request.set(boost::beast::http::field::host, "localhost");
    if (!encoding.empty()) { request.set(boost::beast::http::field::content_encoding, encoding); }
    request.body() = std::move(body);
    request.prepare_payload();
    boost::beast::http::write(stream, request);

    boost::beast::flat_buffer                                     buffer;
    boost::beast::http::response<boost::beast::http::string_body> response;
    boost::beast::http::read(stream, buffer, response);
    return response.result();
}

// The forked server inherits the log location, which also places its registry dumps.
static void StartTestServer(TestServerProcess& server, std::filesystem::path const& serverLog, bool captureStderr = false)
{
    ::setenv("NATIVE_CKPT_SERVER_LOG", serverLog.c_str(), 1);    // NOLINT(concurrency-mt-unsafe)
    REQUIRE_NOTHROW(server.Start(captureStderr ? serverLog : std::filesystem::path{}));
    ::unsetenv("NATIVE_CKPT_SERVER_LOG");    // NOLINT(concurrency-mt-unsafe)
}

static void CheckDuplicateRegistryWarning(std::filesystem::path const& serverLog, uint32_t rank)
{
    std::ifstream log(serverLog);
    REQUIRE(log.good());
    size_t      warnings = 0;
    std::string line;
    while (std::getline(log, line))
    {
        if (!line.contains("duplicate local state registry")) { continue; }
        ++warnings;
        CHECK(line.contains("[WARN]"));
        CHECK(line.contains(fmt::format("#{}]", checkpoint::ipc::SessionId::Default)));
        CHECK(line.contains(fmt::format("rank={}", rank)));
        CHECK(line.contains("unknown reason"));
        CHECK(line.contains("ignoring it"));
    }
    CHECK(warnings == 1);
}

static std::string
FinalizeResultForResponse(boost::beast::http::status status, std::string const& body = {}, bool complete = true, int64_t iteration = 40)
{
    using boost::asio::ip::tcp;
    boost::asio::io_context ioc;
    tcp::acceptor           acceptor(ioc);
    acceptor.open(tcp::v4());
    acceptor.set_option(tcp::acceptor::reuse_address(true));
    acceptor.bind({tcp::v4(), checkpoint::ipc::Message::Port});
    acceptor.listen();
    auto server = std::async(std::launch::async, [&] {
        boost::beast::tcp_stream                                    stream(acceptor.accept());
        boost::beast::flat_buffer                                   buffer;
        boost::beast::http::request<boost::beast::http::empty_body> request;
        boost::beast::http::read(stream, buffer, request);

        boost::beast::http::response<boost::beast::http::empty_body> response{status, 11};
        response.set(boost::beast::http::field::content_type, "text/plain");
        response.content_length(body.size() + static_cast<size_t>(!complete));
        boost::beast::http::response_serializer<boost::beast::http::empty_body> serializer(response);
        boost::beast::http::write_header(stream, serializer);
        for (size_t offset = 0; offset < body.size(); offset += 7)
        {
            auto const part = std::string_view{body}.substr(offset, 7);
            boost::asio::write(stream.socket(), boost::asio::buffer(part));
        }
        return std::string{request.target()};
    });

    std::string failure;
    try
    {
        WaitForCkptFinalize(0, iteration);
    } catch (std::runtime_error const& error) { failure = error.what(); }
    CHECK(server.get() == fmt::format("/api/wait-for-ckpt-finalize/0/{}", iteration));
    return failure;
}

TEST_CASE("finalize uses the completed HTTP status and error body", "[server][finalize][tier1]")
{
    namespace http = boost::beast::http;
    if (TryConnectToCheckpointService())
    {
        SvcStop();
        for (size_t attempt = 0; attempt < 50 && TryConnectToCheckpointService(); ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        REQUIRE_FALSE(TryConnectToCheckpointService());
    }
    CHECK(FinalizeResultForResponse(http::status::ok).empty());
    CHECK(FinalizeResultForResponse(http::status::ok, "checkpoint is durable").empty());
    CHECK(FinalizeResultForResponse(http::status::ok, {}, true, -1).empty());

    std::vector<std::string> const failures{
        "snapshot=40 rank=0: shard Input/output error",
        "wait-for-ckpt-finalize timed out for iteration=40",
        "wait-for-ckpt-finalize aborted during server shutdown",
    };
    for (auto const& body : failures)
    {
        INFO(body);
        auto const failure = FinalizeResultForResponse(http::status::internal_server_error, body);
        CHECK(failure.contains("HTTP 500"));
        CHECK(failure.contains(body));
    }
    CHECK_FALSE(FinalizeResultForResponse(http::status::accepted).empty());
    CHECK_FALSE(FinalizeResultForResponse(http::status::not_found, "unknown endpoint").empty());
    CHECK_FALSE(FinalizeResultForResponse(http::status::ok, "checkpoint is durable", false).empty());
    CHECK_FALSE(FinalizeResultForResponse(http::status::ok, {}, false).empty());
}

TEST_CASE("a shard write failure reaches the finalize caller", "[server][finalize][tier2]")
{
    auto const logDir  = CkptDir("finalize_shard_write_failure");
    auto const goodDir = logDir / "iteration30";
    auto const badDir  = logDir / "iteration40";
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(goodDir);
    std::filesystem::create_directories(badDir);
    std::filesystem::create_symlink("/dev/full", badDir / "__0_0.distcp");

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log", true);
    std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
    StateTree            tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    AddTensorShardLeaf(tree, "weight", Tensor::DType::Float32, {values.size()}, reinterpret_cast<uint64_t>(values.data()));
    FinalizeStateRegistry(tree);

    SaveCheckpoint(tree, checkpoint::ipc::SnapshotId{30}, goodDir);
    auto       goodFinalize = std::async(std::launch::async, [] { WaitForCkptFinalize(0, 30); });
    auto const goodStatus   = goodFinalize.wait_for(std::chrono::seconds{10});
    if (goodStatus != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(goodStatus == std::future_status::ready);
    REQUIRE_NOTHROW(goodFinalize.get());
    REQUIRE(std::filesystem::file_size(goodDir / "__0_0.distcp") > 0);
    REQUIRE(std::filesystem::exists(goodDir / ".metadata"));

    auto const nextDir = logDir / "iteration35";
    std::filesystem::create_directories(nextDir);
    SaveCheckpoint(tree, checkpoint::ipc::SnapshotId{35}, nextDir);
    auto       nextFinalize = std::async(std::launch::async, [] {
        WaitForCkptFinalize(0, 35);
        WaitForCkptFinalize(0, 30);
        WaitForCkptFinalize(0, -1);
    });
    auto const nextStatus   = nextFinalize.wait_for(std::chrono::seconds{10});
    if (nextStatus != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(nextStatus == std::future_status::ready);
    REQUIRE_NOTHROW(nextFinalize.get());
    REQUIRE(std::filesystem::exists(nextDir / ".metadata"));
    auto waitForLog = [&](std::string_view marker) {
        for (size_t attempt = 0; attempt < 100; ++attempt)
        {
            auto const contents = ReadWholeFile(logDir / "server.log");
            if (std::string_view{contents.data(), contents.size()}.contains(marker)) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
        }
        return false;
    };
    REQUIRE(waitForLog("Released 1 snapshot buffers after finalizing snapshot=35 rank=0."));
    std::stringstream reactivation;
    serdes::WriteBlobWithSize<uint16_t>(reactivation, goodDir.string());
    serdes::WriteTrivial(reactivation, uint32_t{0});
    serdes::WriteTrivial(reactivation, uint32_t{0});
    auto const payload = reactivation.str();
    auto       queue   = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    queue.Push({checkpoint::ipc::Message::ActivateSnapshot{
                   .sessionId    = checkpoint::ipc::SessionId::Default,
                   .snapshotId   = checkpoint::ipc::SnapshotId{30},
                   .globalRank   = Rank{0},
                   .snapshotSize = sizeof(values),
                   .messageSize  = static_cast<uint32_t>(payload.size()),
               }},
               payload.data(),
               payload.size());
    REQUIRE(waitForLog("already activated or finalized"));

    static constexpr auto WrappedIteration = (int64_t{1} << 32u) + 30;
    auto                  invalidFinalize  = std::async(std::launch::async, [] { WaitForCkptFinalize(0, WrappedIteration); });
    auto const            invalidStatus    = invalidFinalize.wait_for(std::chrono::seconds{10});
    if (invalidStatus != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(invalidStatus == std::future_status::ready);
    std::string invalidFailure;
    try
    {
        invalidFinalize.get();
    } catch (std::runtime_error const& error) { invalidFailure = error.what(); }
    CHECK(invalidFailure.contains("HTTP 500"));
    CHECK(invalidFailure.contains(std::to_string(WrappedIteration)));

    SaveCheckpoint(tree, checkpoint::ipc::SnapshotId{40}, badDir);
    auto       badFinalize = std::async(std::launch::async, [] { WaitForCkptFinalize(0, 40); });
    auto const badStatus   = badFinalize.wait_for(std::chrono::seconds{10});
    if (badStatus != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(badStatus == std::future_status::ready);
    std::string failure;
    try
    {
        badFinalize.get();
    } catch (std::runtime_error const& error) { failure = error.what(); }
    CHECK(failure.contains("snapshot=40"));
    CHECK(failure.contains("distcp:"));
    CHECK(failure.contains("__0_0.distcp"));
    CHECK_FALSE(std::filesystem::exists(badDir / ".metadata"));
    CHECK(std::filesystem::exists(goodDir / ".metadata"));
    auto       spinDown   = std::async(std::launch::async, &SvcSpinDown);
    auto const stopStatus = spinDown.wait_for(std::chrono::seconds{10});
    if (stopStatus != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(stopStatus == std::future_status::ready);
    CHECK_NOTHROW(spinDown.get());
    CHECK(server.WaitForExit());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("large registries keep snapshot descriptors within a shared-memory slot", "[server][snapshot][tier2]")
{
    static constexpr uint32_t TestSessionId = 61;
    auto const                logDir        = CkptDir("large_snapshot_descriptor");
    auto const                checkpointDir = logDir / "iteration40";
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(checkpointDir);
    std::array<char const*, 3> const options{
        "NATIVE_CKPT_SHM_RING_BUFFER_SIZE", "NATIVE_CKPT_SHM_RING_BUFFER_SLOTS", "NATIVE_CKPT_SHM_RING_BUFFER_NUM_QUEUES"};
    std::array<std::string, 3> previous;
    std::array<bool, 3>        present{};
    for (size_t i = 0; i < options.size(); ++i)
    {
        if (auto const* value = ::getenv(options[i]); value != nullptr)
        {
            present[i]  = true;
            previous[i] = value;
        }
    }
    ::setenv(options[0], "32MiB", 1);
    ::setenv(options[1], "512", 1);
    ::setenv(options[2], "1", 1);
    TestServerProcess server;
    StartTestServer(server, logDir / "server.log", true);
    for (size_t i = 0; i < options.size(); ++i)
    {
        if (present[i]) { ::setenv(options[i], previous[i].c_str(), 1); }
        else
        {
            ::unsetenv(options[i]);
        }
    }
    StateTree         tree(checkpoint::ipc::SessionId{TestSessionId}, SingleRankTopo());
    std::string const suffix(128, 'x');
    for (uint32_t index = 0; index < 512; ++index)
    {
        StateNode::Scalar value(index);
        value.replicas                                                                             = {Rank{0}};
        tree.AddChild(StateNode::Handle::Root, fmt::format("scalar_{:04}_{}", index, suffix)).data = value;
    }
    std::array<StateNode::Scalar, 4> values{
        StateNode::Scalar(int32_t{-7}), StateNode::Scalar(uint64_t{91}), StateNode::Scalar(2.5), StateNode::Scalar(true)};
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index].replicas                                                            = {Rank{0}};
        tree.AddChild(StateNode::Handle::Root, fmt::format("typed_scalar{}", index)).data = values[index];
    }
    std::string const bytes                                = "small snapshot payload";
    tree.AddChild(StateNode::Handle::Root, "payload").data = StateNode::DynamicSizeBlob{
        .replicas = {Rank{0}}, .handle = reinterpret_cast<uint64_t>(bytes.data()), .storageSize = bytes.size()};
    std::string const fixedBytes = "fixed prefix";
    Tensor::Dim       fixedShape;
    fixedShape.push_back(1);
    StateNode::FixedSizeBlob fixed(Tensor::DType::UInt8, fixedShape);
    Tensor::DimRange         fixedRange;
    fixedRange.push_back({.begin = 0, .end = 1});
    fixed.AddShard({.rank          = Rank{0},
                    .snapshotDtype = Tensor::DType::UInt8,
                    .handle        = reinterpret_cast<uint64_t>(fixedBytes.data()),
                    .storageSize   = fixedBytes.size(),
                    .sharddim      = fixedRange,
                    .device        = DeviceId{"cpu"}});
    tree.AddChild(StateNode::Handle::Root, "fixed_payload").data = std::move(fixed);
    FinalizeStateRegistry(tree);
    auto& localRegistry = tree;
    for (size_t index = 0; index < values.size(); ++index)
    {
        auto const  name   = fmt::format("typed_scalar{}", index);
        auto const& scalar = std::get<StateNode::Scalar>(localRegistry.GetNode(localRegistry.FindNodeByFQN(name)).data);
        CHECK(scalar.GetType() == values[index].GetType());
        CHECK(scalar == values[index]);
        CHECK(scalar.replicas == values[index].replicas);
        CHECK(std::get<StateNode::Scalar>(tree.GetNode(tree.FindNodeByFQN(name)).data) == values[index]);
    }
    for (uint32_t index = 0; index < 512; ++index)
    {
        auto const name                       = fmt::format("scalar_{:04}_{}", index, suffix);
        auto&      scalar                     = std::get<StateNode::Scalar>(localRegistry.GetNode(localRegistry.FindNodeByFQN(name)).data);
        static_cast<Primitives64Bit&>(scalar) = Primitives64Bit(index);
    }
    for (size_t index = 0; index < values.size(); ++index)
    {
        auto const name                       = fmt::format("typed_scalar{}", index);
        auto&      scalar                     = std::get<StateNode::Scalar>(localRegistry.GetNode(localRegistry.FindNodeByFQN(name)).data);
        static_cast<Primitives64Bit&>(scalar) = static_cast<Primitives64Bit const&>(values[index]);
    }
    auto& payload = std::get<StateNode::DynamicSizeBlob>(localRegistry.GetNode(localRegistry.FindNodeByFQN("payload")).data);
    CHECK(payload.handle == reinterpret_cast<uint64_t>(bytes.data()));
    CHECK(payload.storageSize == bytes.size());
    std::string const currentBytes = "snapshot payload with a changed serialized length";
    payload.handle                 = reinterpret_cast<uint64_t>(currentBytes.data());
    payload.storageSize            = currentBytes.size();
    auto& fixedShards = std::get<StateNode::FixedSizeBlob>(localRegistry.GetNode(localRegistry.FindNodeByFQN("fixed_payload")).data).shards;
    auto  fixedSource = fixedShards.extract(fixedShards.begin());
    fixedSource.value().handle = reinterpret_cast<uint64_t>(fixedBytes.data());
    fixedShards.insert(std::move(fixedSource));
    std::ostringstream descriptor;
    StateSerDes::Write(descriptor, localRegistry, StateNode::Handle::Root);
    auto queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    REQUIRE(descriptor.str().size() > queue.SlotSize());
    REQUIRE(localRegistry.GetTotalPayloadSizeInBytes() < queue.SlotSize());
    REQUIRE_NOTHROW(SaveCheckpoint(localRegistry, checkpoint::ipc::SnapshotId{40}, checkpointDir));
    auto       durable = std::async(std::launch::async, [] { WaitForCkptFinalize(TestSessionId, 40); });
    auto const status  = durable.wait_for(std::chrono::seconds{10});
    if (status != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(status == std::future_status::ready);
    CHECK_NOTHROW(durable.get());
    auto const saved = ReadWholeFile(checkpointDir / "__0_0.distcp");
    CHECK(std::string(saved.begin(), saved.end()) == fixedBytes + currentBytes);
    CHECK(std::filesystem::exists(checkpointDir / ".metadata"));
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("registry replies wait for every local registry and fit queues to the largest", "[shmem][tier2]")
{
    static constexpr uint32_t        RanksPerPeer = 2;
    static constexpr size_t          MiB          = 1ull << 20u;
    std::array<size_t, RanksPerPeer> snapshotSizes{1 * MiB, 16 * MiB};
    auto const                       logDir = CkptDir("largest_registry_sizes_shm");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);

    auto uploadRegistry = [&](uint32_t rank) {
        StateTree tree(checkpoint::ipc::SessionId::Default, SingleRankTopo(rank, RanksPerPeer));
        auto&     node = tree.AddChild(StateNode::Handle::Root, fmt::format("payload{}", rank));
        if (rank == 1)
        {
            node.data = StateNode::DynamicSizeBlob{.replicas = {Rank{rank}}, .handle = 0, .storageSize = snapshotSizes[rank]};
        }
        else
        {
            Tensor::Dim shape;
            shape.push_back(snapshotSizes[rank]);
            DTensor          tensor(Tensor::DType::UInt8, shape);
            Tensor::DimRange ranges;
            ranges.push_back({.begin = 0, .end = snapshotSizes[rank]});
            tensor.AddShard({.rank          = Rank{rank},
                             .snapshotDtype = Tensor::DType::UInt8,
                             .storageSize   = snapshotSizes[rank],
                             .sharddim      = ranges,
                             .device        = DeviceId{"cpu"}});
            node.data = std::move(tensor);
        }
        tree.FinalizeRegistry();
        std::stringstream body;
        StateSerDes::Write(body, tree, StateNode::Handle::Root);
        return PostToTestServer(fmt::format("/api/localstateregistry/{}", checkpoint::ipc::SessionId::Default), body.str());
    };

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log", true);
    std::stringstream activation;
    serdes::WriteBlobWithSize<uint16_t>(activation, "");
    StateSerDes::WriteTopology(activation, SingleRankTopo(0, RanksPerPeer));
    CHECK(PostToTestServer(fmt::format("/api/activatesession/{}", checkpoint::ipc::SessionId::Default), activation.str())
          == boost::beast::http::status::accepted);

    auto smaller = std::async(std::launch::async, uploadRegistry, 0);
    CHECK(smaller.wait_for(std::chrono::milliseconds{200}) == std::future_status::timeout);
    auto duplicate = std::async(std::launch::async, uploadRegistry, 0);
    CHECK(duplicate.wait_for(std::chrono::milliseconds{200}) == std::future_status::timeout);
    CHECK(uploadRegistry(1) == boost::beast::http::status::accepted);
    CHECK(smaller.get() == boost::beast::http::status::accepted);
    CHECK(duplicate.get() == boost::beast::http::status::accepted);
    for (uint8_t queueIndex = 0; queueIndex < RanksPerPeer; ++queueIndex)
    {
        auto queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, queueIndex));
        CHECK(queue.SlotSize() * queue.SlotCount() >= snapshotSizes[1]);
    }
    CHECK(server.Stop());
    CheckDuplicateRegistryWarning(logDir / "server.log", 0);
    std::filesystem::remove_all(logDir);
}

// Each rank owns one slice of a tensor shared by the whole world.
static DTensor SliceOfSharedTensor(uint32_t rank, uint32_t worldSize)
{
    Tensor::Dim fulldim;
    fulldim.push_back(size_t{4} * worldSize);
    DTensor          tensor(Tensor::DType::Float32, fulldim);
    Tensor::DimRange sharddim;
    sharddim.push_back({.begin = size_t{4} * rank, .end = size_t{4} * (rank + 1)});
    tensor.AddShard({
        .rank          = Rank{rank},
        .snapshotDtype = Tensor::DType::Float32,
        .handle        = 0,
        .storageSize   = 4 * Tensor::GetDTypeSize(Tensor::DType::Float32),
        .sharddim      = sharddim,
        .device        = DeviceId{"cpu"},
    });
    return tensor;
}

static std::string SerializedSliceRegistry(RankTopology topo)
{
    auto const rank      = static_cast<uint32_t>(topo.rank);
    auto const worldSize = static_cast<uint32_t>(topo.worldSize);
    StateTree  tree(checkpoint::ipc::SessionId::Default, std::move(topo));
    auto [leafName, parent]              = tree.FindOrAddNodeByFQN("weight");
    tree.AddChild(parent, leafName).data = SliceOfSharedTensor(rank, worldSize);
    tree.FinalizeRegistry();
    std::stringstream body;
    StateSerDes::Write(body, tree, StateNode::Handle::Root);
    return body.str();
}

static std::string SerializedPeerRegistry(StateTree const& tree)
{
    std::ostringstream body;
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = node.GetType();
        if (type != StateNode::Type::DTensor && type != StateNode::Type::FixedSizeBlob) { return; }
        auto const& tensor = type == StateNode::Type::FixedSizeBlob
                                 ? static_cast<DTensor const&>(std::get<StateNode::FixedSizeBlob>(node.data))
                                 : std::get<DTensor>(node.data);
        for (auto const& shard : tensor.shards)
        {
            serdes::WriteTrivial(body, type == StateNode::Type::FixedSizeBlob ? uint8_t{2} : uint8_t{0});
            serdes::WriteTrivial(body, tensor.durableDtype);
            serdes::WriteTrivial(body, tensor.fulldim);
            serdes::WriteTrivial(body, shard);
            serdes::WriteBlobWithSize<uint32_t>(body, tree.GetNodeId(node.handle));
        }
    });
    tree.WalkNodes([&](StateNode const& node) {
        if (node.GetType() != StateNode::Type::DynamicSizeBlob) { return; }
        serdes::WriteTrivial(body, uint8_t{1});
        serdes::WriteTrivial(body, std::get<StateNode::DynamicSizeBlob>(node.data).storageSize);
        serdes::WriteBlobWithSize<uint32_t>(body, tree.GetNodeId(node.handle));
    });
    return body.str();
}

static std::string SerializedSliceFragment(uint32_t rank, uint32_t worldSize)
{
    StateTree tree(checkpoint::ipc::SessionId::Default, SingleRankTopo(rank, worldSize));
    tree.AddChild(StateNode::Handle::Root, "weight").data = SliceOfSharedTensor(rank, worldSize);
    return SerializedPeerRegistry(tree);
}

TEST_CASE("duplicate registries preserve the first upload during forwarding", "[server][registry][tier2]")
{
    using namespace std::chrono_literals;
    namespace http = boost::beast::http;
    using Tcp      = boost::asio::ip::tcp;

    size_t        minimumSendBuffer{};
    size_t        defaultSendBuffer{};
    size_t        maximumSendBuffer{};
    std::ifstream limits("/proc/sys/net/ipv4/tcp_wmem");
    if (!(limits >> minimumSendBuffer >> defaultSendBuffer >> maximumSendBuffer)) { SKIP("TCP send-buffer limits are unavailable"); }
    static constexpr size_t MiB = 1ull << 20u;
    if (maximumSendBuffer > 64 * MiB) { SKIP("TCP send-buffer limit exceeds the bounded registry fixture budget"); }

    boost::asio::io_context ioc;
    Tcp::acceptor           acceptor{ioc};
    acceptor.open(Tcp::v4());
    acceptor.set_option(boost::asio::socket_base::receive_buffer_size{64 * 1024});
    acceptor.bind({boost::asio::ip::address_v4::loopback(), 0});
    acceptor.listen();
    boost::asio::socket_base::receive_buffer_size receiveBuffer;
    acceptor.get_option(receiveBuffer);
    auto const         peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
    RankTopology const topology{.rank = Rank{1}, .worldSize = Rank{2}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {peer, "localhost"}};

    static constexpr std::string_view TensorName = "weight";
    static constexpr std::string_view BlobName   = "metadata";
    auto const streamBudget = std::max(4 * MiB, (2 * maximumSendBuffer) + (4 * static_cast<size_t>(receiveBuffer.value())));
    auto const recordSize
        = sizeof(uint8_t) + sizeof(Tensor::DType) + sizeof(Tensor::Dim) + sizeof(TensorShard) + sizeof(uint32_t) + TensorName.size();
    auto const shardCount   = (streamBudget / recordSize) + 1;
    auto       makeRegistry = [&](size_t blobSize) {
        StateTree   tree(checkpoint::ipc::SessionId::Default, topology);
        Tensor::Dim fulldim;
        fulldim.push_back(shardCount);
        DTensor tensor(Tensor::DType::Float32, fulldim);
        for (size_t index = 0; index < shardCount; ++index)
        {
            Tensor::DimRange sharddim;
            sharddim.push_back({.begin = index, .end = index + 1});
            tensor.AddShard({.rank          = Rank{1},
                             .snapshotDtype = Tensor::DType::Float32,
                             .handle        = index * sizeof(float),
                             .storageSize   = sizeof(float),
                             .sharddim      = sharddim,
                             .device        = DeviceId{"cpu"}});
        }
        tree.AddChild(StateNode::Handle::Root, TensorName).data = std::move(tensor);
        Tensor::Dim blobShape;
        blobShape.push_back(1);
        StateNode::FixedSizeBlob blob(Tensor::DType::UInt8, blobShape);
        Tensor::DimRange         blobRange;
        blobRange.push_back({.begin = 0, .end = 1});
        blob.AddShard({.rank          = Rank{1},
                       .snapshotDtype = Tensor::DType::UInt8,
                       .handle        = shardCount * sizeof(float),
                       .storageSize   = blobSize,
                       .sharddim      = blobRange,
                       .device        = DeviceId{"cpu"}});
        tree.AddChild(StateNode::Handle::Root, BlobName).data = std::move(blob);
        tree.FinalizeRegistry();
        std::stringstream registry;
        StateSerDes::Write(registry, tree, StateNode::Handle::Root);
        return std::make_pair(registry.str(), SerializedPeerRegistry(tree));
    };

    // The stream exceeds the sender and receiver TCP buffer limits.
    auto const original    = makeRegistry(17);
    auto const replacement = makeRegistry(23);
    REQUIRE(original.second.size() > streamBudget);
    auto const logDir = CkptDir("registry_forwarding_generation");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    TestServerProcess server;
    StartTestServer(server, logDir / "server.log", true);
    std::stringstream activation;
    serdes::WriteBlobWithSize<uint16_t>(activation, "");
    StateSerDes::WriteTopology(activation, topology);
    REQUIRE(PostToTestServer(fmt::format("/api/activatesession/{}", checkpoint::ipc::SessionId::Default), activation.str())
            == http::status::accepted);

    std::promise<void>       bodyStarted;
    auto                     firstBody = bodyStarted.get_future();
    std::future<std::string> received;
    struct ReadingGate
    {
        std::promise<void> released;
        bool               open{false};
        ~ReadingGate() { Open(); }
        void Open()
        {
            if (!open)
            {
                released.set_value();
                open = true;
            }
        }
    } gate;
    auto resume       = gate.released.get_future().share();
    received          = std::async(std::launch::async, [&, resume] {
        auto run = [&](auto start) {
            boost::system::error_code error;
            start([&](boost::system::error_code ec, auto...) { error = ec; });
            ioc.run();
            ioc.restart();
            if (error) { throw boost::system::system_error(error); }
        };
        bool started = false;
        try
        {
            boost::beast::tcp_stream  stream{ioc};
            boost::asio::steady_timer deadline{ioc, 30s};
            deadline.async_wait([&](boost::system::error_code ec) {
                if (!ec) { acceptor.cancel(); }
            });
            run([&](auto done) {
                acceptor.async_accept(stream.socket(), [&, done](boost::system::error_code ec) {
                    deadline.cancel();
                    done(ec);
                });
            });
            boost::beast::flat_buffer               buffer;
            http::request_parser<http::string_body> parser;
            parser.body_limit(512 * MiB);
            stream.expires_after(30s);
            run([&](auto done) { http::async_read_header(stream, buffer, parser, done); });
            while (parser.get().body().empty())
            {
                stream.expires_after(30s);
                run([&](auto done) { http::async_read_some(stream, buffer, parser, done); });
            }
            bodyStarted.set_value();
            started = true;
            if (resume.wait_for(30s) != std::future_status::ready) { throw std::runtime_error("registry read was not released"); }
            stream.socket().set_option(boost::asio::socket_base::receive_buffer_size{static_cast<int>(MiB)});
            stream.expires_after(30s);
            run([&](auto done) { http::async_read(stream, buffer, parser, done); });
            http::response<http::empty_body> response{http::status::ok, parser.get().version()};
            response.prepare_payload();
            stream.expires_after(30s);
            run([&](auto done) { http::async_write(stream, response, done); });
            return std::move(parser.get().body());
        } catch (...)
        {
            if (!started) { bodyStarted.set_exception(std::current_exception()); }
            throw;
        }
    });
    auto const target = fmt::format("/api/localstateregistry/{}", checkpoint::ipc::SessionId::Default);
    REQUIRE(PostToTestServer(target, original.first) == http::status::accepted);
    REQUIRE(firstBody.wait_for(30s) == std::future_status::ready);
    firstBody.get();
    CHECK(PostToTestServer(target, replacement.first) == http::status::accepted);
    gate.Open();
    auto const body = received.get();
    CHECK(std::ranges::equal(body, original.second));
    CHECK(server.Stop());
    acceptor.non_blocking(true);
    Tcp::socket               unexpected{ioc};
    boost::system::error_code error;
    acceptor.accept(unexpected, error);
    CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
    auto const dump = ReadWholeFile(logDir / "registry.1.bin");
    CHECK(std::ranges::equal(dump, original.first));
    CheckDuplicateRegistryWarning(logDir / "server.log", 1);
    std::filesystem::remove_all(logDir);
}

static void ActivateAndUploadRegistries(RankTopology const& topo, std::vector<std::string> const& registries)
{
    std::stringstream activation;
    serdes::WriteBlobWithSize<uint16_t>(activation, "");
    StateSerDes::WriteTopology(activation, topo);
    CHECK(PostToTestServer(fmt::format("/api/activatesession/{}", checkpoint::ipc::SessionId::Default), activation.str())
          == boost::beast::http::status::accepted);
    std::vector<std::future<boost::beast::http::status>> uploads;
    uploads.reserve(registries.size());
    for (auto const& registry : registries)
    {
        uploads.push_back(std::async(std::launch::async, [&registry] {
            return PostToTestServer(fmt::format("/api/localstateregistry/{}", checkpoint::ipc::SessionId::Default), registry);
        }));
    }
    for (auto& upload : uploads) { CHECK(upload.get() == boost::beast::http::status::accepted); }
}

static void ActivateTestSnapshot(checkpoint::ipc::MsgQueue&   queue,
                                 std::filesystem::path const& directory,
                                 Rank                         rank,
                                 RankTopology /* topology */,
                                 checkpoint::ipc::SnapshotId snapshotId = checkpoint::ipc::SnapshotId{40})
{
    std::stringstream activation;
    serdes::WriteBlobWithSize<uint16_t>(activation, directory.string());
    serdes::WriteTrivial(activation, uint32_t{0});
    serdes::WriteTrivial(activation, uint32_t{0});
    auto const payload = activation.str();
    queue.Push({checkpoint::ipc::Message::ActivateSnapshot{
                   .sessionId    = checkpoint::ipc::SessionId::Default,
                   .snapshotId   = snapshotId,
                   .globalRank   = rank,
                   .snapshotSize = 4 * sizeof(float),
                   .messageSize  = static_cast<uint32_t>(payload.size()),
               }},
               payload.data(),
               payload.size());
}

static std::string WaitForFinalizeFailure(TestServerProcess& server)
{
    auto       finalize = std::async(std::launch::async, [] { WaitForCkptFinalize(0, 40); });
    auto const status   = finalize.wait_for(std::chrono::seconds{10});
    if (status != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(status == std::future_status::ready);
    std::string failure;
    try
    {
        finalize.get();
    } catch (std::runtime_error const& error) { failure = error.what(); }
    return failure;
}

static std::string WriteCompletionBody(std::string_view failure, std::map<std::string, uint64_t> const& sizes = {})
{
    std::ostringstream body;
    serdes::WriteBlobWithSize<uint32_t>(body, failure);
    serdes::WriteTrivial(body, static_cast<uint32_t>(sizes.size()));
    for (auto const& [fqn, size] : sizes)
    {
        serdes::WriteBlobWithSize<uint32_t>(body, fqn);
        serdes::WriteTrivial(body, size);
    }
    return body.str();
}

template <typename THandler>
static std::future<void> ServePeerRequests(boost::asio::io_context& ioc, boost::asio::ip::tcp::acceptor& acceptor, THandler handler)
{
    return std::async(std::launch::async, [&, requestHandler = std::move(handler)] {
        using namespace std::chrono_literals;
        namespace http = boost::beast::http;
        auto run       = [&](auto start) {
            boost::system::error_code error;
            start([&](boost::system::error_code ec, auto...) { error = ec; });
            ioc.run();
            ioc.restart();
            if (error) { throw boost::system::system_error(error); }
        };
        for (size_t attempt = 0; attempt < 8; ++attempt)
        {
            boost::beast::tcp_stream  stream{ioc};
            boost::asio::steady_timer deadline{ioc, 10s};
            deadline.async_wait([&](boost::system::error_code ec) {
                if (!ec) { acceptor.cancel(); }
            });
            run([&](auto done) {
                acceptor.async_accept(stream.socket(), [&, done](boost::system::error_code ec) {
                    deadline.cancel();
                    done(ec);
                });
            });
            boost::beast::flat_buffer        buffer;
            http::request<http::string_body> request;
            stream.expires_after(10s);
            run([&](auto done) { http::async_read(stream, buffer, request, done); });
            auto const [body, finished] = requestHandler(request);
            http::response<http::string_body> response{http::status::ok, request.version()};
            response.set(http::field::content_type, "text/plain");
            response.body() = body;
            response.prepare_payload();
            stream.expires_after(10s);
            run([&](auto done) { http::async_write(stream, response, done); });
            if (finished) { return; }
        }
        throw std::runtime_error("peer mock received too many requests");
    });
}

TEST_CASE("incomplete tensor data reaches the finalize caller", "[server][finalize][tier2]")
{
    using namespace std::chrono_literals;
    namespace http    = boost::beast::http;
    using Tcp         = boost::asio::ip::tcp;
    auto const logDir = CkptDir("finalize_incomplete_tensor_data");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    boost::asio::io_context ioc;
    Tcp::acceptor           acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    auto const              peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
    RankTopology const topology{.rank = Rank{1}, .worldSize = Rank{2}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {peer, "localhost"}};
    std::string        failureTarget;
    std::string        failureBody;
    http::verb         failureMethod{};
    auto               coordinator = ServePeerRequests(ioc, acceptor, [&](auto const& request) {
        bool const failure = !request.body().empty() && request.target() == "/api/peer/writeckptcomplete/0/1/40";
        if (failure)
        {
            failureTarget = std::string{request.target()};
            failureBody   = request.body();
            failureMethod = request.method();
        }
        return std::pair{std::string{"{}"}, failure};
    });
    TestServerProcess  server;
    StartTestServer(server, logDir / "server.log");
    ActivateAndUploadRegistries(topology, {SerializedSliceRegistry(topology)});
    auto queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    ActivateTestSnapshot(queue, logDir / "iteration40", Rank{1}, topology);
    queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                   .sessionId  = checkpoint::ipc::SessionId::Default,
                   .globalRank = Rank{1},
                   .snapshotId = checkpoint::ipc::SnapshotId{40},
               }},
               nullptr,
               0);
    auto       finalize          = std::async(std::launch::async, [] { WaitForCkptFinalize(0, 40); });
    auto const coordinatorStatus = coordinator.wait_for(10s);
    if (coordinatorStatus != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(coordinatorStatus == std::future_status::ready);
    CHECK_NOTHROW(coordinator.get());
    CHECK(finalize.wait_for(3s) == std::future_status::timeout);
    CHECK(
        PostToTestServer("/api/peer/writeckptcomplete/0/0/40", WriteCompletionBody(fmt::format("rank=1 WriteCkpt failed: {}", failureBody)))
        == http::status::ok);
    auto const status = finalize.wait_for(10s);
    if (status != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(status == std::future_status::ready);
    std::string failure;
    try
    {
        finalize.get();
    } catch (std::runtime_error const& error) { failure = error.what(); }
    CHECK(failure.contains("snapshot=40"));
    CHECK(failure.contains("rank=1"));
    CHECK(failure.contains("Incomplete tensor data"));
    CHECK(failureTarget == "/api/peer/writeckptcomplete/0/1/40");
    CHECK(failureMethod == http::verb::post);
    CHECK(failureBody.contains("Incomplete tensor data"));
    CHECK_FALSE(std::filesystem::exists(logDir / "iteration40" / ".metadata"));
    CHECK(server.Stop());
    acceptor.non_blocking(true);
    Tcp::socket               unexpected{ioc};
    boost::system::error_code error;
    acceptor.accept(unexpected, error);
    CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
    std::filesystem::remove_all(logDir);
}

TEST_CASE("a peer failure reaches the coordinator before local snapshot activation", "[server][finalize][tier2]")
{
    using namespace std::chrono_literals;
    using Tcp         = boost::asio::ip::tcp;
    auto const logDir = CkptDir("finalize_peer_failure_before_activation");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    boost::asio::io_context ioc;
    Tcp::acceptor           acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    auto const              peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
    std::string             resultTarget;
    std::string             resultBody;
    auto                    notified = ServePeerRequests(ioc, acceptor, [&](auto const& request) {
        resultTarget = std::string{request.target()};
        resultBody   = request.body();
        return std::pair{std::string{}, true};
    });
    RankTopology const topology{.rank = Rank{0}, .worldSize = Rank{2}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {"localhost", peer}};
    TestServerProcess  server;
    StartTestServer(server, logDir / "server.log");
    ActivateAndUploadRegistries(topology, {SerializedSliceRegistry(topology)});
    auto finalize = std::async(std::launch::async, [] { WaitForCkptFinalize(0, 40); });
    CHECK(finalize.wait_for(200ms) == std::future_status::timeout);
    CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/1/40", WriteCompletionBody("distcp: peer shard Input/output error"))
          == boost::beast::http::status::ok);
    CHECK(finalize.wait_for(3s) == std::future_status::timeout);
    CHECK(notified.wait_for(200ms) == std::future_status::timeout);
    auto queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    ActivateTestSnapshot(queue, logDir / "iteration40", Rank{0}, topology);
    std::array<float, 4> const values{1.0f, 2.0f, 3.0f, 4.0f};
    queue.Push({checkpoint::ipc::Message::SnapshotData{
                   .sessionId   = checkpoint::ipc::SessionId::Default,
                   .snapshotId  = checkpoint::ipc::SnapshotId{40},
                   .globalRank  = Rank{0},
                   .messageSize = sizeof(values),
                   .offset      = 0,
               }},
               reinterpret_cast<char const*>(values.data()),
               sizeof(values));
    queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                   .sessionId  = checkpoint::ipc::SessionId::Default,
                   .globalRank = Rank{0},
                   .snapshotId = checkpoint::ipc::SnapshotId{40},
               }},
               nullptr,
               0);
    auto const status = finalize.wait_for(10s);
    if (status != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(status == std::future_status::ready);
    std::string failure;
    try
    {
        finalize.get();
    } catch (std::runtime_error const& error) { failure = error.what(); }
    CHECK(failure.contains("HTTP 500"));
    CHECK(failure.contains("snapshot=40"));
    CHECK(failure.contains("rank=1"));
    CHECK(failure.contains("peer shard Input/output error"));
    REQUIRE(notified.wait_for(10s) == std::future_status::ready);
    CHECK_NOTHROW(notified.get());
    CHECK(resultTarget == "/api/peer/writeckptcomplete/0/0/40?fwd=1:2");
    CHECK(resultBody.contains("rank=1"));
    CHECK(resultBody.contains("peer shard Input/output error"));
    CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/1/40", WriteCompletionBody("another error")) == boost::beast::http::status::ok);
    CHECK(server.Stop());
    acceptor.non_blocking(true);
    Tcp::socket               unexpected{ioc};
    boost::system::error_code error;
    acceptor.accept(unexpected, error);
    CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
    std::filesystem::remove_all(logDir);
}

TEST_CASE("a coordinator failure survives later peer snapshot activation", "[server][finalize][tier2]")
{
    using namespace std::chrono_literals;
    using Tcp         = boost::asio::ip::tcp;
    auto const logDir = CkptDir("finalize_result_before_peer_activation");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    boost::asio::io_context ioc;
    Tcp::acceptor           acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    auto const              peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
    boost::asio::io_context downstreamIoc;
    Tcp::acceptor           downstream{downstreamIoc, {boost::asio::ip::address_v4::loopback(), 0}};
    auto const              downstreamPeer = fmt::format("127.0.0.1:{}", downstream.local_endpoint().port());
    RankTopology const      topology{.rank         = Rank{2},
                                     .worldSize    = Rank{4},
                                     .ranksPerPeer = 1,
                                     .peersPerNode = 1,
                                     .peers        = {peer, "unused", "localhost", downstreamPeer}};
    auto                    forwarded = ServePeerRequests(ioc, acceptor, [](auto const&) { return std::pair{std::string{}, true}; });
    TestServerProcess       server;
    StartTestServer(server, logDir / "server.log", true);
    ActivateAndUploadRegistries(topology, {SerializedSliceRegistry(topology)});
    REQUIRE(forwarded.wait_for(10s) == std::future_status::ready);
    CHECK_NOTHROW(forwarded.get());
    std::string const        failure = "Checkpoint finalization failed for snapshot=40 rank=0: disk full";
    std::vector<std::string> forwardedTargets;
    std::vector<std::string> forwardedBodies;
    auto                     delegated = ServePeerRequests(downstreamIoc, downstream, [&](auto const& request) {
        forwardedTargets.emplace_back(request.target());
        forwardedBodies.push_back(request.body());
        CHECK(request[boost::beast::http::field::content_encoding].empty());
        return std::pair{std::string{}, true};
    });
    CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/0/40?fwd=2:4", WriteCompletionBody(failure)) == boost::beast::http::status::ok);
    CHECK(WaitForFinalizeFailure(server).contains(failure));
    auto queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    ActivateTestSnapshot(queue, logDir / "iteration40", Rank{2}, topology);
    queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                   .sessionId  = checkpoint::ipc::SessionId::Default,
                   .globalRank = Rank{2},
                   .snapshotId = checkpoint::ipc::SnapshotId{40},
               }},
               nullptr,
               0);
    bool finalized = false;
    for (size_t attempt = 0; attempt < 100; ++attempt)
    {
        auto const bytes = ReadWholeFile(logDir / "server.log");
        finalized        = std::string_view{bytes.data(), bytes.size()}.contains("OnFinalizeSnapshotAsync_ for rank=2, snapshotId=40");
        if (finalized) { break; }
        std::this_thread::sleep_for(50ms);
    }
    REQUIRE(finalized);
    CHECK(WaitForFinalizeFailure(server).contains(failure));
    REQUIRE(delegated.wait_for(10s) == std::future_status::ready);
    CHECK_NOTHROW(delegated.get());
    CHECK(forwardedTargets == std::vector<std::string>{"/api/peer/writeckptcomplete/0/0/40?fwd=3:4"});
    CHECK(forwardedBodies == std::vector<std::string>{WriteCompletionBody(failure)});
    CHECK(server.Stop());
    acceptor.non_blocking(true);
    Tcp::socket               unexpected{ioc};
    boost::system::error_code error;
    acceptor.accept(unexpected, error);
    CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
    std::filesystem::remove_all(logDir);
}

TEST_CASE("the coordinator notifies peers after metadata is durable", "[server][finalize][tier2]")
{
    using namespace std::chrono_literals;
    namespace http           = boost::beast::http;
    using Tcp                = boost::asio::ip::tcp;
    auto const logDir        = CkptDir("finalize_coordinator_result");
    auto const checkpointDir = logDir / "iteration40";
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(checkpointDir);
    boost::asio::io_context ioc;
    Tcp::acceptor           acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    auto const              peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
    RankTopology const topology{.rank = Rank{0}, .worldSize = Rank{4}, .ranksPerPeer = 2, .peersPerNode = 1, .peers = {"localhost", peer}};
    std::vector<std::string> targets;
    std::string              resultBody;
    std::promise<void>       resultReceived;
    auto                     received = resultReceived.get_future();
    std::promise<void>       releaseAcknowledgment;
    auto                     acknowledgment    = releaseAcknowledgment.get_future();
    bool                     metadataPublished = false;
    auto                     notified          = ServePeerRequests(ioc, acceptor, [&](auto const& request) {
        targets.emplace_back(request.target());
        auto const result = request.target() == "/api/peer/writeckptcomplete/0/0/40?fwd=1:2";
        if (result)
        {
            resultBody        = request.body();
            metadataPublished = std::filesystem::exists(checkpointDir / ".metadata")
                                && std::filesystem::file_size(checkpointDir / ".metadata") > 0
                                && std::filesystem::file_size(checkpointDir / "__0_0.distcp") > 0;
            resultReceived.set_value();
            CHECK(acknowledgment.wait_for(20s) == std::future_status::ready);
        }
        return std::pair{std::string{}, result};
    });
    TestServerProcess        server;
    StartTestServer(server, logDir / "server.log", true);
    auto secondLocal = topology;
    secondLocal.rank = Rank{1};
    ActivateAndUploadRegistries(topology, {SerializedSliceRegistry(topology), SerializedSliceRegistry(secondLocal)});
    CHECK(PostToTestServer("/api/peer/shardedstateregistry/0/2", SerializedSliceFragment(2, 4)) == http::status::ok);
    CHECK(PostToTestServer("/api/peer/shardedstateregistry/0/3", SerializedSliceFragment(3, 4)) == http::status::ok);
    auto                       queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    std::array<float, 4> const values{1.0f, 2.0f, 3.0f, 4.0f};
    for (uint32_t const rank : {0u, 1u})
    {
        ActivateTestSnapshot(queue, checkpointDir, Rank{rank}, topology);
        queue.Push({checkpoint::ipc::Message::SnapshotData{
                       .sessionId   = checkpoint::ipc::SessionId::Default,
                       .snapshotId  = checkpoint::ipc::SnapshotId{40},
                       .globalRank  = Rank{rank},
                       .messageSize = sizeof(values),
                       .offset      = 0,
                   }},
                   reinterpret_cast<char const*>(values.data()),
                   sizeof(values));
    }
    CHECK(PostToTestServer("/api/peer/snapshotcomplete/0/2/40", "") == http::status::ok);
    CHECK(PostToTestServer("/api/peer/snapshotcomplete/0/3/40", "") == http::status::ok);
    queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                   .sessionId  = checkpoint::ipc::SessionId::Default,
                   .globalRank = Rank{1},
                   .snapshotId = checkpoint::ipc::SnapshotId{40},
               }},
               nullptr,
               0);
    queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                   .sessionId  = checkpoint::ipc::SessionId::Default,
                   .globalRank = Rank{0},
                   .snapshotId = checkpoint::ipc::SnapshotId{40},
               }},
               nullptr,
               0);
    bool localFinalized = false;
    for (size_t attempt = 0; attempt < 100; ++attempt)
    {
        auto const bytes = ReadWholeFile(logDir / "server.log");
        localFinalized   = std::string_view{bytes.data(), bytes.size()}.contains("OnFinalizeSnapshotAsync_ for rank=1, snapshotId=40");
        if (localFinalized) { break; }
        std::this_thread::sleep_for(50ms);
    }
    REQUIRE(localFinalized);
    CHECK(std::filesystem::file_size(checkpointDir / "__1_0.distcp") > 0);
    CHECK_FALSE(std::filesystem::exists(checkpointDir / ".metadata"));
    auto waiting = std::async(std::launch::async, [] { WaitForCkptFinalize(0, -1); });
    CHECK(waiting.wait_for(3s) == std::future_status::timeout);
    ActivateTestSnapshot(queue, logDir / "iteration41", Rank{0}, topology, checkpoint::ipc::SnapshotId{41});
    bool activated = false;
    for (size_t attempt = 0; attempt < 100; ++attempt)
    {
        auto const bytes = ReadWholeFile(logDir / "server.log");
        activated        = std::string_view{bytes.data(), bytes.size()}.contains("Activated snapshot=41 for global rank=0");
        if (activated) { break; }
        std::this_thread::sleep_for(50ms);
    }
    REQUIRE(activated);
    CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/2/40", WriteCompletionBody("")) == http::status::ok);
    CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/3/40", WriteCompletionBody("")) == http::status::ok);
    CHECK(received.wait_for(10s) == std::future_status::ready);
    CHECK(waiting.wait_for(3s) == std::future_status::timeout);
    releaseAcknowledgment.set_value();
    REQUIRE(notified.wait_for(10s) == std::future_status::ready);
    CHECK_NOTHROW(notified.get());
    CHECK(targets == std::vector<std::string>{"/api/peer/globalregistry/0?fwd=1:2", "/api/peer/writeckptcomplete/0/0/40?fwd=1:2"});
    CHECK(resultBody == WriteCompletionBody(""));
    CHECK(metadataPublished);
    auto const status = waiting.wait_for(10s);
    if (status != std::future_status::ready) { CHECK(server.Stop()); }
    REQUIRE(status == std::future_status::ready);
    CHECK_NOTHROW(waiting.get());
    CHECK_NOTHROW(WaitForCkptFinalize(0, 40));
    CHECK(server.Stop());
    acceptor.non_blocking(true);
    Tcp::socket               unexpected{ioc};
    boost::system::error_code error;
    acceptor.accept(unexpected, error);
    CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
    std::filesystem::remove_all(logDir);
}

TEST_CASE("peer write completions refresh dynamic blob metadata across snapshots", "[server][finalize][blob][tier2]")
{
    using namespace std::chrono_literals;
    namespace http    = boost::beast::http;
    using Tcp         = boost::asio::ip::tcp;
    auto const logDir = CkptDir("finalize_peer_dynamic_blob_sizes");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    boost::asio::io_context ioc;
    Tcp::acceptor           acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
    auto const              peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
    RankTopology const topology{.rank = Rank{0}, .worldSize = Rank{2}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {"localhost", peer}};
    auto               remoteTopology = topology;
    remoteTopology.rank               = Rank{1};
    StateTree remote(checkpoint::ipc::SessionId::Default, remoteTopology);
    remote.AddChild(StateNode::Handle::Root, "weight").data    = SliceOfSharedTensor(1, 2);
    remote.AddChild(StateNode::Handle::Root, "a_dynamic").data = StateNode::DynamicSizeBlob{.replicas = {Rank{1}}};
    Tensor::Dim shape;
    shape.push_back(1);
    StateNode::FixedSizeBlob fixed(Tensor::DType::UInt8, shape);
    Tensor::DimRange         ranges;
    ranges.push_back({.begin = 0, .end = 1});
    fixed.AddShard(
        {.rank = Rank{1}, .snapshotDtype = Tensor::DType::UInt8, .storageSize = 5, .sharddim = ranges, .device = DeviceId{"cpu"}});
    remote.AddChild(StateNode::Handle::Root, "z_fixed").data = std::move(fixed);
    remote.FinalizeRegistry();
    std::vector<std::string> targets;
    uint32_t                 completed{0};
    auto                     notified = ServePeerRequests(ioc, acceptor, [&](auto const& request) {
        targets.emplace_back(request.target());
        if (request.target().starts_with("/api/peer/writeckptcomplete/"))
        {
            CHECK(request.body() == WriteCompletionBody(""));
            ++completed;
        }
        return std::pair{std::string{}, completed == 2};
    });
    TestServerProcess        server;
    StartTestServer(server, logDir / "server.log", true);
    ActivateAndUploadRegistries(topology, {SerializedSliceRegistry(topology)});
    REQUIRE(PostToTestServer("/api/peer/shardedstateregistry/0/1", SerializedPeerRegistry(remote)) == http::status::ok);
    auto                       queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
    std::array<float, 4> const values{1.0f, 2.0f, 3.0f, 4.0f};
    for (uint32_t iteration = 40; iteration < 42; ++iteration)
    {
        auto const snapshotId    = checkpoint::ipc::SnapshotId{iteration};
        auto const checkpointDir = logDir / fmt::format("iteration{}", iteration);
        auto const blobSize      = iteration == 40 ? uint64_t{3} : uint64_t{9};
        std::filesystem::create_directories(checkpointDir);
        ActivateTestSnapshot(queue, checkpointDir, Rank{0}, topology, snapshotId);
        queue.Push({checkpoint::ipc::Message::SnapshotData{
                       .sessionId   = checkpoint::ipc::SessionId::Default,
                       .snapshotId  = snapshotId,
                       .globalRank  = Rank{0},
                       .messageSize = sizeof(values),
                       .offset      = 0,
                   }},
                   reinterpret_cast<char const*>(values.data()),
                   sizeof(values));
        queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                       .sessionId  = checkpoint::ipc::SessionId::Default,
                       .globalRank = Rank{0},
                       .snapshotId = snapshotId,
                   }},
                   nullptr,
                   0);
        CHECK(PostToTestServer(fmt::format("/api/peer/snapshotcomplete/0/1/{}", iteration), "") == http::status::ok);
        CHECK(PostToTestServer(fmt::format("/api/peer/writeckptcomplete/0/1/{}", iteration),
                               WriteCompletionBody("", {{"a_dynamic", blobSize}}))
              == http::status::ok);
        auto       finalize = std::async(std::launch::async, [iteration] { WaitForCkptFinalize(0, iteration); });
        auto const status   = finalize.wait_for(10s);
        if (status != std::future_status::ready) { CHECK(server.Stop()); }
        REQUIRE(status == std::future_status::ready);
        CHECK_NOTHROW(finalize.get());
        CHECK(std::filesystem::exists(checkpointDir / ".metadata"));
#ifdef PHOENIX_VENV_PYTHON
        Tensor::Dim shardShape;
        shardShape.push_back(values.size());
        auto const tensorBytes = checkpoint::distcp::ComputeShardArchiveSize(Tensor::DType::Float32, shardShape, sizeof(values));
        auto const cmd         = fmt::format("{} -c \"import pickle; m=pickle.load(open(r'{}','rb')); "
                                             "s={{i.fqn:(v.relative_path,v.offset,v.length) for i,v in m.storage_data.items()}}; "
                                             "assert s['a_dynamic']==('__1_0.distcp',{},{}); "
                                             "assert s['z_fixed/shard_0_1']==('__1_0.distcp',{},5)\"",
                                             PHOENIX_VENV_PYTHON,
                                             (checkpointDir / ".metadata").string(),
                                             tensorBytes,
                                             blobSize,
                                             tensorBytes + blobSize);
        REQUIRE(std::system(cmd.c_str()) == 0);    // NOLINT
#endif
    }
    REQUIRE(notified.wait_for(10s) == std::future_status::ready);
    CHECK_NOTHROW(notified.get());
    CHECK(targets
          == std::vector<std::string>{"/api/peer/globalregistry/0?fwd=1:2",
                                      "/api/peer/writeckptcomplete/0/0/40?fwd=1:2",
                                      "/api/peer/globalregistry/0?fwd=1:2",
                                      "/api/peer/writeckptcomplete/0/0/41?fwd=1:2"});
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("peer durability waits for the coordinator notification", "[server][finalize][tier2]")
{
    using namespace std::chrono_literals;
    namespace http = boost::beast::http;
    using Tcp      = boost::asio::ip::tcp;
    for (bool const succeeds : {false, true})
    {
        DYNAMIC_SECTION("coordinator completes=" << succeeds)
        {
            auto const logDir = CkptDir("finalize_peer_global_result");
            std::filesystem::remove_all(logDir);
            std::filesystem::create_directories(logDir);
            boost::asio::io_context ioc;
            Tcp::acceptor           acceptor{ioc, {boost::asio::ip::address_v4::loopback(), 0}};
            auto const              peer = fmt::format("127.0.0.1:{}", acceptor.local_endpoint().port());
            RankTopology const      topology{
                .rank = Rank{2}, .worldSize = Rank{4}, .ranksPerPeer = 2, .peersPerNode = 1, .peers = {peer, "localhost"}};
            std::set<std::string> registryTargets;
            auto                  registered = ServePeerRequests(ioc, acceptor, [&](auto const& request) {
                registryTargets.insert(std::string{request.target()});
                return std::pair{std::string{}, registryTargets.size() == 2};
            });
            TestServerProcess     server;
            ::setenv("NATIVE_CKPT_SERVER_MAX_ACTIVE_SNAPSHOTS", "0", 1);    // NOLINT(concurrency-mt-unsafe)
            StartTestServer(server, logDir / "server.log", true);
            ::unsetenv("NATIVE_CKPT_SERVER_MAX_ACTIVE_SNAPSHOTS");    // NOLINT(concurrency-mt-unsafe)
            auto secondLocal = topology;
            secondLocal.rank = Rank{3};
            ActivateAndUploadRegistries(topology, {SerializedSliceRegistry(topology), SerializedSliceRegistry(secondLocal)});
            REQUIRE(registered.wait_for(10s) == std::future_status::ready);
            CHECK_NOTHROW(registered.get());
            CHECK(registryTargets == std::set<std::string>{"/api/peer/shardedstateregistry/0/2", "/api/peer/shardedstateregistry/0/3"});
            StateTree global(checkpoint::ipc::SessionId::Default, topology);
            auto      tensor = SliceOfSharedTensor(2, 4);
            tensor.shards.insert(*SliceOfSharedTensor(3, 4).shards.begin());
            global.AddChild(StateNode::Handle::Root, "weight").data = std::move(tensor);
            global.FinalizeRegistry();
            std::stringstream registry;
            StateSerDes::Write(registry, global, StateNode::Handle::Root);
            auto const  registryBytes = registry.str();
            std::string globalRegistry;
            {
                boost::iostreams::filtering_ostream compressed;
                compressed.push(boost::iostreams::zlib_compressor{});
                compressed.push(boost::iostreams::back_inserter(globalRegistry));
                compressed.write(registryBytes.data(), static_cast<std::streamsize>(registryBytes.size()));
                boost::iostreams::close(compressed);
            }
            std::promise<void>    firstWrite;
            auto                  firstWritten = firstWrite.get_future();
            std::set<std::string> written;
            auto                  localWrite    = ServePeerRequests(ioc, acceptor, [&](auto const& notification) {
                CHECK(notification.body()
                      == (notification.target().starts_with("/api/peer/writeckptcomplete/") ? WriteCompletionBody("") : std::string{}));
                if (notification.target() == "/api/peer/snapshotcomplete/0/2/40")
                {
                    CHECK(PostToTestServer("/api/peer/globalregistry/0?fwd=1:2", globalRegistry, "deflate") == http::status::ok);
                }
                if (notification.target() == "/api/peer/writeckptcomplete/0/2/40")
                {
                    if (written.insert(std::string{notification.target()}).second) { firstWrite.set_value(); }
                }
                else if (notification.target() == "/api/peer/writeckptcomplete/0/3/40")
                {
                    written.insert(std::string{notification.target()});
                }
                return std::pair{std::string{}, written.size() == 2};
            });
            auto const            checkpointDir = logDir / "iteration40";
            std::filesystem::create_directories(checkpointDir);
            auto                       queue = checkpoint::ipc::MsgQueue::Open(checkpoint::ipc::Message::RequestChannelName(0, 0));
            std::array<float, 4> const values{1.0f, 2.0f, 3.0f, 4.0f};
            for (uint32_t const rank : {2u, 3u})
            {
                ActivateTestSnapshot(queue, checkpointDir, Rank{rank}, topology);
                queue.Push({checkpoint::ipc::Message::SnapshotData{
                               .sessionId   = checkpoint::ipc::SessionId::Default,
                               .snapshotId  = checkpoint::ipc::SnapshotId{40},
                               .globalRank  = Rank{rank},
                               .messageSize = sizeof(values),
                               .offset      = 0,
                           }},
                           reinterpret_cast<char const*>(values.data()),
                           sizeof(values));
            }
            queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                           .sessionId  = checkpoint::ipc::SessionId::Default,
                           .globalRank = Rank{2},
                           .snapshotId = checkpoint::ipc::SnapshotId{40},
                       }},
                       nullptr,
                       0);
            REQUIRE(firstWritten.wait_for(10s) == std::future_status::ready);
            CHECK_NOTHROW(firstWritten.get());
            bool        firstFinalized = false;
            std::string firstLog;
            for (size_t attempt = 0; attempt < 100; ++attempt)
            {
                auto const bytes = ReadWholeFile(logDir / "server.log");
                firstLog.assign(bytes.data(), bytes.size());
                firstFinalized = firstLog.contains("Released 0 snapshot buffers after finalizing snapshot=40 rank=2.");
                if (firstFinalized) { break; }
                std::this_thread::sleep_for(50ms);
            }
            REQUIRE(firstFinalized);
            CHECK_FALSE(std::filesystem::exists(checkpointDir / "__3_0.distcp"));
            queue.Push({checkpoint::ipc::Message::FinalizeSnapshot{
                           .sessionId  = checkpoint::ipc::SessionId::Default,
                           .globalRank = Rank{3},
                           .snapshotId = checkpoint::ipc::SnapshotId{40},
                       }},
                       nullptr,
                       0);
            auto const localStatus = localWrite.wait_for(10s);
            if (localStatus != std::future_status::ready) { CHECK(server.Stop()); }
            REQUIRE(localStatus == std::future_status::ready);
            CHECK_NOTHROW(localWrite.get());
            CHECK(written == std::set<std::string>{"/api/peer/writeckptcomplete/0/2/40", "/api/peer/writeckptcomplete/0/3/40"});
            CHECK(std::filesystem::file_size(checkpointDir / "__2_0.distcp") > 0);
            CHECK(std::filesystem::file_size(checkpointDir / "__3_0.distcp") > 0);
            boost::asio::io_context  clientIoc;
            boost::beast::tcp_stream waitStream{clientIoc};
            waitStream.connect(Tcp::endpoint{boost::asio::ip::address_v4::loopback(), checkpoint::ipc::Message::Port});
            http::request<http::empty_body> request{http::verb::get, "/api/wait-for-ckpt-finalize/0/40", 11};
            request.set(http::field::host, "localhost");
            http::write(waitStream, request);
            std::this_thread::sleep_for(200ms);
            CHECK(waitStream.socket().available() == 0);
            acceptor.non_blocking(true);
            Tcp::socket               unexpected{ioc};
            boost::system::error_code error;
            acceptor.accept(unexpected, error);
            CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
            std::string const failure = "Checkpoint finalization failed for snapshot=40 rank=0: metadata Input/output error";
            auto const        result  = succeeds ? std::string{} : failure;
            CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/0/40?fwd=1:2", WriteCompletionBody(result)) == http::status::ok);
            CHECK(PostToTestServer("/api/peer/writeckptcomplete/0/0/40?fwd=1:2", WriteCompletionBody(result)) == http::status::ok);
            boost::beast::flat_buffer         buffer;
            http::response<http::string_body> response;
            boost::system::error_code         readError;
            waitStream.expires_after(10s);
            http::async_read(waitStream, buffer, response, [&](boost::system::error_code ec, size_t) { readError = ec; });
            clientIoc.run();
            if (readError) { CHECK(server.Stop()); }
            REQUIRE_FALSE(readError);
            CHECK_FALSE(response.chunked());
            CHECK(response[http::field::content_type].starts_with("text/plain"));
            if (succeeds) { CHECK(response.result() == http::status::ok); }
            else
            {
                CHECK(response.result() == http::status::internal_server_error);
                CHECK(response.body().contains(failure));
            }
            CHECK(server.Stop());
            Tcp::socket noFeedback{ioc};
            acceptor.accept(noFeedback, error);
            CHECK((error == boost::asio::error::would_block || error == boost::asio::error::try_again));
            std::filesystem::remove_all(logDir);
        }
    }
}

// The merged dump appears while the server keeps running; returns the ranks owning a slice in it.
static std::set<uint32_t> GlobalDumpSliceOwners(std::filesystem::path const& logDir)
{
    auto const path = logDir / "registry.global.bin";
    for (size_t attempt = 0; attempt < 100 && !std::filesystem::exists(path); ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());
    auto const global = StateSerDes::Read(file);
    auto const weight = global->FindNodeByFQN("weight");
    REQUIRE(weight != StateNode::Handle::Invalid);
    std::set<uint32_t> owners;
    for (auto const& shard : std::get<DTensor>(global->GetNode(weight).data).shards) { owners.insert(static_cast<uint32_t>(shard.rank)); }
    return owners;
}

TEST_CASE("received registries are dumped beside the server log", "[server][tier2]")
{
    auto const logDir = CkptDir("registry_dumps");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    std::ofstream(logDir / "registry.0.bin") << "stale";
    std::vector<std::string> const registries{SerializedSliceRegistry(SingleRankTopo(0, 2)), SerializedSliceRegistry(SingleRankTopo(1, 2))};

    TestServerProcess server;
    StartTestServer(server, logDir / "server.%h.%p.log");
    ActivateAndUploadRegistries(SingleRankTopo(0, 2), registries);
    for (size_t rank = 0; rank < registries.size(); ++rank)
    {
        std::ifstream     dump(logDir / fmt::format("registry.{}.bin", rank), std::ios::binary);
        std::string const bytes((std::istreambuf_iterator<char>(dump)), std::istreambuf_iterator<char>());
        CHECK(bytes == registries[rank]);
    }
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

// Hidden: the merged registry is dumped at checkpoint finalize, which this registration-only harness never reaches.
TEST_CASE("a peer's registry fragment completes the coordinator's merged dump", "[server][tier2][.]")
{
    auto const logDir = CkptDir("registry_dumps_from_peer");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);
    RankTopology const topo{.rank = Rank{0}, .worldSize = Rank{2}, .ranksPerPeer = 1, .peersPerNode = 1, .peers = {"localhost", "peer"}};

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log");
    ActivateAndUploadRegistries(topo, {SerializedSliceRegistry(topo)});
    CHECK(PostToTestServer(fmt::format("/api/peer/shardedstateregistry/{}/1", checkpoint::ipc::SessionId::Default),
                           SerializedSliceFragment(1, 2))
          == boost::beast::http::status::ok);
    CHECK(GlobalDumpSliceOwners(logDir) == std::set<uint32_t>{0, 1});
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("peer registry fragments reject invalid tensor dimensions", "[server][tier2]")
{
    auto const logDir = CkptDir("invalid_peer_registry_fragment");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);

    auto       fragment     = SerializedSliceFragment(1, 2);
    auto const sizeOffset   = sizeof(uint8_t) + sizeof(Tensor::DType) + (sizeof(uint64_t) * Tensor::MaxDims);
    fragment.at(sizeOffset) = static_cast<char>(Tensor::MaxDims + 1);

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log");
    CHECK_THROWS(
        PostToTestServer(fmt::format("/api/peer/shardedstateregistry/{}/1", checkpoint::ipc::SessionId::Default), std::move(fragment)));
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("peer registry fragments reject mismatched shard ranks", "[server][tier2]")
{
    auto const logDir = CkptDir("mismatched_peer_registry_rank");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log");
    CHECK_THROWS(PostToTestServer(fmt::format("/api/peer/shardedstateregistry/{}/1", checkpoint::ipc::SessionId::Default),
                                  SerializedSliceFragment(0, 2)));
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("local registries reject nonlocal DTensor shards", "[server][tier2]")
{
    auto const logDir = CkptDir("nonlocal_local_registry_shard");
    std::filesystem::remove_all(logDir);
    std::filesystem::create_directories(logDir);

    auto const topology = SingleRankTopo();
    StateTree  tree(checkpoint::ipc::SessionId::Default, topology);
    AddShardedTensorLeaf(tree, "weight", Tensor::DType::Float32, {2}, Rank{1}, 0);
    tree.FinalizeRegistry();
    std::stringstream registry;
    StateSerDes::Write(registry, tree, StateNode::Handle::Root);

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log");
    std::stringstream activation;
    serdes::WriteBlobWithSize<uint16_t>(activation, "");
    StateSerDes::WriteTopology(activation, topology);
    CHECK(PostToTestServer(fmt::format("/api/activatesession/{}", checkpoint::ipc::SessionId::Default), activation.str())
          == boost::beast::http::status::accepted);
    CHECK(PostToTestServer(fmt::format("/api/localstateregistry/{}", checkpoint::ipc::SessionId::Default), registry.str())
          == boost::beast::http::status::insufficient_storage);
    CHECK(server.Stop());
    std::filesystem::remove_all(logDir);
}

TEST_CASE("registries are not dumped without a server log directory", "[server][tier2]")
{
    auto const logDir = CkptDir("registry_dumps_without_log_dir");
    std::filesystem::remove_all(logDir);

    TestServerProcess server;
    StartTestServer(server, logDir / "server.log");
    ActivateAndUploadRegistries(SingleRankTopo(0, 1), {SerializedSliceRegistry(SingleRankTopo(0, 1))});
    CHECK(server.Stop());
    CHECK_FALSE(std::filesystem::exists(logDir));
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

TEST_CASE("evicted pending snapshots remain the latest known result", "[server][completion][finalize][tier1]")
{
    Session session(nullptr, checkpoint::ipc::SessionId::Default, std::filesystem::path{}, SingleRankTopo(0, 2));
    auto    completed     = std::make_unique<Session::Snapshot>();
    completed->snapshotId = checkpoint::ipc::SnapshotId{30};
    completed->complete   = true;
    session.snapshots.emplace(completed->snapshotId, std::move(completed));
    auto snapshot        = std::make_unique<Session::Snapshot>();
    snapshot->snapshotId = checkpoint::ipc::SnapshotId{40};
    snapshot->localStates.emplace(Rank{0}, std::make_unique<Session::LocalState>());
    auto* const pending = snapshot.get();
    session.snapshots.emplace(snapshot->snapshotId, std::move(snapshot));
    pending->localStates.clear();
    checkpoint::ipc::SnapshotId latest{checkpoint::ipc::SnapshotId::Invalid};
    REQUIRE(session.ResolveSnapshotId(-1, latest));
    CHECK(latest == checkpoint::ipc::SnapshotId{40});
    CHECK_FALSE(session.snapshots.at(latest)->complete);
    CHECK_FALSE(session.AllSnapshotsComplete());
    pending->complete = true;
    CHECK(session.snapshots.at(latest).get() == pending);
    CHECK(session.snapshots.at(latest)->complete);
    CHECK(session.AllSnapshotsComplete());
}

TEST_CASE("finalize snapshot lookup rejects out-of-range iterations", "[server][completion][finalize][tier1]")
{
    using SnapshotId = checkpoint::ipc::SnapshotId;
    Session session(nullptr, checkpoint::ipc::SessionId::Default, std::filesystem::path{}, SingleRankTopo());
    auto    resolved = SnapshotId::Invalid;
    CHECK(session.ResolveSnapshotId(0, resolved));
    CHECK(resolved == SnapshotId{0});
    auto const maxValid = static_cast<int64_t>(static_cast<uint32_t>(SnapshotId::Invalid)) - 1;
    CHECK(session.ResolveSnapshotId(maxValid, resolved));
    CHECK(resolved == static_cast<SnapshotId>(static_cast<uint32_t>(maxValid)));
    for (auto const iteration : {maxValid + 1, maxValid + 2, (int64_t{1} << 32u) + 40, std::numeric_limits<int64_t>::max()})
    {
        INFO(iteration);
        CHECK_THROWS_AS(session.ResolveSnapshotId(iteration, resolved), std::invalid_argument);
    }
    auto snapshot        = std::make_unique<Session::Snapshot>();
    snapshot->snapshotId = SnapshotId{40};
    session.snapshots.emplace(snapshot->snapshotId, std::move(snapshot));
    CHECK(session.ResolveSnapshotId(-1, resolved));
    CHECK(resolved == SnapshotId{40});
    CHECK_FALSE(session.ResolveSnapshotId(-2, resolved));
    CHECK_FALSE(session.ResolveSnapshotId(std::numeric_limits<int64_t>::min(), resolved));
}

TEST_CASE("distcp shard walk visits TensorShard leaves in insertion order", "[distcp][tier1]")
{
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(64);

    AddTensorShardLeaf(tree, "a", Tensor::DType::Float32, {2, 2}, 0);    // 16 B
    AddTensorShardLeaf(tree, "b", Tensor::DType::Int8, {4}, 16);         //  4 B
    AddTensorShardLeaf(tree, "c", Tensor::DType::Float64, {2}, 32);      // 16 B

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "a");
    REQUIRE(entries[1].fqn == "b");
    REQUIRE(entries[2].fqn == "c");

    REQUIRE(entries[0].dtype == Tensor::DType::Float32);
    REQUIRE(entries[1].dtype == Tensor::DType::Int8);
    REQUIRE(entries[2].dtype == Tensor::DType::Float64);

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
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(128);

    // Fully-occupied 1-D tensor — shape matches fulldim.
    AddTensorShardLeaf(tree, "full", Tensor::DType::Float32, {4}, 0);

    // Non-full 2-D slice: fulldim is 8x4 but this shard covers rows [2,6) and cols [1,3).
    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("slice");
        auto&       leaf        = tree.AddChild(parent, leafName);
        Tensor::Dim fulldim;
        fulldim.push_back(8);
        fulldim.push_back(4);
        DTensor          tnsr(Tensor::DType::Int32, fulldim);
        Tensor::DimRange sharddim;
        sharddim.push_back({.begin = 2, .end = 6});
        sharddim.push_back({.begin = 1, .end = 3});
        tnsr.AddShard({
            .rank          = Rank{0},
            .snapshotDtype = Tensor::DType::Int32,
            .handle        = 16,
            .storageSize   = 4ull * 2u * Tensor::GetDTypeSize(Tensor::DType::Int32),
            .sharddim      = sharddim,
            .device        = DeviceId{"cpu"},
        });
        leaf.data = std::move(tnsr);
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

TEST_CASE("distcp shard walk expands DTensor into per-rank entries", "[distcp][tier1]")
{
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo(0, 2));
    ByteStorage tensorStorage(64);

    auto [leafName, parent] = tree.FindOrAddNodeByFQN("shrd");
    auto&       leaf        = tree.AddChild(parent, leafName);
    Tensor::Dim fulldim;
    fulldim.push_back(8);
    DTensor tnsr(Tensor::DType::Float32, fulldim);

    Tensor::DimRange r0;
    r0.push_back({.begin = 0, .end = 4});
    Tensor::DimRange r1;
    r1.push_back({.begin = 4, .end = 8});
    tnsr.AddShard({.rank          = Rank{0},
                   .snapshotDtype = Tensor::DType::Float32,
                   .handle        = 0,
                   .storageSize   = 16,
                   .sharddim      = r0,
                   .device        = DeviceId{"cpu"}});
    tnsr.AddShard({.rank          = Rank{1},
                   .snapshotDtype = Tensor::DType::Float32,
                   .handle        = 16,
                   .storageSize   = 16,
                   .sharddim      = r1,
                   .device        = DeviceId{"cpu"}});
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
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo(0, 2));
    ByteStorage tensorStorage(64);

    auto [leafName, parent] = tree.FindOrAddNodeByFQN("shrd");
    auto&       leaf        = tree.AddChild(parent, leafName);
    Tensor::Dim fulldim;
    fulldim.push_back(8);
    DTensor tnsr(Tensor::DType::Float32, fulldim);

    Tensor::DimRange r0low;
    r0low.push_back({.begin = 0, .end = 4});
    Tensor::DimRange r0high;
    r0high.push_back({.begin = 4, .end = 8});
    // Insertion order intentionally inverted vs. the std::set order so the
    // test only passes if `tnsr.shards` is iterated in (rank, sharddim) order.
    tnsr.AddShard({.rank          = Rank{0},
                   .snapshotDtype = Tensor::DType::Float32,
                   .handle        = 32,
                   .storageSize   = 16,
                   .sharddim      = r0high,
                   .device        = DeviceId{"cpu"}});
    tnsr.AddShard({.rank          = Rank{0},
                   .snapshotDtype = Tensor::DType::Float32,
                   .handle        = 0,
                   .storageSize   = 16,
                   .sharddim      = r0low,
                   .device        = DeviceId{"cpu"}});

    leaf.data = std::move(tnsr);

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].sharddim[0].begin == 0);
    REQUIRE(entries[1].sharddim[0].begin == 4);
}

TEST_CASE("distcp shard walk handles mixed TensorShard and DTensor nodes", "[distcp][tier1]")
{
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo(0, 2));
    ByteStorage tensorStorage(128);

    AddTensorShardLeaf(tree, "before", Tensor::DType::Float32, {4}, 0);

    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("middle");
        auto&       leaf        = tree.AddChild(parent, leafName);
        Tensor::Dim fulldim;
        fulldim.push_back(8);
        DTensor          tnsr(Tensor::DType::Float32, fulldim);
        Tensor::DimRange r;
        r.push_back({.begin = 0, .end = 8});
        tnsr.AddShard({.rank          = Rank{0},
                       .snapshotDtype = Tensor::DType::Float32,
                       .handle        = 16,
                       .storageSize   = 32,
                       .sharddim      = r,
                       .device        = DeviceId{"cpu"}});
        leaf.data = std::move(tnsr);
    }

    AddTensorShardLeaf(tree, "after", Tensor::DType::Float32, {2}, 48);

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "before");
    REQUIRE(entries[1].fqn == "middle");
    REQUIRE(entries[2].fqn == "after");
}

TEST_CASE("distcp shard walk joins nested dict names with dots", "[distcp][tier1]")
{
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(64);

    AddTensorShardLeaf(tree, "model.layer0.weight", Tensor::DType::Float32, {2}, 0);
    AddTensorShardLeaf(tree, "model.layer0.bias", Tensor::DType::Float32, {2}, 8);
    AddTensorShardLeaf(tree, "model.layer1.weight", Tensor::DType::Float32, {2}, 16);

    auto const entries = CollectShardsForTest(tree, Rank{0}, tensorStorage);

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0].fqn == "model.layer0.weight");
    REQUIRE(entries[1].fqn == "model.layer0.bias");
    REQUIRE(entries[2].fqn == "model.layer1.weight");
}

TEST_CASE("distcp shard walk skips tensors without shards", "[distcp][tier1]")
{
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(16);

    AddTensorShardLeaf(tree, "non_empty", Tensor::DType::Float32, {2}, 0);

    {
        auto [leafName, parent] = tree.FindOrAddNodeByFQN("empty");
        auto& leaf              = tree.AddChild(parent, leafName);
        leaf.data               = DTensor(Tensor::DType::Float32, {});
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
        if (type == StateNode::Type::DTensor)
        {
            auto const& tnsr = std::get<DTensor>(node.data);
            for (auto const& shard : tnsr.shards)
            {
                auto const shardSz = shard.StorageSizeBytes();
                if (shardSz == 0) { continue; }
                Tensor::Dim shape;
                for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
                uint64_t const archiveLen = checkpoint::distcp::ComputeShardArchiveSize(tnsr.durableDtype, shape, shardSz);
                out.push_back({.fqn = tree.GetNodeId(node.handle), .fileOffset = cursor, .length = archiveLen});
                cursor += archiveLen;
            }
        }
    });
    return out;
}

TEST_CASE("distcp::WriteRankShard produces a contiguous file of torch.save ZIPs", "[distcp][tier2]")
{
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(64);
    // Deterministic source bytes — keep them simple so any future stride/dtype
    // bug shows up as a torch.load mismatch on inspection.
    auto* bytes = static_cast<unsigned char*>(tensorStorage.data);
    for (size_t i = 0; i < tensorStorage.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }

    AddShardedTensorLeaf(tree, "a", Tensor::DType::Float32, {2, 2}, Rank{0}, 0);    // 16 B
    AddShardedTensorLeaf(tree, "b", Tensor::DType::Int8, {4}, Rank{0}, 16);         //  4 B
    AddShardedTensorLeaf(tree, "c", Tensor::DType::Float64, {2}, Rank{0}, 32);      // 16 B

    auto outDir = CkptDir("tier2_layout");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, tensorStorage);

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
    StateTree globalTree(checkpoint::ipc::SessionId::Default, SingleRankTopo(0, 2));
    AddShardedTensorLeaf(globalTree, "weight", Tensor::DType::Float32, {2, 2}, Rank{0}, 0);

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
    checkpoint::distcp::WriteRankShard(outDir, Rank{1}, globalTree, emptyTensorStorage);

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
        ls.tree = std::make_unique<StateTree>(checkpoint::ipc::SessionId::Default, SingleRankTopo(r, kWorldSize));
        AddShardedTensorLeaf(*ls.tree, "a", Tensor::DType::Float32, {2, 2}, Rank{r}, 0);    // 16 B
        AddShardedTensorLeaf(*ls.tree, "b", Tensor::DType::Int8, {4}, Rank{r}, 16);         //  4 B
        AddShardedTensorLeaf(*ls.tree, "c", Tensor::DType::Float64, {2}, Rank{r}, 32);      // 16 B
    }

    auto outDir = CkptDir("tier6_concurrent");
    std::filesystem::remove_all(outDir);

    std::vector<std::thread> workers;
    workers.reserve(kWorldSize);
    for (uint32_t r = 0; r < kWorldSize; ++r)
    {
        workers.emplace_back([&snapshot, &outDir, r] {
            auto& ls = snapshot.states[r];
            checkpoint::distcp::WriteRankShard(outDir, Rank{r}, *ls.tree, *ls.tensorStorage);
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
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(256);
    auto*       bytes = static_cast<unsigned char*>(tensorStorage.data);
    for (size_t i = 0; i < tensorStorage.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }

    // Mix of dtypes that produce different `data.pkl` sizes and shard byte
    // counts. FB padding is offset-dependent, so any shape arrangement gets
    // exercised once the prior shards' cumulative size shifts the cursor.
    AddShardedTensorLeaf(tree, "a", Tensor::DType::Float32, {2, 2}, Rank{0}, 0);    // 16 B
    AddShardedTensorLeaf(tree, "b", Tensor::DType::Int8, {4}, Rank{0}, 16);         //  4 B
    AddShardedTensorLeaf(tree, "c", Tensor::DType::Float64, {2}, Rank{0}, 32);      // 16 B
    AddShardedTensorLeaf(tree, "d", Tensor::DType::Int32, {4}, Rank{0}, 64);        // 16 B
    AddShardedTensorLeaf(tree, "e", Tensor::DType::Float32, {8}, Rank{0}, 128);     // 32 B

    auto outDir = CkptDir("predict_consistency");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, tensorStorage);

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
    // global DTensor entries. The empty manifest skips per-shard
    // verification (the multi-node case where rank 0 has no view into other
    // ranks' writeLogs).
    StateTree globalTree(checkpoint::ipc::SessionId::Default, SingleRankTopo(0, 2));
    auto      addShard = [&](std::string_view                                                            fqn,
                             Tensor::DType                                                               dtype,
                             std::vector<size_t> const&                                                  fullShape,
                             std::vector<std::pair<Rank, std::vector<std::pair<size_t, size_t>>>> const& shards) {
        Tensor::Dim fulldim;
        for (auto d : fullShape) { fulldim.push_back(d); }
        auto [leafName, parent] = globalTree.FindOrAddNodeByFQN(fqn);
        auto&   leaf            = globalTree.AddChild(parent, leafName);
        DTensor st(dtype, fulldim);
        for (auto const& [r, ranges] : shards)
        {
            Tensor::DimRange sharddim;
            for (auto const& [b, e] : ranges) { sharddim.push_back({.begin = b, .end = e}); }
            uint64_t storageSize = Tensor::GetDTypeSize(dtype);
            for (auto const& range : sharddim) { storageSize *= range.end - range.begin; }
            st.AddShard({.rank          = r,
                         .snapshotDtype = dtype,
                         .handle        = 0,
                         .storageSize   = storageSize,
                         .sharddim      = sharddim,
                         .device        = DeviceId{"cpu"}});
        }
        leaf.data = std::move(st);
    };
    // Tensor of full shape (8,) split into two halves across ranks 0 and 1.
    addShard("weight", Tensor::DType::Float32, {8}, {{Rank{0}, {{0, 4}}}, {Rank{1}, {{4, 8}}}});
    addShard("bias", Tensor::DType::Int64, {4}, {{Rank{0}, {{0, 2}}}, {Rank{1}, {{2, 4}}}});

    auto outDir = CkptDir("metadata_multi_rank_inference");
    std::filesystem::remove_all(outDir);

    REQUIRE_NOTHROW(checkpoint::distcp::WriteMetadata(outDir, globalTree));
    auto const metaPath = outDir / ".metadata";
    REQUIRE(std::filesystem::exists(metaPath));
    REQUIRE(std::filesystem::file_size(metaPath) > 0);
}

TEST_CASE("distcp Blobs live in their rank's shard file", "[distcp][tier1][metadata][blob]")
{
    // Each rank appends its Blobs in fqn order after its tensor region.
    // The coordinator's FQN-grouped metadata view points at those local files.
    StateTree globalTree(checkpoint::ipc::SessionId::Default, SingleRankTopo(0, 2));

    std::vector<char> const alphaBytes{'a', 'l', 'p', 'h', 'a'};
    std::vector<char> const gammaBytes{'g'};
    std::vector<char> const betaBytes{'b', 'e', 't', 'a'};
    std::vector<char> const deltaBytes{'d', 'e', 'l', 't', 'a'};

    globalTree.AddChild(StateNode::Handle::Root, "alpha").data
        = StateNode::DynamicSizeBlob{.replicas = {Rank{0}, Rank{1}}, .handle = 0, .storageSize = alphaBytes.size()};
    globalTree.AddChild(StateNode::Handle::Root, "beta").data
        = StateNode::DynamicSizeBlob{.replicas = {Rank{1}}, .handle = alphaBytes.size(), .storageSize = betaBytes.size()};
    globalTree.AddChild(StateNode::Handle::Root, "delta").data = StateNode::DynamicSizeBlob{
        .replicas = {Rank{1}}, .handle = alphaBytes.size() + betaBytes.size(), .storageSize = deltaBytes.size()};
    globalTree.AddChild(StateNode::Handle::Root, "gamma").data
        = StateNode::DynamicSizeBlob{.replicas = {Rank{0}}, .handle = alphaBytes.size(), .storageSize = gammaBytes.size()};
    ByteStorage rank0Storage(alphaBytes.size() + gammaBytes.size());
    std::memcpy(rank0Storage[0], alphaBytes.data(), alphaBytes.size());
    std::memcpy(rank0Storage[alphaBytes.size()], gammaBytes.data(), gammaBytes.size());
    ByteStorage rank1Storage(alphaBytes.size() + betaBytes.size() + deltaBytes.size());
    std::memcpy(rank1Storage[0], alphaBytes.data(), alphaBytes.size());
    std::memcpy(rank1Storage[alphaBytes.size()], betaBytes.data(), betaBytes.size());
    std::memcpy(rank1Storage[alphaBytes.size() + betaBytes.size()], deltaBytes.data(), deltaBytes.size());

    auto outDir = CkptDir("blobs_per_rank");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, globalTree, rank0Storage);
    checkpoint::distcp::WriteRankShard(outDir, Rank{1}, globalTree, rank1Storage);
    REQUIRE_NOTHROW(checkpoint::distcp::WriteMetadata(outDir, globalTree));

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
    Tensor::Dim shape;
    shape.push_back(1);
    StateNode::FixedSizeBlob inconsistent(Tensor::DType::UInt8, shape);
    Tensor::DimRange         ranges;
    ranges.push_back({.begin = 0, .end = 1});
    for (uint32_t rank = 0; rank < 2; ++rank)
    {
        inconsistent.AddShard({.rank          = Rank{rank},
                               .snapshotDtype = Tensor::DType::UInt8,
                               .handle        = 0,
                               .storageSize   = alphaBytes.size() - rank,
                               .sharddim      = ranges,
                               .device        = DeviceId{"cpu"}});
    }
    globalTree.AddChild(StateNode::Handle::Root, "inconsistent").data = std::move(inconsistent);
    REQUIRE_THROWS_AS(checkpoint::distcp::WriteMetadata(outDir, globalTree), std::runtime_error);
}

#ifdef PHOENIX_VENV_PYTHON
TEST_CASE("distcp::WriteMetadata emits an unpickle-able .metadata", "[distcp][tier1][metadata]")
{
    // Single-rank case: localTree and globalTree are the same tree of
    // single-shard ShardedTensors. WriteRankShard requires the global-
    // registry shape (asserted internally), and WriteMetadata consumes the
    // resulting writeLog as a verification side-channel.
    StateTree   tree(checkpoint::ipc::SessionId::Default, SingleRankTopo());
    ByteStorage tensorStorage(64);
    auto*       bytes = static_cast<unsigned char*>(tensorStorage.data);
    for (size_t i = 0; i < tensorStorage.size; ++i) { bytes[i] = static_cast<unsigned char>(i); }
    AddShardedTensorLeaf(tree, "weight", Tensor::DType::Float32, {2, 2}, Rank{0}, 0);
    AddShardedTensorLeaf(tree, "block.weight", Tensor::DType::Int64, {3}, Rank{0}, 16);

    auto outDir = CkptDir("metadata_unpickle");
    std::filesystem::remove_all(outDir);

    checkpoint::distcp::WriteRankShard(outDir, Rank{0}, tree, tensorStorage);
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
    int const rc = std::system(cmd.c_str());    // NOLINT
    REQUIRE(rc == 0);
}
#endif

// Runs the in-process server and registry round trip when a serialized registry is supplied.
TEST_CASE("save_then_round_trip_validate", "[round-trip]")
{
    auto const& registryPath = TestRegistryPath();
    if (registryPath.empty()) { SKIP("no --registry path provided"); }
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
        REQUIRE_NOTHROW(WaitForInprocServerWrites());
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

// Custom main registers the optional --registry argument.
int main(int argc, char* argv[])
{
    if (argc == 2 && std::string_view{argv[1]} == ShmTestServerArg)
    {
        RunServer();
        return EXIT_SUCCESS;
    }

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
