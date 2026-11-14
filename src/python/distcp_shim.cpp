// Test-only shim: exposes WriteRankShard and WriteMetadata to Python so the
// hypothesis-driven DCP-load tests (tests/test_dcp_load.py, also
// test_distcp_roundtrip.py) can drive the writer without spinning up the
// server, IPC, or async dispatch.

#include "byte_storage.h"
#include "distcp.h"
#include "distcp_internal.h"
#include "state.h"
#include "tensor.h"

#include <fmt/format.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace
{
// Each entry: (fqn, dtype_name, shape, data_bytes).
struct InputEntry
{
    std::string                fqn;
    Tensor::DType              dtype{};
    std::vector<size_t>        shape;
    std::vector<unsigned char> bytes;
};
}    // namespace
static Tensor::DType ParseDType(std::string const& name)
{
    using DT = Tensor::DType;
    if (name == "float16") { return DT::Float16; }
    if (name == "bfloat16") { return DT::BFloat16; }
    if (name == "float32") { return DT::Float32; }
    if (name == "float64") { return DT::Float64; }
    if (name == "uint8") { return DT::UInt8; }
    if (name == "uint16") { return DT::UInt16; }
    if (name == "uint32") { return DT::UInt32; }
    if (name == "uint64") { return DT::UInt64; }
    if (name == "int8") { return DT::Int8; }
    if (name == "int16") { return DT::Int16; }
    if (name == "int32") { return DT::Int32; }
    if (name == "int64") { return DT::Int64; }
    if (name == "bool") { return DT::Bool; }
    throw std::runtime_error(fmt::format("distcp_shim: unsupported dtype '{}'", name));
}
static InputEntry ConvertEntry(nb::tuple const& t)
{
    if (t.size() != 4) { throw std::runtime_error("distcp_shim: each entry must be (fqn, dtype, shape, bytes)"); }
    InputEntry e;
    e.fqn   = nb::cast<std::string>(t[0]);
    e.dtype = ParseDType(nb::cast<std::string>(t[1]));
    e.shape = nb::cast<std::vector<size_t>>(t[2]);
    if (e.shape.empty() || e.shape.size() > Tensor::MaxDims)
    {
        throw std::runtime_error(fmt::format("distcp_shim: shape rank {} out of range [1, {}]", e.shape.size(), Tensor::MaxDims));
    }
    auto data = nb::cast<nb::bytes>(t[3]);
    e.bytes.resize(data.size());
    std::memcpy(e.bytes.data(), data.c_str(), data.size());
    return e;
}

// Parse Python's (fqn, bytes) object tuples into owned Blob storage. The
// parity shim is single-rank, so every Blob belongs to the calling rank.
static std::vector<std::pair<std::string, std::vector<char>>> ParseObjects(nb::list const& objectsPy)
{
    std::vector<std::pair<std::string, std::vector<char>>> out;
    out.reserve(objectsPy.size());
    for (auto const& object : objectsPy)
    {
        auto entry = nb::cast<nb::tuple>(object);
        if (entry.size() != 2) { throw std::runtime_error("distcp_shim: each object must be (fqn, bytes)"); }
        auto fqn  = nb::cast<std::string>(entry[0]);
        auto data = nb::cast<nb::bytes>(entry[1]);
        out.emplace_back(std::move(fqn), std::vector<char>(data.c_str(), data.c_str() + data.size()));
    }
    return out;
}

static void AddScalars(StateTree& tree, Rank rank, nb::dict const& values)
{
    for (auto const& [key, value] : values)
    {
        auto const        fqn = nb::cast<std::string>(key);
        StateNode::Scalar scalar;
        if (nb::isinstance<bool>(value)) { scalar = StateNode::Scalar(nb::cast<bool>(value)); }
        else if (nb::isinstance<nb::int_>(value))
        {
            if (value < nb::int_(0)) { scalar = StateNode::Scalar(nb::cast<int64_t>(value)); }
            else
            {
                scalar = StateNode::Scalar(nb::cast<uint64_t>(value));
            }
        }
        else if (nb::isinstance<nb::float_>(value)) { scalar = StateNode::Scalar(nb::cast<double>(value)); }
        else
        {
            throw std::invalid_argument("Native scalar values require integers, floats, or booleans");
        }
        scalar.replicas                  = {rank};
        auto const [name, parent]        = tree.FindOrAddNodeByFQN(fqn);
        tree.AddChild(parent, name).data = scalar;
    }
}

static nb::list WriteRankShardImpl(std::string const& outDir,
                                   uint32_t           rank,
                                   nb::list const&    entriesPy,
                                   nb::list const&    objectsPy,
                                   nb::dict const&    scalarsPy,
                                   bool               native)
{
    std::vector<InputEntry> entries;
    entries.reserve(entriesPy.size());
    size_t totalBytes = 0;
    for (auto const& i : entriesPy)
    {
        entries.push_back(ConvertEntry(nb::cast<nb::tuple>(i)));
        totalBytes += entries.back().bytes.size();
    }
    auto const blobBytes = ParseObjects(objectsPy);
    for (auto const& [fqn, bytes] : blobBytes) { totalBytes += bytes.size(); }

    // `ByteStorage`'s aligned_alloc is POSIX-conformant: `size` must be a multiple of
    // `alignof(std::max_align_t)` (typically 16). Hypothesis-drawn tensors
    // smaller than that — e.g. `tensor([0.])` (4 B) — would otherwise fall
    // through to a null return and a bad_alloc. We only use this buffer as a
    // source pointer for miniz to copy from; the actual data layout per
    // shard is driven by `handle`, so a rounded-up buffer is fine.
    constexpr size_t kAlign     = alignof(std::max_align_t);
    size_t const     wantedSize = totalBytes == 0 ? kAlign : totalBytes;
    size_t const     blobSize   = (wantedSize + kAlign - 1) & ~(kAlign - 1);
    ByteStorage      tensorStorage(blobSize);

    // Single-rank topology — multi-rank parity is exercised by sequential calls
    // to write_rank_shard from the test, each with its own (rank, world_size).
    RankTopology topo{
        .rank         = Rank{rank},
        .worldSize    = Rank{1},
        .ranksPerPeer = 1,
        .peersPerNode = 1,
        .peers        = {"localhost"},
    };
    StateTree tree(checkpoint::ipc::SessionId::Default, std::move(topo));
    tree.nativeState = native;
    AddScalars(tree, Rank{rank}, scalarsPy);

    // Build the tree in the post-merge global-registry shape — DTensor
    // leaves keyed by rank — since that's what WriteRankShard's assert
    // expects. For the parity shim every entry is one shard owned by the
    // calling rank covering the full shape.
    size_t offset = 0;
    for (auto const& e : entries)
    {
        std::memcpy(tensorStorage[offset], e.bytes.data(), e.bytes.size());

        Tensor::Dim fulldim;
        for (auto d : e.shape) { fulldim.push_back(d); }
        Tensor::DimRange sharddim;
        for (auto d : e.shape) { sharddim.push_back({.begin = 0, .end = d}); }

        auto [leafName, parent] = tree.FindOrAddNodeByFQN(e.fqn);
        auto&   leaf            = tree.AddChild(parent, leafName);
        DTensor st(e.dtype, fulldim);
        st.AddShard({
            .rank          = Rank{rank},
            .snapshotDtype = e.dtype,
            .handle        = offset,
            .storageSize   = e.bytes.size(),
            .sharddim      = sharddim,
            .device        = DeviceId{"cpu"},
        });
        leaf.data = std::move(st);

        offset += e.bytes.size();
    }
    for (auto const& [fqn, bytes] : blobBytes)
    {
        if (!bytes.empty()) { std::memcpy(tensorStorage[offset], bytes.data(), bytes.size()); }
        auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
        tree.AddChild(parent, leafName).data
            = StateNode::DynamicSizeBlob{.replicas = {Rank{rank}}, .handle = offset, .storageSize = bytes.size()};
        offset += bytes.size();
    }
    checkpoint::distcp::WriteRankShard(std::filesystem::path(outDir), Rank{rank}, tree, tensorStorage);

    std::vector<InputEntry const*> archiveEntries;
    archiveEntries.reserve(entries.size());
    for (auto const& entry : entries) { archiveEntries.push_back(&entry); }
    std::ranges::sort(archiveEntries, {}, [](InputEntry const* entry) { return entry->fqn; });
    std::map<std::string, std::pair<uint64_t, uint64_t>> locations;

    uint64_t cursor = 0;
    for (auto const* entry : archiveEntries)
    {
        Tensor::Dim shape;
        for (auto d : entry->shape) { shape.push_back(d); }
        uint64_t const archiveLen = checkpoint::distcp::ComputeShardArchiveSize(entry->dtype, shape, entry->bytes.size());
        locations.emplace(entry->fqn, std::pair{cursor, archiveLen});
        cursor += archiveLen;
    }
    nb::list out;
    for (auto const& e : entries)
    {
        nb::list sharddim;
        for (auto d : e.shape) { sharddim.append(nb::make_tuple(0ULL, static_cast<uint64_t>(d))); }
        auto const& [offset, archiveLen] = locations.at(e.fqn);
        out.append(nb::make_tuple(e.fqn, sharddim, offset, archiveLen));
    }
    return out;
}

static Tensor::DimRange ParseSharddim(nb::sequence const& sd)
{
    Tensor::DimRange out;
    auto             sdList = nb::cast<nb::list>(sd);
    if (sdList.empty() || sdList.size() > Tensor::MaxDims)
    {
        throw std::runtime_error(fmt::format("distcp_shim: sharddim rank {} out of range [1, {}]", sdList.size(), Tensor::MaxDims));
    }
    for (auto&& i : sdList)
    {
        auto pair = nb::cast<nb::tuple>(i);
        if (pair.size() != 2) { throw std::runtime_error("distcp_shim: sharddim entry must be a (begin, end) tuple"); }
        out.push_back({.begin = nb::cast<size_t>(pair[0]), .end = nb::cast<size_t>(pair[1])});
    }
    return out;
}

// `tensors`: list of (fqn, dtype_name, full_shape, shards) where `shards` is a
// list of (rank, sharddim, file_offset, length). The shim assembles a global
// StateTree with one DTensor per fqn, populated with the per-rank chunks,
// then builds per-rank writeLogs from the same input and calls WriteMetadata.
// The (file_offset, length) values are used as the verification side-channel
// inside WriteMetadata — they're asserted against the analytical prediction.
static void
WriteMetadataImpl(std::string const& outDir, nb::list const& tensors, nb::list const& objects, nb::dict const& scalars, bool native)
{
    RankTopology topo{
        .rank         = Rank{0},
        .worldSize    = Rank{1},
        .ranksPerPeer = 1,
        .peersPerNode = 1,
        .peers        = {"localhost"},
    };
    StateTree tree(checkpoint::ipc::SessionId::Default, std::move(topo));
    tree.nativeState = native;
    AddScalars(tree, Rank{0}, scalars);

    for (auto const& tensor : tensors)
    {
        auto t = nb::cast<nb::tuple>(tensor);
        if (t.size() != 4)
        {
            throw std::runtime_error("distcp_shim: each tensor entry must be (fqn, dtype_name, full_shape, shards_list)");
        }
        auto fqn       = nb::cast<std::string>(t[0]);
        auto dtype     = ParseDType(nb::cast<std::string>(t[1]));
        auto fullShape = nb::cast<std::vector<size_t>>(t[2]);
        auto shards    = nb::cast<nb::list>(t[3]);

        if (fullShape.empty() || fullShape.size() > Tensor::MaxDims)
        {
            throw std::runtime_error(
                fmt::format("distcp_shim: full_shape rank {} out of range [1, {}]", fullShape.size(), Tensor::MaxDims));
        }

        Tensor::Dim fulldim;
        for (auto d : fullShape) { fulldim.push_back(d); }

        auto [leafName, parent] = tree.FindOrAddNodeByFQN(fqn);
        auto&   leaf            = tree.AddChild(parent, leafName);
        DTensor st(dtype, fulldim);

        for (auto&& shard : shards)
        {
            auto entry = nb::cast<nb::tuple>(shard);
            if (entry.size() != 2) { throw std::runtime_error("distcp_shim: each shard entry must be (rank, sharddim)"); }
            auto const rank     = Rank{nb::cast<uint32_t>(entry[0])};
            auto const sharddim = ParseSharddim(nb::cast<nb::sequence>(entry[1]));

            uint64_t storageSize = Tensor::GetDTypeSize(dtype);
            for (auto const& range : sharddim) { storageSize *= range.end - range.begin; }
            st.AddShard({
                .rank          = rank,
                .snapshotDtype = dtype,
                .handle        = 0,
                .storageSize   = storageSize,
                .sharddim      = sharddim,
                .device        = DeviceId{"cpu"},
            });
        }

        leaf.data = std::move(st);
    }
    auto const blobBytes = ParseObjects(objects);
    for (auto const& [fqn, bytes] : blobBytes)
    {
        auto [leafName, parent]              = tree.FindOrAddNodeByFQN(fqn);
        tree.AddChild(parent, leafName).data = StateNode::DynamicSizeBlob{.replicas = {Rank{0}}, .handle = 0, .storageSize = bytes.size()};
    }
    checkpoint::distcp::WriteMetadata(std::filesystem::path(outDir), tree);
}

NB_MODULE(_distcp_shim, m)    // NOLINT
{
    m.doc() = "Single-process shim around distcp::WriteRankShard / WriteMetadata for parity tests.";

    m.def("write_rank_shard",
          &WriteRankShardImpl,
          nb::arg("out_dir"),
          nb::arg("rank"),
          nb::arg("entries"),
          nb::arg("objects") = nb::list(),
          nb::arg("scalars") = nb::dict(),
          nb::arg("native")  = false,
          "Write `<out_dir>/__<rank>_0.distcp` from a list of (fqn, dtype_name, "
          "shape, data_bytes) entries and optional (fqn, bytes) objects. Returns the populated writeLog as a list of "
          "(fqn, sharddim, file_offset, length) tuples.");

    m.def("write_metadata",
          &WriteMetadataImpl,
          nb::arg("out_dir"),
          nb::arg("tensors"),
          nb::arg("objects") = nb::list(),
          nb::arg("scalars") = nb::dict(),
          nb::arg("native")  = false,
          "Write `<out_dir>/.metadata` from a list of (fqn, dtype_name, full_shape, "
          "shards) tuples and optional (fqn, bytes) objects, where each shards entry is (rank, sharddim). The "
          "per-shard (file_offset, length) is inferred analytically from the "
          "shape/dtype — no need to pass it in.");
}
