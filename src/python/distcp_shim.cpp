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

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace
{

DS4Tensor::DType ParseDType(std::string const& name)
{
    using DT = DS4Tensor::DType;
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

// Each entry: (fqn, dtype_name, shape, data_bytes).
struct InputEntry
{
    std::string                fqn;
    DS4Tensor::DType           dtype;
    std::vector<size_t>        shape;
    std::vector<unsigned char> bytes;
};

InputEntry ConvertEntry(nb::tuple const& t)
{
    if (t.size() != 4) { throw std::runtime_error("distcp_shim: each entry must be (fqn, dtype, shape, bytes)"); }
    InputEntry e;
    e.fqn   = nb::cast<std::string>(t[0]);
    e.dtype = ParseDType(nb::cast<std::string>(t[1]));
    e.shape = nb::cast<std::vector<size_t>>(t[2]);
    if (e.shape.empty() || e.shape.size() > DS4Tensor::MaxDims)
    {
        throw std::runtime_error(fmt::format("distcp_shim: shape rank {} out of range [1, {}]", e.shape.size(), DS4Tensor::MaxDims));
    }
    auto data = nb::cast<nb::bytes>(t[3]);
    e.bytes.resize(data.size());
    std::memcpy(e.bytes.data(), data.c_str(), data.size());
    return e;
}

// Parse Python's (fqn, bytes) object tuples into owned Blob storage. The
// parity shim is single-rank, so every Blob belongs to the calling rank.
std::vector<std::pair<std::string, std::vector<char>>> ParseObjects(nb::list const& objectsPy)
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

nb::list WriteRankShardImpl(std::string const& outDir, uint32_t rank, nb::list const& entriesPy, nb::list const& objectsPy)
{
    std::vector<InputEntry> entries;
    entries.reserve(entriesPy.size());
    size_t totalBytes = 0;
    for (auto const& i : entriesPy)
    {
        entries.push_back(ConvertEntry(nb::cast<nb::tuple>(i)));
        totalBytes += entries.back().bytes.size();
    }

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
    StateTree tree(std::move(topo));

    // Build the tree in the post-merge global-registry shape — ShardedTensor
    // leaves keyed by rank — since that's what WriteRankShard's assert
    // expects. For the parity shim every entry is one shard owned by the
    // calling rank covering the full shape.
    size_t offset = 0;
    for (auto const& e : entries)
    {
        std::memcpy(tensorStorage[offset], e.bytes.data(), e.bytes.size());

        DS4Tensor::Dim fulldim;
        for (auto d : e.shape) { fulldim.push_back(d); }
        DS4Tensor::DimRange sharddim;
        for (auto d : e.shape) { sharddim.push_back({.begin = 0, .end = d}); }

        auto [leafName, parent]       = tree.FindOrAddNodeByFQN(e.fqn);
        auto&                    leaf = tree.AddChild(parent, leafName);
        StateNode::ShardedTensor st(e.dtype, fulldim);
        st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = offset, .rank = Rank{rank}});
        leaf.data = std::move(st);

        offset += e.bytes.size();
    }
    auto const                     blobBytes = ParseObjects(objectsPy);
    checkpoint::distcp::BlobsByFqn blobView;
    for (auto const& [fqn, bytes] : blobBytes) { blobView[fqn] = std::string_view(bytes.data(), bytes.size()); }

    checkpoint::distcp::WriteRankShard(std::filesystem::path(outDir), Rank{rank}, tree, tensorStorage, blobView);

    nb::list out;
    uint64_t cursor = 0;
    for (auto const& e : entries)
    {
        DS4Tensor::Dim shape;
        for (auto d : e.shape) { shape.push_back(d); }
        uint64_t const archiveLen = checkpoint::distcp::ComputeShardArchiveSize(e.dtype, shape, e.bytes.size());

        nb::list sharddim;
        for (auto d : e.shape) { sharddim.append(nb::make_tuple(0ULL, static_cast<uint64_t>(d))); }
        out.append(nb::make_tuple(e.fqn, sharddim, cursor, archiveLen));
        cursor += archiveLen;
    }
    return out;
}

DS4Tensor::DimRange ParseSharddim(nb::sequence const& sd)
{
    DS4Tensor::DimRange out;
    auto                sdList = nb::cast<nb::list>(sd);
    if (sdList.empty() || sdList.size() > DS4Tensor::MaxDims)
    {
        throw std::runtime_error(fmt::format("distcp_shim: sharddim rank {} out of range [1, {}]", sdList.size(), DS4Tensor::MaxDims));
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
// StateTree with one ShardedTensor per fqn, populated with the per-rank chunks,
// then builds per-rank writeLogs from the same input and calls WriteMetadata.
// The (file_offset, length) values are used as the verification side-channel
// inside WriteMetadata — they're asserted against the analytical prediction.
void WriteMetadataImpl(std::string const& outDir, nb::list const& tensors, nb::list const& objects)
{
    RankTopology topo{
        .rank         = Rank{0},
        .worldSize    = Rank{1},
        .ranksPerPeer = 1,
        .peersPerNode = 1,
        .peers        = {"localhost"},
    };
    StateTree tree(std::move(topo));

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

        if (fullShape.empty() || fullShape.size() > DS4Tensor::MaxDims)
        {
            throw std::runtime_error(
                fmt::format("distcp_shim: full_shape rank {} out of range [1, {}]", fullShape.size(), DS4Tensor::MaxDims));
        }

        DS4Tensor::Dim fulldim;
        for (auto d : fullShape) { fulldim.push_back(d); }

        auto [leafName, parent]       = tree.FindOrAddNodeByFQN(fqn);
        auto&                    leaf = tree.AddChild(parent, leafName);
        StateNode::ShardedTensor st(dtype, fulldim);

        for (auto&& shard : shards)
        {
            auto entry = nb::cast<nb::tuple>(shard);
            if (entry.size() != 2) { throw std::runtime_error("distcp_shim: each shard entry must be (rank, sharddim)"); }
            auto const rank     = Rank{nb::cast<uint32_t>(entry[0])};
            auto const sharddim = ParseSharddim(nb::cast<nb::sequence>(entry[1]));

            st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = 0, .rank = rank});
        }

        leaf.data = std::move(st);
    }
    auto const                            blobBytes = ParseObjects(objects);
    checkpoint::distcp::BlobMetadataByFqn blobView;
    for (auto const& [fqn, bytes] : blobBytes) { blobView[fqn][Rank{0}] = {.length = bytes.size()}; }
    checkpoint::distcp::WriteMetadata(std::filesystem::path(outDir), tree, blobView);
}

}    // namespace

NB_MODULE(_distcp_shim, m)    // NOLINT
{
    m.doc() = "Single-process shim around distcp::WriteRankShard / WriteMetadata for parity tests.";

    m.def("write_rank_shard",
          &WriteRankShardImpl,
          nb::arg("out_dir"),
          nb::arg("rank"),
          nb::arg("entries"),
          nb::arg("objects") = nb::list(),
          "Write `<out_dir>/__<rank>_0.distcp` from a list of (fqn, dtype_name, "
          "shape, data_bytes) entries and optional (fqn, bytes) objects. Returns the populated writeLog as a list of "
          "(fqn, sharddim, file_offset, length) tuples.");

    m.def("write_metadata",
          &WriteMetadataImpl,
          nb::arg("out_dir"),
          nb::arg("tensors"),
          nb::arg("objects") = nb::list(),
          "Write `<out_dir>/.metadata` from a list of (fqn, dtype_name, full_shape, "
          "shards) tuples and optional (fqn, bytes) objects, where each shards entry is (rank, sharddim). The "
          "per-shard (file_offset, length) is inferred analytically from the "
          "shape/dtype — no need to pass it in.");
}
