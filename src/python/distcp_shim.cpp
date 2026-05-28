// Test-only shim: exposes WriteRankShard and WriteMetadata to Python so the
// hypothesis-driven DCP-load tests (tests/test_dcp_load.py, also
// test_distcp_roundtrip.py) can drive the writer without spinning up the
// server, IPC, or async dispatch.

#include "blob.h"
#include "distcp.h"
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
#include <stdexcept>
#include <string>
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

nb::list WriteRankShardImpl(std::string const& outDir, uint32_t rank, nb::list const& entriesPy)
{
    std::vector<InputEntry> entries;
    entries.reserve(entriesPy.size());
    size_t totalBytes = 0;
    for (auto const& i : entriesPy)
    {
        entries.push_back(ConvertEntry(nb::cast<nb::tuple>(i)));
        totalBytes += entries.back().bytes.size();
    }

    Blob blob(totalBytes == 0 ? 1 : totalBytes);    // aligned_alloc(0) is implementation-defined

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

    size_t offset = 0;
    for (auto const& e : entries)
    {
        std::memcpy(blob[offset], e.bytes.data(), e.bytes.size());

        DS4Tensor tnsr;
        tnsr.dtype  = e.dtype;
        tnsr.handle = offset;
        tnsr.device = DeviceId{"cpu"};
        for (auto d : e.shape)
        {
            tnsr.fulldim.push_back(d);
            tnsr.sharddim.push_back({.begin = 0, .end = d});
        }

        auto [leafName, parent] = tree.FindOrAddNodeByFQN(e.fqn);
        auto& leaf              = tree.AddChild(parent, leafName);
        leaf.data               = StateNode::TensorShard{tnsr};

        offset += e.bytes.size();
    }

    std::vector<checkpoint::distcp::WriteRecord> writeLog;
    checkpoint::distcp::WriteRankShard(std::filesystem::path(outDir), Rank{rank}, tree, blob, writeLog);

    nb::list out;
    for (auto const& rec : writeLog)
    {
        nb::list sharddim;
        for (auto const& r : rec.sharddim) { sharddim.append(nb::make_tuple(r.begin, r.end)); }
        out.append(nb::make_tuple(rec.fqn, sharddim, rec.fileOffset, rec.length));
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
void WriteMetadataImpl(std::string const& outDir, nb::list const& tensors)
{
    RankTopology topo{
        .rank         = Rank{0},
        .worldSize    = Rank{1},
        .ranksPerPeer = 1,
        .peersPerNode = 1,
        .peers        = {"localhost"},
    };
    StateTree tree(std::move(topo));

    // Per-rank writeLog storage. The manifest is assembled *after* the input
    // loop completes (see below) — by that point `writeLogs` is no longer
    // being grown, so the `&writeLogs[i]` pointers stored in the manifest are
    // stable across the WriteMetadata call.
    std::vector<std::vector<checkpoint::distcp::WriteRecord>> writeLogs;
    std::vector<Rank>                                         ranks;

    auto ensureRankSlot = [&](Rank r) -> std::vector<checkpoint::distcp::WriteRecord>& {
        for (size_t i = 0; i < ranks.size(); ++i)
        {
            if (ranks[i] == r) { return writeLogs[i]; }
        }
        ranks.push_back(r);
        writeLogs.emplace_back();
        return writeLogs.back();
    };

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
            if (entry.size() != 4)
            {
                throw std::runtime_error("distcp_shim: each shard entry must be (rank, sharddim, file_offset, length)");
            }
            auto const rank       = Rank{nb::cast<uint32_t>(entry[0])};
            auto const sharddim   = ParseSharddim(nb::cast<nb::sequence>(entry[1]));
            auto const fileOffset = nb::cast<uint64_t>(entry[2]);
            auto const length     = nb::cast<uint64_t>(entry[3]);

            st.AddShard({.sharddim = sharddim, .device = DeviceId{"cpu"}, .handle = 0, .rank = rank});

            auto& writeLog = ensureRankSlot(rank);
            writeLog.push_back(checkpoint::distcp::WriteRecord{
                .fqn        = fqn,
                .sharddim   = sharddim,
                .fileOffset = fileOffset,
                .length     = length,
            });
        }

        leaf.data = std::move(st);
    }

    // Build the manifest after writeLogs is fully populated — vector pointers
    // would otherwise dangle if the inner vectors were later relocated.
    std::vector<checkpoint::distcp::RankManifestRef> manifest;
    manifest.reserve(ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        manifest.push_back(checkpoint::distcp::RankManifestRef{.rank = ranks[i], .writeLog = &writeLogs[i]});
    }

    checkpoint::distcp::WriteMetadata(std::filesystem::path(outDir), tree, std::span{manifest});
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
          "Write `<out_dir>/__<rank>_0.distcp` from a list of (fqn, dtype_name, "
          "shape, data_bytes) entries. Returns the populated writeLog as a list of "
          "(fqn, sharddim, file_offset, length) tuples.");

    m.def("write_metadata",
          &WriteMetadataImpl,
          nb::arg("out_dir"),
          nb::arg("tensors"),
          "Write `<out_dir>/.metadata` from a list of (fqn, dtype_name, full_shape, "
          "shards) tuples, where each shards entry is "
          "(rank, sharddim, file_offset, length).");
}
