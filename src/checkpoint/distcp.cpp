#include "distcp.h"
#include "distcp_internal.h"

#include <fmt/format.h>

#include <miniz/miniz.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "distcp writer assumes a little-endian host: archive/byteorder is hardcoded "
              "to \"little\" and the LE serializers are direct memcpys.");

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

// `__<rank>_0.distcp` is the concatenation of one self-contained `torch.save`
// ZIP per shard. We emit those ZIPs directly in C++:
//   - pickle stream: hand-built from primitives (proto-2 _rebuild_tensor_v2/v3)
//   - ZIP container: miniz `mz_zip_writer_add_mem_ex_v2`, exactly what
//     PyTorchStreamWriter calls into. This gets us CRC32 + standard headers
//     for free.
//   - 64-byte alignment of `archive/data/N` body: replicate
//     PyTorchStreamWriter::writeRecord's FB extra-field padding so byte-parity
//     with `dcp.save` output holds.
//   - `archive/.data/serialization_id`: replicate
//     PyTorchStreamWriter::writeSerializationId — `format("{:020d}{:020d}",
//     combined_record_name_hash, combined_uncomp_crc32)`, where the name hash
//     is `c10::hash_combine` over `std::hash<std::string>` of each record
//     name. std::hash<std::string> is libstdc++-specific; we link the same
//     stdlib torch was built against (gcc-14 libstdc++).
//
// The on-disk layout is byte-identical to the output of
// `torch.distributed.checkpoint.save`; `tests/test_distcp_parity.py` is the
// gating check.

namespace checkpoint::distcp
{

std::string_view DTypeName(DS4Tensor::DType dtype)
{
    using DT = DS4Tensor::DType;
    switch (dtype)
    {
    case DT::Float16: return "float16";
    case DT::BFloat16: return "bfloat16";
    case DT::Float32: return "float32";
    case DT::Float64: return "float64";
    case DT::UInt8: return "uint8";
    case DT::UInt16: return "uint16";
    case DT::UInt32: return "uint32";
    case DT::UInt64: return "uint64";
    case DT::Int8: return "int8";
    case DT::Int16: return "int16";
    case DT::Int32: return "int32";
    case DT::Int64: return "int64";
    case DT::Bool: return "bool";
    case DT::Invalid: [[fallthrough]];
    default: throw std::runtime_error(fmt::format("distcp: unsupported dtype {}", static_cast<int>(dtype)));
    }
}

void CollectShards(StateTree const& tree, Blob const& blob, std::vector<ShardEntry>& out)
{
    auto const* base = static_cast<char const*>(blob.data);
    tree.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::TensorShard)
        {
            auto const& tnsr = std::get<StateNode::TensorShard>(node.data);
            auto const  size = tnsr.ShardSizeBytes();
            if (size == 0) { return; }
            DS4Tensor::Dim shape{};
            for (auto const& r : tnsr.sharddim) { shape.push_back(r.end - r.begin); }
            out.push_back(ShardEntry{
                .srcPtr   = base + tnsr.handle,
                .length   = size,
                .fqn      = tree.GetNodeId(node.handle),
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
                auto const size = shard.ShardSizeBytes(tnsr.dtype);
                if (size == 0) { continue; }
                DS4Tensor::Dim shape{};
                for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
                out.push_back(ShardEntry{
                    .srcPtr   = base + shard.handle,
                    .length   = size,
                    .fqn      = tree.GetNodeId(node.handle),
                    .dtype    = tnsr.dtype,
                    .sharddim = shard.sharddim,
                    .shape    = shape,
                });
            }
        }
    });
}

namespace
{

// ---- Pickle stream emission ----------------------------------------------

struct DtypePickleInfo
{
    bool             useV3;
    std::string_view storageMod;
    std::string_view storageName;
    std::string_view dtypeMod;
    std::string_view dtypeName;
};

DtypePickleInfo PickleInfoForDtype(DS4Tensor::DType dtype)
{
    using DT = DS4Tensor::DType;
    switch (dtype)
    {
    case DT::Float16: return {.useV3 = false, .storageMod = "torch", .storageName = "HalfStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::BFloat16: return {.useV3 = false, .storageMod = "torch", .storageName = "BFloat16Storage", .dtypeMod = "", .dtypeName = ""};
    case DT::Float32: return {.useV3 = false, .storageMod = "torch", .storageName = "FloatStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::Float64: return {.useV3 = false, .storageMod = "torch", .storageName = "DoubleStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::UInt8: return {.useV3 = false, .storageMod = "torch", .storageName = "ByteStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::Int8: return {.useV3 = false, .storageMod = "torch", .storageName = "CharStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::Int16: return {.useV3 = false, .storageMod = "torch", .storageName = "ShortStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::Int32: return {.useV3 = false, .storageMod = "torch", .storageName = "IntStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::Int64: return {.useV3 = false, .storageMod = "torch", .storageName = "LongStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::Bool: return {.useV3 = false, .storageMod = "torch", .storageName = "BoolStorage", .dtypeMod = "", .dtypeName = ""};
    case DT::UInt16:
        return {.useV3 = true, .storageMod = "torch.storage", .storageName = "UntypedStorage", .dtypeMod = "torch", .dtypeName = "uint16"};
    case DT::UInt32:
        return {.useV3 = true, .storageMod = "torch.storage", .storageName = "UntypedStorage", .dtypeMod = "torch", .dtypeName = "uint32"};
    case DT::UInt64:
        return {.useV3 = true, .storageMod = "torch.storage", .storageName = "UntypedStorage", .dtypeMod = "torch", .dtypeName = "uint64"};
    case DT::Invalid: [[fallthrough]];
    default: throw std::runtime_error(fmt::format("distcp: unsupported dtype {}", static_cast<int>(dtype)));
    }
}

void PutU8(std::vector<unsigned char>& b, unsigned char v)
{ b.push_back(v); }

void PutU16LE(std::vector<unsigned char>& b, uint16_t v)
{
    b.push_back(static_cast<unsigned char>(v & 0xff));
    b.push_back(static_cast<unsigned char>(v >> 8));
}

void PutU32LE(std::vector<unsigned char>& b, uint32_t v)
{
    for (int i = 0; i < 4; ++i) { b.push_back(static_cast<unsigned char>((v >> (i * 8)) & 0xff)); }
}

void PutBytes(std::vector<unsigned char>& b, std::string_view sv)
{ b.insert(b.end(), sv.begin(), sv.end()); }

void PickleInt(std::vector<unsigned char>& b, uint64_t v)
{
    if (v < 256)
    {
        PutU8(b, 'K');
        PutU8(b, static_cast<unsigned char>(v));
    }
    else if (v < 65536)
    {
        PutU8(b, 'M');
        PutU16LE(b, static_cast<uint16_t>(v));
    }
    else if (v <= 0x7fffffffull)
    {
        PutU8(b, 'J');
        PutU32LE(b, static_cast<uint32_t>(v));
    }
    else
    {
        std::array<unsigned char, 9> tmp{};
        size_t                       n = 0;
        uint64_t                     x = v;
        while (x != 0)
        {
            tmp[n++] = static_cast<unsigned char>(x & 0xff);
            x >>= 8;
        }
        if (n > 0 && (tmp[n - 1] & 0x80U) != 0U) { tmp[n++] = 0U; }
        PutU8(b, 0x8aU);
        PutU8(b, static_cast<unsigned char>(n));
        for (size_t i = 0; i < n; ++i) { PutU8(b, tmp[i]); }
    }
}

void PickleBinunicode(std::vector<unsigned char>& b, std::string_view s)
{
    PutU8(b, 'X');
    PutU32LE(b, static_cast<uint32_t>(s.size()));
    PutBytes(b, s);
}

void PickleGlobal(std::vector<unsigned char>& b, std::string_view mod, std::string_view name)
{
    PutU8(b, 'c');
    PutBytes(b, mod);
    PutU8(b, '\n');
    PutBytes(b, name);
    PutU8(b, '\n');
}

void PickleBinput(std::vector<unsigned char>& b, uint8_t idx)
{
    PutU8(b, 'q');
    PutU8(b, idx);
}

void PickleTupleClose(std::vector<unsigned char>& b, size_t n)
{
    switch (n)
    {
    case 0: PutU8(b, ')'); return;
    case 1: PutU8(b, 0x85); return;
    case 2: PutU8(b, 0x86); return;
    case 3: PutU8(b, 0x87); return;
    default: PutU8(b, 't'); return;
    }
}

// Precondition: shape is non-empty. CollectShards builds shapes from
// `sharddim`, which is non-empty for every tensor we write.
std::vector<unsigned char> BuildDataPkl(DS4Tensor::DType dtype, DS4Tensor::Dim const& shape)
{
    auto const info = PickleInfoForDtype(dtype);

    uint64_t numel = 1;
    for (auto d : shape) { numel *= d; }

    // Row-major contiguous stride. inplace_vector lacks `resize`, so seed with
    // a placeholder then overwrite via operator[].
    DS4Tensor::Dim stride;
    for (size_t i = 0; i < shape.size(); ++i) { stride.push_back(0); }
    stride[shape.size() - 1] = 1;
    for (size_t i = shape.size() - 1; i-- > 0;) { stride[i] = stride[i + 1] * shape[i + 1]; }

    uint64_t const storageSize = info.useV3 ? numel * DynamicSharded4TensorTraits::GetDTypeSize(dtype) : numel;

    std::vector<unsigned char> out;
    out.reserve(192);

    uint8_t memo    = 0;
    auto    memoize = [&]() { PickleBinput(out, memo++); };

    PutU8(out, 0x80);
    PutU8(out, 0x02);

    PickleGlobal(out, "torch._utils", info.useV3 ? "_rebuild_tensor_v3" : "_rebuild_tensor_v2");
    memoize();

    PutU8(out, '(');
    {
        PutU8(out, '(');
        PickleBinunicode(out, "storage");
        memoize();
        PickleGlobal(out, info.storageMod, info.storageName);
        memoize();
        PickleBinunicode(out, "0");
        memoize();
        PickleBinunicode(out, "cpu");
        memoize();
        PickleInt(out, storageSize);
        PutU8(out, 't');
        memoize();
        PutU8(out, 'Q');
    }

    PickleInt(out, 0);

    if (shape.size() >= 4) { PutU8(out, '('); }
    for (auto d : shape) { PickleInt(out, d); }
    PickleTupleClose(out, shape.size());
    memoize();

    if (stride.size() >= 4) { PutU8(out, '('); }
    for (auto s : stride) { PickleInt(out, s); }
    PickleTupleClose(out, stride.size());
    memoize();

    PutU8(out, 0x89);

    PickleGlobal(out, "collections", "OrderedDict");
    memoize();
    PutU8(out, ')');
    PutU8(out, 'R');
    memoize();

    if (info.useV3)
    {
        PickleGlobal(out, info.dtypeMod, info.dtypeName);
        memoize();
    }

    PutU8(out, 't');
    memoize();
    PutU8(out, 'R');
    memoize();
    PutU8(out, '.');

    return out;
}

// ---- miniz ZIP archive helper --------------------------------------------

// 64-byte alignment of `archive/data/N` mirrors what PyTorchStreamWriter does
// for zero-copy mmap reads downstream. Without it, torch.load still works but
// byte-parity with `dcp.save` breaks.
constexpr uint64_t KFieldAlignment = 64;

// Replicates `caffe2::serialize::detail::getOffset` / `getPadding` from
// pytorch/v2.11.0. Returns the FB-prefixed extra-field bytes to attach to a
// local file header so the entry's data starts on a 64-byte boundary.
//
// `cursor` is the archive offset the next LFH will land at (miniz's
// `m_archive_size`). The extra field is laid out as:
//   'F','B', <padding_size LE 2 bytes>, 'Z' * padding_size
// Total length = padding_size + 4 (the 4-byte FB header itself).
std::vector<unsigned char> ComputePadding(size_t cursor, size_t filenameSize, size_t dataSize)
{
    // Mirror the upstream offset computation for ZIP32 entries; ZIP64 entries
    // need extra header bytes (we throw on those upstream — see WriteRankShard).
    // 30 = sizeof(LFH); +4 = FB extra-field header bytes (2 id + 2 length).
    constexpr size_t kLocalFileHeaderSize = 30;
    constexpr size_t kFbExtraHeaderBytes  = sizeof(uint16_t) * 2;
    size_t           start                = cursor + kLocalFileHeaderSize + filenameSize + kFbExtraHeaderBytes;
    if (dataSize >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX)
    {
        // The upstream code adds extra bytes here for ZIP64 LFH fields. We
        // gate ZIP64 out at the caller, so this branch shouldn't fire.
        throw std::runtime_error(fmt::format("distcp: ZIP entry too large for ZIP32 (cursor={}, dataSize={})", cursor, dataSize));
    }
    size_t const mod         = start % KFieldAlignment;
    size_t const nextOffset  = (mod == 0) ? start : (start + KFieldAlignment - mod);
    size_t const paddingSize = nextOffset - start;

    std::vector<unsigned char> extra;
    extra.resize(paddingSize + 4, 'Z');
    extra[0] = 'F';
    extra[1] = 'B';
    extra[2] = static_cast<unsigned char>(paddingSize & 0xff);
    extra[3] = static_cast<unsigned char>((paddingSize >> 8) & 0xff);
    return extra;
}

// `c10::hash_combine` from c10/util/hash.h: deterministic mixer used in the
// serialization_id record-name hash.
uint64_t HashCombine(uint64_t seed, uint64_t value)
{ return seed ^ (value + 0x9e3779b9ULL + (seed << 6U) + (seed >> 2U)); }

// libstdc++'s `std::hash<std::string>` (used by torch's writer via
// `c10::hash<std::string>{}(...)` -> std::hash) — we just call it directly.
uint64_t StdHashString(std::string const& s)
{ return std::hash<std::string>{}(s); }

// Writes one self-contained torch.save-compatible ZIP archive containing the
// 7 entries for a single shard. Returns the bytes appended.
//
// Layout: `archive/data.pkl`, `archive/byteorder`, `archive/.format_version`,
// `archive/.storage_alignment`, `archive/data/0`, `archive/version`,
// `archive/.data/serialization_id`. The data/0 entry is FB-padded to 64-byte
// alignment so zero-copy mmap downstream lines up.
//
// The archive is built in-memory via miniz's heap writer, then dumped to `fd`
// in one shot. Memory cost = ZIP container overhead + tensor body (the
// tensor body is the only large field). For multi-gigabyte tensors this is
// the same RSS hit `dcp.save` takes; revisit if it shows up in a profile.
uint64_t WriteShardArchive(int fd, ShardEntry const& s, std::string_view contextName)
{
    std::vector<unsigned char> dataPkl = BuildDataPkl(s.dtype, s.shape);

    // Per-entry plan: filename + content pointer + content size.
    struct EntryPlan
    {
        std::string                fullName;
        void const*                data;
        size_t                     size;
        std::vector<unsigned char> padding;    // 0 for non-data entries
    };

    constexpr char const* kArchivePrefix = "archive/";
    auto                  joinName       = [](std::string_view sub) {
        std::string name;
        name.reserve(8 + sub.size());
        name.append(kArchivePrefix);
        name.append(sub);
        return name;
    };

    static constexpr std::string_view KByteorderBytes        = "little";
    static constexpr std::string_view KFormatVersionBytes    = "1";
    static constexpr std::string_view KStorageAlignmentBytes = "64";
    static constexpr std::string_view KVersionBytes          = "3\n";

    // Record order matches torch.serialization._save / PyTorchStreamWriter::
    // writeEndOfFile exactly:
    //   data.pkl, .format_version, .storage_alignment, byteorder,
    //   data/N..., version, .data/serialization_id
    // The serialization_id entry is appended after combined-CRC is known.
    std::array<EntryPlan, 6> entries{
        EntryPlan{.fullName = joinName("data.pkl"), .data = dataPkl.data(), .size = dataPkl.size(), .padding = {}},
        EntryPlan{
            .fullName = joinName(".format_version"), .data = KFormatVersionBytes.data(), .size = KFormatVersionBytes.size(), .padding = {}},
        EntryPlan{.fullName = joinName(".storage_alignment"),
                  .data     = KStorageAlignmentBytes.data(),
                  .size     = KStorageAlignmentBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = joinName("byteorder"), .data = KByteorderBytes.data(), .size = KByteorderBytes.size(), .padding = {}},
        EntryPlan{.fullName = joinName("data/0"), .data = s.srcPtr, .size = s.length, .padding = {}},
        EntryPlan{.fullName = joinName("version"), .data = KVersionBytes.data(), .size = KVersionBytes.size(), .padding = {}},
    };

    // Set up miniz heap writer. MZ_ZIP_FLAG_WRITE_ZIP64 forces the ZIP64 EOCD
    // + locator at finalize time — matches PyTorchStreamWriter's "universally
    // ZIP64" guarantee. Per-entry LFH/CD records stay ZIP32 unless their size
    // or offset overflow 32 bits.
    mz_zip_archive zip{};
    if (mz_zip_writer_init_heap_v2(&zip,
                                   /*size_to_reserve_at_beginning=*/0,
                                   /*initial_allocation_size=*/std::max<size_t>(1024, s.length),
                                   MZ_ZIP_FLAG_WRITE_ZIP64)
        != MZ_TRUE)
    {
        throw std::runtime_error(fmt::format("distcp: mz_zip_writer_init_heap_v2 failed for '{}'", contextName));
    }

    // `combined_uncomp_crc32_` from PyTorchStreamWriter is hash_combine'd across
    // CRCs in write order; the record-name hash is hash_combine'd across the
    // unordered_set iteration order — and the set holds BARE names (without
    // the `archive/` prefix) because writeRecord does `files_written_.insert(name)`
    // with the un-prefixed argument.
    uint64_t                        combinedCrc = 0;
    std::unordered_set<std::string> bareNamesWritten;
    std::vector<size_t>             lfhOffsets;
    lfhOffsets.reserve(entries.size() + 1);

    auto addEntry = [&](EntryPlan& plan, std::string_view bareName) {
        plan.padding = ComputePadding(zip.m_archive_size, plan.fullName.size(), plan.size);
        lfhOffsets.push_back(static_cast<size_t>(zip.m_archive_size));
        if (mz_zip_writer_add_mem_ex_v2(&zip,
                                        plan.fullName.c_str(),
                                        plan.data,
                                        plan.size,
                                        /*pComment=*/nullptr,
                                        /*comment_size=*/0,
                                        /*level_and_flags=*/0,    // STORED, no compression
                                        /*uncomp_size=*/0,
                                        /*uncomp_crc32=*/0,    // miniz computes CRC32 itself
                                        /*last_modified=*/nullptr,
                                        /*user_extra_data=*/reinterpret_cast<char const*>(plan.padding.data()),
                                        /*user_extra_data_len=*/static_cast<mz_uint16>(plan.padding.size()),
                                        /*user_extra_data_central=*/nullptr,
                                        /*user_extra_data_central_len=*/0)
            != MZ_TRUE)
        {
            // Don't `mz_zip_writer_end` here — the outer catch does it once.
            throw std::runtime_error(fmt::format("distcp: mz_zip_writer_add_mem_ex_v2 failed for '{}'", plan.fullName));
        }

        // Track contributions to serialization_id: see comment block at file top.
        // Upstream reads the CRC out of miniz's data-descriptor block; we
        // recompute it directly (same CRC32 of the same bytes) and feed
        // hash_combine in write order — matches upstream's accumulator path
        // (caffe2/serialize/inline_container.cc::ostream_write_func).
        // mz_crc32 returns mz_ulong (64-bit on x86_64). CRC32 fits in 32 bits
        // by definition; the static_cast is just to silence -Wconversion.
        auto const entryCrc = static_cast<uint32_t>(mz_crc32(MZ_CRC32_INIT, static_cast<unsigned char const*>(plan.data), plan.size));
        combinedCrc         = HashCombine(combinedCrc, entryCrc);
        bareNamesWritten.insert(std::string{bareName});
    };

    // Bare record names that PyTorchStreamWriter::files_written_ contains, in
    // write order — upstream stores names without the `archive/` prefix. Used
    // to reproduce the serialization_id name-hash leg.
    static constexpr std::array<std::string_view, 6> KBareNames{
        "data.pkl",
        ".format_version",
        ".storage_alignment",
        "byteorder",
        "data/0",
        "version",
    };

    try
    {
        for (size_t i = 0; i < entries.size(); ++i) { addEntry(entries[i], KBareNames[i]); }

        // Record-name hash: iterate `bareNamesWritten` (the unordered_set) in
        // its hash-bucket order, like PyTorchStreamWriter::writeSerializationId.
        // Identical std::hash<string> + bucket layout (same libstdc++ ABI as
        // torch's build) reproduces the upstream iteration order. Verified
        // empirically against torch 2.11 with gcc-14.
        uint64_t combinedRecordNameHash = 0;
        for (auto const& name : bareNamesWritten) { combinedRecordNameHash = HashCombine(combinedRecordNameHash, StdHashString(name)); }

        // serialization_id: "{:020d}{:020d}", record-name hash, combined CRC.
        std::string const serializationId = fmt::format("{:020d}{:020d}", combinedRecordNameHash, combinedCrc);

        EntryPlan idPlan{
            .fullName = joinName(".data/serialization_id"),
            .data     = serializationId.data(),
            .size     = serializationId.size(),
            .padding  = {},
        };
        // Important: do NOT mix the id record itself into combinedCrc / name hash —
        // upstream excludes it (see PyTorchStreamWriter::writeSerializationId, which
        // checks `files_written_.find(kSerializationIdRecordName) == end()` before
        // computing).
        idPlan.padding = ComputePadding(zip.m_archive_size, idPlan.fullName.size(), idPlan.size);
        lfhOffsets.push_back(static_cast<size_t>(zip.m_archive_size));
        if (mz_zip_writer_add_mem_ex_v2(&zip,
                                        idPlan.fullName.c_str(),
                                        idPlan.data,
                                        idPlan.size,
                                        nullptr,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr,
                                        reinterpret_cast<char const*>(idPlan.padding.data()),
                                        static_cast<mz_uint16>(idPlan.padding.size()),
                                        nullptr,
                                        0)
            != MZ_TRUE)
        {
            throw std::runtime_error(fmt::format("distcp: mz_zip_writer_add_mem_ex_v2 failed for '{}'", idPlan.fullName));
        }

    } catch (...)
    {
        mz_zip_writer_end(&zip);
        throw;
    }

    // Flush the in-memory ZIP to fd. `mz_zip_writer_finalize_heap_archive`
    // internally finalizes and detaches the heap buffer in one shot — calling
    // `mz_zip_writer_finalize_archive` first would put the state machine in
    // "already finalized" and the detach call would fail.
    void*  archiveBufRaw = nullptr;
    size_t archiveSize   = 0;
    if (mz_zip_writer_finalize_heap_archive(&zip, &archiveBufRaw, &archiveSize) != MZ_TRUE)
    {
        mz_zip_writer_end(&zip);
        throw std::runtime_error(fmt::format("distcp: mz_zip_writer_finalize_heap_archive failed for '{}'", contextName));
    }
    mz_zip_writer_end(&zip);

    // Own the heap buffer for the rest of this scope. mz_free is the documented
    // counterpart to the malloc performed by `finalize_heap_archive`.
    std::unique_ptr<unsigned char, decltype(&::mz_free)> archiveBuf{static_cast<unsigned char*>(archiveBufRaw), &::mz_free};

    // Zero last-mod time/date in every LFH and CD record. PyTorchStreamWriter is
    // built against a vendored miniz with MINIZ_NO_TIME, which makes those
    // fields stable zeros. Our vcpkg miniz lacks that knob (it would require
    // a source rebuild), so we patch the finalized buffer to match.
    unsigned char* const bytes = archiveBuf.get();
    for (size_t off : lfhOffsets)
    {
        // LFH: time at +10 (2B), date at +12 (2B). Zero 4 contiguous bytes.
        std::memset(bytes + off + 10, 0, 4);
    }
    // Walk the central directory: EOCD lives in the last 22 bytes (we emit no
    // comment), CD offset/size in EOCD bytes [12..20). With MZ_ZIP_FLAG_WRITE_ZIP64
    // the EOCD CD fields hold 0xFFFFFFFF sentinels iff the real values overflow
    // 32 bits; in that case the truthy values live in the ZIP64 EOCD record. The
    // upstream `s.length < MZ_UINT32_MAX` guard at the call-site means we don't
    // hit that branch today — but the parser asserts it anyway, since walking
    // garbage would otherwise be a buffer-overrun surface.
    if (archiveSize >= 22)
    {
        unsigned char const* eocd = bytes + archiveSize - 22;
        if (eocd[0] == 'P' && eocd[1] == 'K' && eocd[2] == 0x05 && eocd[3] == 0x06)
        {
            uint32_t cdSize{};
            uint32_t cdOffset{};
            std::memcpy(&cdSize, eocd + 12, 4);
            std::memcpy(&cdOffset, eocd + 16, 4);

            if (cdSize == 0xffffffffU || cdOffset == 0xffffffffU)
            {
                throw std::runtime_error(fmt::format("distcp: ZIP64 sentinel in EOCD cd_offset/cd_size for '{}'; this means a "
                                                     "shard crossed the 4 GiB boundary that ZIP64 support hasn't been wired in for yet",
                                                     contextName));
            }

            size_t       cursor = cdOffset;
            size_t const cdEnd  = static_cast<size_t>(cdOffset) + static_cast<size_t>(cdSize);
            while (cursor + 46 <= cdEnd)
            {
                if (bytes[cursor] != 'P' || bytes[cursor + 1] != 'K' || bytes[cursor + 2] != 0x01 || bytes[cursor + 3] != 0x02)
                {
                    break;    // not a CD entry; stop defensively
                }
                // CD record: time at +12 (2B), date at +14 (2B). Zero 4 bytes.
                std::memset(bytes + cursor + 12, 0, 4);
                uint16_t fnlen{};
                uint16_t extlen{};
                uint16_t cmtlen{};
                std::memcpy(&fnlen, bytes + cursor + 28, 2);
                std::memcpy(&extlen, bytes + cursor + 30, 2);
                std::memcpy(&cmtlen, bytes + cursor + 32, 2);
                cursor += 46U + fnlen + extlen + cmtlen;
            }
        }
    }

    // Stream the buffer to fd. Loop handles short writes / EINTR; Linux's
    // single-write limit is ~2 GiB.
    auto const* p   = bytes;
    size_t      rem = archiveSize;
    while (rem > 0)
    {
        ssize_t n = ::write(fd, p, rem);
        if (n < 0)
        {
            if (errno == EINTR) { continue; }
            throw std::system_error(
                errno, std::generic_category(), fmt::format("distcp: write({}) failed with {} bytes remaining", contextName, rem));
        }
        p += static_cast<size_t>(n);
        rem -= static_cast<size_t>(n);
    }

    return archiveSize;
}

// ---- `.metadata` pickle emission -----------------------------------------
//
// Hand-emits a proto-2 pickle stream describing
// `torch.distributed.checkpoint.metadata.Metadata`. The loader accepts the
// equivalent object graph just as well as torch's proto-4 output (verified
// by the step-9 loader-tolerance probes — see plan.md "step 9 verification"
// entry). Sharing proto-2 with `BuildDataPkl` keeps the opcode surface
// small. No memoization — strict byte parity isn't required.
//
// Every dataclass instance uses the default `__reduce_ex__(2)` shape:
//   GLOBAL <mod> <name>  EMPTY_TUPLE  NEWOBJ  EMPTY_DICT  MARK
//     <field-name>  <field-value>  ...
//   SETITEMS  BUILD
// Default-None fields are omitted (matches torch's writer).

constexpr std::string_view KMetadataModule   = "torch.distributed.checkpoint.metadata";
constexpr std::string_view KFilesystemModule = "torch.distributed.checkpoint.filesystem";

void EmitNewObjBuildBegin(std::vector<unsigned char>& b, std::string_view mod, std::string_view name)
{
    PickleGlobal(b, mod, name);
    PutU8(b, ')');     // EMPTY_TUPLE
    PutU8(b, 0x81);    // NEWOBJ — push cls.__new__(cls, *())
    PutU8(b, '}');     // EMPTY_DICT — start of state dict
    PutU8(b, '(');     // MARK
}

void EmitNewObjBuildEnd(std::vector<unsigned char>& b)
{
    PutU8(b, 'u');    // SETITEMS — fold marked (key, value) pairs into the dict
    PutU8(b, 'b');    // BUILD — apply state dict to the instance
}

// torch.Size(<dims-tuple>) via REDUCE.
template <typename TInts> void EmitTorchSize(std::vector<unsigned char>& b, TInts const& dims)
{
    PickleGlobal(b, "torch", "Size");
    if (dims.size() >= 4) { PutU8(b, '('); }
    for (auto d : dims) { PickleInt(b, static_cast<uint64_t>(d)); }
    PickleTupleClose(b, dims.size());
    PutU8(b, 0x85);    // TUPLE1 — wrap the dims tuple as the single ctor arg
    PutU8(b, 'R');     // REDUCE — call torch.Size(dims_tuple)
}

void EmitTorchDtype(std::vector<unsigned char>& b, DS4Tensor::DType dtype)
{
    // torch.<dtype> is a module-level attribute (singleton); GLOBAL pushes it directly.
    PickleGlobal(b, "torch", DTypeName(dtype));
}

void EmitLayoutStrided(std::vector<unsigned char>& b)
{
    // `torch.strided` is a module-level singleton — GLOBAL resolves it directly.
    // dcp.save uses a longer `_get_layout(...)` REDUCE path; this shorter form
    // is semantically equivalent (verified by-hand: `pickle.loads(b'\\x80\\x02ctorch\\nstrided\\n.')`
    // returns the same singleton). `TensorProperties.__setstate__` just assigns
    // it straight to `self.layout` — no enum decode.
    PickleGlobal(b, "torch", "strided");
}

void EmitMemoryFormatContiguousEnum(std::vector<unsigned char>& b)
{
    // TensorProperties.__getstate__ encodes memory_format via the
    // `_MEM_FORMAT_ENCODING` Enum that lives next to TensorProperties (in
    // `torch.distributed.checkpoint.metadata`, *not* `torch.memory_format`).
    // `__setstate__` compares against `TORCH_CONTIGUOUS_FORMAT` (== 0) and
    // resolves back to the singleton. Emitting the actual `torch.contiguous_format`
    // object here would break the equality check and raise at load time.
    PickleGlobal(b, KMetadataModule, "_MEM_FORMAT_ENCODING");
    PickleInt(b, 0);    // TORCH_CONTIGUOUS_FORMAT
    PutU8(b, 0x85);     // TUPLE1
    PutU8(b, 'R');      // REDUCE — _MEM_FORMAT_ENCODING(0)
}

void EmitTensorProperties(std::vector<unsigned char>& b, DS4Tensor::DType dtype)
{
    // TensorProperties has a custom __getstate__/__setstate__ that uses a
    // 5-tuple (not a dict): (dtype, layout, requires_grad, mem_format_encoding,
    // pin_memory). The MARK..TUPLE..BUILD shape feeds that tuple directly into
    // __setstate__.
    PickleGlobal(b, KMetadataModule, "TensorProperties");
    PutU8(b, ')');     // EMPTY_TUPLE
    PutU8(b, 0x81);    // NEWOBJ
    PutU8(b, '(');     // MARK
    EmitTorchDtype(b, dtype);
    EmitLayoutStrided(b);
    PutU8(b, 0x89);    // NEWFALSE (requires_grad)
    EmitMemoryFormatContiguousEnum(b);
    PutU8(b, 0x89);    // NEWFALSE (pin_memory)
    PutU8(b, 't');     // TUPLE — fold the marked values into a 5-tuple
    PutU8(b, 'b');     // BUILD — apply via __setstate__
}

void EmitChunkStorageMetadata(std::vector<unsigned char>& b, DS4Tensor::DimRange const& sharddim)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "ChunkStorageMetadata");

    PickleBinunicode(b, "offsets");
    DS4Tensor::Dim offsets;
    for (auto const& r : sharddim) { offsets.push_back(r.begin); }
    EmitTorchSize(b, offsets);

    PickleBinunicode(b, "sizes");
    DS4Tensor::Dim sizes;
    for (auto const& r : sharddim) { sizes.push_back(r.end - r.begin); }
    EmitTorchSize(b, sizes);

    EmitNewObjBuildEnd(b);
}

void EmitTensorStorageMetadata(std::vector<unsigned char>&             b,
                               DS4Tensor::DType                        dtype,
                               DS4Tensor::Dim const&                   fulldim,
                               std::vector<DS4Tensor::DimRange> const& chunks)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "TensorStorageMetadata");

    PickleBinunicode(b, "properties");
    EmitTensorProperties(b, dtype);

    PickleBinunicode(b, "size");
    EmitTorchSize(b, fulldim);

    PickleBinunicode(b, "chunks");
    PutU8(b, ']');    // EMPTY_LIST
    PutU8(b, '(');    // MARK
    for (auto const& chunk : chunks) { EmitChunkStorageMetadata(b, chunk); }
    PutU8(b, 'e');    // APPENDS

    EmitNewObjBuildEnd(b);
}

void EmitMetadataIndex(std::vector<unsigned char>& b, std::string_view fqn, DS4Tensor::DimRange const& sharddim)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "MetadataIndex");

    PickleBinunicode(b, "fqn");
    PickleBinunicode(b, fqn);

    PickleBinunicode(b, "offset");
    DS4Tensor::Dim offsets;
    for (auto const& r : sharddim) { offsets.push_back(r.begin); }
    EmitTorchSize(b, offsets);

    PickleBinunicode(b, "index");
    PickleInt(b, 0);    // always 0 in single-server v1 (no multi-storage-id splits)

    EmitNewObjBuildEnd(b);
}

void EmitStorageInfo(std::vector<unsigned char>& b, std::string_view relativePath, uint64_t offset, uint64_t length)
{
    EmitNewObjBuildBegin(b, KFilesystemModule, "_StorageInfo");

    PickleBinunicode(b, "relative_path");
    PickleBinunicode(b, relativePath);

    PickleBinunicode(b, "offset");
    PickleInt(b, offset);

    PickleBinunicode(b, "length");
    PickleInt(b, length);

    // `transform_descriptors` defaults to None and is omitted (torch's writer
    // does the same — verified in the disassembly probe).

    EmitNewObjBuildEnd(b);
}

// Find the WriteRecord that this rank wrote for (fqn, sharddim). Linear scan
// per the plan; upgrade to a map if it ever shows up in a profile.
WriteRecord const*
LookupWriteRecord(std::span<RankManifestRef const> manifest, Rank rank, std::string const& fqn, DS4Tensor::DimRange const& sharddim)
{
    for (auto const& ref : manifest)
    {
        if (ref.rank != rank) { continue; }
        if (ref.writeLog == nullptr) { return nullptr; }
        for (auto const& wr : *ref.writeLog)
        {
            if (wr.fqn == fqn && wr.sharddim == sharddim) { return &wr; }
        }
        return nullptr;
    }
    return nullptr;
}

std::vector<unsigned char> BuildMetadataPkl(StateTree const& globalRegistry, std::span<RankManifestRef const> manifest)
{
    std::vector<unsigned char> b;
    b.reserve(2048);

    PutU8(b, 0x80);    // PROTO
    PutU8(b, 0x02);    // protocol 2

    EmitNewObjBuildBegin(b, KMetadataModule, "Metadata");

    // `state_dict_metadata`: dict[fqn -> TensorStorageMetadata]
    PickleBinunicode(b, "state_dict_metadata");
    PutU8(b, '}');    // EMPTY_DICT
    PutU8(b, '(');    // MARK
    globalRegistry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::ShardedTensor) { return; }
        auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
        // Dedupe replicas by sharddim so each unique chunk is emitted once.
        // Mirrors DebugWriteNativeMetadataInfo_ in server.cpp; without this
        // the two views of the registry disagree and verify_native_server
        // fires. The proper fix is to drop replica_id != 0 at registration
        // — tracked in ankurv/megatron-lm#6.
        std::vector<DS4Tensor::DimRange> chunkRanges;
        chunkRanges.reserve(tnsr.shards.size());
        DS4Tensor::DimRange lastShard{};
        for (auto const& shard : tnsr.shards)
        {
            if (shard.ShardSizeBytes(tnsr.dtype) == 0) { continue; }
            if (lastShard == shard.sharddim) { continue; }
            lastShard = shard.sharddim;
            chunkRanges.push_back(shard.sharddim);
        }
        if (chunkRanges.empty()) { return; }
        auto const fqn = globalRegistry.GetNodeId(node.handle);
        PickleBinunicode(b, fqn);
        EmitTensorStorageMetadata(b, tnsr.dtype, tnsr.fulldim, chunkRanges);
    });
    PutU8(b, 'u');    // SETITEMS

    // `storage_data`: dict[MetadataIndex -> _StorageInfo]
    PickleBinunicode(b, "storage_data");
    PutU8(b, '}');
    PutU8(b, '(');
    globalRegistry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::ShardedTensor) { return; }
        auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
        auto const  fqn  = globalRegistry.GetNodeId(node.handle);
        // Same dedupe as the state_dict_metadata loop above (ankurv/megatron-lm#6).
        // The surviving _StorageInfo points at the lowest-rank replica;
        // dcp.load reads bytes from that file. Until #6 is fixed, the
        // other replica ranks still carry orphan bytes in their .distcp.
        DS4Tensor::DimRange lastShard{};
        for (auto const& shard : tnsr.shards)
        {
            if (shard.ShardSizeBytes(tnsr.dtype) == 0) { continue; }
            if (lastShard == shard.sharddim) { continue; }
            lastShard       = shard.sharddim;
            auto const* rec = LookupWriteRecord(manifest, shard.rank, fqn, shard.sharddim);
            if (rec == nullptr)
            {
                throw std::runtime_error(fmt::format("distcp::WriteMetadata: no WriteRecord for rank {} fqn '{}' (writeLog populated?)",
                                                     static_cast<uint32_t>(shard.rank),
                                                     fqn));
            }
            EmitMetadataIndex(b, fqn, shard.sharddim);
            auto const relPath = fmt::format("__{}_0.distcp", static_cast<uint32_t>(shard.rank));
            EmitStorageInfo(b, relPath, rec->fileOffset, rec->length);
        }
    });
    PutU8(b, 'u');    // SETITEMS

    // `planner_data`, `storage_meta`, `version` omitted — dataclass defaults
    // handle them (loader tolerates absence — verified in step-9 probes).

    EmitNewObjBuildEnd(b);
    PutU8(b, '.');    // STOP
    return b;
}

}    // namespace

void WriteRankShard(std::filesystem::path const& ckptDir,
                    Rank                         rank,
                    StateTree const&             localRegistry,
                    Blob const&                  blob,
                    std::vector<WriteRecord>&    writeLog)
{
    std::vector<ShardEntry> shards;
    CollectShards(localRegistry, blob, shards);

    std::filesystem::create_directories(ckptDir);

    auto const filename = fmt::format("__{}_0.distcp", static_cast<uint32_t>(rank));
    auto const filepath = ckptDir / filename;

    int const fd = ::open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) { throw std::system_error(errno, std::generic_category(), fmt::format("distcp: open({}) failed", filepath.string())); }

    uint64_t cursor = 0;
    try
    {
        writeLog.reserve(writeLog.size() + shards.size());
        for (auto& s : shards)
        {
            // ZIP32 entry size limit: 4 GiB. ZIP64 requires extending miniz's
            // per-entry call site (the m_zip64 flag) plus the alignment math;
            // defer until we see a real large-shard case. The archive's EOCD
            // is already ZIP64 (MZ_ZIP_FLAG_WRITE_ZIP64), but per-entry LFH
            // sizes are still 32-bit.
            if (s.length >= MZ_UINT32_MAX)
            {
                throw std::runtime_error(
                    fmt::format("distcp: shard '{}' is {} bytes (>= 4 GiB); ZIP64 support is not yet implemented", s.fqn, s.length));
            }

            uint64_t const archiveLen = WriteShardArchive(fd, s, s.fqn);

            writeLog.push_back(WriteRecord{
                .fqn        = std::move(s.fqn),
                .sharddim   = s.sharddim,
                .fileOffset = cursor,
                .length     = archiveLen,
            });
            cursor += archiveLen;
        }

        if (::fsync(fd) != 0)
        {
            throw std::system_error(errno, std::generic_category(), fmt::format("distcp: fsync({}) failed", filepath.string()));
        }
    } catch (...)
    {
        ::close(fd);
        throw;
    }

    if (::close(fd) != 0)
    {
        throw std::system_error(errno, std::generic_category(), fmt::format("distcp: close({}) failed", filepath.string()));
    }
}

void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry, std::span<RankManifestRef const> manifest)
{
    auto const pkl = BuildMetadataPkl(globalRegistry, manifest);

    std::filesystem::create_directories(ckptDir);
    auto const filepath = ckptDir / ".metadata";

    int const fd = ::open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) { throw std::system_error(errno, std::generic_category(), fmt::format("distcp: open({}) failed", filepath.string())); }

    try
    {
        auto const* p   = pkl.data();
        size_t      rem = pkl.size();
        while (rem > 0)
        {
            ssize_t const n = ::write(fd, p, rem);
            if (n < 0)
            {
                if (errno == EINTR) { continue; }
                throw std::system_error(errno,
                                        std::generic_category(),
                                        fmt::format("distcp: write({}) failed with {} bytes remaining", filepath.string(), rem));
            }
            p += static_cast<size_t>(n);
            rem -= static_cast<size_t>(n);
        }
        if (::fsync(fd) != 0)
        {
            throw std::system_error(errno, std::generic_category(), fmt::format("distcp: fsync({}) failed", filepath.string()));
        }
    } catch (...)
    {
        ::close(fd);
        throw;
    }

    if (::close(fd) != 0)
    {
        throw std::system_error(errno, std::generic_category(), fmt::format("distcp: close({}) failed", filepath.string()));
    }
}

}    // namespace checkpoint::distcp
