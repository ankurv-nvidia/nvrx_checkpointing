#include "distcp.h"
#include "distcp_internal.h"
#include "state.h"

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
#include <unordered_map>
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

namespace
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

// Precondition: shape is non-empty. WriteRankShard builds shapes from
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

// Layout constants shared between the actual writer (WriteShardArchive) and
// the length predictor (ComputeShardArchiveSize). The predictor is what lets
// rank 0 emit `.metadata` storage_info entries for OTHER ranks' shards
// without ever seeing their local registry: archive size is fully determined
// by (dtype, shape, dataSize), so the layout simulator below produces the
// same numbers WriteShardArchive would record in a writeLog. See
// `[distcp][tier1][predict]` for the self-consistency test that gates this.
//
// 64-byte alignment of `archive/data/N` mirrors what PyTorchStreamWriter does
// for zero-copy mmap reads downstream. Without it, torch.load still works but
// byte-parity with `dcp.save` breaks.
constexpr uint64_t KFieldAlignment      = 64;
constexpr uint64_t LocalFileHeaderSize  = 30;
constexpr uint64_t CentralDirHeaderSize = 46;
constexpr uint64_t FbExtraHeaderBytes   = sizeof(uint16_t) * 2;    // 2-byte ID + 2-byte length
constexpr uint64_t EocdSize             = 22;
// Forced by MZ_ZIP_FLAG_WRITE_ZIP64 at writer init — present even when
// per-entry sizes fit in 32 bits. Sizes from miniz_zip.c.
constexpr uint64_t Zip64EocdRecordSize  = 56;
constexpr uint64_t Zip64EocdLocatorSize = 20;
// `archive/.data/serialization_id` body is always `fmt::format("{:020d}{:020d}", ...)`.
constexpr uint64_t SerializationIdSize = 40;
// Every entry with `uncomp_size > 0` gets a post-body data descriptor:
// 4-byte signature + 4-byte CRC + 4-byte comp_size + 4-byte uncomp_size.
// Miniz emits the 32-bit variant unless the LFH itself needed a ZIP64 extra
// (`pExtra_data != NULL`), which only happens when size or offset crosses
// 4 GiB — gated out by WriteRankShard. See miniz_zip.c:3505-3533.
constexpr uint64_t DataDescriptorSize = 16;

// Per-entry extra-field length needed to push the entry's data body onto a
// `kFieldAlignment` boundary. Replicates `caffe2::serialize::detail::getOffset`
// / `getPadding` from pytorch/v2.11.0. `cursor` is the archive offset the next
// LFH will land at (miniz's `m_archive_size`).
uint64_t ComputeFbExtraSize(uint64_t cursor, uint64_t filenameSize)
{
    uint64_t const start       = cursor + LocalFileHeaderSize + filenameSize + FbExtraHeaderBytes;
    uint64_t const mod         = start % KFieldAlignment;
    uint64_t const paddingSize = (mod == 0) ? 0 : (KFieldAlignment - mod);
    return paddingSize + FbExtraHeaderBytes;
}

// Materializes the FB extra-field bytes for WriteShardArchive to hand to
// miniz. Layout: `'F','B', <padding_size LE 2 bytes>, 'Z' * padding_size`.
// Total length = padding_size + 4.
std::vector<unsigned char> BuildFbExtra(size_t cursor, size_t filenameSize, size_t dataSize)
{
    if (dataSize >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX)
    {
        // The upstream code adds extra bytes here for ZIP64 LFH fields. We
        // gate ZIP64 out at the caller, so this branch shouldn't fire.
        throw std::runtime_error(fmt::format("distcp: ZIP entry too large for ZIP32 (cursor={}, dataSize={})", cursor, dataSize));
    }
    uint64_t const extraSize   = ComputeFbExtraSize(cursor, filenameSize);
    uint64_t const paddingSize = extraSize - FbExtraHeaderBytes;

    std::vector<unsigned char> extra;
    extra.resize(static_cast<size_t>(extraSize), 'Z');
    extra[0] = 'F';
    extra[1] = 'B';
    extra[2] = static_cast<unsigned char>(paddingSize & 0xff);
    extra[3] = static_cast<unsigned char>((paddingSize >> 8) & 0xff);
    return extra;
}

// Filenames for the 7 archive entries, in write order. Shared between the
// writer and the size predictor so they can't drift apart.
constexpr std::array<std::string_view, 7> EntryFullNames{
    "archive/data.pkl",
    "archive/.format_version",
    "archive/.storage_alignment",
    "archive/byteorder",
    "archive/data/0",
    "archive/version",
    "archive/.data/serialization_id",
};
// Bare names that PyTorchStreamWriter::files_written_ stores (no `archive/`
// prefix), in the same order as EntryFullNames. Used by WriteShardArchive
// to reproduce the serialization_id name-hash leg. The 7th entry,
// `.data/serialization_id`, is intentionally excluded from the hash.
constexpr std::array<std::string_view, 6> EntryBareNames{
    "data.pkl",
    ".format_version",
    ".storage_alignment",
    "byteorder",
    "data/0",
    "version",
};
// Constant-body bytes for the fixed entries (indices 1, 2, 3, 5 in
// EntryFullNames). The two variable entries — `data.pkl` (index 0) and
// `data/0` (index 4) — are produced per-shard by the caller. Sizes drive
// the analytical size predictor; bytes also feed the actual writer.
constexpr std::string_view FormatVersionBytes    = "1";
constexpr std::string_view StorageAlignmentBytes = "64";
constexpr std::string_view ByteorderBytes        = "little";
constexpr std::string_view VersionBytes          = "3\n";

}    // anonymous namespace

uint64_t ComputeShardArchiveSize(DS4Tensor::DType dtype, DS4Tensor::Dim const& shape, uint64_t shardDataSize)
{
    if (shardDataSize >= MZ_UINT32_MAX)
    {
        throw std::runtime_error(
            fmt::format("distcp: ComputeShardArchiveSize: shardDataSize {} >= 4 GiB (ZIP64 LFH not implemented)", shardDataSize));
    }

    auto const dataPkl = BuildDataPkl(dtype, shape);

    // Per-entry payload size, indexed in lockstep with EntryFullNames.
    std::array<uint64_t, 7> const dataSizes{
        dataPkl.size(),
        FormatVersionBytes.size(),
        StorageAlignmentBytes.size(),
        ByteorderBytes.size(),
        shardDataSize,
        VersionBytes.size(),
        SerializationIdSize,
    };

    // LFH + filename + FB extra + data + data descriptor, per entry. Cursor
    // advancement mirrors miniz `m_archive_size` so each ComputeFbExtraSize
    // sees the same alignment input that WriteShardArchive's BuildFbExtra
    // would — getting the cursor wrong shifts every subsequent entry's FB
    // padding off the 64-byte grid too, so the descriptor MUST be folded into
    // cursor advancement, not just added at the end.
    uint64_t cursor = 0;
    for (size_t i = 0; i < EntryFullNames.size(); ++i)
    {
        uint64_t const fnSize    = EntryFullNames.at(i).size();
        uint64_t const dataSize  = dataSizes.at(i);
        uint64_t const extraSize = ComputeFbExtraSize(cursor, fnSize);
        cursor += LocalFileHeaderSize + fnSize + extraSize + dataSize;
        // All seven of our entries have non-zero body, so the descriptor
        // always fires. The `if` keeps the formula honest if a future
        // change ever introduces a zero-body entry.
        if (dataSize > 0) { cursor += DataDescriptorSize; }
    }

    // Central directory: 46-byte fixed header + filename per entry. We pass
    // null/0 for both `pExtra` and `user_extra_data_central`, and all our
    // LFH offsets/sizes fit in 32 bits, so miniz adds no zip64 extra to CD
    // records either (verified against miniz_zip.c:3184-3221).
    for (auto const& fn : EntryFullNames) { cursor += CentralDirHeaderSize + fn.size(); }

    // MZ_ZIP_FLAG_WRITE_ZIP64 forces m_zip64 at writer init, so finalize
    // always emits ZIP64 EOCD record + locator before the plain EOCD trailer
    // — even when nothing in the archive actually needs 64-bit fields
    // (miniz_zip.c:4442-4470).
    cursor += Zip64EocdRecordSize + Zip64EocdLocatorSize + EocdSize;

    return cursor;
}

namespace
{

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
uint64_t WriteShardArchive(int                   fd,
                           DS4Tensor::DType      dtype,
                           DS4Tensor::Dim const& shape,
                           char const*           srcPtr,
                           uint64_t              length,
                           std::string_view      contextName)
{
    std::vector<unsigned char> dataPkl = BuildDataPkl(dtype, shape);

    // Per-entry plan: filename + content pointer + content size.
    struct EntryPlan
    {
        std::string                fullName;
        void const*                data;
        size_t                     size;
        std::vector<unsigned char> padding;    // 0 for non-data entries
    };

    // Record order matches torch.serialization._save / PyTorchStreamWriter::
    // writeEndOfFile exactly:
    //   data.pkl, .format_version, .storage_alignment, byteorder,
    //   data/N..., version, .data/serialization_id
    // Filenames live in `EntryFullNames` so the size predictor uses the same
    // strings. The serialization_id entry is appended after combined-CRC is known.
    enum Idx : size_t
    {
        DataPklIdx    = 0,
        FormatVerIdx  = 1,
        StorageAlnIdx = 2,
        ByteOrderIdx  = 3,
        DataZeroIdx   = 4,
        VersionIdx    = 5,
        SerialIdIdx   = 6,
    };
    std::array<EntryPlan, 6> entries{
        EntryPlan{.fullName = std::string{EntryFullNames[DataPklIdx]}, .data = dataPkl.data(), .size = dataPkl.size(), .padding = {}},
        EntryPlan{.fullName = std::string{EntryFullNames[FormatVerIdx]},
                  .data     = FormatVersionBytes.data(),
                  .size     = FormatVersionBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{EntryFullNames[StorageAlnIdx]},
                  .data     = StorageAlignmentBytes.data(),
                  .size     = StorageAlignmentBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{EntryFullNames[ByteOrderIdx]},
                  .data     = ByteorderBytes.data(),
                  .size     = ByteorderBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{EntryFullNames[DataZeroIdx]}, .data = srcPtr, .size = length, .padding = {}},
        EntryPlan{
            .fullName = std::string{EntryFullNames[VersionIdx]}, .data = VersionBytes.data(), .size = VersionBytes.size(), .padding = {}},
    };

    // Set up miniz heap writer. MZ_ZIP_FLAG_WRITE_ZIP64 forces the ZIP64 EOCD
    // + locator at finalize time — matches PyTorchStreamWriter's "universally
    // ZIP64" guarantee. Per-entry LFH/CD records stay ZIP32 unless their size
    // or offset overflow 32 bits.
    mz_zip_archive zip{};
    if (mz_zip_writer_init_heap_v2(&zip,
                                   /*size_to_reserve_at_beginning=*/0,
                                   /*initial_allocation_size=*/std::max<size_t>(1024, length),
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
        plan.padding = BuildFbExtra(zip.m_archive_size, plan.fullName.size(), plan.size);
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

    try
    {
        for (size_t i = 0; i < entries.size(); ++i) { addEntry(entries.at(i), EntryBareNames.at(i)); }

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
            .fullName = std::string{EntryFullNames[SerialIdIdx]},
            .data     = serializationId.data(),
            .size     = serializationId.size(),
            .padding  = {},
        };
        // Important: do NOT mix the id record itself into combinedCrc / name hash —
        // upstream excludes it (see PyTorchStreamWriter::writeSerializationId, which
        // checks `files_written_.find(kSerializationIdRecordName) == end()` before
        // computing).
        idPlan.padding = BuildFbExtra(zip.m_archive_size, idPlan.fullName.size(), idPlan.size);
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

// torch.distributed.checkpoint.metadata.BytesStorageMetadata() — an empty
// dataclass (no fields). Marks `fqn` in state_dict_metadata as a non-tensor
// (ShardedObject) entry; the DCP load planner reads its bytes via storage_data.
void EmitBytesStorageMetadata(std::vector<unsigned char>& b)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "BytesStorageMetadata");
    EmitNewObjBuildEnd(b);
}

// MetadataIndex for a bytes (ShardedObject) entry. Unlike the tensor index, a
// bytes index carries no offset/index — both default to None, which torch's
// writer omits — so the restored key equals MetadataIndex(fqn).
void EmitMetadataIndexBytes(std::vector<unsigned char>& b, std::string_view fqn)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "MetadataIndex");
    PickleBinunicode(b, "fqn");
    PickleBinunicode(b, fqn);
    EmitNewObjBuildEnd(b);
}

std::vector<unsigned char> BuildMetadataPkl(StateTree const& globalRegistry)
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
        assert(type != StateNode::Type::TensorShard);
        if (type != StateNode::Type::ShardedTensor) { return; }
        auto const&                      tnsr = std::get<StateNode::ShardedTensor>(node.data);
        std::vector<DS4Tensor::DimRange> chunkRanges;
        chunkRanges.reserve(tnsr.shards.size());
        for (auto const& shard : tnsr.shards)
        {
            if (shard.ShardSizeBytes(tnsr.dtype) == 0) { continue; }
            // tnsr.shards is ordered by (rank, sharddim), so replicas of one chunk held on
            // different ranks are not adjacent. Dedup against every chunk collected so far,
            // not just the previous shard, so each chunk is listed once: a consecutive-only
            // check leaves duplicate ChunkStorageMetadata entries for cross-node replicas.
            bool alreadyEmitted = false;
            for (auto const& emitted : chunkRanges)
            {
                if (emitted == shard.sharddim)
                {
                    alreadyEmitted = true;
                    break;
                }
            }
            if (alreadyEmitted) { continue; }
            chunkRanges.push_back(shard.sharddim);
        }
        if (chunkRanges.empty()) { return; }
        auto const fqn = globalRegistry.GetNodeId(node.handle);
        PickleBinunicode(b, fqn);
        EmitTensorStorageMetadata(b, tnsr.dtype, tnsr.fulldim, chunkRanges);
    });
    // ShardedObjects (Blob nodes) — torch.save'd bytes the load path reads back.
    globalRegistry.WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
        PickleBinunicode(b, globalRegistry.GetNodeId(node.handle));
        EmitBytesStorageMetadata(b);
    });
    PutU8(b, 'u');    // SETITEMS

    // `storage_data`: dict[MetadataIndex -> _StorageInfo]
    PickleBinunicode(b, "storage_data");
    PutU8(b, '}');
    PutU8(b, '(');
    std::unordered_map<Rank, uint64_t> rankCursors;
    globalRegistry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::ShardedTensor) { return; }
        auto const&         tnsr = std::get<StateNode::ShardedTensor>(node.data);
        auto const          fqn  = globalRegistry.GetNodeId(node.handle);
        DS4Tensor::DimRange lastShard{};
        for (auto const& shard : tnsr.shards)
        {
            auto const shardSize = shard.ShardSizeBytes(tnsr.dtype);
            if (shardSize == 0) { continue; }
            DS4Tensor::Dim shardShape;
            for (auto const& r : shard.sharddim) { shardShape.push_back(r.end - r.begin); }
            uint64_t const archiveLen = ComputeShardArchiveSize(tnsr.dtype, shardShape, shardSize);
            uint64_t&      cursor     = rankCursors[shard.rank];
            uint64_t       curCursor  = cursor;
            cursor += archiveLen;

            if (lastShard == shard.sharddim) { continue; }
            lastShard = shard.sharddim;

            // Emit only the first replica of each sharddim; later replicas share
            // the same MetadataIndex key and would just overwrite the prior
            // entry. The cursor must still advance for every shard — WriteRankShard
            // wrote each one to that rank's .distcp file, so skipping the
            // advance would leave subsequent non-replica shards on the same rank
            // pointing at a wrong offset.
            EmitMetadataIndex(b, fqn, shard.sharddim);
            auto const relPath = fmt::format("__{}_0.distcp", static_cast<uint32_t>(shard.rank));
            EmitStorageInfo(b, relPath, curCursor, archiveLen);
        }
    });
    // ShardedObject bytes are all appended to rank 0's shard file, after rank 0's
    // tensors (rankCursors[0] now holds rank 0's tensor total). WriteRankShard for
    // rank 0 writes them in the same WalkNodes order, so these offsets line up.
    uint64_t& objCursor = rankCursors[static_cast<Rank>(0)];
    globalRegistry.WalkNodes([&](StateNode const& node) {
        if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
        auto const& blob = std::get<StateNode::Blob>(node.data);
        EmitMetadataIndexBytes(b, globalRegistry.GetNodeId(node.handle));
        EmitStorageInfo(b, "__0_0.distcp", objCursor, blob.size());
        objCursor += blob.size();
    });
    PutU8(b, 'u');    // SETITEMS

    // `planner_data`, `storage_meta`, `version` omitted — dataclass defaults
    // handle them (loader tolerates absence — verified in step-9 probes).

    EmitNewObjBuildEnd(b);
    PutU8(b, '.');    // STOP
    return b;
}

}    // namespace

void WriteRankShard(std::filesystem::path const& ckptDir, Rank rank, StateTree const& registry, Blob const& blob)
{
    std::filesystem::create_directories(ckptDir);

    auto const filename = fmt::format("__{}_0.distcp", static_cast<uint32_t>(rank));
    auto const filepath = ckptDir / filename;

    int const fd = ::open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) { throw std::system_error(errno, std::generic_category(), fmt::format("distcp: open({}) failed", filepath.string())); }

    uint64_t cursor = 0;
    try
    {
        auto const* base = static_cast<char const*>(blob.data);
        registry.WalkNodes([&](StateNode const& node) {
            auto const type = static_cast<StateNode::Type>(node.data.index());
            // The registry handed in is always the post-merge global registry
            // (ShardedTensor entries keyed by rank). Tests construct it with
            // the same shape.
            assert(type != StateNode::Type::TensorShard);
            if (type != StateNode::Type::ShardedTensor) { return; }
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            auto const  fqn  = registry.GetNodeId(node.handle);
            for (auto const& shard : tnsr.shards)
            {
                if (shard.rank != rank) { continue; }
                auto const size = shard.ShardSizeBytes(tnsr.dtype);
                if (size == 0) { continue; }
                DS4Tensor::Dim shape{};
                for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
                // ZIP32 entry size limit: 4 GiB. ZIP64 requires extending miniz's
                // per-entry call site (the m_zip64 flag) plus the alignment math;
                // defer until we see a real large-shard case. The archive's EOCD
                // is already ZIP64 (MZ_ZIP_FLAG_WRITE_ZIP64), but per-entry LFH
                // sizes are still 32-bit.
                if (size >= MZ_UINT32_MAX)
                {
                    throw std::runtime_error(
                        fmt::format("distcp: shard '{}' is {} bytes (>= 4 GiB); ZIP64 support is not yet implemented", fqn, size));
                }
                uint64_t const predicted = ComputeShardArchiveSize(tnsr.dtype, shape, size);
                uint64_t const actual    = WriteShardArchive(fd, tnsr.dtype, shape, base + shard.handle, size, fqn);
                if (predicted != actual)
                {
                    throw std::runtime_error(fmt::format("distcp: ComputeShardArchiveSize/WriteShardArchive disagree for '{}' "
                                                         "(dtype={}, shape=[{}], dataSize={}): predicted={} actual={}",
                                                         fqn,
                                                         DTypeName(tnsr.dtype),
                                                         fmt::join(shape, ","),
                                                         size,
                                                         predicted,
                                                         actual));
                }
                cursor += actual;
            }
        });

        // ShardedObject bytes live only in rank 0's shard file (the metadata points
        // every BytesStorageMetadata entry at __0_0.distcp). Append them after the
        // tensors, in the same WalkNodes order BuildMetadataPkl used for the offsets.
        if (static_cast<uint32_t>(rank) == 0)
        {
            registry.WalkNodes([&](StateNode const& node) {
                if (static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::Blob) { return; }
                auto const& objBytes = std::get<StateNode::Blob>(node.data);
                char const* p        = objBytes.data();
                size_t      rem      = objBytes.size();
                while (rem > 0)
                {
                    ssize_t const n = ::write(fd, p, rem);
                    if (n < 0)
                    {
                        if (errno == EINTR) { continue; }
                        throw std::system_error(errno, std::generic_category(), fmt::format("distcp: write({}) failed", filepath.string()));
                    }
                    p += n;
                    rem -= static_cast<size_t>(n);
                }
                cursor += objBytes.size();
            });
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

void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry)
{
    auto const pkl = BuildMetadataPkl(globalRegistry);

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

    // Megatron uses this marker to recognize the directory as a torch_dist
    // checkpoint. Publish it from the daemon alongside .metadata so the
    // training process does not write checkpoint sidecars in the exposed path.
    static constexpr std::string_view ConfigJson = R"({"sharded_backend": "torch_dist", "sharded_backend_version": 1})";
    auto const                        configPath = ckptDir / "metadata.json";
    int const configFd = ::open(configPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (configFd < 0)
    {
        throw std::system_error(errno, std::generic_category(), fmt::format("distcp: open({}) failed", configPath.string()));
    }
    try
    {
        char const* p   = ConfigJson.data();
        size_t      rem = ConfigJson.size();
        while (rem > 0)
        {
            ssize_t const n = ::write(configFd, p, rem);
            if (n < 0)
            {
                if (errno == EINTR) { continue; }
                throw std::system_error(errno,
                                        std::generic_category(),
                                        fmt::format("distcp: write({}) failed with {} bytes remaining", configPath.string(), rem));
            }
            p += n;
            rem -= static_cast<size_t>(n);
        }
        if (::fsync(configFd) != 0)
        {
            throw std::system_error(errno, std::generic_category(), fmt::format("distcp: fsync({}) failed", configPath.string()));
        }
    } catch (...)
    {
        ::close(configFd);
        throw;
    }
    if (::close(configFd) != 0)
    {
        throw std::system_error(errno, std::generic_category(), fmt::format("distcp: close({}) failed", configPath.string()));
    }
}

}    // namespace checkpoint::distcp
