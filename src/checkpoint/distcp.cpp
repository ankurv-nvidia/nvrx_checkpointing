#include "distcp.h"
#include "distcp_internal.h"
#include "state.h"

#include <miniz/miniz.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "distcp writer assumes a little-endian host: archive/byteorder is hardcoded "
              "to \"little\" and the LE serializers are direct memcpys.");

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

using ByteBuffer = std::vector<char>;

static constexpr uint8_t  ByteMask                  = 0xffU;
static constexpr uint64_t PickleBinIntMax           = 0x7fffffffULL;
static constexpr size_t   PickleLongBufferSize      = 9;
static constexpr uint8_t  PickleLong1Opcode         = 0x8aU;
static constexpr uint8_t  PickleNewObjOpcode        = 0x81U;
static constexpr uint8_t  PickleTuple1Opcode        = 0x85U;
static constexpr uint8_t  PickleTuple2Opcode        = 0x86U;
static constexpr uint8_t  PickleTuple3Opcode        = 0x87U;
static constexpr uint8_t  PickleNewFalseOpcode      = 0x89U;
static constexpr size_t   InitialPickleReserveBytes = 192;

static std::string_view DTypeName(DS4Tensor::DType dtype)
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
namespace
{
struct DtypePickleInfo
{
    bool             useV3;
    std::string_view storageMod;
    std::string_view storageName;
    std::string_view dtypeMod;
    std::string_view dtypeName;
};
}    // namespace

static DtypePickleInfo PickleInfoForDtype(DS4Tensor::DType dtype)
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

static void PutU8(ByteBuffer& b, uint8_t v)
{ b.push_back(static_cast<char>(v)); }

static void PutU16LE(ByteBuffer& b, uint16_t v)
{
    b.push_back(static_cast<char>(v & ByteMask));
    b.push_back(static_cast<char>(v >> 8));
}

static void PutU32LE(ByteBuffer& b, uint32_t v)
{
    for (int i = 0; i < 4; ++i) { b.push_back(static_cast<char>((v >> (i * 8)) & ByteMask)); }
}

static void PutBytes(ByteBuffer& b, std::string_view sv)
{ b.insert(b.end(), sv.begin(), sv.end()); }

static void PickleInt(ByteBuffer& b, uint64_t v)
{
    if (v < 256)
    {
        PutU8(b, 'K');
        PutU8(b, static_cast<uint8_t>(v));
    }
    else if (v < 65536)
    {
        PutU8(b, 'M');
        PutU16LE(b, static_cast<uint16_t>(v));
    }
    else if (v <= PickleBinIntMax)
    {
        PutU8(b, 'J');
        PutU32LE(b, static_cast<uint32_t>(v));
    }
    else
    {
        std::array<uint8_t, PickleLongBufferSize> tmp{};
        size_t                                    n = 0;
        uint64_t                                  x = v;
        while (x != 0)
        {
            tmp[n++] = static_cast<uint8_t>(x & ByteMask);
            x >>= 8;
        }
        if (n > 0 && (tmp[n - 1] & 0x80U) != 0U) { tmp[n++] = 0U; }
        PutU8(b, PickleLong1Opcode);
        PutU8(b, static_cast<uint8_t>(n));
        for (size_t i = 0; i < n; ++i) { PutU8(b, tmp[i]); }
    }
}

static void PickleBinunicode(ByteBuffer& b, std::string_view s)
{
    PutU8(b, 'X');
    PutU32LE(b, static_cast<uint32_t>(s.size()));
    PutBytes(b, s);
}

static void PickleGlobal(ByteBuffer& b, std::string_view mod, std::string_view name)
{
    PutU8(b, 'c');
    PutBytes(b, mod);
    PutU8(b, '\n');
    PutBytes(b, name);
    PutU8(b, '\n');
}

static void PickleBinput(ByteBuffer& b, uint8_t idx)
{
    PutU8(b, 'q');
    PutU8(b, idx);
}

static void PickleTupleClose(ByteBuffer& b, size_t n)
{
    switch (n)
    {
    case 0: PutU8(b, ')'); return;
    case 1: PutU8(b, PickleTuple1Opcode); return;
    case 2: PutU8(b, PickleTuple2Opcode); return;
    case 3: PutU8(b, PickleTuple3Opcode); return;
    default: PutU8(b, 't'); return;
    }
}

// Precondition: shape is non-empty. WriteRankShard builds shapes from
// `sharddim`, which is non-empty for every tensor we write.
static ByteBuffer BuildDataPkl(DS4Tensor::DType dtype, DS4Tensor::Dim const& shape)
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

    ByteBuffer out;
    out.reserve(InitialPickleReserveBytes);

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

    PutU8(out, PickleNewFalseOpcode);

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
constexpr uint64_t   DataDescriptorSize             = 16;
constexpr size_t     LocalTimestampOffset           = 10;
constexpr size_t     CentralDirTimestampOffset      = 12;
constexpr size_t     TimestampFieldBytes            = sizeof(uint16_t) * 2;
constexpr size_t     EocdCentralDirSizeOffset       = 12;
constexpr size_t     EocdCentralDirOffsetOffset     = 16;
constexpr size_t     CentralDirFilenameLengthOffset = 28;
constexpr size_t     CentralDirExtraLengthOffset    = 30;
constexpr size_t     CentralDirCommentLengthOffset  = 32;
constexpr uint32_t   Zip32Sentinel                  = std::numeric_limits<uint32_t>::max();
constexpr std::array EocdSignature{'P', 'K', '\x05', '\x06'};
constexpr std::array CentralDirSignature{'P', 'K', '\x01', '\x02'};

// Per-entry extra-field length needed to push the entry's data body onto a
// `kFieldAlignment` boundary. Replicates `caffe2::serialize::detail::getOffset`
// / `getPadding` from pytorch/v2.11.0. `cursor` is the archive offset the next
// LFH will land at (miniz's `m_archive_size`).
static uint64_t ComputeFbExtraSize(uint64_t cursor, uint64_t filenameSize)
{
    uint64_t const start       = cursor + LocalFileHeaderSize + filenameSize + FbExtraHeaderBytes;
    uint64_t const mod         = start % KFieldAlignment;
    uint64_t const paddingSize = (mod == 0) ? 0 : (KFieldAlignment - mod);
    return paddingSize + FbExtraHeaderBytes;
}

// Materializes the FB extra-field bytes for WriteShardArchive to hand to
// miniz. Layout: `'F','B', <padding_size LE 2 bytes>, 'Z' * padding_size`.
// Total length = padding_size + 4.
static ByteBuffer BuildFbExtra(size_t cursor, size_t filenameSize, size_t dataSize)
{
    if (dataSize >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX)
    {
        // The upstream code adds extra bytes here for ZIP64 LFH fields. We
        // gate ZIP64 out at the caller, so this branch shouldn't fire.
        throw std::runtime_error(fmt::format("distcp: ZIP entry too large for ZIP32 (cursor={}, dataSize={})", cursor, dataSize));
    }
    uint64_t const extraSize   = ComputeFbExtraSize(cursor, filenameSize);
    uint64_t const paddingSize = extraSize - FbExtraHeaderBytes;

    ByteBuffer extra;
    extra.resize(static_cast<size_t>(extraSize), 'Z');
    extra[0] = 'F';
    extra[1] = 'B';
    extra[2] = static_cast<char>(paddingSize & ByteMask);
    extra[3] = static_cast<char>((paddingSize >> 8) & ByteMask);
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

uint64_t ComputeShardArchiveSize(DS4Tensor::DType dtype, DS4Tensor::Dim const& shape, uint64_t shardDataSize)
{
    if (shardDataSize >= MZ_UINT32_MAX)
    {
        throw std::runtime_error(
            fmt::format("distcp: ComputeShardArchiveSize: shardDataSize {} >= 4 GiB (ZIP64 LFH not implemented)", shardDataSize));
    }

    auto const dataPkl = checkpoint::distcp::BuildDataPkl(dtype, shape);

    // Per-entry payload size, indexed in lockstep with EntryFullNames.
    std::array<uint64_t, 7> const dataSizes{
        dataPkl.size(),
        checkpoint::distcp::FormatVersionBytes.size(),
        checkpoint::distcp::StorageAlignmentBytes.size(),
        checkpoint::distcp::ByteorderBytes.size(),
        shardDataSize,
        checkpoint::distcp::VersionBytes.size(),
        checkpoint::distcp::SerializationIdSize,
    };

    // LFH + filename + FB extra + data + data descriptor, per entry. Cursor
    // advancement mirrors miniz `m_archive_size` so each ComputeFbExtraSize
    // sees the same alignment input that WriteShardArchive's BuildFbExtra
    // would — getting the cursor wrong shifts every subsequent entry's FB
    // padding off the 64-byte grid too, so the descriptor MUST be folded into
    // cursor advancement, not just added at the end.
    uint64_t cursor = 0;
    for (size_t i = 0; i < checkpoint::distcp::EntryFullNames.size(); ++i)
    {
        uint64_t const fnSize    = checkpoint::distcp::EntryFullNames.at(i).size();
        uint64_t const dataSize  = dataSizes.at(i);
        uint64_t const extraSize = checkpoint::distcp::ComputeFbExtraSize(cursor, fnSize);
        cursor += checkpoint::distcp::LocalFileHeaderSize + fnSize + extraSize + dataSize;
        // All seven of our entries have non-zero body, so the descriptor
        // always fires. The `if` keeps the formula honest if a future
        // change ever introduces a zero-body entry.
        if (dataSize > 0) { cursor += checkpoint::distcp::DataDescriptorSize; }
    }

    // Central directory: 46-byte fixed header + filename per entry. We pass
    // null/0 for both `pExtra` and `user_extra_data_central`, and all our
    // LFH offsets/sizes fit in 32 bits, so miniz adds no zip64 extra to CD
    // records either (verified against miniz_zip.c:3184-3221).
    for (auto const& fn : checkpoint::distcp::EntryFullNames) { cursor += checkpoint::distcp::CentralDirHeaderSize + fn.size(); }

    // MZ_ZIP_FLAG_WRITE_ZIP64 forces m_zip64 at writer init, so finalize
    // always emits ZIP64 EOCD record + locator before the plain EOCD trailer
    // — even when nothing in the archive actually needs 64-bit fields
    // (miniz_zip.c:4442-4470).
    cursor += checkpoint::distcp::Zip64EocdRecordSize + checkpoint::distcp::Zip64EocdLocatorSize + checkpoint::distcp::EocdSize;

    return cursor;
}

// `c10::hash_combine` from c10/util/hash.h: deterministic mixer used in the
// serialization_id record-name hash.
static constexpr uint64_t HashCombineConstant  = 0x9e3779b9ULL;
static constexpr uint32_t HashCombineLeftShift = 6U;

static uint64_t HashCombine(uint64_t seed, uint64_t value)
{ return seed ^ (value + HashCombineConstant + (seed << HashCombineLeftShift) + (seed >> 2U)); }

// libstdc++'s `std::hash<std::string>` (used by torch's writer via
// `c10::hash<std::string>{}(...)` -> std::hash) — we just call it directly.
static uint64_t StdHashString(std::string const& s)
{ return std::hash<std::string>{}(s); }

static std::ofstream OpenOutputFile(std::filesystem::path const& path)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) { throw std::runtime_error(fmt::format("distcp: open({}) failed", path.string())); }
    return output;
}

static void WriteBytes(std::ostream& output, char const* data, size_t size, std::string_view contextName)
{
    constexpr auto maxWriteSize = static_cast<size_t>(std::numeric_limits<std::streamsize>::max());
    while (size > 0)
    {
        auto const writeSize = std::min(size, maxWriteSize);
        output.write(data, static_cast<std::streamsize>(writeSize));
        if (!output) { throw std::runtime_error(fmt::format("distcp: write({}) failed with {} bytes remaining", contextName, size)); }
        data += writeSize;
        size -= writeSize;
    }
}

static void FinishOutputFile(std::ofstream& output, std::filesystem::path const& path)
{
    output.flush();
    if (!output) { throw std::runtime_error(fmt::format("distcp: flush({}) failed", path.string())); }
    output.close();
    if (output.fail()) { throw std::runtime_error(fmt::format("distcp: close({}) failed", path.string())); }
}

// Writes one self-contained torch.save-compatible ZIP archive containing the
// 7 entries for a single shard. Returns the bytes appended.
//
// Layout: `archive/data.pkl`, `archive/byteorder`, `archive/.format_version`,
// `archive/.storage_alignment`, `archive/data/0`, `archive/version`,
// `archive/.data/serialization_id`. The data/0 entry is FB-padded to 64-byte
// alignment so zero-copy mmap downstream lines up.
//
// The archive is built in-memory via miniz's heap writer, then dumped to `output`
// in one shot. Memory cost = ZIP container overhead + tensor body (the
// tensor body is the only large field). For multi-gigabyte tensors this is
// the same RSS hit `dcp.save` takes; revisit if it shows up in a profile.
static uint64_t WriteShardArchive(std::ostream&         output,
                                  DS4Tensor::DType      dtype,
                                  DS4Tensor::Dim const& shape,
                                  char const*           srcPtr,
                                  uint64_t              length,
                                  std::string_view      contextName)
{
    ByteBuffer dataPkl = checkpoint::distcp::BuildDataPkl(dtype, shape);

    // Per-entry plan: filename + content pointer + content size.
    struct EntryPlan
    {
        std::string fullName;
        void const* data;
        size_t      size;
        ByteBuffer  padding;    // 0 for non-data entries
    };

    // Record order matches torch.serialization._save / PyTorchStreamWriter::
    // writeEndOfFile exactly:
    //   data.pkl, .format_version, .storage_alignment, byteorder,
    //   data/N..., version, .data/serialization_id
    // Filenames live in `EntryFullNames` so the size predictor uses the same
    // strings. The serialization_id entry is appended after combined-CRC is known.
    static constexpr size_t DataPklIdx    = 0;
    static constexpr size_t FormatVerIdx  = 1;
    static constexpr size_t StorageAlnIdx = 2;
    static constexpr size_t ByteOrderIdx  = 3;
    static constexpr size_t DataZeroIdx   = 4;
    static constexpr size_t VersionIdx    = 5;
    static constexpr size_t SerialIdIdx   = 6;

    std::array<EntryPlan, checkpoint::distcp::EntryFullNames.size() - 1> entries{
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[DataPklIdx]},
                  .data     = dataPkl.data(),
                  .size     = dataPkl.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[FormatVerIdx]},
                  .data     = checkpoint::distcp::FormatVersionBytes.data(),
                  .size     = checkpoint::distcp::FormatVersionBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[StorageAlnIdx]},
                  .data     = checkpoint::distcp::StorageAlignmentBytes.data(),
                  .size     = checkpoint::distcp::StorageAlignmentBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[ByteOrderIdx]},
                  .data     = checkpoint::distcp::ByteorderBytes.data(),
                  .size     = checkpoint::distcp::ByteorderBytes.size(),
                  .padding  = {}},
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[DataZeroIdx]}, .data = srcPtr, .size = length, .padding = {}},
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[VersionIdx]},
                  .data     = checkpoint::distcp::VersionBytes.data(),
                  .size     = checkpoint::distcp::VersionBytes.size(),
                  .padding  = {}},
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
        plan.padding = checkpoint::distcp::BuildFbExtra(zip.m_archive_size, plan.fullName.size(), plan.size);
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
                                        /*user_extra_data=*/plan.padding.data(),
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
        auto const entryCrc = static_cast<uint32_t>(mz_crc32(MZ_CRC32_INIT, static_cast<uint8_t const*>(plan.data), plan.size));
        combinedCrc         = HashCombine(combinedCrc, entryCrc);
        bareNamesWritten.insert(std::string{bareName});
    };

    try
    {
        for (size_t i = 0; i < entries.size(); ++i) { addEntry(entries.at(i), checkpoint::distcp::EntryBareNames.at(i)); }

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
            .fullName = std::string{checkpoint::distcp::EntryFullNames[SerialIdIdx]},
            .data     = serializationId.data(),
            .size     = serializationId.size(),
            .padding  = {},
        };
        // Important: do NOT mix the id record itself into combinedCrc / name hash —
        // upstream excludes it (see PyTorchStreamWriter::writeSerializationId, which
        // checks `files_written_.find(kSerializationIdRecordName) == end()` before
        // computing).
        idPlan.padding = checkpoint::distcp::BuildFbExtra(zip.m_archive_size, idPlan.fullName.size(), idPlan.size);
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
                                        idPlan.padding.data(),
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

    // Flush the in-memory ZIP to the output stream. `mz_zip_writer_finalize_heap_archive`
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
    std::unique_ptr<char, decltype(&::mz_free)> archiveBuf{static_cast<char*>(archiveBufRaw), &::mz_free};

    // Zero last-mod time/date in every LFH and CD record. PyTorchStreamWriter is
    // built against a vendored miniz with MINIZ_NO_TIME, which makes those
    // fields stable zeros. Our vcpkg miniz lacks that knob (it would require
    // a source rebuild), so we patch the finalized buffer to match.
    char* const bytes = archiveBuf.get();
    for (size_t off : lfhOffsets)
    {
        // LFH: time at +10 (2B), date at +12 (2B). Zero 4 contiguous bytes.
        std::memset(bytes + off + LocalTimestampOffset, 0, TimestampFieldBytes);
    }
    // Walk the central directory: EOCD lives in the last 22 bytes (we emit no
    // comment), CD offset/size in EOCD bytes [12..20). With MZ_ZIP_FLAG_WRITE_ZIP64
    // the EOCD CD fields hold 0xFFFFFFFF sentinels iff the real values overflow
    // 32 bits; in that case the truthy values live in the ZIP64 EOCD record. The
    // upstream `s.length < MZ_UINT32_MAX` guard at the call-site means we don't
    // hit that branch today — but the parser asserts it anyway, since walking
    // garbage would otherwise be a buffer-overrun surface.
    if (archiveSize >= EocdSize)
    {
        char const* eocd = bytes + archiveSize - EocdSize;
        if (std::equal(EocdSignature.begin(), EocdSignature.end(), eocd))
        {
            uint32_t cdSize{};
            uint32_t cdOffset{};
            std::memcpy(&cdSize, eocd + EocdCentralDirSizeOffset, sizeof(cdSize));
            std::memcpy(&cdOffset, eocd + EocdCentralDirOffsetOffset, sizeof(cdOffset));

            if (cdSize == Zip32Sentinel || cdOffset == Zip32Sentinel)
            {
                throw std::runtime_error(fmt::format("distcp: ZIP64 sentinel in EOCD cd_offset/cd_size for '{}'; this means a "
                                                     "shard crossed the 4 GiB boundary that ZIP64 support hasn't been wired in for yet",
                                                     contextName));
            }

            size_t       cursor = cdOffset;
            size_t const cdEnd  = static_cast<size_t>(cdOffset) + static_cast<size_t>(cdSize);
            while (cursor + CentralDirHeaderSize <= cdEnd)
            {
                if (!std::equal(CentralDirSignature.begin(), CentralDirSignature.end(), bytes + cursor))
                {
                    break;    // not a CD entry; stop defensively
                }
                // CD record: time at +12 (2B), date at +14 (2B). Zero 4 bytes.
                std::memset(bytes + cursor + CentralDirTimestampOffset, 0, TimestampFieldBytes);
                uint16_t fnlen{};
                uint16_t extlen{};
                uint16_t cmtlen{};
                std::memcpy(&fnlen, bytes + cursor + CentralDirFilenameLengthOffset, sizeof(fnlen));
                std::memcpy(&extlen, bytes + cursor + CentralDirExtraLengthOffset, sizeof(extlen));
                std::memcpy(&cmtlen, bytes + cursor + CentralDirCommentLengthOffset, sizeof(cmtlen));
                cursor += CentralDirHeaderSize + fnlen + extlen + cmtlen;
            }
        }
    }

    WriteBytes(output, bytes, archiveSize, contextName);

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

static void EmitNewObjBuildBegin(ByteBuffer& b, std::string_view mod, std::string_view name)
{
    checkpoint::distcp::PickleGlobal(b, mod, name);
    PutU8(b, ')');                   // EMPTY_TUPLE
    PutU8(b, PickleNewObjOpcode);    // NEWOBJ — push cls.__new__(cls, *())
    PutU8(b, '}');                   // EMPTY_DICT — start of state dict
    PutU8(b, '(');                   // MARK
}

static void EmitNewObjBuildEnd(ByteBuffer& b)
{
    PutU8(b, 'u');    // SETITEMS — fold marked (key, value) pairs into the dict
    PutU8(b, 'b');    // BUILD — apply state dict to the instance
}

// torch.Size(<dims-tuple>) via REDUCE.
template <typename TInts> static void EmitTorchSize(ByteBuffer& b, TInts const& dims)
{
    checkpoint::distcp::PickleGlobal(b, "torch", "Size");
    if (dims.size() >= 4) { PutU8(b, '('); }
    for (auto d : dims) { checkpoint::distcp::PickleInt(b, static_cast<uint64_t>(d)); }
    PickleTupleClose(b, dims.size());
    PutU8(b, PickleTuple1Opcode);    // TUPLE1 — wrap the dims tuple as the single ctor arg
    PutU8(b, 'R');                   // REDUCE — call torch.Size(dims_tuple)
}

static void EmitTorchDtype(ByteBuffer& b, DS4Tensor::DType dtype)
{
    // torch.<dtype> is a module-level attribute (singleton); GLOBAL pushes it directly.
    checkpoint::distcp::PickleGlobal(b, "torch", checkpoint::distcp::DTypeName(dtype));
}

static void EmitLayoutStrided(ByteBuffer& b)
{
    // `torch.strided` is a module-level singleton — GLOBAL resolves it directly.
    // dcp.save uses a longer `_get_layout(...)` REDUCE path; this shorter form
    // is semantically equivalent (verified by-hand: `pickle.loads(b'\\x80\\x02ctorch\\nstrided\\n.')`
    // returns the same singleton). `TensorProperties.__setstate__` just assigns
    // it straight to `self.layout` — no enum decode.
    checkpoint::distcp::PickleGlobal(b, "torch", "strided");
}

static void EmitMemoryFormatContiguousEnum(ByteBuffer& b)
{
    // TensorProperties.__getstate__ encodes memory_format via the
    // `_MEM_FORMAT_ENCODING` Enum that lives next to TensorProperties (in
    // `torch.distributed.checkpoint.metadata`, *not* `torch.memory_format`).
    // `__setstate__` compares against `TORCH_CONTIGUOUS_FORMAT` (== 0) and
    // resolves back to the singleton. Emitting the actual `torch.contiguous_format`
    // object here would break the equality check and raise at load time.
    checkpoint::distcp::PickleGlobal(b, KMetadataModule, "_MEM_FORMAT_ENCODING");
    checkpoint::distcp::PickleInt(b, 0);    // TORCH_CONTIGUOUS_FORMAT
    PutU8(b, PickleTuple1Opcode);           // TUPLE1
    PutU8(b, 'R');                          // REDUCE — _MEM_FORMAT_ENCODING(0)
}

static void EmitTensorProperties(ByteBuffer& b, DS4Tensor::DType dtype)
{
    // TensorProperties has a custom __getstate__/__setstate__ that uses a
    // 5-tuple (not a dict): (dtype, layout, requires_grad, mem_format_encoding,
    // pin_memory). The MARK..TUPLE..BUILD shape feeds that tuple directly into
    // __setstate__.
    checkpoint::distcp::PickleGlobal(b, KMetadataModule, "TensorProperties");
    PutU8(b, ')');                   // EMPTY_TUPLE
    PutU8(b, PickleNewObjOpcode);    // NEWOBJ
    PutU8(b, '(');                   // MARK
    EmitTorchDtype(b, dtype);
    EmitLayoutStrided(b);
    PutU8(b, PickleNewFalseOpcode);    // NEWFALSE (requires_grad)
    EmitMemoryFormatContiguousEnum(b);
    PutU8(b, PickleNewFalseOpcode);    // NEWFALSE (pin_memory)
    PutU8(b, 't');                     // TUPLE — fold the marked values into a 5-tuple
    PutU8(b, 'b');                     // BUILD — apply via __setstate__
}

static void EmitChunkStorageMetadata(ByteBuffer& b, DS4Tensor::DimRange const& sharddim)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "ChunkStorageMetadata");

    checkpoint::distcp::PickleBinunicode(b, "offsets");
    DS4Tensor::Dim offsets;
    for (auto const& r : sharddim) { offsets.push_back(r.begin); }
    EmitTorchSize(b, offsets);

    checkpoint::distcp::PickleBinunicode(b, "sizes");
    DS4Tensor::Dim sizes;
    for (auto const& r : sharddim) { sizes.push_back(r.end - r.begin); }
    EmitTorchSize(b, sizes);

    EmitNewObjBuildEnd(b);
}

static void EmitTensorStorageMetadata(ByteBuffer&                             b,
                                      DS4Tensor::DType                        dtype,
                                      DS4Tensor::Dim const&                   fulldim,
                                      std::vector<DS4Tensor::DimRange> const& chunks)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "TensorStorageMetadata");

    checkpoint::distcp::PickleBinunicode(b, "properties");
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

static void EmitMetadataIndex(ByteBuffer& b, std::string_view fqn, DS4Tensor::DimRange const& sharddim)
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

static void EmitStorageInfo(ByteBuffer& b, std::string_view relativePath, uint64_t offset, uint64_t length)
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
static void EmitBytesStorageMetadata(ByteBuffer& b)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "BytesStorageMetadata");
    EmitNewObjBuildEnd(b);
}

// MetadataIndex for a bytes (ShardedObject) entry. Unlike the tensor index, a
// bytes index carries no offset/index — both default to None, which torch's
// writer omits — so the restored key equals MetadataIndex(fqn).
static void EmitMetadataIndexBytes(ByteBuffer& b, std::string_view fqn)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "MetadataIndex");
    PickleBinunicode(b, "fqn");
    PickleBinunicode(b, fqn);
    EmitNewObjBuildEnd(b);
}

static std::vector<DS4Tensor::DimRange> CollectChunkRanges(StateNode::ShardedTensor const& tensor)
{
    std::vector<DS4Tensor::DimRange> chunkRanges;
    chunkRanges.reserve(tensor.shards.size());
    for (auto const& shard : tensor.shards)
    {
        if (shard.ShardSizeBytes(tensor.dtype) == 0) { continue; }
        // Replicas can be separated by rank ordering, so compare with every
        // range already collected rather than only the previous shard.
        if (std::ranges::find(chunkRanges, shard.sharddim) == chunkRanges.end()) { chunkRanges.push_back(shard.sharddim); }
    }
    return chunkRanges;
}

static ByteBuffer BuildMetadataPkl(StateTree const& globalRegistry, BlobMetadataByFqn const& blobs)
{
    ByteBuffer b;
    b.reserve(2048);
    std::unordered_set<std::string> tensorFqns;

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
        auto const& tnsr        = std::get<StateNode::ShardedTensor>(node.data);
        auto const  chunkRanges = CollectChunkRanges(tnsr);
        if (chunkRanges.empty()) { return; }
        auto const fqn = globalRegistry.GetNodeId(node.handle);
        tensorFqns.insert(fqn);
        PickleBinunicode(b, fqn);
        EmitTensorStorageMetadata(b, tnsr.dtype, tnsr.fulldim, chunkRanges);
    });
    // Blobs are opaque torch.save'd bytes. Tensor and Blob fqns share
    // one metadata namespace, so reject an accidental collision explicitly.
    for (auto const& [fqn, copies] : blobs)
    {
        if (copies.empty()) { throw std::runtime_error(fmt::format("distcp: Blob fqn '{}' has no owning rank", fqn)); }
        auto const expectedLength = copies.begin()->second.length;
        for (auto const& [rank, meta] : copies)
        {
            if (meta.length != expectedLength)
            {
                throw std::runtime_error(fmt::format("distcp: replicated Blob fqn '{}' has inconsistent lengths {} and {} on rank {}",
                                                     fqn,
                                                     expectedLength,
                                                     meta.length,
                                                     rank));
            }
        }
        if (tensorFqns.contains(fqn)) { throw std::runtime_error(fmt::format("distcp: fqn '{}' is both a tensor and a Blob", fqn)); }
        PickleBinunicode(b, fqn);
        EmitBytesStorageMetadata(b);
    }
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
    // Each rank writes every local Blob in fqn order, including replicas. The
    // FQN-first view lets us advance every rank's physical cursor in that same
    // order while selecting the lowest-rank replica for metadata.
    std::unordered_map<Rank, uint64_t> blobRegionCursors;
    for (auto const& [fqn, copies] : blobs)
    {
        auto const ownerRank = copies.begin()->first;
        for (auto const& [rank, meta] : copies)
        {
            auto& regionCursor = blobRegionCursors[rank];
            if (rank == ownerRank)
            {
                EmitMetadataIndexBytes(b, fqn);
                EmitStorageInfo(
                    b, fmt::format("__{}_0.distcp", static_cast<uint32_t>(rank)), rankCursors[rank] + regionCursor, meta.length);
            }
            regionCursor += meta.length;
        }
    }
    PutU8(b, 'u');    // SETITEMS

    // `planner_data`, `storage_meta`, `version` omitted — dataclass defaults
    // handle them (loader tolerates absence — verified in step-9 probes).

    EmitNewObjBuildEnd(b);
    PutU8(b, '.');    // STOP
    return b;
}

void WriteRankShard(std::filesystem::path const& ckptDir,
                    Rank                         rank,
                    StateTree const&             registry,
                    ByteStorage const&           tensorStorage,
                    BlobsByFqn const&            blobs)
{
    auto const filename = fmt::format("__{}_0.distcp", static_cast<uint32_t>(rank));
    auto const filepath = ckptDir / filename;

    bool hasPayload = !blobs.empty();
    if (!hasPayload)
    {
        registry.WalkNodes([&](StateNode const& node) {
            if (hasPayload || static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::ShardedTensor) { return; }
            auto const& tnsr = std::get<StateNode::ShardedTensor>(node.data);
            // A zero-element tensor is still a DCP write item and therefore
            // still produces a torch.save archive in the NVRx writer.
            hasPayload = std::ranges::any_of(tnsr.shards, [&](auto const& shard) { return shard.rank == rank; });
        });
    }
    if (!hasPayload)
    {
        // NVRx creates a file only for a non-empty local write bucket. Remove
        // an older partial/stale path so a retried save has the same file set.
        std::error_code ec;
        std::filesystem::remove(filepath, ec);
        if (ec) { throw std::system_error(ec, fmt::format("distcp: remove({}) failed", filepath.string())); }
        return;
    }

    std::filesystem::create_directories(ckptDir);

    auto        output = OpenOutputFile(filepath);
    uint64_t    cursor = 0;
    auto const* base   = static_cast<char const*>(tensorStorage.data);
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
            uint64_t const actual    = WriteShardArchive(output, tnsr.dtype, shape, base + shard.handle, size, fqn);
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

    // Append this rank's Blobs after its tensors in fqn order, matching
    // the per-rank order used by BuildMetadataPkl.
    for (auto const& [fqn, bytes] : blobs)
    {
        WriteBytes(output, bytes.data(), bytes.size(), filepath.string());
        cursor += bytes.size();
    }
    FinishOutputFile(output, filepath);
}

void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry, BlobMetadataByFqn const& blobs)
{
    auto const pkl = BuildMetadataPkl(globalRegistry, blobs);

    std::filesystem::create_directories(ckptDir);
    auto const filepath = ckptDir / ".metadata";

    auto output = OpenOutputFile(filepath);
    WriteBytes(output, pkl.data(), pkl.size(), filepath.string());
    FinishOutputFile(output, filepath);

    // Megatron uses this marker to recognize the directory as a torch_dist
    // checkpoint. Publish it from the daemon alongside .metadata so the
    // training process does not write checkpoint sidecars in the exposed path.
    static constexpr std::string_view ConfigJson   = R"({"sharded_backend": "torch_dist", "sharded_backend_version": 1})";
    auto const                        configPath   = ckptDir / "metadata.json";
    auto                              configOutput = OpenOutputFile(configPath);
    WriteBytes(configOutput, ConfigJson.data(), ConfigJson.size(), configPath.string());
    FinishOutputFile(configOutput, configPath);
}

}    // namespace checkpoint::distcp
