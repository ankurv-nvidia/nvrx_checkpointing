#include "distcp.h"
#include "distcp_internal.h"
#include "host_dequant.h"
#include "state.h"

#define MINIZ_NO_ZLIB_APIS
#include <miniz/miniz.h>
#include <zlib.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "distcp writer assumes a little-endian host: archive/byteorder is hardcoded "
              "to \"little\" and the LE serializers are direct memcpys.");

// Each rank shard stores tensor archives and raw scalar and blob payloads.

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

static std::string_view DTypeName(Tensor::DType dtype)
{
    using DT = Tensor::DType;
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
    case DT::FP8E4M3: return "fp8e4m3";
    case DT::MXFP8: return "mxfp8";
    case DT::NVFP4: return "nvfp4";
    case DT::Invalid: [[fallthrough]];
    default: throw std::runtime_error(fmt::format("distcp: unsupported dtype {}", static_cast<int>(dtype)));
    }
}
namespace
{
// A snapshotted non-tensor state value.
struct BlobMeta
{ uint64_t length{}; };

// FQN-first ordering matches the physical order used by every rank's shard.
// The ordered inner map makes the lowest-rank replica the deterministic owner.
using BlobMetaByRank    = std::map<Rank, BlobMeta>;
using BlobMetadataByFqn = std::map<std::string, BlobMetaByRank>;

// One rank's per-save Blob payloads (fqn-sorted). The referenced registry
// and snapshot values outlive WriteRankShard.
using BlobsByFqn = std::map<std::string, std::string_view>;

struct DtypePickleInfo
{
    bool             useV3;
    std::string_view storageMod;
    std::string_view storageName;
    std::string_view dtypeMod;
    std::string_view dtypeName;
};
}    // namespace

static DtypePickleInfo PickleInfoForDtype(Tensor::DType dtype)
{
    using DT = Tensor::DType;
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
    case DT::FP8E4M3:
    case DT::MXFP8:
    case DT::NVFP4:
    case DT::Invalid: [[fallthrough]];
    default: throw std::runtime_error(fmt::format("distcp: unsupported dtype {}", static_cast<int>(dtype)));
    }
}

static void PutU8(ByteBuffer& b, uint8_t v)
{ b.push_back(static_cast<char>(v)); }

static void PutU16LE(ByteBuffer& b, uint16_t v)
{
    b.push_back(static_cast<char>(v & ByteMask));
    b.push_back(static_cast<char>(v >> 8u));
}

static void PutU32LE(ByteBuffer& b, uint32_t v)
{
    for (unsigned i = 0; i < 4; ++i) { b.push_back(static_cast<char>((v >> (i * 8u)) & ByteMask)); }
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
            x >>= 8u;
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

static ByteBuffer BuildDataPkl(Tensor::DType dtype, Tensor::Dim const& shape)
{
    auto const info = PickleInfoForDtype(dtype);

    uint64_t numel = 1;
    for (auto d : shape) { numel *= d; }

    Tensor::Dim stride;
    for (size_t i = 0; i < shape.size(); ++i) { stride.push_back(0); }
    if (!shape.empty())
    {
        stride[shape.size() - 1] = 1;
        for (size_t i = shape.size() - 1; i-- > 0;) { stride[i] = stride[i + 1] * shape[i + 1]; }
    }

    uint64_t const storageSize = info.useV3 ? numel * Tensor::GetDTypeSize(dtype) : numel;

    ByteBuffer out;
    out.reserve(InitialPickleReserveBytes);

    uint8_t memo    = 0;
    auto    memoize = [&] { PickleBinput(out, memo++); };

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

// Archive bodies use PyTorch's 64-byte FB padding and ZIP64 trailer.
constexpr uint64_t   KFieldAlignment                = 64;
constexpr uint64_t   LocalFileHeaderSize            = 30;
constexpr uint64_t   CentralDirHeaderSize           = 46;
constexpr uint64_t   FbExtraHeaderBytes             = sizeof(uint16_t) * 2;
constexpr uint64_t   EocdSize                       = 22;
constexpr uint64_t   Zip64EocdRecordSize            = 56;
constexpr uint64_t   Zip64EocdLocatorSize           = 20;
constexpr uint64_t   SerializationIdSize            = 40;
constexpr uint64_t   DataDescriptorSize             = 16;
constexpr size_t     LocalTimestampOffset           = 10;
constexpr size_t     CentralDirTimestampOffset      = 12;
constexpr size_t     LocalVersionNeededOffset       = 4;
constexpr size_t     CentralVersionNeededOffset     = 6;
constexpr size_t     LocalCompressionMethodOffset   = 8;
constexpr size_t     CentralCompressionMethodOffset = 10;
constexpr size_t     ZipHeaderFieldBytes            = sizeof(uint16_t);
constexpr size_t     TimestampFieldBytes            = sizeof(uint16_t) * 2;
constexpr size_t     CentralDirFilenameLengthOffset = 28;
constexpr size_t     CentralDirExtraLengthOffset    = 30;
constexpr size_t     CentralDirCommentLengthOffset  = 32;
constexpr std::array CentralDirSignature{'P', 'K', '\x01', '\x02'};
constexpr std::array LocalFileHeaderSignature{'P', 'K', '\x03', '\x04'};
constexpr std::array DataDescriptorSignature{'P', 'K', '\x07', '\x08'};

static uint64_t ComputeFbExtraSize(uint64_t cursor, uint64_t filenameSize)
{
    uint64_t const start       = cursor + LocalFileHeaderSize + filenameSize + FbExtraHeaderBytes;
    uint64_t const mod         = start % KFieldAlignment;
    uint64_t const paddingSize = (mod == 0) ? 0 : (KFieldAlignment - mod);
    return paddingSize + FbExtraHeaderBytes;
}

static ByteBuffer BuildFbExtra(size_t cursor, size_t filenameSize, size_t dataSize)
{
    if (dataSize >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX)
    {
        throw std::runtime_error(fmt::format("distcp: ZIP entry too large for ZIP32 (cursor={}, dataSize={})", cursor, dataSize));
    }
    uint64_t const extraSize   = ComputeFbExtraSize(cursor, filenameSize);
    uint64_t const paddingSize = extraSize - FbExtraHeaderBytes;

    ByteBuffer extra;
    extra.resize(static_cast<size_t>(extraSize), 'Z');
    extra[0] = 'F';
    extra[1] = 'B';
    extra[2] = static_cast<char>(paddingSize & ByteMask);
    extra[3] = static_cast<char>((paddingSize >> 8u) & ByteMask);
    return extra;
}

constexpr std::array<std::string_view, 7> EntryFullNames{
    "archive/data.pkl",
    "archive/.format_version",
    "archive/.storage_alignment",
    "archive/byteorder",
    "archive/data/0",
    "archive/version",
    "archive/.data/serialization_id",
};

// The serialization identifier is excluded from its own name hash.
constexpr std::array<std::string_view, 6> EntryBareNames{
    "data.pkl",
    ".format_version",
    ".storage_alignment",
    "byteorder",
    "data/0",
    "version",
};
constexpr std::string_view FormatVersionBytes    = "1";
constexpr std::string_view StorageAlignmentBytes = "64";
constexpr std::string_view ByteorderBytes        = "little";
constexpr std::string_view VersionBytes          = "3\n";

uint64_t ComputeShardArchiveSize(Tensor::DType dtype, Tensor::Dim const& shape, uint64_t shardDataSize)
{
    if (shardDataSize >= MZ_UINT32_MAX)
    {
        throw std::runtime_error(
            fmt::format("distcp: ComputeShardArchiveSize: shardDataSize {} >= 4 GiB (ZIP64 LFH not implemented)", shardDataSize));
    }

    auto const dataPkl = checkpoint::distcp::BuildDataPkl(dtype, shape);

    std::array<uint64_t, 7> const dataSizes{
        dataPkl.size(),
        checkpoint::distcp::FormatVersionBytes.size(),
        checkpoint::distcp::StorageAlignmentBytes.size(),
        checkpoint::distcp::ByteorderBytes.size(),
        shardDataSize,
        checkpoint::distcp::VersionBytes.size(),
        checkpoint::distcp::SerializationIdSize,
    };

    uint64_t cursor = 0;
    for (size_t i = 0; i < checkpoint::distcp::EntryFullNames.size(); ++i)
    {
        uint64_t const fnSize    = checkpoint::distcp::EntryFullNames.at(i).size();
        uint64_t const dataSize  = dataSizes.at(i);
        uint64_t const extraSize = checkpoint::distcp::ComputeFbExtraSize(cursor, fnSize);
        cursor += checkpoint::distcp::LocalFileHeaderSize + fnSize + extraSize + dataSize;
        if (dataSize > 0) { cursor += checkpoint::distcp::DataDescriptorSize; }
    }

    for (auto const& fn : checkpoint::distcp::EntryFullNames) { cursor += checkpoint::distcp::CentralDirHeaderSize + fn.size(); }

    cursor += checkpoint::distcp::Zip64EocdRecordSize + checkpoint::distcp::Zip64EocdLocatorSize + checkpoint::distcp::EocdSize;

    return cursor;
}

static constexpr uint64_t HashCombineConstant  = 0x9e3779b9ULL;
static constexpr uint32_t HashCombineLeftShift = 6U;

static uint64_t HashCombine(uint64_t seed, uint64_t value)
{ return seed ^ (value + HashCombineConstant + (seed << HashCombineLeftShift) + (seed >> 2U)); }

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
namespace
{
struct StreamingZipOutput
{
    std::ostream&    output;    // NOLINT
    std::string_view contextName;
    uint64_t         nextOffset{};
    uint32_t         lastDescriptorCrc{};
    bool             sawDescriptor{};
};
}    // namespace

static size_t StreamZipWrite(void* opaque, mz_uint64 fileOffset, void const* buffer, size_t size)
{
    auto& state = *static_cast<StreamingZipOutput*>(opaque);
    if (fileOffset != state.nextOffset) { return 0; }

    auto const* bytes = static_cast<char const*>(buffer);
    if (size == DataDescriptorSize && std::equal(DataDescriptorSignature.begin(), DataDescriptorSignature.end(), bytes))
    {
        std::memcpy(&state.lastDescriptorCrc, bytes + sizeof(uint32_t), sizeof(state.lastDescriptorCrc));
        state.sawDescriptor = true;
    }

    char const* writeData = bytes;
    ByteBuffer  patched;
    if (size == LocalFileHeaderSize && std::equal(LocalFileHeaderSignature.begin(), LocalFileHeaderSignature.end(), bytes))
    {
        patched.assign(bytes, bytes + size);
        std::memset(patched.data() + LocalVersionNeededOffset, 0, ZipHeaderFieldBytes);
        std::memset(patched.data() + LocalCompressionMethodOffset, 0, ZipHeaderFieldBytes);
        std::memset(patched.data() + LocalTimestampOffset, 0, TimestampFieldBytes);
        writeData = patched.data();
    }
    else if (size >= CentralDirHeaderSize && std::equal(CentralDirSignature.begin(), CentralDirSignature.end(), bytes))
    {
        patched.assign(bytes, bytes + size);
        size_t cursor = 0;
        while (cursor + CentralDirHeaderSize <= patched.size()
               && std::equal(CentralDirSignature.begin(), CentralDirSignature.end(), patched.data() + cursor))
        {
            std::memset(patched.data() + cursor + CentralVersionNeededOffset, 0, ZipHeaderFieldBytes);
            std::memset(patched.data() + cursor + CentralCompressionMethodOffset, 0, ZipHeaderFieldBytes);
            std::memset(patched.data() + cursor + CentralDirTimestampOffset, 0, TimestampFieldBytes);
            uint16_t filenameLength{};
            uint16_t extraLength{};
            uint16_t commentLength{};
            std::memcpy(&filenameLength, patched.data() + cursor + CentralDirFilenameLengthOffset, sizeof(filenameLength));
            std::memcpy(&extraLength, patched.data() + cursor + CentralDirExtraLengthOffset, sizeof(extraLength));
            std::memcpy(&commentLength, patched.data() + cursor + CentralDirCommentLengthOffset, sizeof(commentLength));
            cursor += CentralDirHeaderSize + filenameLength + extraLength + commentLength;
        }
        writeData = patched.data();
    }

    try
    {
        WriteBytes(state.output, writeData, size, state.contextName);
    } catch (...) { return 0; }
    state.nextOffset += size;
    return size;
}

static void FinishOutputFile(std::ofstream& output, std::filesystem::path const& path)
{
    output.flush();
    if (!output) { throw std::runtime_error(fmt::format("distcp: flush({}) failed", path.string())); }
    output.close();
    if (output.fail()) { throw std::runtime_error(fmt::format("distcp: close({}) failed", path.string())); }
}

static uint64_t LogicalShardSizeBytes(TensorShard const& shard, Tensor::DType dtype)
{
    uint64_t size = Tensor::GetDTypeSize(dtype);
    for (auto const& range : shard.sharddim) { size *= range.end - range.begin; }
    return size;
}
namespace
{
struct CBData
{
    std::span<char const> snapshotBytes;
    TensorShard           shard;
    DType                 durableDtype;
    uint64_t              bytesRead{};
    std::exception_ptr    failure{};
};
}    // namespace

static size_t ReadDurableTensorBytes(void* cbDataPtr, mz_uint64 fileOffset, void* buffer, size_t size)
{
    auto& cbData = *static_cast<CBData*>(cbDataPtr);
    if (cbData.failure != nullptr) { return 0; }
    try
    {
        if (fileOffset != cbData.bytesRead) { throw std::runtime_error("distcp: non-sequential logical tensor read"); }
        auto const logicalSize = LogicalShardSizeBytes(cbData.shard, cbData.durableDtype);
        if (cbData.bytesRead > logicalSize) { throw std::runtime_error("distcp: quantized read exceeds logical tensor size"); }
        if ((cbData.bytesRead % sizeof(uint16_t)) != 0) { throw std::runtime_error("distcp: unaligned quantized streaming offset"); }
        auto const readSize = static_cast<size_t>(std::min<uint64_t>(size, logicalSize - cbData.bytesRead));
        if ((readSize % sizeof(uint16_t)) != 0) { throw std::runtime_error("distcp: unaligned quantized streaming read"); }
        checkpoint::host_dequant::DequantizeToBFloat16Range(
            cbData.snapshotBytes,
            cbData.bytesRead / sizeof(uint16_t),
            std::span<uint16_t>(static_cast<uint16_t*>(buffer), readSize / sizeof(uint16_t)));
        cbData.bytesRead += readSize;
        return readSize;
    } catch (...)
    {
        cbData.failure = std::current_exception();
        return 0;
    }
}

static uint64_t WriteShardArchive(std::ostream&         output,
                                  Tensor::DType         durableDtype,
                                  Tensor::Dim const&    shape,
                                  TensorShard const&    shard,
                                  std::span<char const> snapshot,
                                  std::string_view      contextName)
{
    ByteBuffer dataPkl     = checkpoint::distcp::BuildDataPkl(durableDtype, shape);
    auto const logicalSize = LogicalShardSizeBytes(shard, durableDtype);
    auto const storageSize = shard.StorageSizeBytes();
    if (shard.handle > snapshot.size() || storageSize > snapshot.size() - shard.handle)
    {
        throw std::runtime_error(fmt::format("distcp: shard '{}' storage range [{}:{}) exceeds snapshot size {}",
                                             contextName,
                                             shard.handle,
                                             shard.handle + storageSize,
                                             snapshot.size()));
    }
    auto const storageBytes = snapshot.subspan(static_cast<size_t>(shard.handle), static_cast<size_t>(storageSize));
    bool const dequantize   = durableDtype != shard.snapshotDtype;
    if (dequantize)
    {
        if (durableDtype != Tensor::DType::BFloat16
            || (shard.snapshotDtype != Tensor::DType::FP8E4M3 && shard.snapshotDtype != Tensor::DType::MXFP8
                && shard.snapshotDtype != Tensor::DType::NVFP4)
            || !checkpoint::host_dequant::HasStorageHeader(storageBytes))
        {
            throw std::runtime_error(fmt::format("distcp: shard '{}' has incompatible durable and snapshot dtypes", contextName));
        }
    }
    else if (storageSize != logicalSize)
    {
        throw std::runtime_error(
            fmt::format("distcp: shard '{}' has {} physical bytes and {} logical bytes", contextName, storageSize, logicalSize));
    }

    struct EntryPlan
    {
        std::string fullName;
        void const* data;
        size_t      size;
        bool        streamData{};
        ByteBuffer  padding;
    };

    static constexpr size_t DataPklIdx    = 0;
    static constexpr size_t FormatVerIdx  = 1;
    static constexpr size_t StorageAlnIdx = 2;
    static constexpr size_t ByteOrderIdx  = 3;
    static constexpr size_t DataZeroIdx   = 4;
    static constexpr size_t VersionIdx    = 5;
    static constexpr size_t SerialIdIdx   = 6;

    std::array<EntryPlan, checkpoint::distcp::EntryFullNames.size() - 1> entries{
        EntryPlan{
            .fullName = std::string{checkpoint::distcp::EntryFullNames[DataPklIdx]},
            .data     = dataPkl.data(),
            .size     = dataPkl.size(),
            .padding  = {},
        },
        EntryPlan{
            .fullName = std::string{checkpoint::distcp::EntryFullNames[FormatVerIdx]},
            .data     = checkpoint::distcp::FormatVersionBytes.data(),
            .size     = checkpoint::distcp::FormatVersionBytes.size(),
            .padding  = {},
        },
        EntryPlan{
            .fullName = std::string{checkpoint::distcp::EntryFullNames[StorageAlnIdx]},
            .data     = checkpoint::distcp::StorageAlignmentBytes.data(),
            .size     = checkpoint::distcp::StorageAlignmentBytes.size(),
            .padding  = {},
        },
        EntryPlan{
            .fullName = std::string{checkpoint::distcp::EntryFullNames[ByteOrderIdx]},
            .data     = checkpoint::distcp::ByteorderBytes.data(),
            .size     = checkpoint::distcp::ByteorderBytes.size(),
            .padding  = {},
        },
        EntryPlan{
            .fullName   = std::string{checkpoint::distcp::EntryFullNames[DataZeroIdx]},
            .data       = storageBytes.data(),
            .size       = logicalSize,
            .streamData = dequantize,
            .padding    = {},
        },
        EntryPlan{
            .fullName = std::string{checkpoint::distcp::EntryFullNames[VersionIdx]},
            .data     = checkpoint::distcp::VersionBytes.data(),
            .size     = checkpoint::distcp::VersionBytes.size(),
            .padding  = {},
        },
    };

    mz_zip_archive     zip{};
    StreamingZipOutput streamOutput{.output = output, .contextName = contextName};
    zip.m_pWrite     = &StreamZipWrite;
    zip.m_pIO_opaque = &streamOutput;
    if (mz_zip_writer_init_v2(&zip, /*existing_size=*/0, MZ_ZIP_FLAG_WRITE_ZIP64) != MZ_TRUE)
    {
        throw std::runtime_error(fmt::format("distcp: mz_zip_writer_init_v2 failed for '{}'", contextName));
    }

    uint64_t                        combinedCrc = 0;
    std::unordered_set<std::string> bareNamesWritten;
    CBData                          cbData{
        .snapshotBytes = storageBytes,
        .shard         = shard,
        .durableDtype  = durableDtype,
        .bytesRead     = 0,
    };

    auto addEntry = [&](EntryPlan& plan, std::string_view bareName) {
        plan.padding               = checkpoint::distcp::BuildFbExtra(zip.m_archive_size, plan.fullName.size(), plan.size);
        streamOutput.sawDescriptor = false;
        mz_uint32 entryCrc{};
        mz_bool   added{};
        if (plan.streamData)
        {
            added = mz_zip_writer_add_read_buf_callback(&zip,
                                                        plan.fullName.c_str(),
                                                        &ReadDurableTensorBytes,
                                                        &cbData,
                                                        plan.size,
                                                        /*pFile_time=*/nullptr,
                                                        /*pComment=*/nullptr,
                                                        /*comment_size=*/0,
                                                        /*level_and_flags=*/MZ_NO_COMPRESSION,
                                                        /*user_extra_data=*/plan.padding.data(),
                                                        /*user_extra_data_len=*/static_cast<mz_uint16>(plan.padding.size()),
                                                        /*user_extra_data_central=*/nullptr,
                                                        /*user_extra_data_central_len=*/0);
            if (cbData.failure != nullptr) { std::rethrow_exception(cbData.failure); }
            if (cbData.bytesRead != plan.size)
            {
                throw std::runtime_error(fmt::format(
                    "distcp: logical tensor read for '{}' produced {} of {} bytes", plan.fullName, cbData.bytesRead, plan.size));
            }
        }
        else
        {
            entryCrc = static_cast<mz_uint32>(
                ::crc32_z(/*crc=*/0L, reinterpret_cast<Bytef const*>(plan.data), static_cast<z_size_t>(plan.size)));    // NOLINT
            added = mz_zip_writer_add_mem_ex_v2(&zip,
                                                plan.fullName.c_str(),
                                                plan.data,
                                                plan.size,
                                                /*pComment=*/nullptr,
                                                /*comment_size=*/0,
                                                /*level_and_flags=*/MZ_ZIP_FLAG_COMPRESSED_DATA,
                                                /*uncomp_size=*/plan.size,
                                                /*uncomp_crc32=*/entryCrc,
                                                /*last_modified=*/nullptr,
                                                /*user_extra_data=*/plan.padding.data(),
                                                /*user_extra_data_len=*/static_cast<mz_uint16>(plan.padding.size()),
                                                /*user_extra_data_central=*/nullptr,
                                                /*user_extra_data_central_len=*/0);
        }
        if (added != MZ_TRUE) { throw std::runtime_error(fmt::format("distcp: miniz failed to add '{}'", plan.fullName)); }

        if (!streamOutput.sawDescriptor)
        {
            throw std::runtime_error(fmt::format("distcp: miniz omitted data descriptor for '{}'", plan.fullName));
        }
        if (plan.streamData) { entryCrc = streamOutput.lastDescriptorCrc; }
        else if (streamOutput.lastDescriptorCrc != entryCrc)
        {
            throw std::runtime_error(fmt::format("distcp: data descriptor CRC mismatch for '{}'", plan.fullName));
        }
        combinedCrc = HashCombine(combinedCrc, entryCrc);
        bareNamesWritten.insert(std::string{bareName});
    };

    try
    {
        for (size_t i = 0; i < entries.size(); ++i) { addEntry(entries.at(i), checkpoint::distcp::EntryBareNames.at(i)); }

        uint64_t combinedRecordNameHash = 0;
        for (auto const& name : bareNamesWritten) { combinedRecordNameHash = HashCombine(combinedRecordNameHash, StdHashString(name)); }

        std::string const serializationId = fmt::format("{:020d}{:020d}", combinedRecordNameHash, combinedCrc);

        EntryPlan idPlan{
            .fullName = std::string{checkpoint::distcp::EntryFullNames[SerialIdIdx]},
            .data     = serializationId.data(),
            .size     = serializationId.size(),
            .padding  = {},
        };
        idPlan.padding = checkpoint::distcp::BuildFbExtra(zip.m_archive_size, idPlan.fullName.size(), idPlan.size);
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

    if (mz_zip_writer_finalize_archive(&zip) != MZ_TRUE)
    {
        mz_zip_writer_end(&zip);
        throw std::runtime_error(fmt::format("distcp: mz_zip_writer_finalize_archive failed for '{}'", contextName));
    }
    auto const archiveSize = static_cast<size_t>(zip.m_archive_size);
    mz_zip_writer_end(&zip);

    return archiveSize;
}

constexpr std::string_view KMetadataModule   = "torch.distributed.checkpoint.metadata";
constexpr std::string_view KFilesystemModule = "torch.distributed.checkpoint.filesystem";

static void EmitNewObjBuildBegin(ByteBuffer& b, std::string_view mod, std::string_view name)
{
    checkpoint::distcp::PickleGlobal(b, mod, name);
    PutU8(b, ')');
    PutU8(b, PickleNewObjOpcode);
    PutU8(b, '}');
    PutU8(b, '(');
}

static void EmitNewObjBuildEnd(ByteBuffer& b)
{
    PutU8(b, 'u');
    PutU8(b, 'b');
}

template <typename TInts> static void EmitTorchSize(ByteBuffer& b, TInts const& dims)
{
    checkpoint::distcp::PickleGlobal(b, "torch", "Size");
    if (dims.size() >= 4) { PutU8(b, '('); }
    for (auto d : dims) { checkpoint::distcp::PickleInt(b, static_cast<uint64_t>(d)); }
    PickleTupleClose(b, dims.size());
    PutU8(b, PickleTuple1Opcode);
    PutU8(b, 'R');
}

static void EmitTorchDtype(ByteBuffer& b, Tensor::DType dtype)
{ checkpoint::distcp::PickleGlobal(b, "torch", checkpoint::distcp::DTypeName(dtype)); }

static void EmitLayoutStrided(ByteBuffer& b)
{ checkpoint::distcp::PickleGlobal(b, "torch", "strided"); }

static void EmitMemoryFormatContiguousEnum(ByteBuffer& b)
{
    checkpoint::distcp::PickleGlobal(b, KMetadataModule, "_MEM_FORMAT_ENCODING");
    checkpoint::distcp::PickleInt(b, 0);
    PutU8(b, PickleTuple1Opcode);
    PutU8(b, 'R');
}

static void EmitTensorProperties(ByteBuffer& b, Tensor::DType dtype)
{
    checkpoint::distcp::PickleGlobal(b, KMetadataModule, "TensorProperties");
    PutU8(b, ')');
    PutU8(b, PickleNewObjOpcode);
    PutU8(b, '(');
    EmitTorchDtype(b, dtype);
    EmitLayoutStrided(b);
    PutU8(b, PickleNewFalseOpcode);
    EmitMemoryFormatContiguousEnum(b);
    PutU8(b, PickleNewFalseOpcode);
    PutU8(b, 't');
    PutU8(b, 'b');
}

static void EmitChunkStorageMetadata(ByteBuffer& b, Tensor::DimRange const& sharddim)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "ChunkStorageMetadata");

    checkpoint::distcp::PickleBinunicode(b, "offsets");
    Tensor::Dim offsets;
    for (auto const& r : sharddim) { offsets.push_back(r.begin); }
    EmitTorchSize(b, offsets);

    checkpoint::distcp::PickleBinunicode(b, "sizes");
    Tensor::Dim sizes;
    for (auto const& r : sharddim) { sizes.push_back(r.end - r.begin); }
    EmitTorchSize(b, sizes);

    EmitNewObjBuildEnd(b);
}

static void
EmitTensorStorageMetadata(ByteBuffer& b, Tensor::DType dtype, Tensor::Dim const& fulldim, std::vector<Tensor::DimRange> const& chunks)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "TensorStorageMetadata");

    checkpoint::distcp::PickleBinunicode(b, "properties");
    EmitTensorProperties(b, dtype);

    PickleBinunicode(b, "size");
    EmitTorchSize(b, fulldim);

    PickleBinunicode(b, "chunks");
    PutU8(b, ']');
    PutU8(b, '(');
    for (auto const& chunk : chunks) { EmitChunkStorageMetadata(b, chunk); }
    PutU8(b, 'e');

    EmitNewObjBuildEnd(b);
}

static void EmitMetadataIndex(ByteBuffer& b, std::string_view fqn, Tensor::DimRange const& sharddim)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "MetadataIndex");

    PickleBinunicode(b, "fqn");
    PickleBinunicode(b, fqn);

    PickleBinunicode(b, "offset");
    Tensor::Dim offsets;
    for (auto const& r : sharddim) { offsets.push_back(r.begin); }
    EmitTorchSize(b, offsets);

    PickleBinunicode(b, "index");
    PickleInt(b, 0);

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

    EmitNewObjBuildEnd(b);
}

static std::string_view ScalarTypeName(StateNode::Scalar const& scalar)
{
    auto const type = scalar.GetType();
    if (Primitives64Bit::Type::IsBoolean(type)) { return "bool"; }
    if (Primitives64Bit::Type::IsUnsigned(type)) { return "uint64"; }
    if (Primitives64Bit::Type::IsSigned(type)) { return "int64"; }
    if (Primitives64Bit::Type::IsFloat(type)) { return "float64"; }
    throw std::runtime_error("distcp: unsupported scalar type");
}

static void EmitBytesStorageMetadata(ByteBuffer& b, StateNode::Scalar const* scalar)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "BytesStorageMetadata");
    if (scalar != nullptr)
    {
        PickleBinunicode(b, "nvrx_scalar_type");
        PickleBinunicode(b, ScalarTypeName(*scalar));
    }
    EmitNewObjBuildEnd(b);
}

static void EmitMetadataIndexBytes(ByteBuffer& b, std::string_view fqn)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "MetadataIndex");
    PickleBinunicode(b, "fqn");
    PickleBinunicode(b, fqn);
    EmitNewObjBuildEnd(b);
}

static std::vector<Tensor::DimRange> CollectChunkRanges(DTensor const& tensor)
{
    std::vector<Tensor::DimRange> chunkRanges;
    chunkRanges.reserve(tensor.shards.size());
    for (auto const& shard : tensor.shards)
    {
        if (LogicalShardSizeBytes(shard, tensor.durableDtype) == 0) { continue; }
        if (std::ranges::find(chunkRanges, shard.sharddim) == chunkRanges.end()) { chunkRanges.push_back(shard.sharddim); }
    }
    return chunkRanges;
}

static std::string BlobShardKey(std::string_view key, Tensor::Dim const& fullShape, Tensor::DimRange const& ranges)
{
    std::string result(key);
    result += "/shard_";
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        if (i != 0) { result += '.'; }
        result += std::to_string(ranges[i].begin);
    }
    result += '_';
    for (size_t i = 0; i < fullShape.size(); ++i)
    {
        if (i != 0) { result += '.'; }
        result += std::to_string(fullShape[i]);
    }
    return result;
}

static BlobsByFqn CollectBlobs(StateTree const& registry, ByteStorage const& storage, Rank rank)
{
    BlobsByFqn blobs;
    auto       add = [&](std::string key, uint64_t handle, uint64_t size) {
        if (handle > storage.size || size > storage.size - handle)
        {
            throw std::out_of_range(fmt::format("Blob '{}' exceeds the snapshot payload", key));
        }
        auto const* data = size == 0 ? "" : static_cast<char const*>(storage.data) + handle;
        if (!blobs.emplace(std::move(key), std::string_view(data, size)).second)
        {
            throw std::runtime_error("Snapshot contains duplicate blob keys");
        }
    };
    registry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::DynamicSizeBlob)
        {
            auto const& blob = std::get<StateNode::DynamicSizeBlob>(node.data);
            if (blob.replicas.contains(rank)) { add(registry.GetNodeId(node.handle), blob.handle, blob.storageSize); }
        }
        else if (type == StateNode::Type::FixedSizeBlob)
        {
            auto const& blob = std::get<StateNode::FixedSizeBlob>(node.data);
            for (auto const& shard : blob.shards)
            {
                if (shard.rank == rank)
                {
                    add(BlobShardKey(registry.GetNodeId(node.handle), blob.fulldim, shard.sharddim), shard.handle, shard.storageSize);
                }
            }
        }
    });
    return blobs;
}

static std::array<char, sizeof(uint64_t)> EncodeScalar(StateNode::Scalar const& scalar)
{
    auto const type = scalar.GetType();
    uint64_t   bits{};
    if (Primitives64Bit::Type::IsFloat(type)) { bits = std::bit_cast<uint64_t>(scalar.Cast<double>()); }
    else if (Primitives64Bit::Type::IsSigned(type)) { bits = std::bit_cast<uint64_t>(scalar.Cast<int64_t>()); }
    else if (Primitives64Bit::Type::IsUnsigned(type) || Primitives64Bit::Type::IsBoolean(type)) { bits = scalar.Cast<uint64_t>(); }
    else
    {
        throw std::runtime_error("distcp: unsupported scalar type");
    }
    std::array<char, sizeof(bits)> bytes{};
    for (size_t index = 0; index < bytes.size(); ++index) { bytes[index] = static_cast<char>((bits >> (index * 8)) & ByteMask); }
    return bytes;
}

static BlobMetadataByFqn CollectBlobMetadata(StateTree const& registry, std::unordered_map<std::string, uint64_t> const& dynamicBlobSizes)
{
    BlobMetadataByFqn blobs;
    registry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type == StateNode::Type::DynamicSizeBlob)
        {
            auto const& blob   = std::get<StateNode::DynamicSizeBlob>(node.data);
            auto const  fqn    = registry.GetNodeId(node.handle);
            auto const  size   = dynamicBlobSizes.find(fqn);
            auto const  length = size == dynamicBlobSizes.end() ? blob.storageSize : size->second;
            for (auto rank : blob.replicas) { blobs[fqn][rank] = {.length = length}; }
        }
        else if (type == StateNode::Type::FixedSizeBlob)
        {
            auto const& blob = std::get<StateNode::FixedSizeBlob>(node.data);
            for (auto const& shard : blob.shards)
            {
                blobs[BlobShardKey(registry.GetNodeId(node.handle), blob.fulldim, shard.sharddim)][shard.rank]
                    = {.length = shard.storageSize};
            }
        }
        else if (type == StateNode::Type::Scalar)
        {
            auto const& scalar = std::get<StateNode::Scalar>(node.data);
            auto const  fqn    = registry.GetNodeId(node.handle);
            for (auto rank : scalar.replicas) { blobs[fqn][rank] = {.length = sizeof(uint64_t)}; }
        }
    });
    return blobs;
}

template <typename TCall> static void WalkTensorNodes(StateTree const& registry, TCall const& call)
{
    std::vector<std::pair<std::string, StateNode const*>> nodes;
    registry.WalkNodes([&](StateNode const& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::DTensor))
        {
            nodes.emplace_back(registry.GetNodeId(node.handle), &node);
        }
    });
    std::ranges::sort(nodes, {}, [](auto const& entry) { return entry.first; });
    for (auto const& entry : nodes) { call(*entry.second); }
}

static ByteBuffer BuildMetadataPkl(StateTree const& globalRegistry, BlobMetadataByFqn const& blobs)
{
    ByteBuffer b;
    b.reserve(2048);
    std::unordered_set<std::string> tensorFqns;

    PutU8(b, 0x80);
    PutU8(b, 0x02);

    EmitNewObjBuildBegin(b, KMetadataModule, "Metadata");

    PickleBinunicode(b, "state_dict_metadata");
    PutU8(b, '}');
    PutU8(b, '(');
    WalkTensorNodes(globalRegistry, [&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::DTensor) { return; }
        auto const& tnsr        = std::get<DTensor>(node.data);
        auto const  chunkRanges = CollectChunkRanges(tnsr);
        if (chunkRanges.empty()) { return; }
        auto const fqn = globalRegistry.GetNodeId(node.handle);
        tensorFqns.insert(fqn);
        PickleBinunicode(b, fqn);
        EmitTensorStorageMetadata(b, tnsr.durableDtype, tnsr.fulldim, chunkRanges);
    });
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
        auto const  handle = globalRegistry.FindNodeByFQN(fqn);
        auto const* scalar
            = handle == StateNode::Handle::Invalid ? nullptr : std::get_if<StateNode::Scalar>(&globalRegistry.GetNode(handle).data);
        EmitBytesStorageMetadata(b, scalar);
    }
    PutU8(b, 'u');

    PickleBinunicode(b, "storage_data");
    PutU8(b, '}');
    PutU8(b, '(');
    std::unordered_map<Rank, uint64_t> rankCursors;
    WalkTensorNodes(globalRegistry, [&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::DTensor) { return; }
        auto const&                     tnsr = std::get<DTensor>(node.data);
        auto const                      fqn  = globalRegistry.GetNodeId(node.handle);
        std::optional<Tensor::DimRange> lastShard;
        for (auto const& shard : tnsr.shards)
        {
            auto const shardSize = LogicalShardSizeBytes(shard, tnsr.durableDtype);
            if (shardSize == 0) { continue; }
            Tensor::Dim shardShape;
            for (auto const& r : shard.sharddim) { shardShape.push_back(r.end - r.begin); }
            uint64_t const archiveLen = ComputeShardArchiveSize(tnsr.durableDtype, shardShape, shardSize);
            uint64_t&      cursor     = rankCursors[shard.rank];
            uint64_t       curCursor  = cursor;
            cursor += archiveLen;

            if (lastShard && *lastShard == shard.sharddim) { continue; }
            lastShard = shard.sharddim;

            // Replica records advance physical offsets but share one metadata key.
            EmitMetadataIndex(b, fqn, shard.sharddim);
            auto const relPath = fmt::format("__{}_0.distcp", static_cast<uint32_t>(shard.rank));
            EmitStorageInfo(b, relPath, curCursor, archiveLen);
        }
    });
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
    PutU8(b, 'u');

    EmitNewObjBuildEnd(b);
    PutU8(b, '.');
    return b;
}

std::unordered_map<std::string, uint64_t>
WriteRankShard(std::filesystem::path const& ckptDir, Rank rank, StateTree const& registry, ByteStorage const& tensorStorage)
{
    BlobsByFqn                                                blobs = CollectBlobs(registry, tensorStorage, rank);
    std::map<std::string, std::array<char, sizeof(uint64_t)>> scalarData;
    std::unordered_map<std::string, uint64_t>                 dynamicBlobSizes;
    registry.WalkNodes([&](StateNode const& node) {
        if (auto const* blob = std::get_if<StateNode::DynamicSizeBlob>(&node.data))
        {
            if (blob->replicas.contains(rank)) { dynamicBlobSizes.emplace(registry.GetNodeId(node.handle), blob->storageSize); }
        }
        else if (auto const* scalar = std::get_if<StateNode::Scalar>(&node.data); scalar != nullptr && scalar->replicas.contains(rank))
        {
            auto const fqn               = registry.GetNodeId(node.handle);
            auto const [entry, inserted] = scalarData.emplace(fqn, EncodeScalar(*scalar));
            if (!inserted || !blobs.emplace(fqn, std::string_view(entry->second.data(), entry->second.size())).second)
            {
                throw std::runtime_error("Snapshot contains duplicate scalar keys");
            }
        }
    });
    auto const filename = fmt::format("__{}_0.distcp", static_cast<uint32_t>(rank));
    auto const filepath = ckptDir / filename;

    bool hasPayload = !blobs.empty();
    if (!hasPayload)
    {
        WalkTensorNodes(registry, [&](StateNode const& node) {
            if (hasPayload || static_cast<StateNode::Type>(node.data.index()) != StateNode::Type::DTensor) { return; }
            auto const& tnsr = std::get<DTensor>(node.data);
            // Zero-element tensors still produce an archive.
            hasPayload = std::ranges::any_of(tnsr.shards, [&](auto const& shard) { return shard.rank == rank; });
        });
    }
    if (!hasPayload)
    {
        std::error_code ec;
        std::filesystem::remove(filepath, ec);
        if (ec) { throw std::system_error(ec, fmt::format("distcp: remove({}) failed", filepath.string())); }
        return dynamicBlobSizes;
    }

    std::filesystem::create_directories(ckptDir);

    auto        output        = OpenOutputFile(filepath);
    uint64_t    cursor        = 0;
    auto const* base          = static_cast<char const*>(tensorStorage.data);
    auto const  snapshotBytes = std::span<char const>(base, tensorStorage.size);
    WalkTensorNodes(registry, [&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::DTensor) { return; }
        auto const& tnsr = std::get<DTensor>(node.data);
        auto const  fqn  = registry.GetNodeId(node.handle);
        for (auto const& shard : tnsr.shards)
        {
            if (shard.rank != rank) { continue; }
            auto const logicalSize = LogicalShardSizeBytes(shard, tnsr.durableDtype);
            if (logicalSize == 0) { continue; }
            Tensor::Dim shape{};
            for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }

            if (logicalSize >= MZ_UINT32_MAX)
            {
                throw std::runtime_error(
                    fmt::format("distcp: shard '{}' is {} bytes (>= 4 GiB); ZIP64 support is not yet implemented", fqn, logicalSize));
            }
            uint64_t const predicted = ComputeShardArchiveSize(tnsr.durableDtype, shape, logicalSize);
            uint64_t const actual    = WriteShardArchive(output, tnsr.durableDtype, shape, shard, snapshotBytes, fqn);
            if (predicted != actual)
            {
                throw std::runtime_error(fmt::format("distcp: ComputeShardArchiveSize/WriteShardArchive disagree for '{}' "
                                                     "(dtype={}, shape=[{}], dataSize={}): predicted={} actual={}",
                                                     fqn,
                                                     DTypeName(tnsr.durableDtype),
                                                     fmt::join(shape, ","),
                                                     logicalSize,
                                                     predicted,
                                                     actual));
            }
            cursor += actual;
        }
    });

    // Blob payloads follow tensor archives in FQN order.
    for (auto const& [fqn, bytes] : blobs)
    {
        WriteBytes(output, bytes.data(), bytes.size(), filepath.string());
        cursor += bytes.size();
    }
    FinishOutputFile(output, filepath);
    return dynamicBlobSizes;
}

void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& snapshotRegistry)
{ WriteMetadata(ckptDir, snapshotRegistry, {}); }

void WriteMetadata(std::filesystem::path const&                     ckptDir,
                   StateTree const&                                 snapshotRegistry,
                   std::unordered_map<std::string, uint64_t> const& dynamicBlobSizes)
{
    BlobMetadataByFqn const& blobs = CollectBlobMetadata(snapshotRegistry, dynamicBlobSizes);
    auto const               pkl   = BuildMetadataPkl(snapshotRegistry, blobs);

    std::filesystem::create_directories(ckptDir);
    auto const filepath = ckptDir / ".metadata";

    auto output = OpenOutputFile(filepath);
    WriteBytes(output, pkl.data(), pkl.size(), filepath.string());
    FinishOutputFile(output, filepath);

    // The format marker makes the directory loadable as a torch_dist checkpoint.
    static constexpr std::string_view ConfigJson      = R"({"sharded_backend": "torch_dist", "sharded_backend_version": 1})";
    auto const                        configPath      = ckptDir / "metadata.json";
    bool                              hasNativeSchema = false;
    snapshotRegistry.WalkNodes([&](StateNode const& node) {
        constexpr std::string_view Prefix = "__nvrx_native_state_rank_";
        constexpr std::string_view Suffix = "__";
        auto const                 name   = std::string_view(node.name);
        if (node.parent != StateNode::Handle::Root || node.GetType() != StateNode::Type::DynamicSizeBlob || !name.starts_with(Prefix)
            || !name.ends_with(Suffix) || name.size() <= Prefix.size() + Suffix.size())
        {
            return;
        }
        auto const rank = name.substr(Prefix.size(), name.size() - Prefix.size() - Suffix.size());
        if (std::ranges::all_of(rank, [](char digit) { return digit >= '0' && digit <= '9'; })) { hasNativeSchema = true; }
    });
    if (hasNativeSchema)
    {
        std::filesystem::remove(configPath);
        return;
    }
    auto configOutput = OpenOutputFile(configPath);
    WriteBytes(configOutput, ConfigJson.data(), ConfigJson.size(), configPath.string());
    FinishOutputFile(configOutput, configPath);
}

namespace
{
struct PickleValue
{
    enum class Kind : uint8_t
    {
        None,
        Integer,
        String,
        Sequence,
        Dictionary,
        Global,
        Object,
        Mark
    };
    using Ref = std::shared_ptr<PickleValue>;
    Kind                             kind{Kind::None};
    uint64_t                         integer{};
    bool                             negative{};
    std::string                      text;
    std::vector<Ref>                 items;
    std::vector<std::pair<Ref, Ref>> entries;
    Ref                              state;
};

using PickleRef = PickleValue::Ref;

struct PickleReader
{
    std::span<char const>  bytes;
    size_t                 cursor{};
    std::vector<PickleRef> stack;
    std::vector<PickleRef> memo;

    explicit PickleReader(std::span<char const> input LFTBND) : bytes(input) {}

    std::span<char const> Take(uint64_t count)
    {
        if (count > bytes.size() - cursor) { throw std::runtime_error("distcp: truncated pickle metadata"); }
        auto result = bytes.subspan(cursor, static_cast<size_t>(count));
        cursor += static_cast<size_t>(count);
        return result;
    }
    uint64_t Number(size_t width)
    {
        uint64_t   result{};
        auto const data = Take(width);
        for (size_t index = 0; index < width; ++index)
        {
            result |= static_cast<uint64_t>(static_cast<unsigned char>(data[index])) << (index * 8);
        }
        return result;
    }
    std::string Text(uint64_t length)
    {
        auto const data = Take(length);
        return {data.data(), data.size()};
    }
    std::string Line()
    {
        auto const start = cursor;
        while (cursor < bytes.size() && bytes[cursor] != '\n') { ++cursor; }
        if (cursor == bytes.size()) { throw std::runtime_error("distcp: unterminated pickle string"); }
        auto result = std::string(bytes.data() + start, cursor - start);
        ++cursor;
        return result;
    }
    static PickleRef Make(PickleValue::Kind kind)
    {
        auto value  = std::make_shared<PickleValue>();
        value->kind = kind;
        return value;
    }
    void PushText(std::string text, PickleValue::Kind kind = PickleValue::Kind::String)
    {
        auto value  = Make(kind);
        value->text = std::move(text);
        stack.push_back(std::move(value));
    }
    void PushNumber(uint64_t number, bool negative = false)
    {
        auto value      = Make(PickleValue::Kind::Integer);
        value->integer  = number;
        value->negative = negative;
        stack.push_back(std::move(value));
    }
    PickleRef Pop()
    {
        if (stack.empty()) { throw std::runtime_error("distcp: invalid pickle stack"); }
        auto value = stack.back();
        stack.pop_back();
        return value;
    }
    size_t Mark() const
    {
        for (size_t index = stack.size(); index != 0; --index)
        {
            if (stack[index - 1]->kind == PickleValue::Kind::Mark) { return index - 1; }
        }
        throw std::runtime_error("distcp: missing pickle mark");
    }
    void Sequence(size_t count)
    {
        if (count > stack.size()) { throw std::runtime_error("distcp: invalid pickle sequence"); }
        auto       value = Make(PickleValue::Kind::Sequence);
        auto const start = stack.size() - count;
        value->items.assign(stack.begin() + static_cast<std::ptrdiff_t>(start), stack.end());
        stack.resize(start);
        stack.push_back(std::move(value));
    }
    void Store(size_t index)
    {
        if (stack.empty() || index > bytes.size()) { throw std::runtime_error("distcp: invalid pickle memo index"); }
        if (memo.size() <= index) { memo.resize(index + 1); }
        memo[index] = stack.back();
    }
    void Recall(size_t index)
    {
        if (index >= memo.size() || !memo[index]) { throw std::runtime_error("distcp: missing pickle memo value"); }
        stack.push_back(memo[index]);
    }
    void Object()
    {
        auto args = Pop();
        auto type = Pop();
        if (type->kind != PickleValue::Kind::Global || args->kind != PickleValue::Kind::Sequence)
        {
            throw std::runtime_error("distcp: invalid pickle object");
        }
        if (type->text == "torch.Size")
        {
            if (args->items.size() != 1 || args->items[0]->kind != PickleValue::Kind::Sequence)
            {
                throw std::runtime_error("distcp: invalid tensor size metadata");
            }
            stack.push_back(args->items[0]);
            return;
        }
        auto value   = Make(PickleValue::Kind::Object);
        value->text  = type->text;
        value->items = args->items;
        stack.push_back(std::move(value));
    }
    PickleRef Read()
    {
        while (cursor < bytes.size())
        {
            auto const opcode = Number(1);
            switch (opcode)
            {
            case 0x80:
                if (Number(1) > 5) { throw std::runtime_error("distcp: unsupported pickle protocol"); }
                break;
            case 0x95:
                if (Number(8) > bytes.size() - cursor) { throw std::runtime_error("distcp: truncated pickle frame"); }
                break;
            case '.':
                if (stack.size() != 1 || cursor != bytes.size()) { throw std::runtime_error("distcp: invalid pickle result"); }
                return stack.back();
            case '(': stack.push_back(Make(PickleValue::Kind::Mark)); break;
            case 'N': stack.push_back(Make(PickleValue::Kind::None)); break;
            case 0x88: PushNumber(1); break;
            case 0x89: PushNumber(0); break;
            case 'K': PushNumber(Number(1)); break;
            case 'M': PushNumber(Number(2)); break;
            case 'J':
            {
                auto const number = Number(4);
                PushNumber(number, (number & 0x80000000ULL) != 0);
                break;
            }
            case 0x8a:
            case 0x8b:
            {
                auto const length = Number(opcode == 0x8a ? 1 : 4);
                if (length > 9) { throw std::runtime_error("distcp: pickle integer exceeds 64 bits"); }
                auto data  = Take(length);
                auto value = uint64_t{};
                for (size_t index = 0; index < std::min<size_t>(data.size(), 8); ++index)
                {
                    value |= static_cast<uint64_t>(static_cast<unsigned char>(data[index])) << (index * 8);
                }
                if (length == 9 && data[8] != 0) { throw std::runtime_error("distcp: pickle integer exceeds 64 bits"); }
                PushNumber(value, length != 0 && (static_cast<unsigned char>(data.back()) & 0x80U) != 0);
                break;
            }
            case 'G':
                Take(8);
                stack.push_back(Make(PickleValue::Kind::None));
                break;
            case 'X':
            case 'B':
            case 'T': PushText(Text(Number(4))); break;
            case 0x8c:
            case 'C':
            case 'U': PushText(Text(Number(1))); break;
            case 0x8d:
            case 0x8e:
            case 0x96: PushText(Text(Number(8))); break;
            case 'c':
            {
                auto module = Line();
                auto name   = Line();
                PushText(module + "." + name, PickleValue::Kind::Global);
                break;
            }
            case 0x93:
            {
                auto name   = Pop();
                auto module = Pop();
                if (name->kind != PickleValue::Kind::String || module->kind != PickleValue::Kind::String)
                {
                    throw std::runtime_error("distcp: invalid pickle global");
                }
                PushText(module->text + "." + name->text, PickleValue::Kind::Global);
                break;
            }
            case ')':
            case ']': stack.push_back(Make(PickleValue::Kind::Sequence)); break;
            case '}': stack.push_back(Make(PickleValue::Kind::Dictionary)); break;
            case 0x85: Sequence(1); break;
            case 0x86: Sequence(2); break;
            case 0x87: Sequence(3); break;
            case 't':
            case 'l':
            {
                auto const mark = Mark();
                Sequence(stack.size() - mark - 1);
                auto value = Pop();
                Pop();
                stack.push_back(std::move(value));
                break;
            }
            case 'd':
            {
                auto const mark = Mark();
                if (((stack.size() - mark - 1) % 2) != 0) { throw std::runtime_error("distcp: invalid pickle dictionary"); }
                auto value = Make(PickleValue::Kind::Dictionary);
                for (size_t index = mark + 1; index < stack.size(); index += 2)
                {
                    value->entries.emplace_back(stack[index], stack[index + 1]);
                }
                stack.resize(mark);
                stack.push_back(std::move(value));
                break;
            }
            case 'a':
            {
                auto value = Pop();
                if (stack.empty() || stack.back()->kind != PickleValue::Kind::Sequence)
                {
                    throw std::runtime_error("distcp: invalid pickle append");
                }
                stack.back()->items.push_back(std::move(value));
                break;
            }
            case 'e':
            case 'u':
            {
                auto const mark = Mark();
                if (mark == 0) { throw std::runtime_error("distcp: invalid pickle container"); }
                auto& target = stack[mark - 1];
                if (opcode == 'e' && target->kind == PickleValue::Kind::Sequence)
                {
                    target->items.insert(target->items.end(), stack.begin() + static_cast<std::ptrdiff_t>(mark + 1), stack.end());
                }
                else if (opcode == 'u' && target->kind == PickleValue::Kind::Dictionary && ((stack.size() - mark - 1) % 2) == 0)
                {
                    for (size_t index = mark + 1; index < stack.size(); index += 2)
                    {
                        target->entries.emplace_back(stack[index], stack[index + 1]);
                    }
                }
                else
                {
                    throw std::runtime_error("distcp: invalid pickle container contents");
                }
                stack.resize(mark);
                break;
            }
            case 's':
            {
                auto value = Pop();
                auto key   = Pop();
                if (stack.empty() || stack.back()->kind != PickleValue::Kind::Dictionary)
                {
                    throw std::runtime_error("distcp: invalid pickle item");
                }
                stack.back()->entries.emplace_back(std::move(key), std::move(value));
                break;
            }
            case 0x81:
            case 'R': Object(); break;
            case 0x92:
            {
                auto kwargs = Pop();
                if (kwargs->kind != PickleValue::Kind::Dictionary) { throw std::runtime_error("distcp: invalid pickle keyword arguments"); }
                Object();
                break;
            }
            case 'b':
            {
                auto state = Pop();
                if (stack.empty() || stack.back()->kind != PickleValue::Kind::Object)
                {
                    throw std::runtime_error("distcp: invalid pickle object state");
                }
                stack.back()->state = std::move(state);
                break;
            }
            case 'Q':
            {
                auto value  = Make(PickleValue::Kind::Object);
                value->text = "persistent";
                value->items.push_back(Pop());
                stack.push_back(std::move(value));
                break;
            }
            case 'q': Store(static_cast<size_t>(Number(1))); break;
            case 'r': Store(static_cast<size_t>(Number(4))); break;
            case 0x94: Store(memo.size()); break;
            case 'h': Recall(static_cast<size_t>(Number(1))); break;
            case 'j': Recall(static_cast<size_t>(Number(4))); break;
            case '0': Pop(); break;
            case '1': stack.resize(Mark()); break;
            case '2':
                if (stack.empty()) { throw std::runtime_error("distcp: invalid pickle duplicate"); }
                stack.push_back(stack.back());
                break;
            default: throw std::runtime_error(fmt::format("distcp: unsupported pickle opcode {:#x}", opcode));
            }
        }
        throw std::runtime_error("distcp: missing pickle stop");
    }
};

static PickleRef Field(PickleRef const& object, std::string_view name, bool required = true)
{
    auto value = object;
    if (value && value->kind == PickleValue::Kind::Object) { value = value->state; }
    if (value && value->kind == PickleValue::Kind::Dictionary)
    {
        for (auto item = value->entries.rbegin(); item != value->entries.rend(); ++item)
        {
            if (item->first->kind == PickleValue::Kind::String && item->first->text == name) { return item->second; }
        }
    }
    if (required) { throw std::runtime_error(fmt::format("distcp: missing metadata field '{}'", name)); }
    return {};
}

static uint64_t MetadataNumber(PickleRef const& value)
{
    if (!value || value->kind != PickleValue::Kind::Integer || value->negative)
    {
        throw std::runtime_error("distcp: metadata requires an unsigned integer");
    }
    return value->integer;
}

static std::string const& MetadataText(PickleRef const& value)
{
    if (!value || value->kind != PickleValue::Kind::String) { throw std::runtime_error("distcp: metadata requires a string"); }
    return value->text;
}

static Tensor::Dim MetadataDimensions(PickleRef const& value)
{
    if (!value || value->kind != PickleValue::Kind::Sequence || value->items.size() > Tensor::MaxDims)
    {
        throw std::runtime_error("distcp: invalid tensor dimensions");
    }
    Tensor::Dim result;
    for (auto const& dimension : value->items) { result.push_back(MetadataNumber(dimension)); }
    return result;
}

static uint64_t CheckedMultiply(uint64_t left, uint64_t right)
{
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) { throw std::overflow_error("distcp: tensor size overflow"); }
    return left * right;
}

static uint64_t ElementSize(Tensor::DType dtype)
{ return dtype == Tensor::DType::Bool ? uint64_t{1} : Tensor::GetDTypeSize(dtype); }

static uint64_t DenseSize(Tensor::DimRange const& ranges, Tensor::DType dtype)
{
    uint64_t result = ElementSize(dtype);
    for (auto const range : ranges)
    {
        if (range.end < range.begin) { throw std::runtime_error("distcp: inverted tensor range"); }
        result = CheckedMultiply(result, range.end - range.begin);
    }
    return result;
}

static Tensor::DType MetadataDtype(PickleRef const& properties)
{
    auto value = Field(properties, "dtype", false);
    if (!value && properties->state && properties->state->kind == PickleValue::Kind::Sequence && !properties->state->items.empty())
    {
        value = properties->state->items[0];
    }
    if (!value || value->kind != PickleValue::Kind::Global || !value->text.starts_with("torch."))
    {
        throw std::runtime_error("distcp: invalid tensor datatype");
    }
    auto const name = std::string_view(value->text).substr(6);
    for (auto dtype : {Tensor::DType::Float16,
                       Tensor::DType::BFloat16,
                       Tensor::DType::Float32,
                       Tensor::DType::Float64,
                       Tensor::DType::UInt8,
                       Tensor::DType::UInt16,
                       Tensor::DType::UInt32,
                       Tensor::DType::UInt64,
                       Tensor::DType::Int8,
                       Tensor::DType::Int16,
                       Tensor::DType::Int32,
                       Tensor::DType::Int64,
                       Tensor::DType::Bool})
    {
        if (DTypeName(dtype) == name) { return dtype; }
    }
    throw std::runtime_error(fmt::format("distcp: unsupported tensor datatype '{}'", name));
}

struct ReadLocation
{
    std::string path;
    uint64_t    offset{};
    uint64_t    length{};
};

using ReadIndex = std::pair<std::string, std::vector<uint64_t>>;

static ByteStorage ReadFileRange(std::filesystem::path const& directory, ReadLocation const& location)
{
    auto const relative = std::filesystem::path(location.path);
    if (relative.empty() || relative.is_absolute()) { throw std::runtime_error("distcp: invalid checkpoint payload path"); }
    for (auto const& component : relative)
    {
        if (component == "..") { throw std::runtime_error("distcp: checkpoint payload path escapes its directory"); }
    }
    auto const    path = directory / relative;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) { throw std::runtime_error(fmt::format("distcp: cannot open checkpoint payload '{}'", path.string())); }
    auto const length = stream.tellg();
    if (length < 0 || location.offset > static_cast<uint64_t>(length) || location.length > static_cast<uint64_t>(length) - location.offset
        || location.length > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw std::runtime_error(fmt::format("distcp: truncated checkpoint payload '{}'", location.path));
    }
    ByteStorage bytes(static_cast<size_t>(location.length));
    stream.seekg(static_cast<std::streamoff>(location.offset));
    if (location.length != 0)
    {
        stream.read(static_cast<char*>(bytes.data), static_cast<std::streamsize>(location.length));
        if (!stream) { throw std::runtime_error("distcp: cannot read checkpoint payload"); }
    }
    return bytes;
}

static ByteBuffer ZipEntry(mz_zip_archive& archive, std::string const& name)
{
    auto const               index = mz_zip_reader_locate_file(&archive, name.c_str(), nullptr, 0);
    mz_zip_archive_file_stat stat{};
    if (index < 0 || mz_zip_reader_file_stat(&archive, static_cast<mz_uint>(index), &stat) != MZ_TRUE
        || !std::in_range<size_t>(stat.m_uncomp_size))
    {
        throw std::runtime_error(fmt::format("distcp: missing tensor archive entry '{}'", name));
    }
    ByteBuffer result(static_cast<size_t>(stat.m_uncomp_size));
    char       empty{};
    if (mz_zip_reader_extract_to_mem(&archive, static_cast<mz_uint>(index), result.empty() ? &empty : result.data(), result.size(), 0)
        != MZ_TRUE)
    {
        throw std::runtime_error(fmt::format("distcp: corrupt tensor archive entry '{}'", name));
    }
    return result;
}

static ByteStorage ReadTensorArchive(ByteStorage const& bytes, Tensor::DimRange const& ranges, Tensor::DType dtype)
{
    mz_zip_archive archive{};
    if (mz_zip_reader_init_mem(&archive, bytes.data, bytes.size, 0) != MZ_TRUE)
    {
        throw std::runtime_error("distcp: invalid tensor ZIP archive");
    }
    try
    {
        std::string prefix;
        for (mz_uint index = 0; index < mz_zip_reader_get_num_files(&archive); ++index)
        {
            mz_zip_archive_file_stat stat{};
            if (mz_zip_reader_file_stat(&archive, index, &stat) != MZ_TRUE)
            {
                throw std::runtime_error("distcp: invalid tensor ZIP directory");
            }
            auto const name = std::string_view(stat.m_filename);
            if (name.ends_with("/data.pkl"))
            {
                prefix = name.substr(0, name.size() - std::string_view("data.pkl").size());
                break;
            }
        }
        if (prefix.empty()) { throw std::runtime_error("distcp: tensor archive has no metadata"); }
        if (mz_zip_reader_locate_file(&archive, (prefix + "byteorder").c_str(), nullptr, 0) >= 0)
        {
            auto const byteorder = ZipEntry(archive, prefix + "byteorder");
            if (std::string_view(byteorder.data(), byteorder.size()) != "little")
            {
                throw std::runtime_error("distcp: tensor archive byte order is unsupported");
            }
        }
        auto const metadataBytes = ZipEntry(archive, prefix + "data.pkl");
        auto const tensor        = PickleReader(metadataBytes).Read();
        if (tensor->kind != PickleValue::Kind::Object
            || (tensor->text != "torch._utils._rebuild_tensor_v2" && tensor->text != "torch._utils._rebuild_tensor_v3"
                && tensor->text != "torch._utils._rebuild_tensor")
            || tensor->items.size() < 4)
        {
            throw std::runtime_error("distcp: unsupported tensor archive metadata");
        }
        auto const shape   = MetadataDimensions(tensor->items[2]);
        auto const strides = MetadataDimensions(tensor->items[3]);
        if (shape.size() != ranges.size() || strides.size() != ranges.size())
        {
            throw std::runtime_error("distcp: tensor archive dimensions disagree with checkpoint metadata");
        }
        auto size = ElementSize(dtype);
        for (size_t axis = 0; axis < ranges.size(); ++axis)
        {
            if (shape[axis] != ranges[axis].end - ranges[axis].begin)
            {
                throw std::runtime_error("distcp: tensor archive shape disagrees with checkpoint metadata");
            }
            size = CheckedMultiply(size, shape[axis]);
        }
        auto const& persistent = tensor->items[0];
        if (persistent->text != "persistent" || persistent->items.size() != 1 || persistent->items[0]->kind != PickleValue::Kind::Sequence
            || persistent->items[0]->items.size() < 3)
        {
            throw std::runtime_error("distcp: invalid tensor storage reference");
        }
        auto const& storageType  = persistent->items[0]->items[1];
        auto const  expectedType = PickleInfoForDtype(dtype);
        if (storageType->kind != PickleValue::Kind::Global
            || storageType->text != std::string(expectedType.storageMod) + "." + std::string(expectedType.storageName))
        {
            throw std::runtime_error("distcp: tensor archive datatype disagrees with checkpoint metadata");
        }
        if (expectedType.useV3
            && (tensor->items.size() <= 6 || tensor->items[6]->kind != PickleValue::Kind::Global
                || tensor->items[6]->text != "torch." + std::string(DTypeName(dtype))))
        {
            throw std::runtime_error("distcp: untyped tensor archive datatype disagrees with checkpoint metadata");
        }
        ByteStorage result(static_cast<size_t>(size));
        if (size != 0)
        {
            auto const storageKey    = MetadataText(persistent->items[0]->items[2]);
            auto const raw           = ZipEntry(archive, prefix + "data/" + storageKey);
            auto const storageOffset = MetadataNumber(tensor->items[1]);
            auto const elementSize   = ElementSize(dtype);
            auto const elements      = size / elementSize;
            uint64_t   blockElements{1};
            for (size_t axis = shape.size(); axis != 0; --axis)
            {
                if (shape[axis - 1] > 1 && strides[axis - 1] != blockElements) { break; }
                blockElements = CheckedMultiply(blockElements, shape[axis - 1]);
            }
            auto const blockBytes = CheckedMultiply(blockElements, elementSize);
            for (uint64_t element = 0; element < elements; element += blockElements)
            {
                auto remainder = element;
                auto source    = storageOffset;
                for (size_t axis = shape.size(); axis != 0; --axis)
                {
                    auto const coordinate = remainder % shape[axis - 1];
                    remainder /= shape[axis - 1];
                    auto const delta = CheckedMultiply(coordinate, strides[axis - 1]);
                    if (source > std::numeric_limits<uint64_t>::max() - delta)
                    {
                        throw std::overflow_error("distcp: tensor storage offset overflow");
                    }
                    source += delta;
                }
                auto const sourceByte = CheckedMultiply(source, elementSize);
                if (sourceByte > raw.size() || blockBytes > raw.size() - sourceByte)
                {
                    throw std::runtime_error("distcp: tensor storage is shorter than its view");
                }
                std::memcpy(
                    static_cast<char*>(result.data) + element * elementSize, raw.data() + sourceByte, static_cast<size_t>(blockBytes));
            }
        }
        mz_zip_reader_end(&archive);
        return result;
    } catch (...)
    {
        mz_zip_reader_end(&archive);
        throw;
    }
}

static StateNode& LoadedNode(StateTree& registry LFTBND, std::string_view path)
{
    auto   parent = StateNode::Handle::Root;
    size_t start{};
    for (;;)
    {
        auto const separator = path.find('.', start);
        auto const name      = path.substr(start, separator == std::string_view::npos ? path.size() - start : separator - start);
        if (name.empty()) { throw std::runtime_error("distcp: empty checkpoint address component"); }
        auto [found, handle] = registry.FindChildNodeIfExists(parent, name);
        if (!found) { handle = registry.AddChild(parent, name).handle; }
        auto& node = registry.GetNode(handle);
        if (separator == std::string_view::npos) { return node; }
        if (node.GetType() == StateNode::Type::Invalid) { node.data = StateNode::Dict{}; }
        if (node.GetType() != StateNode::Type::Dict) { throw std::runtime_error("distcp: overlapping checkpoint addresses"); }
        parent = handle;
        start  = separator + 1;
    }
}

static uint64_t CopyIntersection(ByteStorage const&      source,
                                 Tensor::DimRange const& sourceRanges,
                                 ByteStorage&            target,
                                 Tensor::DimRange const& targetRanges,
                                 Tensor::DType           dtype)
{
    Tensor::DimRange intersection;
    for (size_t axis = 0; axis < sourceRanges.size(); ++axis)
    {
        auto const begin = std::max(sourceRanges[axis].begin, targetRanges[axis].begin);
        auto const end   = std::min(sourceRanges[axis].end, targetRanges[axis].end);
        if (end <= begin) { return 0; }
        intersection.push_back({.begin = begin, .end = end});
    }
    auto const elementSize = ElementSize(dtype);
    auto const size        = DenseSize(intersection, dtype);
    uint64_t   blockElements{1};
    size_t     blockAxis = intersection.size();
    for (size_t axis = intersection.size(); axis != 0; --axis)
    {
        blockAxis     = axis - 1;
        blockElements = CheckedMultiply(blockElements, intersection[blockAxis].end - intersection[blockAxis].begin);
        if (intersection[blockAxis] != sourceRanges[blockAxis] || intersection[blockAxis] != targetRanges[blockAxis]) { break; }
    }
    auto const blockBytes = CheckedMultiply(blockElements, elementSize);
    auto const blocks     = size / blockBytes;
    for (uint64_t block = 0; block < blocks; ++block)
    {
        auto     remainder = block;
        uint64_t sourceElement{}, targetElement{}, sourceStride{1}, targetStride{1};
        for (size_t axis = intersection.size(); axis != 0; --axis)
        {
            auto const index      = axis - 1;
            auto       coordinate = intersection[index].begin;
            if (index < blockAxis)
            {
                coordinate += remainder % (intersection[index].end - intersection[index].begin);
                remainder /= intersection[index].end - intersection[index].begin;
            }
            sourceElement += CheckedMultiply(coordinate - sourceRanges[index].begin, sourceStride);
            targetElement += CheckedMultiply(coordinate - targetRanges[index].begin, targetStride);
            sourceStride = CheckedMultiply(sourceStride, sourceRanges[index].end - sourceRanges[index].begin);
            targetStride = CheckedMultiply(targetStride, targetRanges[index].end - targetRanges[index].begin);
        }
        std::memcpy(static_cast<char*>(target.data) + targetElement * elementSize,
                    static_cast<char const*>(source.data) + sourceElement * elementSize,
                    static_cast<size_t>(blockBytes));
    }
    return size;
}
}    // namespace

std::vector<ByteStorage> ReadCheckpoint(std::filesystem::path const& ckptDir, StateTree& localRegistry, bool allowPartialLoad)
{
    auto metadataStorage
        = ReadFileRange(ckptDir, {.path = ".metadata", .offset = 0, .length = std::filesystem::file_size(ckptDir / ".metadata")});
    auto const metadata    = PickleReader({static_cast<char const*>(metadataStorage.data), metadataStorage.size}).Read();
    auto const definitions = Field(metadata, "state_dict_metadata");
    auto const locations   = Field(metadata, "storage_data");
    if (definitions->kind != PickleValue::Kind::Dictionary || locations->kind != PickleValue::Kind::Dictionary)
    {
        throw std::runtime_error("distcp: checkpoint metadata dictionaries are invalid");
    }
    std::map<ReadIndex, ReadLocation> storage;
    for (auto const& [key, value] : locations->entries)
    {
        auto const            offset = Field(key, "offset", false);
        std::vector<uint64_t> coordinates;
        if (offset && offset->kind != PickleValue::Kind::None)
        {
            auto const shape = MetadataDimensions(offset);
            coordinates.assign(shape.begin(), shape.end());
        }
        auto const   fqn = MetadataText(Field(key, "fqn"));
        ReadLocation location{.path   = MetadataText(Field(value, "relative_path")),
                              .offset = MetadataNumber(Field(value, "offset")),
                              .length = MetadataNumber(Field(value, "length"))};
        if (!storage.emplace(ReadIndex{fqn, std::move(coordinates)}, std::move(location)).second)
        {
            throw std::runtime_error("distcp: duplicate checkpoint storage index");
        }
    }
    std::map<std::string, PickleRef> entries;
    for (auto const& [key, value] : definitions->entries)
    {
        if (!entries.emplace(MetadataText(key), value).second) { throw std::runtime_error("distcp: duplicate checkpoint key"); }
    }
    std::vector<std::pair<std::string, DTensor>> requests;
    localRegistry.WalkNodes([&](StateNode const& node) {
        if (node.GetType() == StateNode::Type::DTensor)
        {
            requests.emplace_back(localRegistry.GetNodeId(node.handle), std::get<DTensor>(node.data));
        }
    });
    std::vector<ByteStorage> buffers;
    for (auto const& [fqn, definition] : entries)
    {
        if (definition->text != std::string(KMetadataModule) + ".BytesStorageMetadata") { continue; }
        auto const location = storage.find({fqn, {}});
        if (location == storage.end()) { throw std::runtime_error(fmt::format("distcp: missing byte storage for '{}'", fqn)); }
        auto       bytes      = ReadFileRange(ckptDir, location->second);
        auto const scalarType = Field(definition, "nvrx_scalar_type", false);
        if (scalarType)
        {
            if (bytes.size != sizeof(uint64_t)) { throw std::runtime_error("distcp: scalar payload contains exactly eight bytes"); }
            uint64_t raw{};
            std::memcpy(&raw, bytes.data, sizeof(raw));
            auto const&       type = MetadataText(scalarType);
            StateNode::Scalar scalar;
            if (type == "int64") { scalar = StateNode::Scalar(std::bit_cast<int64_t>(raw)); }
            else if (type == "uint64") { scalar = StateNode::Scalar(raw); }
            else if (type == "float64") { scalar = StateNode::Scalar(std::bit_cast<double>(raw)); }
            else if (type == "bool" && raw <= 1) { scalar = StateNode::Scalar(raw != 0); }
            else
            {
                throw std::runtime_error("distcp: invalid checkpoint scalar datatype or value");
            }
            scalar.replicas                     = {localRegistry.GlobalRank()};
            LoadedNode(localRegistry, fqn).data = std::move(scalar);
        }
        else
        {
            LoadedNode(localRegistry, fqn).data = StateNode::DynamicSizeBlob{
                .replicas = {localRegistry.GlobalRank()}, .handle = reinterpret_cast<uint64_t>(bytes.data), .storageSize = bytes.size};
            buffers.push_back(std::move(bytes));
        }
    }
    for (auto const& [fqn, request] : requests)
    {
        auto const entry = entries.find(fqn);
        if (entry == entries.end())
        {
            if (!allowPartialLoad) { throw std::runtime_error(fmt::format("distcp: missing checkpoint tensor '{}'", fqn)); }
            LoadedNode(localRegistry, fqn).data = std::monostate{};
            continue;
        }
        if (entry->second->text == std::string(KMetadataModule) + ".BytesStorageMetadata") { continue; }
        if (entry->second->text != std::string(KMetadataModule) + ".TensorStorageMetadata")
        {
            throw std::runtime_error("distcp: unsupported checkpoint storage metadata");
        }
        auto const dtype     = MetadataDtype(Field(entry->second, "properties"));
        auto const fullShape = MetadataDimensions(Field(entry->second, "size"));
        if (request.fulldim != fullShape)
        {
            throw std::runtime_error(fmt::format("distcp: checkpoint tensor '{}' shape or datatype does not match destination", fqn));
        }
        auto const chunks = Field(entry->second, "chunks");
        if (chunks->kind != PickleValue::Kind::Sequence) { throw std::runtime_error("distcp: invalid checkpoint tensor chunks"); }
        DTensor loaded(dtype, fullShape);
        for (auto const& requestedShard : request.shards)
        {
            if (requestedShard.sharddim.size() != fullShape.size()) { throw std::runtime_error("distcp: invalid destination shard rank"); }
            for (size_t axis = 0; axis < fullShape.size(); ++axis)
            {
                if (requestedShard.sharddim[axis].end > fullShape[axis])
                {
                    throw std::runtime_error("distcp: destination shard exceeds tensor");
                }
            }
            auto const                    size = DenseSize(requestedShard.sharddim, dtype);
            ByteStorage                   destination(static_cast<size_t>(size));
            uint64_t                      copied{};
            std::vector<Tensor::DimRange> covered;
            for (auto const& chunk : chunks->items)
            {
                auto const offsets = MetadataDimensions(Field(chunk, "offsets"));
                auto const sizes   = MetadataDimensions(Field(chunk, "sizes"));
                if (offsets.size() != fullShape.size() || sizes.size() != fullShape.size())
                {
                    throw std::runtime_error("distcp: invalid checkpoint chunk rank");
                }
                Tensor::DimRange      ranges;
                std::vector<uint64_t> index;
                for (size_t axis = 0; axis < fullShape.size(); ++axis)
                {
                    if (offsets[axis] > fullShape[axis] || sizes[axis] > fullShape[axis] - offsets[axis])
                    {
                        throw std::runtime_error("distcp: checkpoint chunk exceeds tensor");
                    }
                    ranges.push_back({.begin = offsets[axis], .end = offsets[axis] + sizes[axis]});
                    index.push_back(offsets[axis]);
                }
                if (size == 0 || !DTensor::Intersects(ranges, requestedShard.sharddim)) { continue; }
                for (auto const& previous : covered)
                {
                    if (DTensor::Intersects(previous, ranges)) { throw std::runtime_error("distcp: overlapping saved tensor chunks"); }
                }
                covered.push_back(ranges);
                auto const location = storage.find({fqn, std::move(index)});
                if (location == storage.end()) { throw std::runtime_error("distcp: missing tensor chunk storage"); }
                auto       archive = ReadFileRange(ckptDir, location->second);
                auto const source  = ReadTensorArchive(archive, ranges, dtype);
                auto const count   = CopyIntersection(source, ranges, destination, requestedShard.sharddim, dtype);
                if (copied > size || count > size - copied) { throw std::runtime_error("distcp: overlapping tensor chunk coverage"); }
                copied += count;
            }
            if (copied != size) { throw std::runtime_error(fmt::format("distcp: incomplete tensor chunk coverage for '{}'", fqn)); }
            auto shard          = requestedShard;
            shard.snapshotDtype = dtype;
            shard.device        = DeviceId("cpu");
            shard.handle        = reinterpret_cast<uint64_t>(destination.data);
            shard.storageSize   = size;
            loaded.shards.insert(shard);
            buffers.push_back(std::move(destination));
        }
        LoadedNode(localRegistry, fqn).data = std::move(loaded);
    }
    return buffers;
}

}    // namespace checkpoint::distcp
