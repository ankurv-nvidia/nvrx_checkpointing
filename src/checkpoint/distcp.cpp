#include "distcp.h"
#include "distcp_internal.h"
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
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
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

// Each rank shard concatenates byte-compatible torch.save ZIP archives.

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
    stride[shape.size() - 1] = 1;
    for (size_t i = shape.size() - 1; i-- > 0;) { stride[i] = stride[i + 1] * shape[i + 1]; }

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

static uint64_t WriteShardArchive(std::ostream&      output,
                                  Tensor::DType      dtype,
                                  Tensor::Dim const& shape,
                                  char const*        srcPtr,
                                  uint64_t           length,
                                  std::string_view   contextName)
{
    ByteBuffer dataPkl = checkpoint::distcp::BuildDataPkl(dtype, shape);

    struct EntryPlan
    {
        std::string fullName;
        void const* data;
        size_t      size;
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
        EntryPlan{.fullName = std::string{checkpoint::distcp::EntryFullNames[DataZeroIdx]}, .data = srcPtr, .size = length, .padding = {}},
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

    auto addEntry = [&](EntryPlan& plan, std::string_view bareName) {
        plan.padding               = checkpoint::distcp::BuildFbExtra(zip.m_archive_size, plan.fullName.size(), plan.size);
        streamOutput.sawDescriptor = false;
        auto const entryCrc        = static_cast<mz_uint32>(
            ::crc32_z(/*crc=*/0L, reinterpret_cast<Bytef const*>(plan.data), static_cast<z_size_t>(plan.size)));    // NOLINT
        if (mz_zip_writer_add_mem_ex_v2(&zip,
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
                                        /*user_extra_data_central_len=*/0)
            != MZ_TRUE)
        {
            throw std::runtime_error(fmt::format("distcp: mz_zip_writer_add_mem_ex_v2 failed for '{}'", plan.fullName));
        }

        if (!streamOutput.sawDescriptor)
        {
            throw std::runtime_error(fmt::format("distcp: miniz omitted data descriptor for '{}'", plan.fullName));
        }
        if (streamOutput.lastDescriptorCrc != entryCrc)
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

static void EmitBytesStorageMetadata(ByteBuffer& b)
{
    EmitNewObjBuildBegin(b, KMetadataModule, "BytesStorageMetadata");
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
        if (shard.StorageSizeBytes() == 0) { continue; }
        if (std::ranges::find(chunkRanges, shard.sharddim) == chunkRanges.end()) { chunkRanges.push_back(shard.sharddim); }
    }
    return chunkRanges;
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
    globalRegistry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        assert(type != StateNode::Type::Tensor);
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
        EmitBytesStorageMetadata(b);
    }
    PutU8(b, 'u');

    PickleBinunicode(b, "storage_data");
    PutU8(b, '}');
    PutU8(b, '(');
    std::unordered_map<Rank, uint64_t> rankCursors;
    globalRegistry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        if (type != StateNode::Type::DTensor) { return; }
        auto const&      tnsr = std::get<DTensor>(node.data);
        auto const       fqn  = globalRegistry.GetNodeId(node.handle);
        Tensor::DimRange lastShard{};
        for (auto const& shard : tnsr.shards)
        {
            auto const shardSize = shard.StorageSizeBytes();
            if (shardSize == 0) { continue; }
            Tensor::Dim shardShape;
            for (auto const& r : shard.sharddim) { shardShape.push_back(r.end - r.begin); }
            uint64_t const archiveLen = ComputeShardArchiveSize(tnsr.durableDtype, shardShape, shardSize);
            uint64_t&      cursor     = rankCursors[shard.rank];
            uint64_t       curCursor  = cursor;
            cursor += archiveLen;

            if (lastShard == shard.sharddim) { continue; }
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
        return;
    }

    std::filesystem::create_directories(ckptDir);

    auto        output = OpenOutputFile(filepath);
    uint64_t    cursor = 0;
    auto const* base   = static_cast<char const*>(tensorStorage.data);
    registry.WalkNodes([&](StateNode const& node) {
        auto const type = static_cast<StateNode::Type>(node.data.index());
        assert(type != StateNode::Type::Tensor);
        if (type != StateNode::Type::DTensor) { return; }
        auto const& tnsr = std::get<DTensor>(node.data);
        auto const  fqn  = registry.GetNodeId(node.handle);
        for (auto const& shard : tnsr.shards)
        {
            if (shard.rank != rank) { continue; }
            auto const size = shard.StorageSizeBytes();
            if (size == 0) { continue; }
            Tensor::Dim shape{};
            for (auto const& r : shard.sharddim) { shape.push_back(r.end - r.begin); }
            if (size >= MZ_UINT32_MAX)
            {
                throw std::runtime_error(
                    fmt::format("distcp: shard '{}' is {} bytes (>= 4 GiB); ZIP64 support is not yet implemented", fqn, size));
            }
            uint64_t const predicted = ComputeShardArchiveSize(tnsr.durableDtype, shape, size);
            uint64_t const actual    = WriteShardArchive(output, tnsr.durableDtype, shape, base + shard.handle, size, fqn);
            if (predicted != actual)
            {
                throw std::runtime_error(fmt::format("distcp: ComputeShardArchiveSize/WriteShardArchive disagree for '{}' "
                                                     "(dtype={}, shape=[{}], dataSize={}): predicted={} actual={}",
                                                     fqn,
                                                     DTypeName(tnsr.durableDtype),
                                                     fmt::join(shape, ","),
                                                     size,
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
}

void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry, BlobMetadataByFqn const& blobs)
{
    auto const pkl = BuildMetadataPkl(globalRegistry, blobs);

    std::filesystem::create_directories(ckptDir);
    auto const filepath = ckptDir / ".metadata";

    auto output = OpenOutputFile(filepath);
    WriteBytes(output, pkl.data(), pkl.size(), filepath.string());
    FinishOutputFile(output, filepath);

    // The format marker makes the directory loadable as a torch_dist checkpoint.
    static constexpr std::string_view ConfigJson   = R"({"sharded_backend": "torch_dist", "sharded_backend_version": 1})";
    auto const                        configPath   = ckptDir / "metadata.json";
    auto                              configOutput = OpenOutputFile(configPath);
    WriteBytes(configOutput, ConfigJson.data(), ConfigJson.size(), configPath.string());
    FinishOutputFile(configOutput, configPath);
}

}    // namespace checkpoint::distcp
