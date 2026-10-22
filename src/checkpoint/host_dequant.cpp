#include "host_dequant.h"
#include "quant.h"

#include <fmt/format.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

// NOLINTBEGIN(readability-magic-numbers)
namespace checkpoint::host_dequant
{

static constexpr uint64_t MXFP8BlockSize          = 32;
static constexpr uint64_t RowScaleAlignment       = 128;
static constexpr uint64_t RowScaleStrideAlignment = 4;
static constexpr uint64_t NVFP4BlockSize          = 16;
static constexpr uint64_t NVFP4ScaleRowsAlignment = 128;
static constexpr uint64_t NVFP4ScaleColsAlignment = 4;

static uint64_t RoundUp(uint64_t value, uint64_t alignment)
{ return ((value + alignment - 1) / alignment) * alignment; }

static float DecodeE4M3(uint8_t value)
{
    uint8_t const sign     = value >> 7U;
    uint8_t const exponent = static_cast<uint8_t>(value >> 3U) & uint8_t{0x0fU};
    uint8_t const mantissa = value & 0x07U;

    if (exponent == 0x0fU && mantissa == 0x07U) { return std::bit_cast<float>(0x7fffffffU); }

    float decoded = 0.0F;
    if (exponent == 0) { decoded = std::ldexp(static_cast<float>(mantissa), -9); }
    else
    {
        decoded = std::ldexp(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10);
    }
    return sign == 0 ? decoded : -decoded;
}

static uint16_t FloatToBFloat16(float value)
{
    auto bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0) { return 0x7fffU; }
    bits += 0x00007fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

static uint16_t ConvertScalar(uint8_t value, uint8_t biasedScale)
{
    uint32_t const scaleBits = biasedScale == 0xffU ? 0x7fffffffU
                               : biasedScale == 0U  ? 0x00400000U
                                                    : static_cast<uint32_t>(biasedScale) << 23U;
    return FloatToBFloat16(std::bit_cast<float>(scaleBits) * DecodeE4M3(value));
}

static float DecodeE2M1(uint8_t value)
{
    static constexpr std::array<float, 16> Values{
        0.0F,
        0.5F,
        1.0F,
        1.5F,
        2.0F,
        3.0F,
        4.0F,
        6.0F,
        -0.0F,
        -0.5F,
        -1.0F,
        -1.5F,
        -2.0F,
        -3.0F,
        -4.0F,
        -6.0F,
    };
    return Values[value & 0x0fU];
}

static float NVFP4E4M3ScaleBound(quant::NVFP4E4M3Mode mode)
{
    switch (mode)
    {
    case quant::NVFP4E4M3Mode::ScaleBound448: return 448.0F;
    case quant::NVFP4E4M3Mode::ScaleBound256: return 256.0F;
    default: throw std::runtime_error("NVFP4 E4M3 scale bound is invalid");
    }
}

static uint16_t ConvertNVFP4Scalar(uint8_t value, uint8_t e4m3Scale, float amax, quant::NVFP4E4M3Mode mode)
{
    float const factorInv  = 1.0F / (6.0F * NVFP4E4M3ScaleBound(mode));
    float const finalScale = DecodeE4M3(e4m3Scale) * amax * factorInv;
    return FloatToBFloat16(DecodeE2M1(value) * finalScale);
}

static constexpr std::array<uint16_t, 65536> const& ConversionTable()
{
    static std::array<uint16_t, 65536> const Table = [] {
        std::array<uint16_t, 65536> values{};
        for (uint32_t scale = 0; scale < 256; ++scale)
        {
            for (uint32_t value = 0; value < 256; ++value)
            {
                values[(scale << 8U) | value] = ConvertScalar(static_cast<uint8_t>(value), static_cast<uint8_t>(scale));
            }
        }
        return values;
    }();
    return Table;
}

static quant::Component const&
RequireComponent(quant::Component const& comp LFTBND, std::string_view const name, std::string_view const encoding)
{
    if (comp.handle == 0 || comp.sizeBytes == 0) { throw std::runtime_error(fmt::format("{} storage is missing {}", encoding, name)); }
    return comp;
}

static uint64_t GemmSwizzledScaleIndex(uint64_t row, uint64_t col, uint64_t numTilesX)
{
    constexpr uint64_t tileDimX = 4;
    constexpr uint64_t tileDimY = 128;
    constexpr uint64_t tileSize = tileDimX * tileDimY;
    uint64_t const     tileX    = col / tileDimX;
    uint64_t const     tileY    = row / tileDimY;
    uint64_t const     inTileX  = col % tileDimX;
    uint64_t const     inTileY  = row % tileDimY;
    return (((tileY * numTilesX) + tileX) * tileSize) + ((inTileY % 32) * 16) + ((inTileY / 32) * 4) + inTileX;
}

bool HasStorageHeader(std::span<char const> bytes)
{ return bytes.size() >= quant::Magic.size() && std::memcmp(bytes.data(), quant::Magic.data(), quant::Magic.size()) == 0; }

uint16_t FP8E4M3ToBFloat16(uint8_t value, float scaleInv)
{ return FloatToBFloat16(DecodeE4M3(value) * scaleInv); }

uint16_t MXFP8E4M3ToBFloat16(uint8_t value, uint8_t biasedScale)
{ return ConvertScalar(value, biasedScale); }

uint16_t NVFP4E2M1ToBFloat16(uint8_t value, uint8_t e4m3Scale, float amax, quant::NVFP4E4M3Mode e4m3Mode)
{ return ConvertNVFP4Scalar(value, e4m3Scale, amax, e4m3Mode); }

void MXFP8E4M3ToBFloat16(std::span<char const> const& storage, std::span<uint16_t> output)
{ MXFP8E4M3ToBFloat16Range(storage, 0, output); }

void FP8E4M3ToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output)
{
    auto const& header = *reinterpret_cast<quant::Header const*>(storage.data());    // NOLINT
    if (header.encoding != quant::Encoding::FP8E4M3)
    {
        throw std::runtime_error(fmt::format("Unsupported host dequantization encoding {}", static_cast<uint8_t>(header.encoding)));
    }
    if (header.scaleLayout != quant::ScaleLayout::Compact || header.rowwiseAmax.handle != 0)
    {
        throw std::runtime_error("FP8 E4M3 storage has invalid components");
    }

    uint64_t const rows     = header.logicalRows;
    uint64_t const cols     = header.logicalCols;
    uint64_t const elements = rows * cols;
    if (elementOffset > elements || output.size() > elements - elementOffset)
    {
        throw std::runtime_error(fmt::format(
            "FP8 E4M3 output range [{}:{}) exceeds logical shape {}x{}", elementOffset, elementOffset + output.size(), rows, cols));
    }

    auto const& data  = RequireComponent(header.rowwiseData, "data", "FP8 E4M3");
    auto const& scale = RequireComponent(header.rowwiseScaleInv, "scale", "FP8 E4M3");
    if (data.rows != rows || data.cols != cols)
    {
        throw std::runtime_error(
            fmt::format("FP8 E4M3 data shape {}x{} does not match logical shape {}x{}", data.rows, data.cols, rows, cols));
    }
    if (scale.rows != 1 || scale.cols != 1 || scale.sizeBytes != sizeof(float))
    {
        throw std::runtime_error(
            fmt::format("FP8 E4M3 scale shape {}x{} with {} bytes is not a float scalar", scale.rows, scale.cols, scale.sizeBytes));
    }

    float scaleInv{};
    auto  scaleBytes = header.ScaleInvSubSpan<char const>();
    std::memcpy(&scaleInv, scaleBytes.data(), sizeof(scaleInv));
    std::array<uint16_t, 256> conversion{};
    for (uint16_t value = 0; value < conversion.size(); ++value)
    {
        conversion[value] = FP8E4M3ToBFloat16(static_cast<uint8_t>(value), scaleInv);
    }
    auto const dataBytes = header.DataSubSpan<uint8_t const>();
    for (size_t i = 0; i < output.size(); ++i) { output[i] = conversion[dataBytes[elementOffset + i]]; }
}

void MXFP8E4M3ToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output)
{
    auto const& header = *reinterpret_cast<quant::Header const*>(storage.data());    // NOLINT
    if (header.encoding != quant::Encoding::MXFP8E4M3)
    {
        throw std::runtime_error(fmt::format("Unsupported host dequantization encoding {}", static_cast<uint8_t>(header.encoding)));
    }
    if (header.scaleLayout != quant::ScaleLayout::Compact) { throw std::runtime_error("MXFP8 GEMM-swizzled scales are unsupported"); }
    if (header.rowwiseAmax.handle != 0 || header.rowwiseScaleInv.handle == 0 || header.rowwiseData.handle == 0)
    {
        throw std::runtime_error("MXFP8 storage has invalid components");
    }

    uint64_t const rows     = header.logicalRows;
    uint64_t const cols     = header.logicalCols;
    uint64_t const elements = rows * cols;
    if (elementOffset > elements || output.size() > elements - elementOffset)
    {
        throw std::runtime_error(fmt::format(
            "MXFP8 output range [{}:{}) exceeds logical shape {}x{}", elementOffset, elementOffset + output.size(), rows, cols));
    }

    auto const& data  = RequireComponent(header.rowwiseData, "rowwise data", "MXFP8");
    auto const& scale = RequireComponent(header.rowwiseScaleInv, "rowwise scales", "MXFP8");
    if (data.rows != rows || data.cols != cols)
    {
        throw std::runtime_error(
            fmt::format("MXFP8 rowwise data shape {}x{} does not match logical shape {}x{}", data.rows, data.cols, rows, cols));
    }
    uint64_t const expectedScaleRows = RoundUp(rows, RowScaleAlignment);
    uint64_t const expectedScaleCols = RoundUp((cols + MXFP8BlockSize - 1) / MXFP8BlockSize, RowScaleStrideAlignment);
    if (scale.rows != expectedScaleRows || scale.cols != expectedScaleCols)
    {
        throw std::runtime_error(fmt::format(
            "MXFP8 rowwise scale shape {}x{} does not match expected {}x{}", scale.rows, scale.cols, expectedScaleRows, expectedScaleCols));
    }

    auto const  dataBytes     = header.DataSubSpan<uint8_t const>();
    auto const  scaleInvBytes = header.ScaleInvSubSpan<uint8_t const>();
    auto const& conversion    = ConversionTable();
    uint64_t    index         = elementOffset;
    uint64_t    row           = elementOffset / cols;
    uint64_t    col           = elementOffset % cols;
    for (auto& outputValue : output)
    {
        uint64_t const scaleIndex = (row * scale.cols) + (col / MXFP8BlockSize);

        auto const conversionIndex = static_cast<size_t>((static_cast<uint16_t>(scaleInvBytes[scaleIndex]) << 8U) | dataBytes[index]);
        outputValue                = conversion[conversionIndex];
        ++index;
        if (++col == cols)
        {
            col = 0;
            ++row;
        }
    }
}

void NVFP4E2M1ToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output)
{
    auto const& header = *reinterpret_cast<quant::Header const*>(storage.data());    // NOLINT
    if (header.encoding != quant::Encoding::NVFP4E2M1)
    {
        throw std::runtime_error(fmt::format("Unsupported host dequantization encoding {}", static_cast<uint8_t>(header.encoding)));
    }
    if (header.rowwiseData.handle == 0 || header.rowwiseScaleInv.handle == 0 || header.rowwiseAmax.handle == 0)
    {
        throw std::runtime_error("NVFP4 storage has invalid components");
    }

    uint64_t const rows = header.logicalRows;
    uint64_t const cols = header.logicalCols;
    if ((cols % NVFP4BlockSize) != 0) { throw std::runtime_error(fmt::format("NVFP4 has invalid logical shape {}x{}", rows, cols)); }
    uint64_t const elements = rows * cols;
    if (elementOffset > elements || output.size() > elements - elementOffset)
    {
        throw std::runtime_error(fmt::format(
            "NVFP4 output range [{}:{}) exceeds logical shape {}x{}", elementOffset, elementOffset + output.size(), rows, cols));
    }

    auto const& data  = RequireComponent(header.rowwiseData, "rowwise data", "NVFP4");
    auto const& scale = RequireComponent(header.rowwiseScaleInv, "rowwise scales", "NVFP4");
    auto const& amax  = RequireComponent(header.rowwiseAmax, "rowwise amax", "NVFP4");
    if (data.rows != rows || data.cols != cols / 2)
    {
        throw std::runtime_error(
            fmt::format("NVFP4 rowwise data shape {}x{} does not match logical shape {}x{}", data.rows, data.cols, rows, cols));
    }
    uint64_t const expectedScaleRows = RoundUp(rows, NVFP4ScaleRowsAlignment);
    uint64_t const expectedScaleCols = RoundUp(cols / NVFP4BlockSize, NVFP4ScaleColsAlignment);
    if (scale.rows != expectedScaleRows || scale.cols != expectedScaleCols)
    {
        throw std::runtime_error(fmt::format(
            "NVFP4 rowwise scale shape {}x{} does not match expected {}x{}", scale.rows, scale.cols, expectedScaleRows, expectedScaleCols));
    }
    uint64_t const expectedAmaxCount = header.rowScaledNVFP4 ? rows : 1;
    if (amax.sizeBytes != expectedAmaxCount * sizeof(float))
    {
        throw std::runtime_error(
            fmt::format("NVFP4 rowwise amax has {} bytes, expected {}", amax.sizeBytes, expectedAmaxCount * sizeof(float)));
    }
    uint64_t const scaleBlocks    = cols / NVFP4BlockSize;
    uint64_t const numScaleTilesX = RoundUp(scaleBlocks, 4) / 4;
    float const    factorInv      = 1.0F / (6.0F * NVFP4E4M3ScaleBound(header.nvfp4E4M3Mode));
    uint64_t       index          = elementOffset;
    uint64_t       row            = elementOffset / cols;
    uint64_t       col            = elementOffset % cols;
    auto           rowAmaxData    = header.AmaxSubSpan();
    auto           rowAmax        = rowAmaxData[header.rowScaledNVFP4 ? row : 0];
    uint64_t       lastScaleIndex = std::numeric_limits<uint64_t>::max();
    float          finalScale{};
    auto           scaleBytes = header.ScaleInvSubSpan<uint8_t>();
    auto           dataBytes  = header.DataSubSpan<uint8_t>();

    for (auto& outputValue : output)
    {
        uint64_t const scaleCol   = col / NVFP4BlockSize;
        uint64_t const scaleIndex = header.scaleLayout == quant::ScaleLayout::GemmSwizzled
                                        ? GemmSwizzledScaleIndex(row, scaleCol, numScaleTilesX)
                                        : (row * scale.cols) + scaleCol;
        if (scaleIndex != lastScaleIndex)
        {
            finalScale     = DecodeE4M3(scaleBytes[scaleIndex]) * rowAmax * factorInv;
            lastScaleIndex = scaleIndex;
        }
        uint8_t const packed = dataBytes[index / 2];
        uint8_t const value  = (index & 1U) == 0 ? packed & 0x0fU : packed >> 4U;
        outputValue          = FloatToBFloat16(DecodeE2M1(value) * finalScale);
        ++index;
        if (++col == cols)
        {
            col = 0;
            ++row;
            if (row < rows && header.rowScaledNVFP4) { rowAmax = rowAmaxData[row]; }
            lastScaleIndex = std::numeric_limits<uint64_t>::max();
        }
    }
}

void DequantizeToBFloat16Range(std::span<char const> const& storage, uint64_t elementOffset, std::span<uint16_t> output)
{
    auto const& header = *reinterpret_cast<quant::Header const*>(storage.data());    // NOLINT
    switch (header.encoding)
    {
    case quant::Encoding::FP8E4M3: FP8E4M3ToBFloat16Range(storage, elementOffset, output); return;
    case quant::Encoding::MXFP8E4M3: MXFP8E4M3ToBFloat16Range(storage, elementOffset, output); return;
    case quant::Encoding::NVFP4E2M1: NVFP4E2M1ToBFloat16Range(storage, elementOffset, output); return;
    default: throw std::runtime_error(fmt::format("Unsupported host dequantization encoding {}", static_cast<uint8_t>(header.encoding)));
    }
}

}    // namespace checkpoint::host_dequant
// NOLINTEND(readability-magic-numbers)
