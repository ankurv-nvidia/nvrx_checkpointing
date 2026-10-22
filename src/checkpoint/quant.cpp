#include "quant.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace checkpoint::quant
{
static auto HeaderKey(Header const& header)
{ return std::pair{header.device, header.rowwiseData.handle}; }

DType SnapshotDType(Encoding const encoding)
{
    switch (encoding)
    {
    case Encoding::FP8E4M3: return DType::FP8E4M3;
    case Encoding::MXFP8E4M3: return DType::MXFP8;
    case Encoding::NVFP4E2M1: return DType::NVFP4;
    default: break;
    }
    throw std::runtime_error("Quantized storage encoding is invalid");
}

void FinalizeHeaders(HeaderMap& headers)
{
    std::ranges::sort(headers, {}, HeaderKey);
    auto output = headers.begin();
    for (auto input = headers.begin(); input != headers.end(); ++input)
    {
        if (output != headers.begin() && HeaderKey(*(output - 1)) == HeaderKey(*input))
        {
            if (*(output - 1) != *input) { throw std::runtime_error("Quantized tensor handle has conflicting headers"); }
            continue;
        }
        if (output != input) { *output = *input; }
        ++output;
    }
    headers.erase(output, headers.end());
}

Header const& FindHeader(HeaderMap const& headers, DeviceId const device, uint64_t const captureHandle)
{
    auto const key   = std::pair{device, captureHandle};
    auto const match = std::ranges::lower_bound(headers, key, {}, HeaderKey);
    if (match == headers.end() || HeaderKey(*match) != key) { throw std::runtime_error("Quantized tensor header is not attached"); }
    return *match;
}

void SerializeSnapshotSegments(uint64_t                                                 tensorHandle,
                               uint64_t const                                           offset,
                               uint64_t const                                           size,
                               std::function<void(DeviceId, uint64_t, uint64_t)> const& appendDevice)
{
    if (tensorHandle == 0) { throw std::runtime_error("Quantized storage handle is null"); }
    auto const& source = *reinterpret_cast<checkpoint::quant::Header const*>(static_cast<uintptr_t>(tensorHandle));    // NOLINT
    if (source.magic != Magic || source.version != CurrentVersion || source.logicalRows == 0 || source.logicalCols == 0)
    {
        throw std::runtime_error("Quantized storage header is invalid");
    }
    if (source.encoding == Encoding::NVFP4E2M1)
    {
        if (source.nvfp4E4M3Mode != NVFP4E4M3Mode::ScaleBound448 && source.nvfp4E4M3Mode != NVFP4E4M3Mode::ScaleBound256)
        {
            throw std::runtime_error("NVFP4 E4M3 scale bound is invalid");
        }
    }
    else if ((source.encoding != Encoding::FP8E4M3 && source.encoding != Encoding::MXFP8E4M3) || source.rowwiseAmax.sizeBytes != 0)
    {
        throw std::runtime_error("Quantized storage components are invalid");
    }
    if (offset > source.totalSizeBytes || size > source.totalSizeBytes - offset)
    {
        throw std::runtime_error("Quantized storage range is invalid");
    }

    uint64_t const end     = offset + size;
    uint64_t       emitted = 0;
    auto           appendOverlap
        = [&](DeviceId const device, uint64_t const sourceHandle, uint64_t const serializedOffset, uint64_t const regionSize) {
              auto const begin = std::max(offset, serializedOffset);
              auto const limit = std::min(end, serializedOffset + regionSize);
              if (begin >= limit) { return; }
              appendDevice(device, sourceHandle + begin - serializedOffset, limit - begin);
              emitted += limit - begin;
          };

    appendOverlap("cpu", tensorHandle, 0, sizeof(Header));
    uint64_t componentOffset = sizeof(Header);
    auto     appendComponent = [&](Component const& component) {
        if (component.handle == 0 || component.sizeBytes == 0 || component.rows == 0 || component.cols == 0)
        {
            throw std::runtime_error("Quantized storage component is invalid");
        }
        appendOverlap(source.device, component.handle, componentOffset, component.sizeBytes);
        componentOffset += component.sizeBytes;
    };
    appendComponent(source.rowwiseData);
    appendComponent(source.rowwiseScaleInv);
    if (source.encoding == Encoding::NVFP4E2M1) { appendComponent(source.rowwiseAmax); }
    if (emitted != size) { throw std::runtime_error("Quantized storage does not cover its snapshot range"); }
}

}    // namespace checkpoint::quant
