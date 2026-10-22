#pragma once

#include "tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>
#include <vector>

namespace checkpoint::quant
{

inline constexpr std::array<uint8_t, 8> Magic{'N', 'V', 'R', 'X', 'Q', 'N', 'T', 'Z'};
inline constexpr uint8_t                CurrentVersion = 1;

enum class NVFP4E4M3Mode : uint8_t
{
    ScaleBound448,
    ScaleBound256,
};

enum class Encoding : uint8_t
{
    MXFP8E4M3 = 1,
    NVFP4E2M1,
    FP8E4M3,
};

enum class ScaleLayout : uint8_t
{
    Compact = 0,
    GemmSwizzled,
};

struct Component
{
    uint64_t handle{};
    uint64_t sizeBytes{};
    uint64_t rows{};
    uint64_t cols{};

    bool operator==(Component const& other) const = default;
};

struct Header
{
    std::array<uint8_t, 8> magic{Magic};

    uint64_t totalSizeBytes{};
    uint64_t logicalRows{};
    uint64_t logicalCols{};

    uint8_t                version{CurrentVersion};
    Encoding               encoding{Encoding::MXFP8E4M3};
    ScaleLayout            scaleLayout{ScaleLayout::Compact};
    bool                   rowScaledNVFP4{};
    NVFP4E4M3Mode          nvfp4E4M3Mode{NVFP4E4M3Mode::ScaleBound448};
    std::array<uint8_t, 3> reserved{};

    DeviceId  device;
    Component rowwiseData{};
    Component rowwiseScaleInv{};
    Component rowwiseAmax{};

    bool operator==(Header const& other) const = default;

    template <typename T>
        requires(sizeof(T) == 1)
    [[nodiscard]] constexpr auto DataSubSpan() const
    {
        auto spn = std::span(reinterpret_cast<T const*>(this), totalSizeBytes);    // NOLINT
        return spn.subspan(sizeof(Header), rowwiseData.sizeBytes);
    }

    template <typename T>
        requires(sizeof(T) == 1)
    [[nodiscard]] constexpr auto ScaleInvSubSpan() const
    {
        auto spn = std::span(reinterpret_cast<T const*>(this), totalSizeBytes);    // NOLINT
        return spn.subspan(sizeof(Header) + rowwiseData.sizeBytes, rowwiseScaleInv.sizeBytes);
    }

    [[nodiscard]] constexpr auto AmaxSubSpan() const
    {
        auto const* ptr = reinterpret_cast<char const*>(this)    // NOLINT
                          + (sizeof(Header) + rowwiseData.sizeBytes + rowwiseScaleInv.sizeBytes);
        return std::span(reinterpret_cast<float const*>(ptr), rowwiseAmax.sizeBytes / sizeof(float));    // NOLINT
    }
};

using HeaderMap = std::vector<Header>;

[[nodiscard]] DType SnapshotDType(Encoding encoding);

void                        FinalizeHeaders(HeaderMap& headers);
[[nodiscard]] Header const& FindHeader(HeaderMap const& headers, DeviceId device, uint64_t captureHandle);

void SerializeSnapshotSegments(uint64_t                                                 tensorHandle,
                               uint64_t                                                 offset,
                               uint64_t                                                 size,
                               std::function<void(DeviceId, uint64_t, uint64_t)> const& appendDevice);

static_assert(std::is_trivially_copyable_v<Component>);
static_assert(std::is_trivially_copyable_v<Header>);
static_assert(sizeof(Component) == 32);
static_assert(sizeof(Header) == 144);    // NOLINT(readability-magic-numbers)

}    // namespace checkpoint::quant
