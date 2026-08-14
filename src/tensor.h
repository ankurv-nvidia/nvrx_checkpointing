#pragma once

#include "CommonMacros.h"
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NDEBUG
#define TENSOR_SHARD_ASSERTION_ENABLED
#endif

namespace std
{
template <typename T, std::size_t N> struct inplace_vector : std::array<T, N>    // NOLINT
{
    inplace_vector()  = default;    // NOLINT(cppcoreguidelines-pro-type-member-init)
    ~inplace_vector() = default;
    CLASS_DEFAULT_COPY_AND_MOVE(inplace_vector);

    size_t               size_ = 0;                                                  // NOLINT(readability-identifier-naming)
    void                 push_back(T const& value) { this->at(size_++) = value; }    // NOLINT
    [[nodiscard]] size_t size() const { return size_; }                              // NOLINT(readability-identifier-naming)
    [[nodiscard]] auto   end() const                                                 // NOLINT(readability-identifier-naming)
    { return this->begin() + static_cast<std::ptrdiff_t>(size_); }    // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    [[nodiscard]] auto end() { return this->begin() + static_cast<std::ptrdiff_t>(size_); }    // NOLINT(readability-identifier-naming)

    // Element-wise comparisons must respect size_. The inherited std::array operators
    // compare all N slots and would yield wrong answers if any slot beyond size_ ever
    // diverged (e.g. on resize/reuse), silently corrupting std::set<DimRange> ordering.
    bool operator==(inplace_vector const& other) const { return size_ == other.size_ && std::equal(this->begin(), end(), other.begin()); }

    bool operator<(inplace_vector const& other) const
    { return std::lexicographical_compare(this->begin(), end(), other.begin(), other.end()); }
};
}    // namespace std
// Pay-per-use model
// Dynamic Tensor
//      Max Dimensions = 4;
//      Runtime Dimensions
//      Runtime Shards . Full Dim and Shard Dim
//      Mesh
//
// Sharded Tensor
// Static Dimensional Tensor
struct DynamicSharded4TensorTraits
{
    using SizeT = size_t;
    struct Range
    {
        SizeT begin;
        SizeT end;
        bool  operator==(Range const& other) const = default;
        bool  operator<(Range const& other) const
        {
            if (begin < other.begin) { return true; }
            if (begin > other.begin) { return false; }
            return end < other.end;
        }

        [[nodiscard]] bool Intersects(Range const& other) const { return end > other.begin && other.end > begin; }
    };

    static constexpr SizeT MaxDims = 4;
    using Dim                      = std::inplace_vector<SizeT, MaxDims>;
    using DimRange                 = std::inplace_vector<Range, MaxDims>;

    enum class DType : uint8_t
    {
        Invalid = 0,

        Float16,
        BFloat16,
        Float32,
        Float64,

        UInt8,
        UInt16,
        UInt32,
        UInt64,

        Int8,
        Int16,
        Int32,
        Int64,

        Bool,
    };

    static constexpr size_t GetDTypeSize(DType dtype)
    {
        switch (dtype)
        {
        case DType::Float16: return 2;    // NOLINT(bugprone-branch-clone)
        case DType::BFloat16: return 2;
        case DType::Float32: return 4;
        case DType::Float64: return 8;

        case DType::Int8: return 1;
        case DType::Int16: return 2;
        case DType::Int32: return 4;
        case DType::Int64: return 8;

        case DType::UInt8: return 1;    // NOLINT(bugprone-branch-clone)
        case DType::UInt16: return 2;
        case DType::UInt32: return 4;
        case DType::UInt64: return 8;

        case DType::Bool:
        case DType::Invalid: [[fallthrough]];
        default: throw std::runtime_error("Invalid DType:" + std::to_string(static_cast<int>(dtype)));
        }
    }
};

struct DeviceId
{
    constexpr DeviceId() = default;
    constexpr DeviceId(std::array<char, 4> const& strIn) : type(strIn) {}
    template <size_t N> constexpr DeviceId(char const (&strIn)[N]) : DeviceId(std::string_view(strIn, N - 1)) {}    // NOLINT
    constexpr DeviceId(std::string_view const& sv)
    {
        auto it     = sv.find(':');
        it          = (it == std::string_view::npos) ? sv.size() : it;
        auto svType = sv.substr(0, it);
        if (svType.size() > 4) { throw std::runtime_error("DeviceId string must be less than 4 characters"); }
        for (size_t i = 0; i < 4 && i < svType.size(); ++i) { type[i] = svType[i]; }    // NOLINT
        ++it;
        while (it < sv.size())
        {
            if (std::isdigit(sv[it]) == 0)
            {
                throw std::runtime_error(fmt::format("DeviceId string after ':' must be a number representing device index: {}", sv));
            }
            index = (index * 10) + static_cast<uint32_t>(sv[it] - '0');    // NOLINT(readability-magic-numbers)
            ++it;
        }
    }
    explicit constexpr DeviceId(std::string const& sv) : DeviceId(std::string_view(sv)) {}
    constexpr auto operator<=>(DeviceId const& other) const = default;
    // constexpr auto operator<=>(std::string_view const& other) const { return *this <=> DeviceId(other); }

    constexpr operator uint64_t() const
    {
        return (static_cast<uint64_t>(*reinterpret_cast<uint32_t const*>(type.data())) << 32) | static_cast<uint64_t>(index);    // NOLINT
    }

    [[nodiscard]] bool IsType(std::string_view const& typeStr) const
    {
        return typeStr.size() <= 4
               && std::equal(typeStr.begin(),
                             typeStr.end(),
                             type.begin(),
                             type.begin() + typeStr.size());    // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    std::array<char, 4> type{};
    uint32_t            index = 0;
};

template <> struct std::hash<DeviceId>
{
    constexpr size_t operator()(DeviceId const& device) const { return static_cast<uint64_t>(device); }
};
template <size_t TDims> struct StaticShardedTensorTraits
{
    using Dim                     = std::array<size_t, TDims>;
    static constexpr bool Sharded = true;
};

template <typename Traits>
concept ConceptTensorTraits = requires {
    typename Traits::Dim;
    typename Traits::DimRange;
    Traits::MaxDims;
};

template <ConceptTensorTraits Traits> struct TensorT
{
    static constexpr size_t MaxDims = Traits::MaxDims;

    using DimRange = Traits::DimRange;
    using Dim      = Traits::Dim;
    using Range    = Traits::Range;
    using SizeT    = Traits::SizeT;
    using DType    = Traits::DType;

    DimRange      sharddim{};
    Dim           fulldim{};
    Traits::DType dtype{};
    uint64_t      handle = 0;
    // DeviceGUID    deviceGUID{0};
    DeviceId device{"cpu"};

    bool operator==(TensorT<Traits> const& other) const = default;

    [[nodiscard]] uint64_t ShardSizeBytes() const
    {
        if (sharddim.size() == 0) { return 0; }
        uint64_t size = 1;
        for (auto const& r : sharddim) { size *= (r.end - r.begin); }
        return size * static_cast<uint64_t>(Traits::GetDTypeSize(dtype));
    }

    [[nodiscard]] uint64_t FullSizeBytes() const
    {
        if (fulldim.size() == 0) { return 0; }
        uint64_t size = 1;
        for (auto const& r : fulldim) { size *= r; }
        return size * static_cast<uint64_t>(Traits::GetDTypeSize(dtype));
    }

    void AssertShardIsValid() const
    {
#ifdef TENSOR_SHARD_ASSERTION_ENABLED
        for (size_t dim = 0; dim < sharddim.size(); ++dim)
        {
            auto const& r = sharddim[dim];
            if (r.end > fulldim[dim])
            {
                throw std::runtime_error(fmt::format(
                    "[native_ckpt] : Invalid shard range: end {} is greater than full dim {} for dim {}", r.end, fulldim[dim], dim));
            }
            if (r.begin >= r.end)
            {
                throw std::runtime_error(
                    fmt::format("[native_ckpt] : Invalid shard range: begin {} is not less than end {} for dim {}", r.begin, r.end, dim));
            }
            if (fulldim[dim] > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(fmt::format("[native_ckpt] : Full dim {} is too large for dim {}", fulldim[dim], dim));
            }
        }
#endif
    }
};

enum class Rank : uint32_t
{
    Default = 0,
    Invalid = std::numeric_limits<uint32_t>::max()
};

struct RankTopology
{
    Rank     rank{Rank::Invalid};
    Rank     worldSize{Rank::Invalid};
    uint32_t ranksPerPeer{};
    uint32_t peersPerNode{};

    void AssertValid() const;

    std::vector<std::string> peers;
};

template <ConceptTensorTraits Traits> struct RankedTensorShardT
{
    Traits::DimRange sharddim{};
    DeviceId         device{"cpu"};
    uint64_t         handle = 0;
    Rank             rank{Rank::Invalid};
    uint32_t         padding{};

    bool operator==(RankedTensorShardT<Traits> const& other) const = default;

    bool operator<(RankedTensorShardT<Traits> const& other) const
    {
        if (rank < other.rank) { return true; }
        if (rank > other.rank) { return false; }
        return sharddim < other.sharddim;
    }

    [[nodiscard]] uint64_t ShardSizeBytes(Traits::DType const& dtype) const
    {
        if (sharddim.size() == 0) { return 0; }
        uint64_t size = 1;
        for (auto const& r : sharddim) { size *= (r.end - r.begin); }
        return size * static_cast<uint64_t>(Traits::GetDTypeSize(dtype));
    }

    void AssertShardIsValid([[maybe_unused]] Traits::Dim const& fulldim) const
    {
#ifdef TENSOR_SHARD_ASSERTION_ENABLED
        for (size_t dim = 0; dim < sharddim.size(); ++dim)
        {
            auto const& r = sharddim[dim];
            if (r.end > fulldim[dim])
            {
                throw std::runtime_error(fmt::format(
                    "[native_ckpt] : Invalid shard range: end {} is greater than full dim {} for dim {}", r.end, fulldim[dim], dim));
            }
            if (r.begin >= r.end)
            {
                throw std::runtime_error(
                    fmt::format("[native_ckpt] : Invalid shard range: begin {} is not less than end {} for dim {}", r.begin, r.end, dim));
            }
            if (fulldim[dim] > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(fmt::format("[native_ckpt] : Full dim {} is too large for dim {}", fulldim[dim], dim));
            }
        }
#endif
    }
};

template <ConceptTensorTraits Traits> struct TensorShardMapT
{
    using Tensor            = TensorT<Traits>;
    using RankedTensorShard = RankedTensorShardT<Traits>;

    using DimRange = Traits::DimRange;
    using Dim      = Traits::Dim;
    using Range    = Traits::Range;
    using SizeT    = Traits::SizeT;
    using DType    = Traits::DType;

    Traits::DType dtype{};
    Dim           fulldim{};

    std::set<RankedTensorShard> shards;
    // Runtime info
    std::vector<DimRange> holes;

    TensorShardMapT() = default;
    TensorShardMapT(DType dtypeIn, Dim const& fulldimIn) : dtype(dtypeIn), fulldim(fulldimIn)
    {
        // Initially the entire tensor is a hole
        holes.push_back(DimRange{});
        for (size_t i = 0; i < fulldim.size(); ++i) { holes.back().push_back({0, fulldim[i]}); }
    }

    bool operator==(TensorShardMapT<Traits> const& other) const
    { return dtype == other.dtype && fulldim == other.fulldim && shards == other.shards; }

    static bool Intersects(DimRange const& r1, DimRange const& r2)
    {
        for (auto const& [dim1, dim2] : std::ranges::zip_view(r1, r2))    // NOLINT(readability-use-anyofallof)
        {
            if (dim1.end <= dim2.begin || dim2.end <= dim1.begin) { return false; }
        }
        return true;
    }

    static bool Contains(DimRange const& outer, DimRange const& inner)
    {
        for (auto const& [dimOuter, dimInner] : std::ranges::zip_view(outer, inner))    // NOLINT(readability-use-anyofallof)
        {
            if (dimOuter.begin > dimInner.begin || dimOuter.end < dimInner.end) { return false; }
        }
        return true;
    }

    void AddShard(RankedTensorShard const& rankTensor)
    {
        if (rankTensor.ShardSizeBytes(dtype) == 0) { return; }
        shards.insert({rankTensor.sharddim, rankTensor.device, rankTensor.handle, rankTensor.rank});
        // For each hole that the shard intersects with
        // We could produce 2*MaxDim new holes in the worst case
        std::vector<DimRange> newHoles;
        for (size_t i = 0; i < holes.size(); i++)
        {
            auto& hole = holes[i];
            if (!Intersects(hole, rankTensor.sharddim)) { continue; }
            if (Contains(rankTensor.sharddim, hole))
            {
                hole = {};    // Mark as empty hole. Will be cleanup in 2nd pass
                continue;
            }
            // Subtract the shard from the hole by peeling slices dimension by dimension.
            // In each dimension, carve off the part of the remaining region that lies
            // outside the shard, then constrain the remaining region to the shard
            // before moving to the next dimension.
            DimRange remaining = hole;
            for (size_t dim = 0; dim < hole.size(); ++dim)
            {
                // Left slice: portion of remaining that is before the shard in this dim
                if (remaining[dim].begin < rankTensor.sharddim[dim].begin)
                {
                    DimRange newHole = remaining;
                    newHole[dim].end = rankTensor.sharddim[dim].begin;
                    newHoles.push_back(newHole);
                }
                // Right slice: portion of remaining that is after the shard in this dim
                if (remaining[dim].end > rankTensor.sharddim[dim].end)
                {
                    DimRange newHole   = remaining;
                    newHole[dim].begin = rankTensor.sharddim[dim].end;
                    newHoles.push_back(newHole);
                }
                // Constrain remaining to the shard bounds in this dimension
                remaining[dim].begin = std::max(remaining[dim].begin, rankTensor.sharddim[dim].begin);
                remaining[dim].end   = std::min(remaining[dim].end, rankTensor.sharddim[dim].end);
            }
            hole = {};    // Mark as empty; replaced by the new holes above
        }
        // Add new holes and remove any that were marked empty
        holes.erase(std::remove_if(holes.begin(), holes.end(), [](auto const& h) { return h.size() == 0; }), holes.end());
        holes.insert(holes.end(), newHoles.begin(), newHoles.end());
    }

    [[nodiscard]] uint64_t FullDimSizeBytes() const
    {
        if (fulldim.size() == 0) { return 0; }
        uint64_t size = 1;
        for (auto const& r : fulldim) { size *= r; }
        return size * static_cast<uint64_t>(Traits::GetDTypeSize(dtype));
    }
};

template <ConceptTensorTraits Traits> struct std::hash<RankedTensorShardT<Traits>>
{
    constexpr size_t operator()(RankedTensorShardT<Traits> const& shard) const
    { return std::hash<uint32_t>{}(shard.rank) ^ std::hash<DeviceId>{}(shard.device) ^ std::hash<uint64_t>{}(shard.handle); }
};

template <typename T, size_t N> struct std::hash<std::inplace_vector<T, N>>
{
    constexpr size_t operator()(std::inplace_vector<T, N> const& obj) const
    {
        return std::accumulate(
            obj.begin(), obj.end(), static_cast<size_t>(0), [](size_t acc, T const& value) { return acc ^ std::hash<T>{}(value); });
    }
};

template <> struct std::hash<DynamicSharded4TensorTraits::Range>
{
    constexpr size_t operator()(DynamicSharded4TensorTraits::Range const& obj) const
    { return std::hash<size_t>{}(obj.begin) ^ std::hash<size_t>{}(obj.end); }
};

using DS4Tensor = TensorT<DynamicSharded4TensorTraits>;

// NOLINTNEXTLINE(readability-magic-numbers)
static_assert(sizeof(DS4Tensor) == 136, "Unexpected padding in DS4Tensor. Size should be exactly 136 bytes for efficient serialization");

using DS4TensorShardMap = TensorShardMapT<DynamicSharded4TensorTraits>;
