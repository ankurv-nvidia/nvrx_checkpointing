#pragma once

#include "CommonMacros.h"

#include <fmt/ranges.h>
#include <fmt/std.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef NDEBUG
#define TENSOR_SHARD_ASSERTION_ENABLED
#endif

namespace std
{
// NOLINTBEGIN(readability-identifier-naming,cppcoreguidelines-pro-bounds-pointer-arithmetic )
template <typename T, uint8_t N> class inplace_vector    // NOLINT
{
    public:
    constexpr inplace_vector() = default;    // NOLINT
    ~inplace_vector()          = default;
    CLASS_DEFAULT_COPY_AND_MOVE(inplace_vector);

    void push_back(T const& value)
    {
        values_.at(size_) = value;
        ++size_;
    }
    [[nodiscard]] size_t size() const
    {
        if (size_ > N) { throw std::length_error("inplace_vector size exceeds capacity"); }
        return size_;
    }
    [[nodiscard]] bool empty() const { return size() == 0; }

    [[nodiscard]] auto begin() const LFTBND { return values_.begin(); }
    [[nodiscard]] auto begin() LFTBND { return values_.begin(); }
    [[nodiscard]] auto end() const LFTBND { return begin() + static_cast<std::ptrdiff_t>(size()); }
    [[nodiscard]] auto end() LFTBND { return begin() + static_cast<std::ptrdiff_t>(size()); }

    [[nodiscard]] T const& operator[](size_t index) const LFTBND
    {
        if (index >= size()) { throw std::out_of_range("inplace_vector index exceeds size"); }
        return values_[index];
    }
    [[nodiscard]] T& operator[](size_t index) LFTBND
    {
        if (index >= size()) { throw std::out_of_range("inplace_vector index exceeds size"); }
        return values_[index];
    }

    bool operator==(inplace_vector const& other) const { return size_ == other.size_ && std::equal(this->begin(), end(), other.begin()); }
    bool operator<(inplace_vector const& other) const
    { return std::lexicographical_compare(this->begin(), end(), other.begin(), other.end()); }

    private:
    std::array<T, N>                    values_{};
    uint8_t                             size_{};
    std::array<uint8_t, alignof(T) - 1> padding_{};
};
// NOLINTEND(readability-identifier-naming,cppcoreguidelines-pro-bounds-pointer-arithmetic )

}    // namespace std

struct DeviceId
{
    constexpr DeviceId() = default;
    constexpr explicit DeviceId(std::array<char, 4> const& strIn) : type(strIn) {}
    template <size_t N> constexpr DeviceId(char const (&strIn)[N]) : DeviceId(std::string_view(strIn, N - 1)) {}    // NOLINT
    constexpr explicit DeviceId(std::string_view const& sv)
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

    constexpr operator uint64_t() const    // NOLINT
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

enum class Rank : uint32_t
{
    Default = 0,
    Invalid = std::numeric_limits<uint32_t>::max(),
};

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

    MXFP8,
    NVFP4,
    FP8E4M3,
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

template <> struct std::hash<DeviceId>
{
    constexpr size_t operator()(DeviceId const& device) const { return static_cast<uint64_t>(device); }
};

struct TensorShard
{
    static constexpr size_t MaxDims = 4;

    struct Range
    {
        uint64_t begin{};
        uint64_t end{};

        bool operator==(Range const& other) const = default;
        bool operator<(Range const& other) const
        {
            if (begin < other.begin) { return true; }
            if (begin > other.begin) { return false; }
            return end < other.end;
        }

        [[nodiscard]] bool Intersects(Range const& other) const { return end > other.begin && other.end > begin; }
    };

    using Dim      = std::inplace_vector<uint64_t, MaxDims>;
    using DimRange = std::inplace_vector<Range, MaxDims>;

    Rank                   rank{Rank::Invalid};
    DType                  snapshotDtype{DType::Invalid};
    std::array<uint8_t, 3> padding{};    // to avoid Uninitialized memory-reads in valgrind
    uint64_t               handle{};
    uint64_t               storageSize{};
    DimRange               sharddim;

    DeviceId device;    // Only for snapshotting. Try to avoid using it

    bool operator==(TensorShard const& other) const = default;

    bool operator<(TensorShard const& other) const
    {
        if (rank < other.rank) { return true; }
        if (rank > other.rank) { return false; }
        return sharddim < other.sharddim;
    }

    [[nodiscard]] uint64_t StorageSizeBytes() const { return storageSize; }

    void AssertShardIsValid([[maybe_unused]] Dim const& fulldim) const
    {
        (void)sharddim.size();
        (void)fulldim.size();
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

struct Tensor
{
    using DType    = ::DType;
    using Dim      = TensorShard::Dim;
    using DimRange = TensorShard::DimRange;
    using Range    = TensorShard::Range;

    static constexpr auto MaxDims = TensorShard::MaxDims;

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
        case DType::FP8E4M3:
        case DType::MXFP8:
        case DType::NVFP4:
        case DType::Invalid: [[fallthrough]];
        default: throw std::runtime_error("Invalid DType:" + std::to_string(static_cast<int>(dtype)));
        }
    }

    DType                   durableDtype{DType::Invalid};
    DType                   snapshotDtype{DType::Invalid};
    std::array<uint8_t, 6U> padding{};    // NOLINT(readability-magic-numbers)
    uint64_t                handle{};
    uint64_t                storageSize{};
    Dim                     fulldim;

    DeviceId device;    // Only for snapshotting. Try to avoid using it

    [[nodiscard]] uint64_t StorageSizeBytes() const { return storageSize; }

    bool operator==(Tensor const& other) const = default;

    void AssertShardIsValid() const { (void)fulldim.size(); }
};
struct DTensor
{
    using DimRange = Tensor::DimRange;
    using Shard    = TensorShard;

    Tensor::DType durableDtype{Tensor::DType::Invalid};
    Tensor::Dim   fulldim;

    std::set<TensorShard> shards;
    // Runtime info
    std::vector<Tensor::DimRange> holes;

    DTensor() = default;
    DTensor(Tensor::DType durableDtypeIn, Tensor::Dim const& fulldimIn) : durableDtype(durableDtypeIn), fulldim(fulldimIn)
    {
        // Initially the entire tensor is a hole
        holes.emplace_back();
        for (auto const i : fulldim) { holes.back().push_back({.begin = 0, .end = i}); }
    }

    bool operator==(DTensor const& other) const
    { return durableDtype == other.durableDtype && fulldim == other.fulldim && shards == other.shards; }

    static bool Intersects(DimRange const& r1, DimRange const& r2)
    {
        for (auto const [dim1, dim2] : std::ranges::zip_view(r1, r2))    // NOLINT(readability-use-anyofallof)
        {
            if (dim1.end <= dim2.begin || dim2.end <= dim1.begin) { return false; }
        }
        return true;
    }

    static bool Contains(DimRange const& outer, DimRange const& inner)
    {
        for (auto const [dimOuter, dimInner] : std::ranges::zip_view(outer, inner))    // NOLINT(readability-use-anyofallof)
        {
            if (dimOuter.begin > dimInner.begin || dimOuter.end < dimInner.end) { return false; }
        }
        return true;
    }

    void AddShard(TensorShard const& rankTensor)
    {
        if (rankTensor.StorageSizeBytes() == 0) { throw std::runtime_error("Tensor shard storage size must be nonzero"); }
        if (shards.empty() && holes.empty())
        {
            holes.emplace_back();
            for (auto const dim : fulldim) { holes.back().push_back({.begin = 0, .end = dim}); }
        }
        shards.insert(rankTensor);
        // For each hole that the shard intersects with
        // We could produce 2*MaxDim new holes in the worst case
        std::vector<DimRange> newHoles;
        for (auto& hole : holes)
        {
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
        holes.erase(std::remove_if(holes.begin(), holes.end(), [](auto const& h) { return h.size() == 0; }), holes.end());    // NOLINT
        holes.insert(holes.end(), newHoles.begin(), newHoles.end());
    }
};

template <> struct std::hash<TensorShard>
{
    constexpr size_t operator()(TensorShard const& shard) const
    { return std::hash<uint32_t>{}(static_cast<uint32_t>(shard.rank)) ^ std::hash<uint64_t>{}(shard.handle); }
};

template <typename T, size_t N> struct std::hash<std::inplace_vector<T, N>>
{
    constexpr size_t operator()(std::inplace_vector<T, N> const& obj) const
    {
        return std::accumulate(
            obj.begin(), obj.end(), static_cast<size_t>(0), [](size_t acc, T const& value) { return acc ^ std::hash<T>{}(value); });
    }
};

template <> struct std::hash<Tensor::Range>
{
    constexpr size_t operator()(Tensor::Range const& obj) const { return std::hash<size_t>{}(obj.begin) ^ std::hash<size_t>{}(obj.end); }
};

// NOLINTBEGIN(readability-magic-numbers)
static_assert(sizeof(DeviceId) == 8);
static_assert(sizeof(Tensor::Range) == 16);
static_assert(sizeof(Tensor::Dim) == 40);
static_assert(sizeof(Tensor::DimRange) == 72);
static_assert(sizeof(Tensor) == 72);
static_assert(sizeof(TensorShard) == 104);
// NOLINTEND(readability-magic-numbers)
static_assert(std::is_trivially_copyable_v<DeviceId>);
static_assert(std::is_trivially_copyable_v<DType>);
static_assert(std::is_trivially_copyable_v<Tensor::Range>);
static_assert(std::is_trivially_copyable_v<Tensor::Dim>);
static_assert(std::is_trivially_copyable_v<Tensor::DimRange>);
static_assert(std::is_trivially_copyable_v<Tensor>);
static_assert(std::is_trivially_copyable_v<TensorShard>);
static_assert(std::is_standard_layout_v<DeviceId>);
static_assert(std::is_standard_layout_v<DType>);
static_assert(std::is_standard_layout_v<Tensor::Range>);
static_assert(std::is_standard_layout_v<Tensor::Dim>);
static_assert(std::is_standard_layout_v<Tensor::DimRange>);
static_assert(std::is_standard_layout_v<Tensor>);
static_assert(std::is_standard_layout_v<TensorShard>);
static_assert(std::has_unique_object_representations_v<DeviceId>);
static_assert(std::has_unique_object_representations_v<DType>);
static_assert(std::has_unique_object_representations_v<Tensor::Range>);
static_assert(std::has_unique_object_representations_v<Tensor::Dim>);
static_assert(std::has_unique_object_representations_v<Tensor::DimRange>);
static_assert(std::has_unique_object_representations_v<Tensor>);
static_assert(std::has_unique_object_representations_v<TensorShard>);
