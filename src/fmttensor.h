#pragma once
#include "tensor.h"

#include <fmt/ranges.h>
#include <fmt/std.h>
template <> struct fmt::formatter<Rank> : fmt::formatter<std::string_view>
{
    auto format(Rank const& rank, fmt::format_context& ctx) const    // NOLINT
    {
        if (rank == Rank::Invalid) { return fmt::format_to(ctx.out(), "?rank?"); }
        return fmt::format_to(ctx.out(), "{}", static_cast<uint32_t>(rank));
    }
};
template <> struct fmt::formatter<DeviceId> : fmt::formatter<std::string_view>
{
    auto format(DeviceId const& device, fmt::format_context& ctx) const    // NOLINT
    {
        return fmt::format_to(
            ctx.out(), "{}:{}", std::string_view(device.type.data(), strnlen(device.type.data(), device.type.size())), device.index);
    }
};

template <> struct fmt::formatter<Tensor::DType> : fmt::formatter<std::string_view>
{
    auto format(Tensor::DType const& dtype, fmt::format_context& ctx) const    // NOLINT
    {
        switch (dtype)
        {
        case Tensor::DType::Float16: return fmt::format_to(ctx.out(), "Float16");
        case Tensor::DType::BFloat16: return fmt::format_to(ctx.out(), "BFloat16");
        case Tensor::DType::Float32: return fmt::format_to(ctx.out(), "Float32");
        case Tensor::DType::Float64: return fmt::format_to(ctx.out(), "Float64");
        case Tensor::DType::Int8: return fmt::format_to(ctx.out(), "Int8");
        case Tensor::DType::Int16: return fmt::format_to(ctx.out(), "Int16");
        case Tensor::DType::Int32: return fmt::format_to(ctx.out(), "Int32");
        case Tensor::DType::Int64: return fmt::format_to(ctx.out(), "Int64");
        case Tensor::DType::UInt8: return fmt::format_to(ctx.out(), "UInt8");
        case Tensor::DType::UInt16: return fmt::format_to(ctx.out(), "UInt16");
        case Tensor::DType::UInt32: return fmt::format_to(ctx.out(), "UInt32");
        case Tensor::DType::UInt64: return fmt::format_to(ctx.out(), "UInt64");
        case Tensor::DType::Bool: return fmt::format_to(ctx.out(), "Bool");
        case Tensor::DType::FP8E4M3: return fmt::format_to(ctx.out(), "FP8E4M3");
        case Tensor::DType::MXFP8: return fmt::format_to(ctx.out(), "MXFP8");
        case Tensor::DType::NVFP4: return fmt::format_to(ctx.out(), "NVFP4");
        case Tensor::DType::Invalid: return fmt::format_to(ctx.out(), "Invalid");
        default: return fmt::format_to(ctx.out(), "Unknown");
        }
    }
};

template <> struct fmt::formatter<Tensor::Range> : fmt::formatter<std::string_view>
{
    auto format(Tensor::Range const& obj, fmt::format_context& ctx) const    // NOLINT
    { return fmt::format_to(ctx.out(), "({}, {})", obj.begin, obj.end); }
};

template <> struct fmt::formatter<Tensor::DimRange> : fmt::formatter<std::string_view>
{
    auto format(Tensor::DimRange const& obj, fmt::format_context& ctx) const    // NOLINT
    { return fmt::format_to(ctx.out(), "{}", fmt::join(obj, ",")); }
};

template <> struct fmt::formatter<Tensor> : fmt::formatter<std::string_view>
{
    auto format(Tensor const& tensor, fmt::format_context& ctx) const    // NOLINT
    {
        fmt::format_to(ctx.out(),
                       "Tensor(handle={:x} bytes={} durableDtype={} snapshotDtype={} fulldim=[",
                       tensor.handle,
                       tensor.storageSize,
                       tensor.durableDtype,
                       tensor.snapshotDtype);
        for (size_t i = 0; i < tensor.fulldim.size(); ++i)
        {
            if (i > 0) { fmt::format_to(ctx.out(), ", "); }
            fmt::format_to(ctx.out(), "{}", tensor.fulldim[i]);    // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
        fmt::format_to(ctx.out(), "])");
        return ctx.out();
    }
};

template <> struct fmt::formatter<TensorShard> : fmt::formatter<std::string_view>
{
    auto format(TensorShard const& obj, fmt::format_context& ctx) const    // NOLINT
    { return fmt::format_to(ctx.out(), "rank={} sharddim=[{}]", obj.rank, obj.sharddim); }
};

template <> struct fmt::formatter<DTensor> : fmt::formatter<std::string_view>
{
    auto format(DTensor const& tensor, fmt::format_context& ctx) const    // NOLINT
    {
        return fmt::format_to(ctx.out(),
                              "DTensor(durableDtype={} fulldim=[{}], holes={}, nshards={})",
                              tensor.durableDtype,
                              tensor.fulldim,
                              tensor.holes,
                              tensor.shards.size());
    }
};

#ifndef NDEBUG
inline void RankTopology::AssertValid() const
{
    auto numPeers = (static_cast<uint32_t>(worldSize) + ranksPerPeer - 1) / ranksPerPeer;
    if (numPeers != peers.size())
    {
        throw std::runtime_error(fmt::format(
            "[native_ckpt][{}] Number of peers {} in root state does not match global world size {}", rank, numPeers, worldSize));
    }
    if (rank == Rank::Invalid || worldSize == Rank::Invalid)
    {
        throw std::runtime_error(fmt::format("[native_ckpt][{}] Rank and world size must be set", rank));
    }
    if (rank >= worldSize)
    {
        throw std::runtime_error(fmt::format("[native_ckpt][{}] Rank must be less than world size={}", rank, worldSize));
    }
    if (static_cast<uint32_t>(worldSize) % ranksPerPeer != 0)
    {
        throw std::runtime_error(
            fmt::format("[native_ckpt][{}] Worldsize={} must be divisible by ranks-per-peer={}", rank, worldSize, ranksPerPeer));
    }
    if ((static_cast<uint32_t>(worldSize) / ranksPerPeer) % peersPerNode != 0)
    {
        throw std::runtime_error(fmt::format("[native_ckpt][{}] Number of peers={} must be divisible by peers per node={}",
                                             rank,
                                             static_cast<uint32_t>(worldSize) / ranksPerPeer,
                                             peersPerNode));
    }
}
#endif
