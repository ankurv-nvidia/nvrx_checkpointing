#pragma once
#include "CommonMacros.h"
#include <fmt/format.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <span>

struct ByteStorage
{
    void*  data{nullptr};
    size_t size{0};

    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wallocator-wrappers")
    static void* PadAlignedAlloc(size_t alignment, size_t size)
    {
        if (size == 0) { return nullptr; }
        auto paddedSize = (size + alignment - 1) / alignment * alignment;
        return std::aligned_alloc(alignment, paddedSize);    // NOLINT
    }
    SUPPRESS_WARNINGS_END

    ByteStorage() = default;
    explicit ByteStorage(size_t sizeIn) : data(PadAlignedAlloc(alignof(std::max_align_t), sizeIn)), size(sizeIn)
    {
        if (sizeIn != 0 && data == nullptr)
        {
            fmt::print(stderr, "[native_ckpt][ERROR] Failed to allocate byte storage of size {} bytes\n", sizeIn);
            throw std::bad_alloc();
        }
    }
    ~ByteStorage()
    {
        if (data != nullptr) { std::free(data); }    // NOLINT
    }

    ByteStorage(ByteStorage const& other)            = delete;
    ByteStorage& operator=(ByteStorage const& other) = delete;
    ByteStorage(ByteStorage&& other) noexcept : data(other.data), size(other.size)
    {
        other.data = nullptr;
        other.size = 0;
    }
    ByteStorage& operator=(ByteStorage&& other) noexcept
    {
        if (this != &other)
        {
            if (data != nullptr) { std::free(data); }    // NOLINT
            data       = other.data;
            size       = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void* operator[](size_t index) const { return static_cast<char*>(data) + index; }    // NOLINT

    template <typename T> T& At(size_t index)
    {
        if (index >= size) { throw std::out_of_range(fmt::format("ByteStorage access out of range: index {} >= size {}", index, size)); }
        return reinterpret_cast<T*>(data)[index];    // NOLINT
    }
    template <typename T> [[nodiscard]] T const& At(size_t index) const
    {
        if (index >= size) { throw std::out_of_range(fmt::format("ByteStorage access out of range: index {} >= size {}", index, size)); }
        return reinterpret_cast<T const*>(data)[index];    // NOLINT
    }

    template <typename T> [[nodiscard]] std::span<T const> AsSpan(size_t index, size_t count) const
    {
        auto* ptr = &At<T>(index);
        if (index + (count * sizeof(T)) > size)
        {
            throw std::out_of_range(fmt::format(
                "ByteStorage access out of range: index {} + count {} * sizeof(T) {} > size {}", index, count, sizeof(T), size));
        }
        return std::span<T const>{ptr, count};
    }
};
