#pragma once
#include <fmt/format.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <span>
struct Blob
{
    void*  data{nullptr};
    size_t size{0};

    Blob() = default;
    Blob(size_t sizeIn) : data(std::aligned_alloc(alignof(std::max_align_t), sizeIn)), size(sizeIn)
    {
        if (data == nullptr)
        {
            fmt::print(stderr, "[native_ckpt][ERROR] Failed to allocate blob of size {} bytes\n", sizeIn);
            throw std::bad_alloc();
        }
    }
    ~Blob()
    {
        if (data != nullptr) { std::free(data); }    // NOLINT
    }

    Blob(Blob const& other)            = delete;
    Blob& operator=(Blob const& other) = delete;
    Blob(Blob&& other) noexcept : data(other.data), size(other.size)
    {
        other.data = nullptr;
        other.size = 0;
    }
    Blob& operator=(Blob&& other) noexcept
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
        if (index >= size) { throw std::out_of_range(fmt::format("Blob access out of range: index {} >= size {}", index, size)); }
        return reinterpret_cast<T*>(data)[index];    // NOLINT
    }
    template <typename T> T const& At(size_t index) const
    {
        if (index >= size) { throw std::out_of_range(fmt::format("Blob access out of range: index {} >= size {}", index, size)); }
        return reinterpret_cast<T const*>(data)[index];    // NOLINT
    }

    template <typename T> std::span<T const> AsSpan(size_t index, size_t count) const
    {
        auto* ptr = &At<T>(index);
        if (index + (count * sizeof(T)) > size)
        {
            throw std::out_of_range(
                fmt::format("Blob access out of range: index {} + count {} * sizeof(T) {} > size {}", index, count, sizeof(T), size));
        }
        return std::span<T const>{ptr, count};
    }
};
