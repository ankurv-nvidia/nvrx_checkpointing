#pragma once
#include "CommonMacros.h"

import std.compat;

namespace serdes
{
template <typename T> std::ostream& WriteTrivial(std::ostream& ostrm, T const& obj)
{
    return ostrm.write(reinterpret_cast<char const*>(&obj), sizeof(T));    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

template <typename T> T ReadTrivial(std::istream& istrm)
{
    T obj{};
    istrm.read(reinterpret_cast<char*>(&obj), sizeof(T));    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return obj;
}

template <typename T> std::ostream& WriteBlob(std::ostream& ostrm LFTBND, std::span<T> const& obj)
{
    if (obj.empty()) { return ostrm; }
    return ostrm.write(obj.data(), static_cast<std::streamsize>(obj.size() * sizeof(T)));
}

inline std::string ReadBlob(std::istream& istrm, uint32_t size)
{
    std::string result(size, '\0');
    istrm.read(result.data(), static_cast<std::streamsize>(size));
    return result;
}

template <typename TSize> inline std::string ReadBlobWithSize(std::istream& is)
{
    TSize size;
    is.read(reinterpret_cast<char*>(&size), sizeof(TSize));    // NOLINT
    std::string data(size, '\0');
    is.read(data.data(), static_cast<std::streamsize>(size));    // NOLINT
    return data;
}

template <typename TSize> auto WriteBlobWithSize(std::ostream& os LFTBND, std::string_view const& data) -> std::ostream&
{
    auto size = static_cast<TSize>(data.size());
    os.write(reinterpret_cast<char const*>(&size), sizeof(TSize));    // NOLINT
    os.write(data.data(), static_cast<std::streamsize>(size));        // NOLINT
    return os;
}

// Counts the bytes streamed through it without storing them.
class ByteCounter final : public std::streambuf
{
    public:
    [[nodiscard]] size_t Count() const { return _count; }

    protected:
    std::streamsize xsputn(char const* /*data*/, std::streamsize size) override
    {
        _count += static_cast<size_t>(size);
        return size;
    }

    int_type overflow(int_type ch) override
    {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) { ++_count; }
        return traits_type::not_eof(ch);
    }

    private:
    size_t _count = 0;
};

// Appends the bytes streamed through it to a caller-owned string.
class StringSink final : public std::streambuf
{
    public:
    explicit StringSink(std::string& out LFTBND) : _out(&out) {}

    protected:
    std::streamsize xsputn(char const* data, std::streamsize size) override
    {
        _out->append(data, static_cast<size_t>(size));
        return size;
    }

    int_type overflow(int_type ch) override
    {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) { _out->push_back(traits_type::to_char_type(ch)); }
        return traits_type::not_eof(ch);
    }

    private:
    std::string* _out;
};

}    // namespace serdes
