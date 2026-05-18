#pragma once
#include <istream>
#include <ostream>
#include <span>
#include <string>

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

template <typename T> std::ostream& WriteBlob(std::ostream& ostrm, std::span<T> const& obj)
{
    if (obj.size() == 0) { return ostrm; }
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

template <typename TSize> auto WriteBlobWithSize(std::ostream& os, std::string_view const& data) -> std::ostream&
{
    auto size = static_cast<TSize>(data.size());
    os.write(reinterpret_cast<char const*>(&size), sizeof(TSize));    // NOLINT
    os.write(data.data(), static_cast<std::streamsize>(size));        // NOLINT
    return os;
}

}    // namespace serdes
