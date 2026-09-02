#pragma once
#include "messages.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace checkpoint::telemetry::span
{
struct Id
{
    ipc::SessionId  sessionId{};
    Rank            rank{};
    ipc::SnapshotId snapshotId{};

    constexpr bool operator==(Id const&) const = default;
};

struct CkptSave
{
    static constexpr std::string_view Name = "ckpt.save";

    struct Begin
    {
        Id                       id        = {};
        std::chrono::nanoseconds startedAt = {};
    };

    struct End
    {
        Id                        id                = {};
        std::chrono::microseconds d2hDuration       = {};
        std::chrono::microseconds snapshotDuration  = {};
        std::chrono::microseconds totalDuration     = {};
        uint64_t                  numTensors        = {};
        uint64_t                  d2hRequestCount   = {};
        uint64_t                  totalTensorBytes  = {};
        uint64_t                  totalWrittenBytes = {};
    };
};

struct SvrFinalize
{
    static constexpr std::string_view Name = "ckpt.svr.finalize";

    struct Begin
    {
        Id id{};
    };

    struct End
    {
        Id id{};
    };
};

struct SvrWriteMetadata
{
    static constexpr std::string_view Name = "ckpt.svr.write_metadata";

    struct Begin
    {
        Id id{};
    };

    struct End
    {
        Id id{};
    };
};

struct SvrDurable
{
    static constexpr std::string_view Name = "ckpt.svr.durable";

    struct Begin
    {
        Id                                    id{};
        std::chrono::system_clock::time_point startedAt;
    };
    struct End
    {
        Id id{};
    };
};

}    // namespace checkpoint::telemetry::span

namespace checkpoint::telemetry
{

void Init();
void Shutdown();
template <typename TSpan> struct Emitter;
template <> struct Emitter<span::CkptSave::Begin>
{
    static void Emit(span::CkptSave::Begin const& fields);
};
template <> struct Emitter<span::CkptSave::End>
{
    static void Emit(span::CkptSave::End const& fields);
};

template <> struct Emitter<span::SvrFinalize::Begin>
{
    static void Emit(span::SvrFinalize::Begin const& fields);
};
template <> struct Emitter<span::SvrFinalize::End>
{
    static void Emit(span::SvrFinalize::End const& fields);
};

template <> struct Emitter<span::SvrWriteMetadata::Begin>
{
    static void Emit(span::SvrWriteMetadata::Begin const& fields);
};
template <> struct Emitter<span::SvrWriteMetadata::End>
{
    static void Emit(span::SvrWriteMetadata::End const& fields);
};

template <> struct Emitter<span::SvrDurable::Begin>
{
    static void Emit(span::SvrDurable::Begin const& fields);
};
template <> struct Emitter<span::SvrDurable::End>
{
    static void Emit(span::SvrDurable::End const& fields);
};

struct Stream
{
    template <typename T> Stream& operator<<(T const& value)
    {
        Emitter<T>::Emit(value);
        return *this;
    }
};

void InitLocalCollector(std::filesystem::path const& filePath);
void InitRelayCollector(std::string_view const& relayToRemote);
void Collect(std::string_view const& payload);

struct OtlpResponse
{
    uint16_t    status;
    std::string body;
};

inline constexpr size_t MaxOtlpRequestBytes = 64UL * 1024 * 1024;
OtlpResponse            ReceiveOtlp(std::string_view const& path,
                                    std::string_view const& contentType,
                                    std::string_view const& contentEncoding,
                                    std::string_view const& payload);

}    // namespace checkpoint::telemetry

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define PERF() (::checkpoint::telemetry::Stream{})
// NOLINTEND(cppcoreguidelines-macro-usage)
