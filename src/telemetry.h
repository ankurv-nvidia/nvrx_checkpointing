#pragma once
#include "CommonMacros.h"
#include "messages.h"
#include "process_memory.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
namespace checkpoint::telemetry
{
struct Id
{
    ipc::SessionId  sessionId{};
    Rank            rank{};
    ipc::SnapshotId snapshotId{};

    constexpr bool operator==(Id const&) const = default;
};
}    // namespace checkpoint::telemetry

namespace checkpoint::telemetry::span
{

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
    { Id id{}; };

    struct End
    { Id id{}; };
};

struct SvrWriteMetadata
{
    static constexpr std::string_view Name = "ckpt.svr.write_metadata";

    struct Begin
    { Id id{}; };

    struct End
    { Id id{}; };
};

struct SvrWaitLocalRegistries
{
    static constexpr std::string_view Name = "ckpt.svr.wait_local_registries";

    struct Begin
    { Id id{}; };

    struct End
    { Id id{}; };
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
    { Id id{}; };
};

}    // namespace checkpoint::telemetry::span

namespace checkpoint::telemetry::event
{
struct ProcessMemory
{
    static constexpr std::string_view Name = "process.memory";

    ProcessMemUsageInfo usage;
    std::string_view    trigger;
};
}    // namespace checkpoint::telemetry::event

namespace checkpoint::telemetry
{

void Init();
void Shutdown();

[[nodiscard]] bool IsEnabled() noexcept;

// Process-memory telemetry. Span events are always sampled; periodic sampling
// remains opt-in through NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS.
class MemorySampler
{
    public:
    explicit MemorySampler();
    ~MemorySampler();
    CLASS_DELETE_COPY_AND_MOVE(MemorySampler);

    void Shutdown();
    void Start();

    static void Log(std::string_view const& trigger);

    private:
    void Run_(std::stop_token& stopToken);

    std::chrono::seconds        _interval{};
    std::mutex                  _waitMutex;
    std::condition_variable_any _wake;
    std::jthread                _thread;
};

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

template <> struct Emitter<span::SvrWaitLocalRegistries::Begin>
{
    static void Emit(span::SvrWaitLocalRegistries::Begin const& fields);
};
template <> struct Emitter<span::SvrWaitLocalRegistries::End>
{
    static void Emit(span::SvrWaitLocalRegistries::End const& fields);
};

template <> struct Emitter<span::SvrDurable::Begin>
{
    static void Emit(span::SvrDurable::Begin const& fields);
};
template <> struct Emitter<span::SvrDurable::End>
{
    static void Emit(span::SvrDurable::End const& fields);
};

template <> struct Emitter<event::ProcessMemory>
{
    static void Emit(event::ProcessMemory const& fields);
};

struct Stream
{
    template <typename T> Stream& operator<<(T const& value) LFTBND
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
SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wunused-macros")
#define PERF() (::checkpoint::telemetry::Stream{})
SUPPRESS_WARNINGS_END
// NOLINTEND(cppcoreguidelines-macro-usage)
