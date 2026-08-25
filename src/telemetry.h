#pragma once
#include "messages.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace checkpoint::telemetry::span
{

struct CkptSave
{
    static constexpr std::string_view Name = "ckpt.save";
    struct Id
    {
        ipc::SessionId  sessionId{};
        Rank            rank{};
        ipc::SnapshotId snapshotId{};
    };

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

}    // namespace checkpoint::telemetry::span

template <> struct fmt::formatter<checkpoint::telemetry::span::CkptSave::Begin> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::telemetry::span::CkptSave::Begin const& evt, fmt::format_context& ctx) const    // NOLINT
    {
        return fmt::format_to(ctx.out(),
                              "[native_ckpt:svr][{}][PERF] Rank={} Begin span={} started_at_ns={} snapshot={}\n",
                              static_cast<uint32_t>(evt.id.sessionId),
                              static_cast<uint32_t>(evt.id.rank),
                              checkpoint::telemetry::span::CkptSave::Name,
                              evt.startedAt.count(),
                              static_cast<uint32_t>(evt.id.snapshotId));
    }
};

template <> struct fmt::formatter<checkpoint::telemetry::span::CkptSave::End> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::telemetry::span::CkptSave::End const& evt, fmt::format_context& ctx) const    // NOLINT
    {
        auto const writeSeconds = static_cast<double>(evt.d2hDuration.count()) / 1e6;    // NOLINT
        return fmt::format_to(
            ctx.out(),
            "[native_ckpt:svr][{}][PERF] Rank={} Finished bw_mbps={:.2f} "
            "| Time(us):  wr_tnsr_us={} snapshot_us={} total_us={} "
            "| Count: num_tensors={} total_d2h={} "
            "| Bytes: total_written_bytes={} avg_d2h_rq={} avg_tnsr={} snapshot={}\n",
            static_cast<uint32_t>(evt.id.sessionId),
            static_cast<uint32_t>(evt.id.rank),
            writeSeconds == 0.0 ? 0.0 : static_cast<double>(evt.totalTensorBytes) / (1024.0 * 1024.0) / writeSeconds,    // NOLINT
            evt.d2hDuration,
            evt.snapshotDuration,
            evt.totalDuration,
            evt.numTensors,
            evt.d2hRequestCount,
            evt.totalWrittenBytes,
            evt.totalTensorBytes / std::max(evt.d2hRequestCount, uint64_t{1}),
            evt.totalTensorBytes / std::max(evt.numTensors, uint64_t{1}),
            static_cast<uint32_t>(evt.id.snapshotId));
    }
};

namespace checkpoint::telemetry
{

struct PerfStream
{};

inline PerfStream operator<<(PerfStream stream, span::CkptSave::Begin const& value)
{
    fmt::print(stderr, "{}", value);
    return stream;
}

inline PerfStream operator<<(PerfStream stream, span::CkptSave::End const& value)
{
    fmt::print(stderr, "{}", value);
    return stream;
}

}    // namespace checkpoint::telemetry

#define PERF() (::checkpoint::telemetry::PerfStream{}) /* NOLINT */
