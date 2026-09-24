#pragma once
#include "shm_ring_buffer.h"
#include "tensor.h"

#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <variant>

namespace checkpoint::ipc
{

enum class SessionId : uint32_t
{
    Default = 0,
    Invalid = std::numeric_limits<uint32_t>::max(),
};

enum class SnapshotId : uint32_t
{
    Default = 0,
    Invalid = std::numeric_limits<uint32_t>::max(),
};
using ::Rank;

/*
A server can contain multiple sessions
A session consist of multiple peers(ranks) participating in a distributed job

Session (Common Across all ranks)
    -> GlobalStateRegistry
        -> LocalStateRegistry (Per Rank)
            -> ReplicatedStateRegistry (Common across all ranks)
            -> ShardedStateRegistry (Per Rank)
    -> Snapshot (name)
        -> GlobalStateRegistry
        -> ShardedStateData
*/
struct Message
{
    // Kept below ip_local_port_range (32768-60999 by default)
    static constexpr uint16_t Port = 4100;

    static std::string RequestChannelName(uint16_t index, uint8_t rank) { return fmt::format("checkpoint_request_{}_{}", index, rank); }

    enum class Type : uint8_t
    {
        Invalid,
        Ping,
        Shutdown,

        ActivateSession,
        FinalizeSession,    // Rendezvous with the mesh

        ActivateSnapshot,
        FinalizeSnapshot,

        // Recieved from local client or from peers
        LocalStateRegistry,    // Combination of Replicated and Sharded State Registry
        LocalStateData,        // Could be recieved multiple times in chunked manner

        StateRegistryFragment,    // The format is differnt from StateRegistry.
        // ShardedStateData,

        Unknown,
    };

    struct Ping
    {
        SessionId sessionId;
    };

    struct Shutdown
    {};

    struct ActivateSession
    {
        SessionId sessionId;
        uint32_t  numActiveSnapshots = 1;
        uint32_t  messageSize;
        /* Message Format
        std::string workdir;
        std::vector<std::string> peers;
        */
    };

    struct FinalizeSession
    {
        SessionId sessionId;
    };

    struct ActivateSnapshot
    {
        SessionId  sessionId;
        SnapshotId snapshotId;
        Rank       globalRank;
        uint64_t   snapshotSize;
        uint32_t   messageSize;
        int32_t    numaNode = -1;    // this rank's GPU-local NUMA node, or -1 if unknown

        std::chrono::nanoseconds startedAt{};
        /*Message Format
        std::string ckptPath;
        Blob and Scalar counts followed by values in this rank's registered StateTree order;
        */
    };

    struct FinalizeSnapshot
    {
        SessionId  sessionId;
        Rank       globalRank;
        SnapshotId snapshotId;

        // Perf metrics
        std::chrono::nanoseconds snapshotDuration{};
        std::chrono::nanoseconds d2hDuration{};
        uint32_t                 d2hRequests{};
        uint64_t                 totalSnapshotBytes{};
    };

    // Combination of Replicated and Sharded State Registry
    struct LocalStateRegistry
    {
        SessionId sessionId;
        uint64_t  messageSize;
    };

    struct StateRegistryFragment
    {
        SessionId sessionId;
        uint32_t  messageSize;
        /*Message Format
         * NumFragments NodeId + NodeData
         */
    };

    struct LocalStateData
    {
        SessionId  sessionId;
        SnapshotId snapshotId;
        Rank       globalRank;
        uint32_t   messageSize;
        uint64_t   offset;
    };

    // struct StateDataFragment {};
    using Header = std::variant<std::monostate,
                                Ping,
                                Shutdown,
                                ActivateSession,
                                FinalizeSession,
                                ActivateSnapshot,
                                FinalizeSnapshot,
                                LocalStateRegistry,
                                LocalStateData,
                                StateRegistryFragment
                                //, StateDataFragment
                                >;

    static_assert(std::is_same_v<std::monostate, std::variant_alternative_t<static_cast<size_t>(Type::Invalid), Header>>);
    static_assert(std::is_same_v<Ping, std::variant_alternative_t<static_cast<size_t>(Type::Ping), Header>>);
    static_assert(std::is_same_v<Shutdown, std::variant_alternative_t<static_cast<size_t>(Type::Shutdown), Header>>);
    static_assert(std::is_same_v<ActivateSession, std::variant_alternative_t<static_cast<size_t>(Type::ActivateSession), Header>>);
    static_assert(std::is_same_v<FinalizeSession, std::variant_alternative_t<static_cast<size_t>(Type::FinalizeSession), Header>>);
    static_assert(std::is_same_v<ActivateSnapshot, std::variant_alternative_t<static_cast<size_t>(Type::ActivateSnapshot), Header>>);
    static_assert(std::is_same_v<FinalizeSnapshot, std::variant_alternative_t<static_cast<size_t>(Type::FinalizeSnapshot), Header>>);
    static_assert(std::is_same_v<LocalStateRegistry, std::variant_alternative_t<static_cast<size_t>(Type::LocalStateRegistry), Header>>);
    static_assert(std::is_same_v<LocalStateData, std::variant_alternative_t<static_cast<size_t>(Type::LocalStateData), Header>>);
    static_assert(
        std::is_same_v<StateRegistryFragment, std::variant_alternative_t<static_cast<size_t>(Type::StateRegistryFragment), Header>>);
    // static_assert(std::is_same_v<StateDataFragment, std::variant_alternative_t<static_cast<size_t>(Type::StateDataFragment), Header>>);
};

using MsgQueue = checkpoint::ipc::ShmRingBuffer<Message::Header>;

struct ErrorAlreadyRunning : std::runtime_error
{
    ErrorAlreadyRunning() : std::runtime_error("Checkpoint server is already running") {}
};

}    // namespace checkpoint::ipc

template <> struct fmt::formatter<checkpoint::ipc::Message::Header> : fmt::formatter<std::string>
{
    auto format(checkpoint::ipc::Message::Header const& hdr,    // NOLINT(readability-convert-member-functions-to-static)
                fmt::format_context&                    ctx) const
    {
        namespace cipc           = checkpoint::ipc;
        std::string_view typeStr = "Invalid";
        switch (static_cast<cipc::Message::Type>(hdr.index()))
        {
        case cipc::Message::Type::Ping: typeStr = "Ping"; break;
        case cipc::Message::Type::Shutdown: typeStr = "Shutdown"; break;
        case cipc::Message::Type::ActivateSession: typeStr = "ActivateSession"; break;
        case cipc::Message::Type::FinalizeSession: typeStr = "FinalizeSession"; break;
        case cipc::Message::Type::ActivateSnapshot: typeStr = "ActivateSnapshot"; break;
        case cipc::Message::Type::FinalizeSnapshot: typeStr = "FinalizeSnapshot"; break;
        case cipc::Message::Type::LocalStateRegistry: typeStr = "LocalStateRegistry"; break;
        case cipc::Message::Type::LocalStateData: typeStr = "LocalStateData"; break;
        case cipc::Message::Type::StateRegistryFragment:
            typeStr = "StateRegistryFragment";
            break;
            // case Message::Type::StateDataFragment: typeStr = "StateDataFragment"; break;
        case cipc::Message::Type::Unknown: typeStr = "Unknown"; break;
        case cipc::Message::Type::Invalid: [[fallthrough]];
        default: break;
        }
        return fmt::format_to(ctx.out(), "MsgHdr(Type={}, Index={})", typeStr, hdr.index());
    }
};
template <> struct fmt::formatter<checkpoint::ipc::SessionId> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::ipc::SessionId const& sessionId, fmt::format_context& ctx) const    // NOLINT
    {
        if (sessionId == checkpoint::ipc::SessionId::Invalid) { return fmt::format_to(ctx.out(), "?sessionId?"); }
        return fmt::format_to(ctx.out(), "{}", static_cast<uint32_t>(sessionId));
    }
};

template <> struct fmt::formatter<checkpoint::ipc::SnapshotId> : fmt::formatter<std::string_view>
{
    auto format(checkpoint::ipc::SnapshotId const& snapshotId, fmt::format_context& ctx) const    // NOLINT
    {
        if (snapshotId == checkpoint::ipc::SnapshotId::Invalid) { return fmt::format_to(ctx.out(), "?snapshotId?"); }
        return fmt::format_to(ctx.out(), "{}", static_cast<uint32_t>(snapshotId));
    }
};
