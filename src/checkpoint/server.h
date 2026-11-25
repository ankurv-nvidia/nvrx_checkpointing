#pragma once
#include "byte_storage.h"
#include "messages.h"
#include "state.h"

#include <filesystem>
#include <string_view>

// Starts the checkpoint server, prints "READY\n" to stdout when the port is bound,
// then blocks until a stop request or until its references drain.
void RunServer(uint16_t index);

void RunInprocServer(uint16_t nodes);
void SpinDownAllInprocServers();

// Returns once this process hosts the server for the peer index or another process serves it.
void EnsureInprocServer(uint16_t index);

using ServerSessionCallback = void (*)(void*                        cbData,
                                       checkpoint::ipc::SessionId   sessionId,
                                       checkpoint::ipc::SnapshotId  snapshotId,
                                       std::string_view const&      name,
                                       std::filesystem::path const& path,
                                       StateTree const&             globalRegistry,
                                       StateTree const&             localRegistry,
                                       StateTree const&             snapshotRegistry,
                                       StateTree const&             dedupedGlobalRegistry,
                                       ByteStorage const&           snapshotData);

void QueryServerSessions(ServerSessionCallback callback, void* cbData);

static constexpr int MaxRetries = 20;

static constexpr auto ServerWaitTimeout              = std::chrono::seconds(900);
static constexpr auto IntervalBetweenRetries         = std::chrono::milliseconds(100);
static constexpr auto ThreadStallWaitForEventTimeout = std::chrono::seconds(30);
