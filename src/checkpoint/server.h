#pragma once
#include "byte_storage.h"
#include "messages.h"
#include "state.h"

#include <filesystem>
#include <string_view>

// The standalone daemon announces readiness and waits for its references to drain.
void RunServer();

void StartOwnedInprocServer();
void WaitForOwnedInprocServer();

void RunInprocServer(uint16_t nodes);

// Waits until every in-process daemon has made its active snapshots durable.
// This is intended for test and benchmark teardown, not the measured save path.
void WaitForInprocServerWrites();

void StopInprocServer();

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
