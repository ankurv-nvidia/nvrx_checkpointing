#pragma once
#include "byte_storage.h"
#include "messages.h"
#include "state.h"

#include <filesystem>
#include <string_view>

// Starts the checkpoint server, prints "READY\n" to stdout when the port is bound,
// then blocks until a shutdown is triggered (either by message or inactivity timeout).
void RunServer();

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
                                       StateTree const&             localState,
                                       ByteStorage const&           tensorStorage);

void QueryServerSessions(ServerSessionCallback callback, void* cbData);
