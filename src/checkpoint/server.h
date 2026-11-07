#pragma once
#include "byte_storage.h"
#include "messages.h"
#include "state.h"

#include <filesystem>
#include <string_view>

// Runs the checkpoint daemon until references expire or shutdown is requested.
void RunServer(uint16_t index = 0);

void StartOwnedInprocServer();
void WaitForOwnedInprocServer();
bool OwnsInprocServer();
bool ReleaseOwnedInprocReference(uint32_t sessionId, std::string_view token);

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
