#pragma once
#include "byte_storage.h"
#include "state.h"
#include "tensor.h"

#include <filesystem>

namespace checkpoint::distcp
{
// Write this rank's tensor shards and local Blobs into __<rank>_0.distcp.
// An empty Blob map means this rank has no non-tensor state payloads.
void WriteRankShard(std::filesystem::path const& ckptDir, Rank rank, StateTree const& snapshotRegistry, ByteStorage const& tensorStorage);

// Emit the global .metadata. Blob ownership is the lowest rank in each FQN's
// rank map.
void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& snapshotRegistry);
}    // namespace checkpoint::distcp
