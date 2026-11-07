#pragma once
#include "byte_storage.h"
#include "state.h"
#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace checkpoint::distcp
{

// A snapshotted non-tensor state value.
struct BlobMeta
{ uint64_t length{}; };

// FQN-first ordering matches the physical order used by every rank's shard.
// The ordered inner map makes the lowest-rank replica the deterministic owner.
using BlobMetaByRank    = std::map<Rank, BlobMeta>;
using BlobMetadataByFqn = std::map<std::string, BlobMetaByRank>;

// One rank's per-save Blob payloads (fqn-sorted). The referenced registry
// and snapshot values outlive WriteRankShard.
using BlobsByFqn       = std::map<std::string, std::string_view>;
using NonTensorOwners  = std::map<std::string, Rank>;
using WriteItemsByRank = std::map<Rank, std::vector<CheckpointWriteItem>>;

NonTensorOwners PlanNonTensorWrites(WriteItemsByRank const& rankItems);

std::vector<char>
BuildCommonStateArchive(StateTree const& localRegistry, NonTensorOwners const* owners = nullptr, Rank ownerRank = Rank{0});

// Write this rank's tensor shards and local Blobs into __<rank>_0.distcp.
// An empty Blob map means this rank has no non-tensor state payloads.
void WriteRankShard(std::filesystem::path const& ckptDir,
                    Rank                         rank,
                    StateTree const&             globalRegistry,
                    ByteStorage const&           tensorStorage,
                    BlobsByFqn const&            blobs);

// Emit the global .metadata. Blob ownership is the lowest rank in each FQN's
// rank map.
void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry, BlobMetadataByFqn const& blobs);
}    // namespace checkpoint::distcp
