#pragma once
#include "byte_storage.h"
#include "state.h"
#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace checkpoint::distcp
{
std::vector<char> BuildCommonStateArchive(StateTree const& snapshotRegistry, ByteStorage const& snapshotData);
std::string       CommonStateKeyForRank(std::string_view key, Rank rank);

std::unordered_map<std::string, uint64_t>
WriteRankShard(std::filesystem::path const& ckptDir, Rank rank, StateTree const& snapshotRegistry, ByteStorage const& tensorStorage);

// Emit the global .metadata. Blob ownership is the lowest rank in each FQN's
// rank map.
void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& snapshotRegistry);
void WriteMetadata(std::filesystem::path const&                     ckptDir,
                   StateTree const&                                 snapshotRegistry,
                   std::unordered_map<std::string, uint64_t> const& dynamicBlobSizes);
}    // namespace checkpoint::distcp
