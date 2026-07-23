#pragma once
#include "byte_storage.h"
#include "state.h"
#include "tensor.h"

#include <filesystem>

namespace checkpoint::distcp
{

void WriteRankShard(std::filesystem::path const& ckptDir, Rank rank, StateTree const& globalRegistry, ByteStorage const& tensorStorage);
void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry);
}    // namespace checkpoint::distcp
