#pragma once
#include "blob.h"
#include "state.h"
#include "tensor.h"

#include <filesystem>

namespace checkpoint::distcp
{

void WriteRankShard(std::filesystem::path const& ckptDir, Rank rank, StateTree const& globalRegistry, Blob const& blob);
void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry);
}    // namespace checkpoint::distcp
