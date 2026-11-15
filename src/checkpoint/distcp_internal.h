#pragma once

#include "tensor.h"

import std.compat;
namespace checkpoint::distcp
{
uint64_t ComputeShardArchiveSize(Tensor::DType dtype, Tensor::Dim const& shape, uint64_t shardDataSize);
}    // namespace checkpoint::distcp
