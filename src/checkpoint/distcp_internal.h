#pragma once
#include "blob.h"
#include "distcp.h"
#include "state.h"
#include "tensor.h"

#include <cstdint>
#include <string_view>
#include <vector>

// Internal extraction helpers shared between `distcp.cpp` (the production
// writer) and `ckpttest` (tier-1 unit tests). Not part of the public writer
// API — consumers outside this library should still go through
// `checkpoint::distcp::WriteRankShard`.

namespace checkpoint::distcp
{

// Per-shard layout record produced by `CollectShards`. `WriteRankShard` builds
// one per shard and emits a self-contained ZIP from these fields directly into
// the rank file.
struct ShardEntry
{
    char const*         srcPtr;
    uint64_t            length;
    StateTree::NodeId   fqn;
    DS4Tensor::DType    dtype;
    DS4Tensor::DimRange sharddim;
    DS4Tensor::Dim      shape;
};

// Walks `tree` and appends one `ShardEntry` per non-empty `TensorShard` /
// `ShardedTensor` shard, in `StateTree::WalkNodes` order. Zero-length shards
// are skipped. `srcPtr` points into `blob.data` at the shard's handle offset.
//
// Precondition: `tree` is a per-rank local registry. `CollectShards` does NOT
// filter `ShardedTensor::shards` by rank — every shard in the set is emitted.
// In production, upstream consolidation hands each rank its own filtered tree;
// tests reconstruct that invariant directly.
void CollectShards(StateTree const& tree, Blob const& blob, std::vector<ShardEntry>& out);

// Maps a `DS4Tensor::DType` to its string `torch.dtype` name (e.g. "float32").
// Used by the parity-test shim to convert names back to dtypes. Throws on
// `DType::Invalid` / unsupported values.
std::string_view DTypeName(DS4Tensor::DType dtype);

}    // namespace checkpoint::distcp
