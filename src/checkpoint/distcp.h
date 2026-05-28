#pragma once
#include "blob.h"
#include "state.h"
#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

// TODO(v2): introduce a `WriterConfig` struct on the entry points below to
// expose tunables. Likely contents when needed:
//   - filesystem-backend selection (POSIX / Lustre / NFS-aware) once a Lustre
//     code path lands. Today the writer is plain POSIX inline.
//   - per-file `pwrite` chunk size and aligned-buffer toggle for `O_DIRECT`.
//   - Lustre `lfs setstripe` parameters (stripe count, stripe size).
//   - cross-rank writer thread pool size, if per-rank `std::async` parallelism
//     ever proves insufficient. Per-rank async is the v1 parallelism model.
// Until any of these need to be configurable, both entry points take only the
// minimum they require — explicit primitives, not Session types — so callers
// can place them anywhere downstream of FinalizeSnapshot validation.

namespace checkpoint::distcp
{

struct WriteRecord
{
    StateTree::NodeId   fqn;
    DS4Tensor::DimRange sharddim;
    uint64_t            fileOffset{0};
    uint64_t            length{0};
};

// Writes one rank's `__<rank>_0.distcp` file under `ckptDir`. Each shard is
// emitted as a self-contained torch.save-compatible ZIP archive, byte-for-byte
// identical to what `torch.distributed.checkpoint.save` would produce for the
// same shard. Walks `localRegistry` to lay out shards within the file, streams
// bytes from `blob.data + handle` per shard, populates `writeLog` (one record
// per shard), and fsyncs the file before returning. Touches no shared state —
// safe to call from any thread once this rank's blob and localRegistry are
// settled.
void WriteRankShard(std::filesystem::path const& ckptDir,
                    Rank                         rank,
                    StateTree const&             localRegistry,
                    Blob const&                  blob,
                    std::vector<WriteRecord>&    writeLog);

// Per-rank input to WriteMetadata: a borrowed pointer to the rank's writeLog
// after WriteRankShard has populated it.
struct RankManifestRef
{
    Rank                            rank;
    std::vector<WriteRecord> const* writeLog;
};

// Writes `<ckptDir>/.metadata` — a raw pickle stream (no ZIP container) of
// `torch.distributed.checkpoint.metadata.Metadata`. Walks `globalRegistry`,
// looks up each shard's `(fileOffset, length)` in the matching rank's writeLog
// (via `manifest`), and hand-emits a proto-2 pickle stream describing the
// `Metadata` object graph. No libpython dependency — see `BuildMetadataPkl` in
// distcp.cpp for the byte-level recipe. `torch.distributed.checkpoint.load`
// is the gating compatibility check (tier 3 in `tests/test_dcp_load.py`).
//
// Caller must guarantee every entry in `manifest` points at a fully-populated
// writeLog before calling. Single-server v1 only.
void WriteMetadata(std::filesystem::path const& ckptDir, StateTree const& globalRegistry, std::span<RankManifestRef const> manifest);

}    // namespace checkpoint::distcp
