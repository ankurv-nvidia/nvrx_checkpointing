# Daemon-owned snapshot planning

## Goal

Register the complete local state dictionary without using `replica_id` to
select data. The daemon mesh gathers all rank metadata, assigns each logical
state value to an eligible rank, and exposes each local rank's snapshot plan.
Training snapshots only its assigned state.

This branch starts at `origin/main` commit `344d5f1c`. This file stays in the
branch during development and is removed before the pull request.

## Agreed registry model

- The global state-registry contains metadata for all ranks and belongs to the
  session.
- Each local state-registry contains all state present on that rank and belongs
  to the session.
- Each snapshot state-registry contains that snapshot's selected local state,
  current scalar values, and handles for its tensor/blob payloads.
- The relationship is snapshot subset local subset global. Ownership can move
  among ranks that already hold the same logical state; planning does not move
  data to ranks without a local copy.
- Session registries and cached plans retain metadata only. Blob bytes live
  outside StateNode; scalar payloads are cleared in session metadata while
  retaining their type.
- Every value-bearing node participates in ownership selection: tensors,
  scalars, fixed-size blobs, and dynamic-size blobs. Dictionaries provide the
  required structure for selected values.

## Registration and planning

1. Traverse every local state entry, including non-main tensor and object
   replicas, lists, and factory results. Preserve complete registration for
   later validation and address refresh.
2. Merge all rank registries at the coordinator. Both local registry arrival
   and remote registry-fragment arrival run the same once-only completion
   check, including fragments queued before session activation.
3. Once the complete global registry is ready, validate it and broadcast it.
   Owner selection follows broadcast on the coordinator and receipt on peers.
   Neither operation waits for LocalStateData or snapshot finalization.
4. Reuse the existing deterministic exact-shard matching and balancing policy,
   extending ownership selection to other state values. Tensor identity uses
   name, dtype, global shape, and exact shard coordinates. Object identity uses
   the corresponding logical object key and coordinates. Replicas are assumed
   interchangeable; payload comparisons are outside this change.
5. Each daemon computes and caches snapshot-registry plans only for its local
   ranks. Preserve session and global-rank identifiers so remote plan lookup
   remains possible to add later, without implementing remote lookup now.

## Client API and overlap

- Add GET `/api/snapshot-plan/{session}/{rank}` to the rank's owning daemon.
  Return the metadata-only selected local registry and its stable identities.
- A pending request waits without occupying handlers needed to receive peer
  registry traffic. Use the existing asynchronous completion-wait pattern.
- Propagate planning failures, timeout, and shutdown interruption. Snapshot
  data copying does not start with a missing or incomplete plan.
- Fetch and cache the plan in asynchronous registration finalization after
  load. Subsequent saves use it with newly refreshed addresses and values.
- A first save without a load performs registration and planning before
  snapshotting; its exposed wait is expected.
- The registered tensor layout, fixed blob sizes, and topology remain fixed
  for the session. Dynamic blob sizes and snapshot values may change. Reject
  incompatible registered metadata changes rather than reusing a stale plan.

## State node and payload representation

- FixedSizeBlob nodes reuse DTensor metadata with a distinct variant/type tag
  for immediate capture and opaque-byte persistence. They support object
  sharding and replication, using the existing four-dimension constraint.
- Treat sharded/replicated ShardedObject entries as fixed-sized blobs. Their
  dimensions describe atomic object positions, not a count of serialized
  bytes. Preserve the existing DCP unique keys for individual object shards.
- Fixed-size replicas of a logical shard have equal registered size and
  contents. Empty blobs are valid and need blob-aware metadata validation.
- DynamicSizeBlob nodes use separate lightweight metadata containing replica
  ranks, a current snapshot size, and a handle. They do not support sharding.
  Replicas have equal content and size within a snapshot. Their changing
  lengths do not require owner replanning.
- Blob payloads and optional payload wrappers are absent from StateNode. A
  handle denotes a client pointer before capture and a packed offset in the
  daemon's snapshot-data buffer afterward.
- Scalars retain replica-location metadata for planning and current primitive
  values only in snapshots. Rank-specific values require distinct identities.
- Dynamic blobs are expected to be rare, such as strings and paths. Extra
  balancing or size optimization for them is outside this change.

## Snapshot transport and persistence

- Reuse existing contiguous packing, driven by the snapshot registry rather
  than the complete local registry.
- Tensor and both blob payload types share the packed LocalStateData stream.
  Keep selected ordering, total byte counts, daemon offsets, and disk writing
  consistent. Send per-snapshot dynamic sizes and scalar values as snapshot
  metadata before allocating or receiving the data stream.
- Snapshot activation metadata travels over HTTP, independently of shared-memory
  slot capacity. The activation header and selected registry are validated
  before LocalStateData arrives. Blob source setters accept explicit handles
  and lengths; callers own those source buffers until capture completes.
- All ranks participate in activation/finalization, including ranks assigned
  no payload bytes.
- Preserve DCP tensor/object/common-state compatibility and current durability
  sequencing. Derive blob metadata from registry nodes instead of maintaining
  competing authoritative session maps.
- Remove the native loader's replica-id preservation workaround when native
  registration no longer consumes the hints.

## Implementation and validation

- [x] Introduce metadata-only scalar/blob representations and serialization.
- [x] Implement reusable all-node ownership selection and local projections.
- [x] Trigger planning at registration completion and expose the local API.
- [x] Fetch plans in the client and pack selected tensor/blob payloads.
- [x] Update Python/Megatron traversal, lifetime management, and hint removal.
- [x] Finish DCP persistence, diagnostics, fixtures, and affected readers.
- [x] Add regression coverage for nonzero/misleading hints, scalar/blob owner
      assignment, and planning completion before any snapshot data arrives.
- [x] Validate multi-rank/peer arrival ordering, responsive plan waits, errors,
      zero-owned ranks, fresh addresses and scalar/blob values, dynamic blob
      size changes, empty blobs, quantized tensors, and DCP load round trips.
- [x] Run available relevant native and Python checks, review the complete diff,
      and record actual results and environment limitations here.

## Progress

Implementation and final diff review are complete. The plan stays in the branch
for the user's review and removal before the pull request.

The user authorizes committing and pushing this branch. Remote validation uses
the full CI workflow, including both wheel builds and the GB300 eight-node
deterministic/performance workloads, plus the GB200 eight-node native workloads.
Build and workload results are recorded here as they become available.

### Implementation details confirmed during validation

- Internal registry serialization carries version 2. The diagnostic reader
  accepts both versioned metadata registries and legacy dumps.
- Peer registration sends complete metadata-only trees, so every node type
  shares the same merge path. Selected snapshot descriptors supply the
  coordinator with current scalar values and dynamic lengths for each snapshot;
  remote snapshot plans are not cached.
- Common-state bytes are dynamic-sized, without the previous padded-capacity
  restriction. Fixed ShardedObject slots retain their fixed-size behavior.
- Snapshot activation uses HTTP because metadata can exceed a shared-memory
  slot. Only packed payloads and small control messages depend on slot capacity.
  The regression saves approximately 92 KiB of metadata with approximately
  60 KiB slots and a small payload.
- Client spin-down waits for the daemon listener to close, fixing a shutdown
  race exposed by consecutive integration sessions.
- The multi-rank fixture uses a stop-aware condition-variable predicate. GDB
  confirmed that its earlier bare wait could miss shutdown after all checkpoint
  work completed. The round-trip test also waits for durability before teardown.
- Generic local traversal still visits all state and serializes immediate blob
  values into temporary client buffers. Only assigned payloads are transferred,
  retained by the daemon, and written to disk. Tensor capture is selected-only.

### Validation results

- Strict Clang 23 build with CUDA enabled: daemon, Python extension, DCP shim,
  benchmark, stress executable, native tests, and wheel build pass.
- Final Python integration/save/registration/quantized/DCP/planning suite:
  **139 passed**.
- Diagnostic, registry-reader, equality, peer-tools, and telemetry Python suite:
  **112 passed**.
- Native planner checks: **3 cases / 45 assertions passed**.
- Native tier-1 checks excluding service/shared-memory cases:
  **35 cases / 315 assertions passed**.
- Native DCP checks: **17 cases / 232 assertions passed**.
- Native service/shared-memory cases, isolated as in CTest:
  **23 cases / 354 assertions passed**.
- Actual two-daemon loopback round trip: **eight CPU ranks, eight replicated
  tensors, five durable snapshots**, followed by independent PyTorch DCP loading
  of all eight tensors. Each snapshot captures **128 bytes from 1024 aggregate
  local tensor bytes**, an eightfold reduction; ownership spans both daemons.
- Independent code review approves the implementation and the follow-up HTTP
  activation and source-buffer lifetime fixes.
- Formatting and `git diff --check` pass. Ruff reports **zero new findings**
  against the target branch; eight pre-existing test lint findings remain.

### Remaining validation limit

One native peer-durability success assertion returned an unexpected HTTP 500 in
an earlier run before the fixture printed response bodies. The case subsequently
passed ten normal runs, a traced run, and the final service suite. A targeted
production review found no completion-state visibility race. The original cause
remains unexplained; response-body and daemon-log context stay in the test for a
future recurrence. No speculative delay or behavior workaround is included.
