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
- [ ] Complete GB200 and GB300 eight-node deterministic/performance validation.

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
- The large-descriptor regression uses a distinct session so its cached client
  plan stays isolated from other daemon lifetimes in a combined native run.
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
- Complete native suite in one process: **64 passed, 1 expected skip, and
  818 assertions passed**. The skip requires an explicit registry fixture, which
  the separate two-daemon round trip supplies.
- Actual two-daemon loopback round trip: **eight CPU ranks, eight replicated
  tensors, five durable snapshots**, followed by independent PyTorch DCP loading
  of all eight tensors. Each snapshot captures **128 bytes from 1024 aggregate
  local tensor bytes**, an eightfold reduction; ownership spans both daemons.
- Independent code review approves the implementation and the follow-up HTTP
  activation and source-buffer lifetime fixes.
- Formatting and `git diff --check` pass. Ruff reports **zero new findings**
  against the target branch; eight pre-existing test lint findings remain.

### Remote validation

- Implementation commit `d2625aa9` is pushed to `origin/no_replica_id`.
- Initial CI run `42350483257` passes both wheel builds before the replacement
  run supersedes it. The follow-up test-isolation commit is `2b17145f`.
- Fresh CI run `42351227684` passes both wheel builds, all four platform image
  builds, **509 Python tests** with 39 expected skips, and the combined native
  suite (**64 passed / 1 expected skip / 818 assertions**). Its CTest job passes
  **83 checks** and all loopback service stress gates.
- GB200 eight-node run `42350485730` passes its ARM64 wheel and image builds.
  Both native `nano3-8x4` workloads run on `dlalgo-ci/oci-hsg`: deterministic
  GitHub job `128341190202` / Slurm `8703635`, and performance GitHub job
  `128341190228` / Slurm `8703632`. Both allocations start at 07:36:26 UTC on
  2026-11-07; runtime setup is active at the latest poll.
- Fresh CI selects both GB300 native `nano4-8x4` workloads on
  `dlalgo-ci/nsc-lhr-slurm-1`: deterministic GitHub job `128342103263` / Slurm
  `658393`, and performance GitHub job `128342103271` / Slurm `658396`. The
  deterministic allocation starts at 07:42:54 UTC on 2026-11-07; performance
  starts at 07:44:11 UTC. All four eight-node allocations are active at the
  07:45:30 UTC poll. GB200 determinism reaches workload launch; the other jobs
  finish runtime setup. No training iteration or checkpoint is observed yet.
- Eight-node workload verdicts and the overall CI gate remain pending.

### First-save regression found in distributed validation

The initial distributed runs fail at the first checkpoint on GB200, GB300, and
H100. All eight completed workloads report the same missing common-state value;
training-phase failures on other ranks are downstream errors. B200 remains
queued at the failure audit.

Registry finalization sorts nodes and changes their handles. The client records
supplied values and empty shards before sorting, then checks those handles after
sorting on a first save without a load. This falsely rejects a supplied value.
Validate the complete local snapshot before finalization so the recorded handles
still identify the scanned nodes. Preserve missing-value rejection for later
saves and keep capture restricted to the daemon-selected plan.

The cold-save regression reproduces the exact remote error before the fix. It
covers mixed tensors, scalars, fixed blobs, an empty fixed blob, and dynamic
common state without an explicit finalization call, then checks refreshed values
and rejection of an omitted registered value. Server integration coverage runs
with the CPU-only native test build, while the wheel retains its CUDA support.
Strict Clang 23 extension rebuilding and all nine focused checks pass after
the fix.
The complete Python and CI helper suite with server tests enabled passes
**547 tests**, with two skips. Formatting, shell syntax, diff checks, and
independent review pass. Corrected remote builds and distributed reruns remain
pending.

The corrected production code is pushed as `7eb93843`. Run `42363305685`
passes both wheels, all four images, 83 CTest checks, and loopback stress. Its
newly enabled wheel IPC checks fail because the build runner's CUDA driver
cannot run the CUDA wheel. That runner limitation is separate from the
first-save defect. The integration checks move to the existing CPU-only CTest
job, using its native artifacts and a source package alias with automatic
rebuilding disabled. Their JUnit report is retained with the CTest artifacts.
This keeps CUDA wheel testing on the cluster jobs and exercises all four
IPC-bearing test files on the CPU-capable build runner.
Fresh strict CPU-only rebuilding passes, and the exact CI pytest invocation
passes **56 tests**, skipping two CUDA-only cases. Independent review, shell
syntax, CI YAML, artifact-path validation, and diff checks pass.

### Corrected distributed checkpoints

- The production correction is `7eb93843`; CI placement is corrected by
  `9177ea34` without further production changes.
- GB200 eight-node performance run `42363307322`, GitHub job `128378275723`,
  JET job `534838130`, Slurm `8707690` successfully saves its first native
  checkpoint at iteration 100 at 10:44:27 UTC on 2026-11-07. This includes
  durability and crosses the original failure point.
- The same native checkpoint loads successfully at iteration 100 at 10:45:46
  UTC. Resumed native checkpoints at iterations 250 and 500 also become
  durable, followed by the final iteration 750. The GB200 performance workload
  finishes successfully at 11:03:14 UTC. GB200 determinism completes native
  save/load/resume through iteration 125 and reference comparison without
  differences; its workload finishes successfully at 11:04:46 UTC.
- Latest full CI run `42365092053` validates the CPU integration harness and
  restarts the GB300 eight-node checks on the same production correction.
  Both wheels, all four images, wheel checks (509 passed / 40 skipped), native
  checks (64 passed / 1 skipped / 818 assertions), 83 CTest checks, the 56 CPU
  integration checks (2 CUDA-only skips), and loopback service stress pass.
- GB300 eight-node jobs `128383813341` / Slurm `661019` (determinism) and
  `128383813238` / Slurm `661018` (performance) remain queued for scheduler
  priority/resources without a start estimate. Both GB200 GitHub workload jobs
  pass. Its separate performance-report job `128387361764` fails only for the
  baseline warnings, so that workflow's overall conclusion is failure.

### Snapshot volume and performance observations

Across all 32 GB200 ranks, full local tensor registries total 415.750 GiB;
selected snapshots contain 353.263 GiB. Planning excludes 62.487 GiB per
checkpoint, a 15.03% reduction. These exact tensor totals hold for checkpoints
100, 250, 500, and 750 and match the unique global-registry shard volume.
All 128 rank/snapshot samples agree with daemon activation records.

Functional success remains separate from performance-baseline warnings. GB200
performance reports bandwidth 8.46% below baseline and write time 13.53% above
baseline. Its mean byte-count mismatch is 23,683.84375 bytes per rank, fully
accounted for by the blob payloads included in the unified snapshot-data metric;
tensor bytes match the baseline exactly. Determinism also flags timing and byte
baselines while its checkpoint comparison succeeds.

Resumed write-time medians vary from 65.991 to 97.630 ms, including one snapshot
faster than the 78.256 ms baseline. Tensor counts and full registry metadata
sizes remain stable. Two outliers on one host stall before activation completes.
The write timer includes HTTP activation, but existing records cannot isolate
its cost or establish a causal timing regression. No baselines are changed.
GB300 outcomes and the overall CI gate remain pending.

### Remaining validation limit

One native peer-durability success assertion returned an unexpected HTTP 500 in
an earlier run before the fixture printed response bodies. The case subsequently
passed ten normal runs, a traced run, and the final service suite. A targeted
production review found no completion-state visibility race. The original cause
remains unexplained; response-body and daemon-log context stay in the test for a
future recurrence. No speculative delay or behavior workaround is included.
