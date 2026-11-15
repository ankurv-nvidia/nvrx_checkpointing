# dtensor branch hand-off

## Current contract

- Provide Megatron-LM `fsdp_dtensor` save/load parity through the native provider.
- Preserve the existing `torch_dist` path and its common-state archive.
- Pass Megatron's preprocessed nested dictionary directly to native capture;
  do not flatten it in the provider or generate Python `torch.save` blobs.
- Keep planning and snapshot persistence in the daemon. Add no provider-owned
  training collectives. Megatron's existing preprocessing remains in use.
- Use main's FQN/global-range ownership policy. Replicas of the same tensor
  chunk and replicated scalar values are assumed identical; distinct global
  ranges are distinct chunks. Do not add tensor writer/source hints.
- Preserve peer-owned expert optimizer keys and rank-dependent RNG dictionaries.
- Scalar values may travel inline in registry exchanges. There is no scalar
  scrubbing flag or coordinator zeroing. Every persisted scalar is an eight-byte
  little-endian byte entry with its primitive datatype in DCP metadata.
- Load into the supplied destination dictionary and retain tensor storage,
  aliases, mutable containers, and runtime object types.
- Use existing C++ sources and minimal Python integration; no Python load
  planner/reader classes, extra native-state header, or global format boolean.

## Code structure

`megatron/save.py` retains Megatron state-dict generation and FSDP preprocessing.
The native binding registers supported Python values and local DTensor storage.
Prepared `torch_dist` registration retains its existing encoding and addresses.

Structured save and load use a common visitor in `src/python/bindings.cpp` for
paths, containers, namespace handling, depth/cycle checks, and tuple identity.
Leaf actions capture values, build load requests, or restore loaded values.
`src/python/torch_bindings.cpp` shares tensor geometry extraction; only capture
obtains payload storage. Load requests accept empty and strided destinations.

Loading has three stages:

1. Scan the destination dictionary into a local request `StateTree`.
2. `distcp::ReadCheckpoint` reads global `.metadata`, locates entries across
   writer files, and fills that tree and owned buffers. Writer rank is independent
   of the destination rank. Tensor views and overlapping shard ranges are handled
   in C++, and pickle globals are parsed as inert data.
3. Traverse the destination again, assign values, and copy tensors into their
   existing local storage. Runtime argument templates support bootstrap loads.

Public APIs are `checkpoint.load_checkpoint(path, state_dict, rank=0,
world_size=1, allow_partial_load=False)` and `checkpoint.read_common_state(path)`.
`megatron/load.py` supplies function hooks and runtime templates. The C++ binding
protects live runtime arguments with one copy during restoration.

Tensor-free containers retain the structured object encoding and per-rank
schema/source descriptions. These describe dictionary reconstruction and RNG
source selection during load; they are not daemon tensor-writer hints. Removing
this representation requires preserving expert/RNG behavior under stock DCP.

`debug.py` uses the C++ reader for native structured state. Its small Python
scalar inspector also supports historical rank bundles with overlapping physical
addresses. Common-archive and separate object manifests remain distinct. Scalar
tests exercise both the Python inspector and the public C++ reader. The
experimental registry debug formats 4–7 are
removed; main's formats 1–3 and the current format 8 remain readable.

## CI and validation

CI pins public Megatron-LM commit `16251ac1221f4814cd39350def05cefa3c36734a`.
`ci/configs/fsdp.yaml` adds the four-GPU eight-expert MoE workload. Its environment
metadata unsets `CUDA_DEVICE_MAX_CONNECTIONS`; other profiles retain main's
existing defaults. `fsdp_2gpu` inherits it with TP1, EP2, and DP2.

Both argument baselines are required: local `devbox_detr-native_fsdp_2gpu_1x2`
and `gb200_detr-native_fsdp_1x4`. Keep generated JET workflows synchronized with
`ci/jobs.json`. Do not weaken comparisons, thresholds, checkpoint counts, or
performance baselines to pass validation.

Validation passes 659 Python tests, 101 focused CPU load/inspection tests,
98 isolated server cases, and 94 of 95 CTest cases. The external-registry fixture
is absent; forwarding passes separately in its network-only namespace. Two CUDA-only cases
are skipped by CPU runtime checks. Native checkpoints load with Python checkpoint
readers disabled. Stock metadata protocols 2, 4, and 5 have interoperability
coverage, including strided tensors, booleans, scalar tensors, and empty tensors.

The two-GPU MoE comparison resumes at iteration 50 and trains through 125.
All five checkpoints match exactly: 3,209 entries at 25/50 and 3,208 at
75/100/125. Worker, epilog, argument, save-span, and telemetry checks pass.
Current validation artifacts are recorded in the accompanying change.

Runtime tests require private user/network/mount/IPC/PID namespaces. The host
checkpoint daemon on port 4100 belongs to another worktree and must not be
stopped. Run each server lifecycle test in a fresh process because the client
queue cache is process-global. CPU imports use the explicit CPU site/build;
source artifact links are restored to the CUDA build afterward.

## Known boundaries

- The four-GPU tensor comparison discrepancy remains unresolved. Failed run
  `43544071178` has 11 changed entries at 25/50 and 219 at 75/100/125, with no
  missing/extra entries or RNG/common differences. The first keys and hashes
  reproduce November 7 run `42407791025`.
- Run `43549290021` passes with a tensor-source override that has since been
  removed. It is not validation of the current ownership policy. Archived
  artifacts do not prove that valid replicas differed. Diagnosis needs per-rank
  FQN, mesh, placements, shape, ranges, and hashes before registration/capture.
- Globally empty DTensor capture omits its key, so strict loading lacks metadata.
  Empty ordinary tensors are supported. The exercised MoE model has no globally
  empty DTensors.
- A complete Torch archive stored as an unmarked generic opaque blob is
  ambiguous with a standard DCP object archive. Structured native byte values
  have explicit kinds and retain exact embedded archive bytes.

## Brief history

- The original work adds an FSDP YAML profile and JET end-to-end checkpoint test.
- November 13: the branch adopts main's daemon-generated snapshot plans;
  peer-owned entries replace the rank-0-only prototype.
- November 14: nested structured capture replaces provider flattening and
  `torch.save` object generation. Tensor source hints are subsequently removed
  under the identical-replica contract.
- November 14: `53bb5ed` persists all scalar nodes independently; `7350231`
  implements native three-stage loading and removes Python planner/codec files.
- November 14: `1fffa556` removes scalar scrubbing and records the binding audit.
  The user then requests shared traversal and broader diff reduction.
- November 15: shared traversal and tensor geometry remove duplicated binding
  code. The writer/session APIs match main; unused test-reader and experimental
  registry formats are removed. Historical scalar inspection stays in Python.

The complete earlier hand-off, discussion, and validation history is preserved
in [commit 1fffa556](https://github.com/ankurv-nvidia/nvrx_checkpointing/blob/1fffa556aa36e3ab52078a0bf22891fd50347641/prompt.md).
Read it locally with `git show 1fffa556:prompt.md` when historical detail is needed.
