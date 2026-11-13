# Hand-off: the `dtensor` branch of nvrx_checkpointing

## Current implementation, 2026-11-13

The sections below this update preserve the original hand-off and discussion. Their line numbers,
branch status, implementation descriptions, and validation commands describe the earlier branch.
The owner subsequently authorized implementation, preserving peer-owned objects, and a one-time
rebase + squash onto the latest main. Those instructions supersede the historical process rules.

The implementation uses main's unified peer registry and daemon-generated capture plans:

- Python calls Megatron's flattening and FSDP preprocessing, serializes object leaves in memory,
  and registers all participating replicas with their original ordered checkpoint descriptions.
- Registration supplies immutable `write_items` and `common_state_key` metadata. The normal
  two-argument `save_async` interface remains unchanged.
- Daemons select tensor and object sources using DCP's ordered load-balancing policy, then return
  local capture registries. Only selected local values enter each snapshot. Peer exchanges contain
  registry metadata, with scalar values scrubbed, and never object or tensor payloads.
- Each owner writes its scalar/object dictionary as a Torch-readable bundle in its local distcp
  file. Dynamic object sizes are refreshed on every save. Peer-only optimizer entries survive.
- The guarded Megatron load adapter restores those bundles into the flattened target state before
  tensor loading. Ordinary torch_dist capture, fixed-size blobs, and format markers retain main's
  behavior.

The provider imports no PyTorch save planner and performs no provider-owned collectives. Tests use
the stock planner as an independent reference. Source selection is covered for two and four ranks
with differing tensor and object replicas, including growing objects. Python set iteration can
change equal-cost DCP tie choices at larger rank counts; parity for those ties remains deferred.

The November 7 four-GPU run `42407791025` failed tensor comparisons: 11 differences at each of
iterations 25 and 50, and 219 at each of 75, 100, and 125 after resume. Objects, keys, shapes, and
layouts matched. Earlier two-GPU results do not establish four-GPU parity. The current implementation
registers all replicas and applies the daemon source policy to tensors as well as objects.

The November 13 four-GPU run `43403781753` matches all checkpoint entries at iterations 25, 50,
75, 100, and 125: 3,327 entries before resume and 3,326 after resume, with no missing, extra, or
changed entries. Both providers resume from iteration 50 and save through 125. Its argument
baseline is recorded from the native run. The final report flags bandwidth of 37,924 MB/s against
the 115,000 MB/s baseline; this metric is informational for the determinism workload. The missing
argument baseline causes the initial workflow failure, and the adopted record passes CI's argument
comparison. There is no controlled comparison against main for this workload's bandwidth.

Fresh validation and model-run results are recorded in the commit accompanying this update.

## Original hand-off

You are taking over the `dtensor` branch of this repository (remote `github`,
`git@github.com:ankurv-nvidia/nvrx_checkpointing.git`). Read this whole file before changing anything.
It gives you the goal, the history, an unfinished review, and code pointers. It deliberately avoids
telling you how the code works: read the pointers and form your own conclusions.

## What the branch is for

In the owner's words, from earlier work:

> Lets fork the dtensor work out into a dtensor branch that can go in earlier and adds a fsdp_dtensor
> path + e2e test

> Can you add a fsdp.yaml file next to nvrx_checkpointing/ci/configs/ultra3fp8.yaml that triggers the
> fsdp_dtensor path and then add that config to the jet-ci so that it runs e2e . That is what i meant
> by an e2e test, not an e2e_fsdp_dtensor.sh script.

The fsdp_dtensor path is Megatron-LM's `--ckpt-format fsdp_dtensor` (Megatron-FSDP), saved through
this repository's native provider instead of Megatron's own DCP save. CI pins Megatron-LM to
`DEFAULT_MEGATRON_COMMIT` (`ci/scripts/slurm_job.py:95`, `16251ac1221f…`).

## Branch state

The base is `67d996b2` (Jet CI integration (#1)). The commits since then, oldest first, are listed
below. Read them with `git log -p 67d996b2..dtensor`; each message states its intent.

| Commit | Subject |
| --- | --- |
| `73f4ae71` | Save fsdp_dtensor checkpoints of DTensor state dictionaries |
| `13c5f67d` | remove comments (the owner's commit) |
| `d61f2d31` | Write each fsdp_dtensor value from the rank DCP writes it from |
| `81b3ec4e` | Trace fsdp_dtensor checkpoints |
| `c1fd66a7` | Run Megatron-FSDP checkpoints end to end in JET CI |
| `879b1c01` | Set CUDA_DEVICE_MAX_CONNECTIONS on Hopper only |

Validation so far:

- Unit tests: the full `tests/` suite passed at `879b1c01` (one test deselected, see Validation tools).
- A local two-GPU emulation of the JET determinism job (see Validation tools) passed at `c1fd66a7`:
  the native and nvrx checkpoints matched on all 3,210 traced entries at iterations 5, 10, 15 and 20.
- The JET workload `gb200-native-detr-fsdp-1x4` has never run in CI. There is no PR, and the owner
  does not want one (see Process rules).

## The review in progress

After `879b1c01` the owner reviewed `megatron/save.py` with the previous agent. Both sides are
reproduced verbatim. The previous agent's answers are its own reading of the code. Treat every claim
in them as a hypothesis and check it against the code pointers before relying on it.

### Owner, message 1

> save.py
> I would like to take a dependence on megatron processing functions as much as possible
> what does preprocess_fsdp_dtensor_state_dict do ? Can we inline it in save.py if its not too much change ?
> same with flatten_state_dict ? seems like we're importing it from an internal namespace / class
>
> Why do we need to import DefaultSavePlanner ?
>
> i would like to avoid anything that needs a collective in save.py
> looks like _dcp_writes needs a collective. Can you explain why is that ?
> There's also a bunch of synchronous saves going on in that file that i find objectionable.
>
> Walk me through the justifications in small sentences so i can ask questions as you're explaining me

### Previous agent, answer 1

Answers in order, numbered so you can point at a line.

**1. `preprocess_fsdp_dtensor_state_dict`**
- It's Megatron's function, not ours (`megatron/training/checkpointing.py:1862` in 16251ac).
- Megatron's own fsdp_dtensor save calls it right before its DCP save (`:965`). We call it at the same point.
- It turns the live Megatron-FSDP state into the on-disk layout:
  - drops FP8 `_extra_state`;
  - splits fused SwiGLU and MLA weights;
  - renumbers local experts to global expert numbers;
  - renames MTP layers;
  - marks unevenly sharded DTensors so DCP writes them correctly.
- Its helpers come to about 1,100 lines (`megatron/core/transformer/fsdp_dtensor_checkpoint.py`).
- Inlining would mean copying Megatron's layout rules. Calling it is exactly the dependence on Megatron you want, so I'd keep it.

**2. `flatten_state_dict`**
- DCP flattens the nested state into dotted keys like `optimizer.state.<param>.exp_avg` before writing.
- We need the same keys, or DCP can't load what we write.
- Right now we import torch's private copy from `torch.distributed.checkpoint._nested_dict`.
- Megatron has its own copy: `megatron.core.dist_checkpointing.strategies.torch.flatten_state_dict`. It only adds Megatron's sharded types as leaves, so DTensor states flatten the same way.
- Switching is a one-line change.

**3. `DefaultSavePlanner`**
- Only `_dcp_writes` uses it.
- It's the planner Megatron's own fsdp_dtensor save hands to DCP.
- I used it to ask: "which rank would DCP pick to write this key?" If `_dcp_writes` goes, this import goes too.

**4. Why `_dcp_writes` needs a collective**
- Many non-tensor values exist on several ranks: `args`, `iteration`, per-parameter optimizer settings, `rng_state`.
- DCP writes each key once. Before `_dcp_writes`, every rank wrote its own copy and the metadata kept one. The extra copies were orphan bytes, which your gap check rejects.
- To write a key once, the ranks holding it must agree on one writer.
- No rank can see what the others hold (expert keys exist only on some ranks), so agreeing needs communication.
- DCP does this on every save: it gathers every rank's plan to rank 0, removes duplicates, and sends each rank its share. I do it once, at the first save.
- I copied DCP's choice because copies can differ. Megatron's fsdp_dtensor `rng_state` key leaves out the data-parallel rank (a TODO in Megatron), so ranks can store different values under one key.
- The code before it also had a collective: an `all_gather_object` to agree on slot sizes.
- The Megatron functions save.py calls may still communicate internally, just as in Megatron's own save. `_dcp_writes` is save.py's only direct collective.

**5. A way to drop the collective**
- The engine already deduplicates tensors on the server side. `DeduplicateTensorReplicas` (`src/checkpoint/server.cpp:312`) picks one owner for each replicated shard, from the registry merged across all ranks.
- Objects skip that step, which is why every rank writes them.
- Extending it to objects removes `_dcp_writes`, `DefaultSavePlanner` and the slot-size agreement from save.py.
- It's a C++ change, so it needs your OK under harness rule 8.
- Caveat: the engine would pick its own owner, not DCP's. That `rng_state` key could then differ from the reference in layouts where a rank's EP rank ≠ its TP rank. The CI layout isn't one of them.

**6. Blocking work in save.py**
- File writes on the training thread:
  - **Dataloader state:** Megatron's `maybe_save_dataloader_state` writes it before the snapshot is submitted, so the tracker never names an iteration without it. It could move into finalize, ahead of the tracker write.
  - **`metadata.json` (torch_dist only):** rank 0 writes it before submission. Megatron's own save writes it at finalize (`dist_checkpointing/serialization.py:436`), so ours should move there.
  - **Tracker file:** a few bytes, written by rank 0 once the checkpoint is durable.
- Waits:
  - **Without `--async-save`:** the save blocks until the checkpoint is durable. That's Megatron's contract for synchronous saves, and the DETR jobs use it.
  - **With `--async-save`:** it still waits until the snapshot is off the GPU, so the next step can't change tensors mid-copy.
  - **Previous save:** each save first waits for the previous one; Megatron allows one outstanding async save.
  - **After a load:** registration blocks until it completes.
- Serialization: objects go through `torch.save` into memory on the training thread. It's CPU time, not disk.

My suggestion: do 2 and the `metadata.json` move now, since both are small, and do 5 if you're OK with the C++ change.

### Owner, message 2

> 1. ok for now but we'll revisit after others have been clarified
> 2. But why do we need it at all if registering the state-dict will construct an fqn for it in the C++ StateTree
> 3. I would like to avoid any use of the torch planner. Lets discuss if thats feasible and if not i need a good understanding of why
> 4., 5. 6. I would like it to go through the regular StateTree registration and daemon-planning and writes . What is missing ?

### Previous agent, answer 2

**2. Why flatten at all**
- The C++ walker does build FQNs, but not the ones DCP uses.
- It splits `args` (a Namespace) into one node per field. DCP stores `args` as one object.
- It names list elements `key.[list]#0`. DCP names them `key.0`.
- It stores ints, strings and tuples as its own typed nodes. A string, for example, becomes raw wide-char bytes (`bindings.cpp:440`).
- DCP stores every leaf as a `torch.save` archive, and Megatron's fsdp_dtensor loader `torch.load`s each one. The engine never writes the int nodes at all, and Megatron's loader can't `torch.load` the typed blobs.
- So Python flattens to DCP's keys and `torch.save`s each leaf before registering.
- Dropping that needs a "DCP mode" in the walker: DCP's flattening rules plus `torch.save` for each leaf. The `torch.save` part already exists as `TorchSaveBlob`.

**3. Avoiding the torch planner**
- It's feasible.
- The planner's only job is to copy DCP's choice of which rank writes a shared key.
- Once the daemons pick an owner for each object (below), save.py needs neither the planner nor a collective.
- What we give up: Megatron's ambiguous `rng_state` key. When its copies differ, the daemon may keep a different rank's copy than DCP. That only happens when a rank's EP rank ≠ its TP rank, which the CI layout never has.

**4–6. Registration → daemon planning → daemon writes**

For objects, what's missing is all in the engine:
- **a.** The merged registry doesn't know which ranks hold an object. A Blob node is just bytes (`state.h:32`), whereas DTensor shards carry their rank.
- **b.** Daemon planning picks owners for tensor shards only. `DeduplicateTensorReplicas` (`server.cpp:312`) skips Blob nodes.
- **c.** So each daemon writes every object its rank registered (`server.cpp:1001`). That's where the orphans come from.
- **d.** The metadata step then requires every copy of an object to have the same length (`distcp.cpp:950`). That's the only reason Python agrees on slot sizes with a collective.
- Fixing a–c means tracking holders, picking one owner, and writing only owned objects. Then d goes away, and save.py just registers padded bytes, with no planner and no collective.

For the synchronous file writes:
- **e.** `metadata.json` and the tracker: the daemons have no way to write small extra files at finalize. A generic hook in the save call (file name → bytes, written by the coordinator after `.metadata`) would cover both, without the FQN hardcode you rejected. The tracker sits outside the iteration directory, so the hook must allow that.
- **f.** Dataloader state: Megatron's `maybe_save_dataloader_state` writes its own files. Routing it through the daemons means capturing its bytes instead, i.e. not calling that Megatron function as-is.
- **g.** The waits stay. With `--async-save` off, the save must be durable when it returns; that's Megatron's contract. Async saves still have to wait until the snapshot is off the GPU.

I'd start with a–d: that one engine change removes `_dcp_writes`, `DefaultSavePlanner` and the collective together. It's a C++ change under harness rule 8. Should I prototype it?

### Where it stands

The owner did not answer that last question and handed the branch to you instead. No code has changed
in response to this review.

## The owner's guidance

From this review:

- "I would like to take a dependence on megatron processing functions as much as possible"
- On `preprocess_fsdp_dtensor_state_dict`: "ok for now but we'll revisit after others have been clarified"
- On `flatten_state_dict`: "seems like we're importing it from an internal namespace / class" and
  "why do we need it at all if registering the state-dict will construct an fqn for it in the C++ StateTree"
- "I would like to avoid any use of the torch planner. Lets discuss if thats feasible and if not i need
  a good understanding of why"
- "i would like to avoid anything that needs a collective in save.py"
- "There's also a bunch of synchronous saves going on in that file that i find objectionable."
- "I would like it to go through the regular StateTree registration and daemon-planning and writes .
  What is missing ?"

From earlier work on this branch:

- On hardcoding in the engine: "I dont like the changes in distcp.cpp . We shouldnt need to hardcode
  based on whether that specific fqn exist. Convince me that its needed if it really is." Commit
  `73f4ae71` removed the FQN-keyed `metadata.json` write from `src/checkpoint/distcp.cpp`.
- On a flavor added only to unset one environment variable: "Creating a flavor for that seems a little
  overkill". Commit `879b1c01` replaced it.
- "no dont open a draft PR . Just push to the branch"
- The owner made the `distcp_trace` gap check in `debug.py` a hard error in commit `182ed095`. Keep it strict.

## Process rules

- Follow the "LLM/agent maintenance constraints" at the top of `ci/scripts/slurm_job.py`. Rule 8
  matters most here: explain and obtain approval before changing Python or C++ checkpointing
  implementation.
- Add new commits only. Never amend, rebase or force-push `dtensor`. Push to `github dtensor` and do
  not open PRs.
- Commit with `git commit -s` and an imperative subject line. This devbox has no signing key, so `-S`
  is not possible.
- Write comments only when the code cannot say it, in at most two lines. Rationale belongs in the
  commit message.
- `pyproject.toml` sets ruff `fix = true` and `unsafe-fixes = true`, so a plain `ruff check` rewrites
  files. The ruff on `PATH` (0.9.10) also strips `# noqa` codes it does not know. Lint with
  `/home/coder/workspace/nvrx_checkpointing/build/python_venv/bin/ruff check --no-fix` and compare
  the findings against `HEAD`.

## Code pointers

Line numbers are at `879b1c01`; function names are stable if they drift.

### This repository

Save path (Python):

- `megatron/save.py:429` `_save_checkpoint_impl`: the per-save flow for both formats.
- `megatron/save.py:359` `register_dtensor_state_dict`, `:343` `_dcp_writes`, `:194` `_snapshot_blob_capacity`,
  and the module caches at `:61-62`.
- `megatron/save.py:231` `_prepare_registration_state_dict`: how torch_dist state is prepared, for contrast.
- `megatron/save.py:390` `_write_checkpoint_tracker`, `:397` `_save_dataloader_state`,
  `:303` `maybe_finalize_async_save`.
- `megatron/save.py:108` `_wrap_load_checkpoint`: registration after a load.
- `megatron/save.py:182` `detect_peers`, `:324` `_ensure_root`.

Registration (C++ binding):

- `src/python/bindings.cpp:587` `RegisterStateDictRecrContainerItem`: the generic walker. Namespaces
  are handled at `:698`, list element naming at `:657` and `:673`, strings by `StringBlob` at `:440`.
- `src/python/torch_bindings.cpp:978` `TryRegisterTorchStateDict`, `:583` `IsMainTorchDTensorReplica`,
  `:598` `TorchDTensorState`, `:821` `TorchSaveBlob`.
- `src/state.h`: `StateNode` (the `Blob` type at `:32`) and `FinalizeRegistry` at `:420`.

Daemon planning and writes (C++):

- `src/checkpoint/server.cpp:890` `Session::OnFinalizeSnapshotAsync_`: merge, broadcast, plan, write.
- `src/checkpoint/server.cpp:312` `DeduplicateTensorReplicas`, called at `:973`. Local blobs are
  collected at `:1001`, and `WriteRankShard` is called at `:1013`.
- `src/checkpoint/distcp.cpp:916` `BuildMetadataPkl` (the replicated-blob length check is at `:950`),
  `:1016` `WriteRankShard` (blob payloads at `:1085`), `:1094` `WriteMetadata`.

Verification:

- `debug.py:319` `distcp_trace` (the gap check is at `:385`), `:268` `_is_plain_dcp`,
  `:273` `load_common_state_dict`. `metadata_compare.py` is the CLI the CI runs.

CI wiring:

- `ci/configs/fsdp.yaml`, which inherits `ci/configs/gpt_moe.yaml`.
- The `fsdp` entry under the gb200 `multinode` list in `ci/jobs.json`. `ci/github/generate_jet_workflows.py`
  regenerates `.github/workflows/jet-*.yml` from that file, and `.github/workflows/ci.yml` lists the
  workloads pull requests run.
- `ci/jet/generate_workloads.py:152` `resolved_workloads`: how a `detr` job becomes
  `--run ref:nvrx --run cmp:native`.
- `ci/scripts/slurm_job.py:2562` `_train_options` (the profile's `ckpt-format` at `:2596`),
  `:2711` `_phase_command` (the Hopper block at `:2734`), `:2014` `DeterministicFlavor`,
  `:1439` `_compare_arguments` (a new workload adopts its argument baselines from its first run).

Tests:

- `tests/test_megatron_save.py:307` and `:349`; the latter is a two-rank test driven by `_plan_and_save`.
- `tests/test_dtensor_registration.py`
- `tests/test_debug_semantic_objects.py:123`
- `tests/test_slurm_job_manual.py:1820` and `:2348`

### Megatron-LM at 16251ac

On this devbox it is checked out at `/home/coder/workspace/scratch/mlm_16251ac`.

- `megatron/training/checkpointing.py:959-1013`: Megatron's own fsdp_dtensor save, with the planner
  at `:975` and the synchronous save at `:1010`.
- `megatron/training/checkpointing.py:1862` `preprocess_fsdp_dtensor_state_dict`. Its handlers are in
  `megatron/core/transformer/fsdp_dtensor_checkpoint.py` and
  `megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py:238`.
- `megatron/training/checkpointing.py:556`: the fsdp_dtensor `rng_state` key. Read its TODO.
- `megatron/training/checkpointing.py:1576` `maybe_save_dataloader_state`.
- `megatron/training/checkpointing.py:2270` onward: the fsdp_dtensor load.
- `megatron/core/dist_checkpointing/strategies/torch.py:107` `flatten_state_dict`.
- `megatron/core/dist_checkpointing/serialization.py:436` `metadata_finalize_fn`.
- `megatron/training/arguments.py:1192` and `:1489-1515`: the `CUDA_DEVICE_MAX_CONNECTIONS` rules.

### torch

In the installed version:

- `torch/distributed/checkpoint/default_planner.py`: `DefaultSavePlanner` and `create_default_local_save_plan`.
- `torch/distributed/checkpoint/_dedup_save_plans.py`: `dedup_save_plans`.
- `torch/distributed/checkpoint/_nested_dict.py` and `_traverse.py`: `flatten_state_dict` and its traversal rules.

## Validation tools

These paths are on this devbox.

- **Python:** use the Megatron virtualenv, `/home/coder/workspace/megatron-lm/.venv/bin/python`.
- **Build:** `/home/coder/workspace/scratch/nvrx_ci_build/build_worktree.sh <src> <build dir> [tests]`
  builds an importable site for a worktree.
  - The site used so far is `/home/coder/workspace/build/nvrx-dtensor/site`.
  - Its Python files were copied in by hand after edits, so rebuild or re-copy after changing Python.
- **Unit tests:** `PYTHONPATH=<site>:<dir with hypothesis> python -m pytest -q -p no:cacheprovider tests
  --deselect tests/test_quantized_binding.py::test_quantized_registration_rejects_components_on_different_devices`.
  - The deselected test also fails on main here.
  - `hypothesis` is not in the virtualenv; `pip install --target <dir> hypothesis` provides it.
- **Local determinism job:** `NVRX_SITE=<site> python /home/coder/workspace/scratch/nvrx_fsdp_ci/drive.py
  <nvrx worktree> <work dir> [--compare-only]`.
  - It emulates the JET determinism job on two GPUs against Megatron 16251ac, comparing an nvrx
    reference run with a native run.
  - It shrinks the parallelism to fit two GPUs, so read it before trusting its results.

## Related work

`.worktrees/native_ckpt-v2` was never pushed. It is based on `73f4ae71` and will need rebasing onto
this branch. Leave it alone unless the owner asks.
