# Manual `slurm_job.py` runs

`ci/scripts/slurm_job.py` supports isolated, named runs for large Ultra v3
experiments. A named run writes its durable state below:

```text
<work-dir>/<name>/
├── checkpoints/
├── phase1-checkpoints/
├── tensorboard/
├── data-cache/
├── deps/
│   ├── megatron-lm/
│   ├── nvidia-resiliency-ext[-<commit>]/
│   ├── python/nvrx_checkpointing/
│   └── vcpkg/
├── build/
│   └── nvrx_checkpointing/
├── logs/
│   ├── <run>.log
│   └── <run>_prep.log
├── gitlab_job_id
├── job_id
└── run.sbatch
```

The launcher writes `job_id` as soon as Slurm accepts the allocation, including
while it is pending. A directly submitted `run.sbatch` writes it when the batch
script starts. CI runs also record `gitlab_job_id`; canceling a GitLab job sends
`scancel` to that exact allocation, with the CI `after_script` as a fallback.
`run.sbatch` has
resolved `#SBATCH` values, absolute output paths, and the non-secret run
environment. Its executable commands refer back to those exports (for example,
`${ENROOT_IMAGE}` and `${SLURMJOB_OPT_RUN_DIR}`), so changing a source checkout or
image does not require editing every `srun`. `#SBATCH` directives remain
resolved because Slurm reads them before the shell can expand variables. The
reproducer can be submitted directly:

```bash
cd /lustre/my-runs/ultra3-64n
sbatch run.sbatch
```

Foreground and background submissions both write one combined training,
reporting, and allocation-cleanup log at `logs/<run>.log`; the prep step keeps
its separate `logs/<run>_prep.log`.

The reproducer explicitly exports the resolved PyTorch allocator, NCCL, UCX,
NVTE, TorchInductor, Triton, and checkpoint-service settings and passes them
through the training container. Immediately before every pretrain phase or
repeat cycle, rank 0 prints an allowlisted snapshot with the
`[slurm_job:env]` prefix. Authentication variables are excluded.

Authentication tokens are intentionally not written to the reproducer. If a
direct re-run has no `CI_JOB_TOKEN`, native wheel setup falls back to a source
build. Its preparation step uses a writable container overlay and runs
`scripts/prepare.sh ubuntu2404` when the custom image does not contain
`clang-22`; the supplied `.sqsh` itself is not overwritten. Native checkpoint
setup and an exact NVIDIA Resiliency Extension install share one preparation
`srun`. nvrx_checkpointing is installed with `pip --target` under
`<run>/deps/python`, so the final `pip list` in `<run>_prep.log` includes its
version. The log also records its selected source commit, installed module
path, shared library, and server path explicitly.

## One-time shell setup

The runner setup normally already provides the first four variables:

```bash
export SLURM_ACCOUNT=coreai_resiliency_osiris
export CLUSTER=ocihsg
export GPUS_PER_NODE=4
export PLATFORM=gb200

export SLURMJOB_OPT_WORK_DIR=/lustre/my-runs
export ENROOT_IMAGE=/lustre/images/custom-ultra3.sqsh
```

`SLURMJOB_OPT_WORK_DIR` must be on storage visible at the same absolute path from every
allocated node.

## Foreground and background runs

Foreground submission waits and streams the training log:

```bash
python3 ci/scripts/slurm_job.py pretrain-native \
  --config ultra3fp8 \
  --nodes 64 \
  --name ultra3-64n
```

Background submission returns as soon as `sbatch` accepts the exclusive
allocation:

```bash
python3 ci/scripts/slurm_job.py pretrain-native \
  --config ultra3fp8 \
  --nodes 128 \
  --name ultra3-128n \
  --background
```

The name can also come from the environment, leaving only the scale on the
command line:

```bash
SLURMJOB_OPT_RUN_NAME=ultra3-256n SLURMJOB_OPT_BACKGROUND=1 \
  python3 ci/scripts/slurm_job.py pretrain-native \
    --config ultra3fp8 \
    --nodes 256
```

When no name is supplied, the launcher generates:

```text
[{gitlab_pipeline_jobid}_]{job_name}_{config}_{nodes}n_mlm.{mlm_abbreviated_sha_commit}_nvrx.{nvrx_abbreviated_sha_commit}_nckpt.{native_checkpointing_abbreviated_sha_commit}
```

The bracketed prefix is included only when GitLab supplies `CI_JOB_ID`. It
gives every job and retry its own run directory. Manual runs omit it without
leaving a leading underscore.
For example, a run may be named
`pretrain-native_perf_nano3_4n_mlm.7b7eef9a_nvrx.ae97937a_nckpt.5e674dbd`.
`job_name` includes the selected flavor so concurrent performance,
deterministic, and valgrind jobs do not share a directory.
An explicit `--name` or `SLURMJOB_OPT_RUN_NAME` may use any of the fields in the
default template, so `--name '{gitlab_pipeline_jobid}_{config}_{nodes}n_nckpt.{native_checkpointing_abbreviated_sha_commit}'`
is expanded before the run directory and Slurm job are created. If a Megatron
or NVRx commit is not selected explicitly, its field expands to `image`.

Use `--dry-run` to generate and inspect `<work-dir>/<name>/run.sbatch` without
submitting:

```bash
SLURMJOB_OPT_RUN_NAME=ultra3-64n-check \
  python3 ci/scripts/slurm_job.py pretrain-native \
    --config ultra3fp8 \
    --nodes 64 \
    --dry-run
```

## Long convergence runs

The default measured phase saves three checkpoints, at iterations 250, 500,
and 750. `--checkpoints-per-run` changes that count. A value of `0` keeps one
training process running and checkpointing every 250 iterations until it
finishes or Slurm reaches the allocation limit:

```bash
python3 ci/scripts/slurm_job.py pretrain-native \
  --config nano3 --nodes 4 --name native-main-long \
  --checkpoints-per-run 0 --time-minutes 240 --background
```

`--repeat` instead restarts the training process inside the same training
`srun`, on the same Slurm allocation, after each checkpoint cycle and loads the
latest checkpoint before continuing. It does not submit or request another
allocation. The phase-1 bootstrap runs only once. With the default
three-checkpoint cycle, cycle 1 writes
250/500/750, cycle 2 writes 1000/1250/1500, and cycle 3 starts at 1750:

```bash
python3 ci/scripts/slurm_job.py pretrain-native \
  --config nano3 --nodes 4 --name native-main-repeated \
  --repeat 4 --time-minutes 240 --background
```

Set `--repeat 0` to repeat until the allocation ends. An unlimited
`--checkpoints-per-run 0` cannot also be repeated because that training process
does not return. Deterministic reference-comparison flavors require finite
counts so both reference and measured phases can run.

These are the two common indefinite convergence modes. Both use the `nano3`
profile on four nodes, start from a clean automatically named directory below
the selected work directory, and remain bounded by the same four-hour Slurm
allocation:

```bash
# One training process keeps checkpointing indefinitely.
python3 ci/scripts/slurm_job.py pretrain-native \
  --config nano3 --nodes 4 --work-dir /lustre/runs \
  --clean-run --checkpoints-per-run 0 --time-minutes 240 --background

# Repeat the default three-checkpoint cycle indefinitely on that allocation.
python3 ci/scripts/slurm_job.py pretrain-native \
  --config nano3 --nodes 4 --work-dir /lustre/runs \
  --clean-run --repeat 0 --time-minutes 240 --background
```

Long-run modes apply the raised training-sample ceiling to both the phase-1
bootstrap and measured phases so the optimizer scheduler saved by phase 1
matches the scheduler loaded for continued training. The configured warmup and
decay sample horizons remain unchanged. Every cycle uses the same
`<run>/tensorboard/` directory and resumes at the saved iteration, so loss and
gradient-norm curves have a continuous TensorBoard x-axis. An unlimited run
that reaches its Slurm limit may end with a `TIMEOUT`; its last completed
checkpoint and TensorBoard events remain in the named run directory.

Use `--clean-run` when intentionally starting over with an existing name. It
removes exactly `<work-dir>/<name>/` on the launcher before creating the new
logs, dependencies, checkpoints, TensorBoard directory, and reproducer. The
generated `run.sbatch` records the same intent and repeats the cleanup when
submitted directly, while preserving itself and its currently open scheduler
log. The cleanup is not repeated by worker ranks:

```bash
python3 ci/scripts/slurm_job.py pretrain-native \
  --config nano3 --nodes 4 --name native-main-repeated \
  --repeat 4 --time-minutes 240 --clean-run --background
```

Every generated batch allocation uses `#SBATCH --exclusive`, explicitly
requests one task per GPU (`#SBATCH --ntasks-per-node=4` on Ultra v3), and
uses Slurm's `singleton` dependency so runs with the same user and job name do
not overlap. Allocations of 16 nodes or more use `#SBATCH --segment=16`;
smaller allocations omit the topology request because Slurm rejects it. Set
`SLURM_SEGMENT` to override either behavior. Every `srun` uses
`--kill-on-bad-exit=1`, so one failed rank ends the distributed step instead
of holding the allocation. The checkpoint service uses a port derived from the
Slurm job ID so an orphaned daemon from an older allocation cannot intercept a
new run. Training
and preparation `srun` steps redirect their output to separate files below
`logs/`. The allocation retains total task counts
(`#SBATCH --ntasks=256/512/1024`) but omits the redundant total GPU request
(`#SBATCH --gpus=...`); GPU placement is expressed by
`#SBATCH --gpus-per-node=4`.

At a training-phase boundary, every task on a node rendezvous before local rank
0 spins down that node's checkpoint server. Shutdown verifies the service is no
longer reachable and removes its shared-memory queues and election lock before
the tasks can enter the next phase.

An allocation-exit trap runs one host-side task per node after the reporting
epilog and stops detached checkpoint servers and removes their shared-memory
queues. The trap also runs when preparation or training fails and preserves the
original failure code. Cleanup must cover every allocated node and run outside
the container so it can see host processes and shared memory.

## Custom source commits

Both overrides have CLI and environment forms:

```bash
export MEGATRON_COMMIT=<40-character-megatron-lm-commit>
export NVRX_COMMIT=<40-character-nvidia-resiliency-ext-commit>

SLURMJOB_OPT_RUN_NAME=ultra3-source-test SLURMJOB_OPT_BACKGROUND=1 \
  python3 ci/scripts/slurm_job.py pretrain-native \
    --config ultra3fp8 \
    --nodes 64
```

Equivalent flags are `--megatron-commit` and `--nvrx-commit`.
`MEGATRON_REPO_URL` / `--megatron-repo-url` and
`NVRX_REPO_URL` / `--nvrx-repo-url` override the corresponding repositories.
The `NVRX_*` source variables are reserved for NVIDIA Resiliency Extension:
in particular, `NVRX_COMMIT` always identifies its commit, never the
nvrx_checkpointing commit.

`NVRX_REPO_ROOT` selects the source checkout for NVIDIA Resiliency Extension.
If it does not exist, the preparation step clones `NVRX_REPO_URL` there. If it
already exists, it is reused without switching or cleaning its working tree.
With `NVRX_COMMIT`, the requested Git object is fetched into that checkout when
needed and installed exactly; without a commit, the existing checkout contents
are installed.

The nvrx_checkpointing source is always the repository containing
`ci/scripts/slurm_job.py`. Native wheels are built from that checkout and
pip-installed under the named run directory. To select a different native
revision, invoke that revision's copy of the launcher. NVIDIA Resiliency
Extension remains independently selectable:

```bash
export NVRX_REPO_ROOT=/lustre/checkouts/nvrx-b
export NVRX_COMMIT=<40-character-nvidia-resiliency-ext-commit>
```

The requested NVIDIA Resiliency Extension commit is installed with
`--no-deps` into the run's `deps/` directory. Its path is placed first on
`PYTHONPATH`, so it replaces the version pre-installed in the custom image
without changing the image's dependency set.

## Environment reference

| CLI | Environment | Meaning |
| --- | --- | --- |
| `--nodes` | `SLURMJOB_OPT_NODES` | Number of allocated nodes |
| `--config` | `SLURMJOB_OPT_CONFIG` | Model configuration |
| `--gpus-per-node` | `GPUS_PER_NODE` | GPUs and tasks per allocated node |
| `--flavor` | `SLURMJOB_OPT_FLAVOR` | Comma-separated flavors when set through the environment |
| `--name` | `SLURMJOB_OPT_RUN_NAME` | Run and scheduler job name |
| `--work-dir` | `SLURMJOB_OPT_WORK_DIR` | Parent of the named run directory |
| `--image` | `ENROOT_IMAGE` | Custom `.sqsh` image |
| `--background` | `SLURMJOB_OPT_BACKGROUND=1` | Nonblocking `sbatch` submission |
| `--megatron-commit` | `MEGATRON_COMMIT` | Exact Megatron-LM commit |
| `--megatron-repo-url` | `MEGATRON_REPO_URL` | Megatron-LM repository |
| `--nvrx-commit` | `NVRX_COMMIT` | Exact NVIDIA Resiliency Extension commit |
| `--nvrx-repo-url` | `NVRX_REPO_URL` | NVIDIA Resiliency Extension repository |
| `--checkpoints-per-run` | `SLURMJOB_OPT_CHECKPOINTS_PER_RUN` | Checkpoints per training process; `0` is unlimited |
| `--repeat` | `SLURMJOB_OPT_REPEAT` | Checkpoint cycles to run; `0` repeats indefinitely |
| `--time-minutes` | `SLURMJOB_OPT_TIME_MINUTES` | Slurm allocation limit in minutes |
| `--clean-run` | `SLURMJOB_OPT_CLEAN_RUN=1` | Remove the named run directory before submission |
| `--dry-run` | `SLURMJOB_OPT_DRY_RUN=1` | Write the reproducer without submitting it |
| — | `NVRX_REPO_ROOT` | NVIDIA Resiliency Extension source checkout/clone path |

For every row, the resolved value follows `built-in default < environment <
command-line argument`. Command-line arguments therefore replace rather than
combine with environment values, including repeated `--flavor` arguments.
