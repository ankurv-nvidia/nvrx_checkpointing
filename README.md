# Using `nvrx_checkpointing` from an sbatch script

For named Ultra v3 runs, background submission, per-run work directories, and
custom image/source commits, see [Manual `slurm_job.py` runs](ci/slurm_job.md).

## Install a wheel from GitLab PyPI

Successful `main` pipelines publish tested x86_64 and aarch64 wheels to the
project's public GitLab PyPI registry. Install the newest wheel compatible with
the current machine (Python 3.12 or newer) with:

```bash
python3 -m pip install --upgrade --pre \
  --index-url https://gitlab-master.nvidia.com/api/v4/projects/381021/packages/pypi/simple \
  nvrx-checkpointing
```

Published versions have the form
`0.0.1.dev<PIPELINE_IID>+g<COMMIT_SHORT_SHA>`. To reproduce a particular
pipeline, replace `VERSION` below with its published version:

```bash
python3 -m pip install \
  --index-url https://gitlab-master.nvidia.com/api/v4/projects/381021/packages/pypi/simple \
  "nvrx-checkpointing==VERSION"
```

Use the public Megatron-LM checkout pinned by this repository
(`aac9b2db7f0d203940f6c0a7f2852a094c18c12e`) and make sure the
`nvrx_checkpointing` package is installed in the training container on every
node.

The only required sbatch change is to run the Megatron entrypoint through the
integration module. For example, change:

```bash
python3 -u "${MEGATRON_LM_DIR}/pretrain_mamba.py" "${OPTIONS[@]}"
```

to:

```bash
python3 -u -m nvrx_checkpointing.megatron.integration "${MEGATRON_LM_DIR}/pretrain_mamba.py" "${OPTIONS[@]}"
```

The same substitution works inside `srun`, `numactl`, or an existing Python
wrapper chain. With `torchrun`, use its module form:

```bash
torchrun <torchrun-options> --module nvrx_checkpointing.megatron.integration "${MEGATRON_LM_DIR}/pretrain_mamba.py" "${OPTIONS[@]}"
```

Each rank must invoke the integration module; calling `install()` in a
separate setup process will not work because the patches are process-local.

### Adapting an existing Megatron workload launcher

Wrap the existing Megatron entrypoint in the `run_cmd` assignment:

```bash
run_cmd="python -u -m nvrx_checkpointing.megatron.integration ${REPO_DIR}/pretrain_hybrid.py ${options}"
```

Keep the existing `--ckpt-format torch_dist`, `--save`, `--load`, and
`--save-interval` arguments. This launcher is native-only and does not patch
Megatron's argument parser or optimizer classes. It stages each checkpoint and
lets the checkpoint daemon persist it. Saves are chained back-to-back: a later
checkpoint waits for the previous one to become durable before taking its next
snapshot. With `--async-save`, the durability wait and tracker-file update run
off the training thread through the package's rank-local queue. If the previous
save is still running when the next one starts, the native provider emits a
flushed warning and appends the time spent waiting in milliseconds on the same
log line.

Comparison harnesses that need a `native`/`nvrx` selector can opt in to
`nvrx_checkpointing.megatron.provider`; reference runs should not install the
native integration. Loading checkpoints created with the historical private
fork and its stringified optimizer dtype keys requires no extra flags.
