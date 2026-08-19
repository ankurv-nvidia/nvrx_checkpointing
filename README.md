# Using `nvrx_checkpointing` from an sbatch script

For named Ultra v3 runs, background submission, per-run work directories, and
custom image/source commits, see [Manual `slurm_job.py` runs](ci/slurm_job.md).

Use the public Megatron-LM checkout pinned by this repository
(`6c344290e54aa58a8be26d6af5ca2d46a3b00948`) and make sure the
`nvrx_checkpointing` package is installed in the training container on every
node.

The only required sbatch change is to run the Megatron entrypoint through the
integration module. For example, change:

```bash
python3 -u "${MEGATRON_LM_DIR}/pretrain_mamba.py" "${OPTIONS[@]}"
```

to:

```bash
python3 -u -m nvrx_checkpointing.megatron_integration "${MEGATRON_LM_DIR}/pretrain_mamba.py" "${OPTIONS[@]}"
```

The same substitution works inside `srun`, `numactl`, or an existing Python
wrapper chain. With `torchrun`, use its module form:

```bash
torchrun <torchrun-options> --module nvrx_checkpointing.megatron_integration "${MEGATRON_LM_DIR}/pretrain_mamba.py" "${OPTIONS[@]}"
```

Each rank must invoke the integration module; calling `install()` in a
separate setup process will not work because the patches are process-local.

Keep the existing `--ckpt-format torch_dist`, `--save`, `--load`, and
`--save-interval` arguments. The launcher adds `--ckpt-provider`; it defaults
to `native`, which selects this repository's checkpoint writer. The native
provider stages each checkpoint synchronously and lets the checkpoint daemon
persist it. Saves are chained back-to-back: a later checkpoint waits for the
previous one to become durable before taking its next snapshot. With
`--async-save`, the durability wait and tracker-file update run off the
training thread through Megatron's async queue. If the previous save is still
running when the next one starts, the native provider emits a flushed warning
and appends the time spent waiting in milliseconds on the same log line.

For an explicit provider selection, add:

```bash
--ckpt-provider native
```

Use `--ckpt-provider nvrx` only when intentionally running Megatron's reference
NVRx path. Loading checkpoints created with the historical patched Megatron
commit requires no extra flags.
