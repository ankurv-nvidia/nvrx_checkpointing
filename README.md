# Using `nvrx_checkpointing` from an sbatch script

Use the public Megatron-LM checkout pinned by this repository
(`1aa880d0ea1dfddef567785ebc9a384f2a600d18`) and make sure the
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
to `native`, which selects this repository's checkpoint writer. Existing
Megatron async-worker flags may remain in the script—the integration disables
Megatron's worker when the native provider is selected.

For an explicit provider selection, add:

```bash
--ckpt-provider native
```

Use `--ckpt-provider nvrx` only when intentionally running Megatron's reference
NVRx path. Loading checkpoints created with the historical patched Megatron
commit requires no extra flags.
