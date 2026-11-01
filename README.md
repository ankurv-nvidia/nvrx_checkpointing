# Using `nvrx_checkpointing` from an sbatch script

For named Ultra v3 runs, background submission, per-run work directories, and
custom image/source commits, see [Manual `slurm_job.py` runs](ci/slurm_job.md).

## Install a wheel from the project's pip index

Every `main` build that passes CI publishes its x86_64 and aarch64 wheels to the
project's pip index on GitHub Pages. Install the newest wheel compatible with
the current machine (Python 3.12 or newer) with:

```bash
python3 -m pip install --upgrade --pre \
  --index-url https://ankurv-nvidia.github.io/nvrx_checkpointing/simple/ \
  nvrx-checkpointing
```

The aarch64 wheel targets Grace (Neoverse V2) CPUs, as in GB200 and GB300
systems.

Published versions have the form `0.0.1.dev<RUN_NUMBER>+g<COMMIT_SHORT_SHA>`,
and the index keeps the newest 50 builds. To reproduce a particular build,
replace `VERSION` below with its published version:

```bash
python3 -m pip install \
  --index-url https://ankurv-nvidia.github.io/nvrx_checkpointing/simple/ \
  "nvrx-checkpointing==VERSION"
```

Use the public Megatron-LM checkout pinned by this repository
(`16251ac1221f4814cd39350def05cefa3c36734a`) and make sure the
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

### Enabling OpenTelemetry

Native checkpoint telemetry is disabled by default. To collect it through the
node-local checkpoint daemon, set its output file for the training ranks and
daemon. `daemon` is the default export mode:

```bash
export NATIVE_CKPT_OTEL_FILE=/path/to/native_ckpt_otel.jsonl
```

The file name accepts the same `%h` (host) and `%p` (process ID)
substitutions as `NATIVE_CKPT_SERVER_LOG`. A relative path is resolved beneath
the directory containing `NATIVE_CKPT_SERVER_LOG`.

Native span begin/end events automatically emit standalone `process.memory`
events recording current and peak RSS plus anonymous, file, shared-memory,
locked, pinned, and swapped bytes. These events include the most recently seen
session, rank, and snapshot IDs without adding memory attributes to the spans.

Set `NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS=1` to additionally emit a
periodic `process.memory` event once per second. Periodic sampling is disabled
by default.

To include Megatron traces and metrics in the same daemon collector, also set:

```bash
export MEGATRON_OTEL_ENABLED=1
export MEGATRON_OTEL_TRACES_ENABLED=1
export MEGATRON_OTEL_METRICS_ENABLED=1
export MEGATRON_OTEL_EXPORT_STRATEGY=all_ranks
export MEGATRON_OTEL_SPAN_GROUPS=default,load_checkpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4100
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

See [OpenTelemetry performance telemetry](docs/OpenTelemetry.md) for export
modes, collector topology, attributes, and analysis commands.

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
native integration.
