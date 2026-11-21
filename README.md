# Using `nvrx_checkpointing` from an sbatch script

## Install a wheel from the project's pip index

Successful `main` wheel builds publish x86_64 and aarch64 wheels to the
project's pip index on GitHub Pages. Install the newest wheel compatible with
the current machine (Python 3.12 or newer) with:

```bash
python3 -m pip install --upgrade --pre \
  --index-url https://ankurv-nvidia.github.io/nvrx_checkpointing/simple/ \
  nvrx-checkpointing
```

The aarch64 wheel targets Grace (Neoverse V2) CPUs, as in GB200 and GB300
systems.

Published versions have the form `0.0.1.dev<COMMIT_COUNT>+g<COMMIT_SHORT_SHA>`,
where `COMMIT_COUNT` is the number of commits in the built commit's history,
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

Native checkpoint telemetry is disabled by default. The checkpoint daemon
exports it over OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT`, the variable the
Megatron exporter also reads, so one setting sends both to an OpenTelemetry
Collector:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

Without an external collector, point the endpoint at the node-local daemon and
name the file the coordinator daemon writes for every node:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4100
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export NATIVE_CKPT_OTEL_FILE=/path/to/native_ckpt_otel.jsonl
```

With no endpoint, `NATIVE_CKPT_OTEL_FILE` alone makes each daemon write its
own telemetry to that file. The file name accepts the same `%h` (host) and `%p`
(process ID) substitutions as `NATIVE_CKPT_SERVER_LOG`. A relative path is
resolved beneath the directory containing `NATIVE_CKPT_SERVER_LOG`.

Native span begin/end events automatically emit standalone `process.memory`
events recording current and peak RSS plus anonymous, file, shared-memory,
locked, pinned, and swapped bytes. These events include the most recently seen
session, rank, and snapshot IDs without adding memory attributes to the spans.

Set `NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS=1` to additionally emit a
periodic `process.memory` event once per second. Periodic sampling is disabled
by default.

To include Megatron traces and metrics at the same endpoint, also set:

```bash
export MEGATRON_OTEL_ENABLED=1
export MEGATRON_OTEL_TRACES_ENABLED=1
export MEGATRON_OTEL_METRICS_ENABLED=1
export MEGATRON_OTEL_EXPORT_STRATEGY=all_ranks
export MEGATRON_OTEL_SPAN_GROUPS=default,load_checkpoint
```

See [OpenTelemetry performance telemetry](docs/OpenTelemetry.md) for the
collection topologies and attributes.

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
