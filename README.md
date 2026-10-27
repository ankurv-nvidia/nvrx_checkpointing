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

Use a Megatron-LM checkout that saves its checkpoints through this package,
such as the `native_ckpt` branch of
<https://github.com/ankurv-nvidia/megatron-lm>, and make sure the
`nvrx_checkpointing` package is installed in the training container on every
node. The Megatron entrypoints run unchanged.

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

Keep the existing Megatron entrypoint and its `--save`, `--load`,
`--save-interval`, and `--async-save` arguments.

Comparison harnesses that need a `native`/`nvrx` selector can opt in to
`nvrx_checkpointing.megatron.provider`.
