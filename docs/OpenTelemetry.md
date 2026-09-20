# OpenTelemetry performance telemetry

The checkpoint daemon emits OpenTelemetry **spans** (and, opt-in, **metrics**) for
every checkpoint save, and forwards its operational **logs** as OTLP log records.
It also accepts OTLP/HTTP protobuf traces, metrics, and logs. All terminal telemetry
is written as **OTLP-JSONL** (one OTLP JSON object per line) and analyzed offline by
`ci/scripts/otel_report.py`.

Telemetry is **off by default**. Native CI jobs driven by `ci/scripts/slurm_job.py`
enable it for both checkpoint providers. When enabled, performance reports consume
the post-run OTLP data and daemon and trainer logs remain diagnostic artifacts. When
both OTel flags are disabled, CI uses the legacy log report instead.

## Enabling

Set on every rank (the daemon inherits its own env):

| Variable | Default | Meaning |
|---|---|---|
| `NATIVE_CKPT_OTEL_EXPORT` | `daemon` | `daemon` — route through the node-local daemon collector, which collects to a single file on rank 0. `file` — each process writes its own OTLP-JSONL. `both` — use both routes. |
| `NATIVE_CKPT_OTEL_FILE` | — | A nonempty value enables native telemetry and selects the collector-local output file. In `file` or `both` mode it also selects each process's output path. `%h` and `%p` expand to the host and process ID, as for `NATIVE_CKPT_SERVER_LOG`. A relative path is resolved beneath the server-log directory. |
| `NATIVE_CKPT_OTEL_COLLECTOR` | — | In `daemon` or `both` export mode, initialize a collector before checkpoint topology exists: `file` writes `NATIVE_CKPT_OTEL_FILE`; any other value relays to that host or `host:port`. |
| `NATIVE_CKPT_OTEL_SERVICE` | — | Override `service.name`. |
| `NATIVE_CKPT_OTEL_METRICS` | `0` | `1` also emits `bw_mbps`, `wr_tnsr_us`, and `total_written_bytes` OTLP histograms. Off by default because the spans already carry every figure; enable for Grafana. |
| `NATIVE_CKPT_OTEL_LOGS` | `1` | `0` suppresses just the log-record signal (spans/metrics unaffected). |
| `NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS` | `0` | Positive interval for periodic `process.memory` events. Native span begin/end events emit standalone memory events without periodic sampling. |
| `NATIVE_CKPT_CI_PHASE` | — | Sets the `nvrx.ci_phase` resource attribute so the analyzer can scope to a phase. |
| `NATIVE_CKPT_LOG_LEVEL` | `info` | Daemon spdlog threshold (`trace`/`debug`/`info`/`warn`/`error`). |

## Architecture

```text
training native OTel SDK ── serialized JSON ──► node-local daemon ──┐
daemon OTel SDK ─────────── serialized JSON ───────────────────────┼─ FileCollector ── JSONL
OTLP/HTTP protobuf ─────────────── terminal decode ────────────────┘
                              └─ RelayCollector ── unchanged OTLP request ──► another daemon
```

- A single TU (`src/checkpoint/logtel.cpp`) contains the OpenTelemetry SDK, spdlog,
  and daemon telemetry collection behind the lightweight `src/telemetry.h` and
  `src/logging.h` facades.
- The daemon holds one active `FileCollector` or `RelayCollector` directly. Checkpoint
  session topology normally selects the route. `NATIVE_CKPT_OTEL_COLLECTOR` selects it
  at startup when the daemon is used only as a collector. OTLP requests received before
  a route exists are buffered up to 64 MiB and drained when one becomes available.
- The OTLP/HTTP receiver accepts binary protobuf on `/v1/traces`, `/v1/metrics`, and
  `/v1/logs`, including gzip request bodies. Requests and decompressed payloads are
  limited to 64 MiB. Success and failure responses use the corresponding OTLP protobuf
  response format.
- A relay forwards the OTLP path, content encoding, and body unchanged, then returns
  the downstream response. It does not decode, delimit, or retry the request; the
  originating OTLP exporter owns retry policy. Multi-hop relay remains valid.
- A terminal collector decodes protobuf and uses OpenTelemetry's OTLP file serializer
  to produce canonical JSON. The file collector alone appends and flushes the JSONL
  newline. Daemon-generated telemetry already reaches it as canonical JSON.
- The existing JSON relay used by daemon-generated telemetry retains its three bounded
  attempts. Its at-least-once delivery can duplicate a batch if a response is lost.
- The daemon maps its own topology policy onto that neutral collector configuration.
  In `daemon` mode rank 0 writes locally, while the other daemons relay to rank 0's
  `/api/telemetry/remote` listener. Every daemon can accept both local and remote input.
- Each training process sends its native telemetry to its node-local daemon in
  `daemon` mode. In `file` mode each native process writes to
  `NATIVE_CKPT_OTEL_FILE`; use `%h` and `%p` in that path to give every process
  a distinct file.

### Spans, logs and attributes

- The daemon carries the perf counters, so it emits the **`ckpt.save`** span once per
  rank at the point it prints the `[PERF]` line, with `nvrx.rank`, `nvrx.bw_mbps`,
  `nvrx.session_id`, `nvrx.snapshot_id`, `nvrx.wr_tnsr_us`, `nvrx.snapshot_us`,
  `nvrx.total_us`, `nvrx.num_tensors`, `nvrx.total_d2h`,
  `nvrx.total_tensor_bytes`, `nvrx.total_written_bytes`, `nvrx.avg_d2h_rq`, and
  `nvrx.avg_tnsr`. The span's start and duration cover the cross-process interval
  recorded by the `PERF() << span::CkptSave::{Begin,End}` events. The daemon also
  emits `ckpt.svr.finalize`, `ckpt.svr.write_metadata`, and `ckpt.svr.durable` from
  independently emitted, ID-matched Begin/End records. `ckpt.svr.durable` covers the
  coordinator's snapshot activation-to-durable interval.
- Every native checkpoint span records current process-memory fields under
  `nvrx.memory.start.*` and `nvrx.memory.end.*`. The zero-duration
  **`process.memory`** span records the same fields under `nvrx.memory.*`, plus
  `nvrx.memory.trigger` and `nvrx.process.role`. Training events include
  `nvrx.rank`, `nvrx.local_rank`, and, once known, `nvrx.snapshot_id` and
  `nvrx.snapshot_bytes`.
- Daemon operational logs (`[native_ckpt:svr][LEVEL] …`) go through spdlog, and a
  custom sink forwards each as an OTLP **log record** with its severity and
  current `nvrx.memory.*` attributes.
- The daemon's `[native_ckpt:svr][{session}][PERF] …` Begin/Finished stderr markers
  remain human-readable diagnostics. CI parses them only when OTel is disabled.
- Resource attributes: `service.name`, `host.name`, `process.pid`,
  `nvrx.process.role`, `nvrx.ci_phase`, and, when supplied by CI, `nemo.run.id`.

## Analysis

```bash
# Human summary of the ckpt.save spans:
python3 ci/scripts/otel_report.py "$NATIVE_CKPT_OTEL_FILE"
```

`otel_report.py` extracts one sample per `ckpt.save` span (keyed on the span's
`nvrx.rank`) and reads OTLP gauges, sums, and histograms. It does not recreate
performance log lines. `tests/test_otel_report.py` covers the extraction and summary
with synthetic OTLP fixtures.

## CI perf gating

`PerfReport` (`ci/scripts/slurm_job.py`) derives every checkpoint-provider metric from
the collected OTLP data after a run:

- Native writer bandwidth, write time, and bytes come from `ckpt.save` spans.
- Save-to-durable time comes from Megatron checkpoint spans for both providers.
  Asynchronous saves use `nvrx.call_idx`, with a cardinality-checked FIFO match when
  the dispatch span does not carry the request index.
- Native coordinator activation-to-durable time comes from `ckpt.svr.durable`.
- Megatron step duration, TFLOP/s/GPU, tokens/s, and allocated GPU memory come from
  `megatron.training.*` metrics. The final cumulative histogram point supplies step
  duration; the final gauge point supplies each other value.
- CUDA allocation, HBM, GPU capacity, and CPU RSS gauges are exported once per rank
  after training and aggregated by rank. `gpu_peak_bytes` is PyTorch's peak live
  allocation; `gpu_hbm_peak_bytes` is the sampled whole-device HBM use and therefore
  also includes reserved allocator memory, CUDA/NCCL state, and other device users.
- The Slurm harness also persists each training rank's process high-water mark and
  reads each native checkpoint daemon's Linux `VmHWM` immediately before shutdown.
  The final report prints `training_process_peak_rss_bytes` across ranks and
  `checkpoint_daemon_peak_rss_bytes` across nodes, taking each process's maximum
  over every phase and cycle in the run. GPU and host peak-memory metrics are
  compared with provider-specific workload baselines.
- The existing one-second GPU sampler also records each training rank's current
  RSS as OTLP `process.memory` events. Each rank sends them through its node-local
  daemon into the single collected OTel JSONL. After each phase, CI combines those
  samples with the native daemon's
  `process.memory` events in one `process_memory_all.svg`: daemon processes use
  blue shades and training processes use red shades. The graph contains host RSS
  only and is generated solely from that OTel JSONL.
- CI compresses the final OTel JSONL and Perfetto JSON as `.gz` files after all
  validation and report generation have completed.

Metrics carrying `dl.rank` retain per-rank medians plus min, median, and max across
ranks in the JSON report and generated baseline candidate. Checkpoint durability is
paired and summarized independently for every Megatron exporter rank. Coordinator-only
durability has no rank aggregation.

OTel takes precedence whenever `NATIVE_CKPT_OTEL_FILE` is nonempty or
`MEGATRON_OTEL_ENABLED` is enabled. Missing, malformed, mismatched, or incomplete
OTLP data then fails validation instead of silently falling back to logs. If both
are disabled, `log_report.py` extracts the legacy performance markers and validates
that every native `ckpt.save` Begin has a schema-valid matching Finished marker.
The wiring:

- `_TRAIN_ENV_DEFAULTS` sets `NATIVE_CKPT_OTEL_EXPORT=daemon`, and each
  telemetry-enabled cycle assigns `NATIVE_CKPT_OTEL_FILE`. Periodic process-memory
  sampling remains opt-in through `NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS`;
  native span begin/end emissions generate standalone memory events without it.
  Megatron sends traces and metrics from every rank to the node-local daemon so
  checkpoint timing summaries retain rank minima, maxima, and medians. The default
  span groups plus `load_checkpoint` cover both checkpoint directions.
- CI starts one daemon per node before the trainer so startup exports reach the
  pre-topology buffer rather than waiting for checkpoint warmup to launch it.
- Image preparation installs the pinned Megatron checkout's `otel` extra; the
  underlying PyTorch base-image tag is unchanged.
- `_run_routine` gives each phase or cycle one `native_ckpt_otel.jsonl` beneath
  `ci-results/otel/{cycle_tag}`. Every node-local daemon relays to the rank-zero
  daemon, which writes that single file. Native checkpoint topology selects the route;
  NVRx runs configure the collector-only daemons explicitly.
- Each cycle requires checkpoint save spans, all four Megatron performance metrics,
  and all four resource gauges from every world rank. Native cycles additionally require
  complete `ckpt.save` rank sets for every session and snapshot.
- `otel_report.py` owns OTLP loading, validation, and metric derivation;
  `log_report.py` owns regex-based log extraction. `slurm_job.py` selects one source
  and applies the shared baseline comparison.

The functional ckptbench gate (`ci/scripts/ckpt_stress.py`) does not gate on perf.

## Notes

- **In-process runs** (`ckpttest` with `RunInprocServer`) put many ranks and daemons
  in one process, which shares one process-global OTel provider. Each save's
  `nvrx.rank` span attribute remains correct. Production daemons run in separate processes.
- The multi-node relay (`/api/telemetry/remote`) is exercised by multi-node native and
  NVRx CI runs; single-node and file-mode runs collect directly.
- Grafana dashboards over this OTLP output are a planned follow-on; point an
  OpenTelemetry Collector / Tempo / Prometheus at the JSONL or enable `otlp-http`.
