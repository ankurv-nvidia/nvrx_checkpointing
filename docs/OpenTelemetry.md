# OpenTelemetry performance telemetry

The checkpoint daemon emits OpenTelemetry **spans** (and, opt-in, **metrics**) for
every checkpoint save, forwards its operational **logs** as OTLP log records, and
exports all of them over **OTLP/HTTP**. It also accepts OTLP/HTTP protobuf traces,
metrics, and logs and can route them into one **OTLP-JSONL** file (one OTLP JSON
object per line).

Telemetry is **off by default**.

## Enabling

Set on every rank (the daemon inherits its own env):

| Variable | Default | Meaning |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | A nonempty value enables native telemetry and exports it to `<endpoint>/v1/{traces,metrics,logs}`. The standard `OTEL_EXPORTER_OTLP_*` variables (protocol, per-signal endpoints, headers, timeout, compression) configure the exporter. Endpoints are plain HTTP. |
| `NATIVE_CKPT_OTEL_FILE` | — | With an endpoint, the daemons route the OTLP they receive: the coordinator writes this file and the others relay to it. Without an endpoint, a nonempty value enables native telemetry and each process writes its own telemetry to this path, with metrics and logs in `.metrics` and `.logs` siblings. `%h` and `%p` expand to the host and process ID, as for `NATIVE_CKPT_SERVER_LOG`. A relative path is resolved beneath the server-log directory. |
| `NATIVE_CKPT_OTEL_COLLECTOR` | — | With both an endpoint and a file, select the route before checkpoint topology exists: `file` writes `NATIVE_CKPT_OTEL_FILE`; any other value relays to that host or `host:port`. |
| `NATIVE_CKPT_OTEL_SERVICE` | — | Override `service.name`. |
| `NATIVE_CKPT_OTEL_METRICS` | `0` | `1` also emits `bw_mbps`, `wr_tnsr_us`, and `total_written_bytes` OTLP histograms. Off by default because the spans already carry every figure; enable for Grafana. |
| `NATIVE_CKPT_OTEL_LOGS` | `1` | `0` suppresses just the log-record signal (spans/metrics unaffected). |
| `NATIVE_CKPT_MEMORY_SAMPLE_INTERVAL_SECONDS` | `0` | Positive interval for periodic `process.memory` events. Native span begin/end events emit standalone memory events without periodic sampling. |
| `NATIVE_CKPT_CI_PHASE` | — | Sets the `nvrx.ci_phase` resource attribute so the analyzer can scope to a phase. |
| `NATIVE_CKPT_LOG_LEVEL` | `info` | Daemon spdlog threshold (`trace`/`debug`/`info`/`warn`/`error`). |

## Architecture

```text
external collector:  daemon SDK, Megatron SDK ── OTLP/HTTP ──► collector ──► JSONL or backend

daemon routing:      daemon SDK, Megatron SDK ── OTLP/HTTP ──► node-local daemon ─┬─ coordinator: FileCollector ── JSONL
                                                                                  └─ others: RelayCollector ── unchanged request ──► coordinator
```

- A single TU (`src/checkpoint/logtel.cpp`) contains the OpenTelemetry SDK, spdlog,
  and daemon telemetry collection behind the lightweight `src/telemetry.h` and
  `src/logging.h` facades.
- `OTEL_EXPORTER_OTLP_ENDPOINT` selects the topology. An external
  OpenTelemetry Collector aggregates every process's telemetry and leaves the daemons'
  routing idle. The node-local daemon's address (`http://127.0.0.1:4100`) together with
  `NATIVE_CKPT_OTEL_FILE` makes the daemons the collector; each daemon's own export
  reaches its own receiver over loopback.
- Daemon routing accepts only protobuf, so it requires
  `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`.
- The daemon holds one active `FileCollector` or `RelayCollector` directly. Checkpoint
  session topology normally selects the route: the coordinator writes the file, and the
  other daemons relay to it. `NATIVE_CKPT_OTEL_COLLECTOR` selects the route at startup
  when the daemon is used only as a collector. OTLP requests received before a route
  exists are buffered up to 64 MiB and drained when one becomes available.
- The OTLP/HTTP receiver accepts binary protobuf on `/v1/traces`, `/v1/metrics`, and
  `/v1/logs`, including gzip request bodies. Requests and decompressed payloads are
  limited to 64 MiB. Success and failure responses use the corresponding OTLP protobuf
  response format.
- A relay forwards the OTLP path, content encoding, and body unchanged, then returns
  the downstream response. It does not decode, delimit, or retry the request; the
  originating OTLP exporter owns retry policy. Multi-hop relay remains valid.
- A terminal collector decodes protobuf and uses OpenTelemetry's OTLP file serializer
  to produce canonical JSON. The file collector alone appends and flushes the JSONL
  newline.

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
  remain human-readable diagnostics.
- Resource attributes: `service.name`, `host.name`, `process.pid`,
  `nvrx.process.role`, `nvrx.ci_phase`, and, when the launcher supplies one, `nemo.run.id`.
- Python checkpoint boundary spans use the application's process-global OpenTelemetry
  provider. They are no-ops when `MEGATRON_OTEL_ENABLED` is not `1` or the
  OpenTelemetry Python API is unavailable; otherwise the training job's provider owns
  sampling, OTLP export, resource attributes, and shutdown. The native provider
  publishes each asynchronous save inside an `nvrx.checkpoint.finalize` span that
  carries `megatron.iteration` and `nvrx.snapshot_id`.

## Notes

- **In-process runs** (`ckpttest` with `RunInprocServer`) put many ranks and daemons
  in one process, which shares one process-global OTel provider. Each save's
  `nvrx.rank` span attribute remains correct. Production daemons run in separate processes.
- Grafana dashboards over this OTLP output are a planned follow-on; an OpenTelemetry
  Collector at the endpoint can forward it to Tempo or Prometheus.
