# OpenTelemetry performance telemetry

The checkpoint daemon emits OpenTelemetry **spans** (and, opt-in, **metrics**) for
every checkpoint save, and forwards its operational **logs** as OTLP log records.
It also accepts OTLP/HTTP protobuf traces, metrics, and logs. All terminal telemetry
is written as **OTLP-JSONL** (one OTLP JSON object per line).

Telemetry is **off by default**.

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
  remain human-readable diagnostics.
- Resource attributes: `service.name`, `host.name`, `process.pid`,
  `nvrx.process.role`, `nvrx.ci_phase`, and, when the launcher supplies one, `nemo.run.id`.
- Python checkpoint boundary spans use the application's process-global OpenTelemetry
  provider. They are no-ops when `MEGATRON_OTEL_ENABLED` is not `1` or the
  OpenTelemetry Python API is unavailable; otherwise the training job's provider owns
  sampling, OTLP export, resource attributes, and shutdown.

## Notes

- **In-process runs** (`ckpttest` with `RunInprocServer`) put many ranks and daemons
  in one process, which shares one process-global OTel provider. Each save's
  `nvrx.rank` span attribute remains correct. Production daemons run in separate processes.
- The multi-node relay (`/api/telemetry/remote`) carries the telemetry of multi-node
  runs; single-node and file-mode runs collect directly.
- Grafana dashboards over this OTLP output are a planned follow-on; point an
  OpenTelemetry Collector / Tempo / Prometheus at the JSONL or enable `otlp-http`.
