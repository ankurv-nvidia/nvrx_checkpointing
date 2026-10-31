---
name: native-perf-baselines
description: Reseed the native checkpoint CI performance baselines until the JET perf workloads stop missing them, and diagnose run-to-run perf variance with per-rank NUMA analysis. Covers where JET runs report their metrics, the baseline-file mechanism, the iterate-by-sampling loop, and the per-rank socket/NUMA breakdown.
when_to_use: A GitHub CI or JET form run has a failing perf report or missed-baseline warnings on gb200/gb300/h100/b200 perf or DETR workloads and you want to reseed baselines; the user asks to "fix the perf baselines", "reseed baselines", "make the perf jobs green"; the user asks why a perf number swings run-to-run or wants a per-rank / NUMA breakdown of bw_mbps.
---

# Native Perf Baselines — reseed & NUMA diagnosis

Reseed the `ci/baselines/*.yaml` perf baselines so the JET workloads stop
missing them. Also: diagnose run-to-run variance by breaking the aggregate `bw_mbps`
into its per-rank samples and per-NUMA-socket groups.

**Trigger prompt to remember:** _"Reseed the native perf baselines for run
&lt;url&gt; — iterate to green, no widening tolerances, and do the per-rank
NUMA analysis."_ (URL optional; default to the latest pull-request CI run.)

## Hard rules (from the user)

1. **Do NOT widen tolerances.** Every job must pass at the default **±5%**.
   The baseline YAML supports a `tolerances:` map and `_compare_to_baseline`
   honors it — **do not use it.** Recenter the baseline instead.
2. **Center on the "most likely" value** from _clean_ runs. Don't chase
   outliers; an occasional outlier run tripping the non-blocking check is
   acceptable. Don't infinite-loop.
   - **When a metric's spread genuinely can't fit ±5%** (e.g. h100 perf's
     residual per-socket NUMA swing), bias the baseline to fail _high, never
     low_: set it no more than **5% above the minimum** clean sample
     (`baseline ≈ min × 1.05`) so the slowest run still passes at the lower
     edge and only a faster-than-expected run trips the non-blocking check.
     The check is a symmetric ±5% band on the raw value, so this applies
     per metric on its raw number — `bw_mbps` (anchor to its min) and
     `wr_tnsr_us` (anchor to its min) are anti-correlated, so prefer the
     metric whose low-side false alarm you most want to avoid rather than
     min-anchoring both at once.
3. **Run ONE run at a time per cluster.** Wait for the current run to finish
   before starting another. Overlapping runs contend for the cluster's shared
   storage and produce false slow-mode outliers. Never cancel a run unless the
   user explicitly requests it.
4. Fetch the remote, then create a separate worktree and branch from the latest
   remote `main`.

## Where the numbers are

- Every JET workload job uploads `report-<workload>`, a short report with each
  metric against its baseline, the options that differ from the recorded run,
  and Megatron's `save-checkpoint` timings. The run's perf report job collects
  them all and fails when any workload missed a baseline.
- The `<workload>` artifact holds the full results: `ci/ci-results/*_report.json`
  (every metric, its baseline and tolerance), the baseline candidate
  `ci/ci-results/baselines/<stem>.yaml`, recorded-argument candidates under
  `ci/ci-runs/**/logs/baselines/`, and the daemon logs
  `ci/ci-results/checkpoint_server.*.log` with one
  `[native_ckpt:svr][...][PERF] Rank=<n> Finished bw_mbps=...` line per rank and save.
- Baseline file = `ci/baselines/<platform>_<job>[_<flavor>]_<config>.yaml`.
  Metrics: `bw_mbps` (higher better), `wr_tnsr_us` (lower better; anti-correlated
  with bw), plus exact-match info metrics (`total_written_bytes`, `gpu_peak_bytes`).
  Adopting a candidate is copying it over the baseline; editing the YAML by hand
  is equivalent.

## The loop

1. Open the run's perf report job summary; for each workload that missed, read
   its `report-<workload>` artifact for measured-vs-baseline and the samples.
2. Fetch the remote, then create a separate worktree and branch from the latest
   remote `main`.
3. Edit the drifting metrics to the most-likely mean. Anti-correlated bw/wr:
   center **both** so either extreme of a clean run stays within ±5%. Verify
   every clean historical sample lands inside ±5% of the new center.
4. **`git add` the files**, create a new commit, and push. Never amend, rebase,
   squash, force-push, or otherwise overwrite commits. Verify the committed
   files with `git show HEAD:<file>`.
5. Wait for the current run to finish before starting another.
6. Each run = one sample per workload. Need more samples cheaply? Dispatch the
   platform's `JET <platform>` form with only that workload ticked; sequential
   single-workload runs are clean (no contention from the rest of the suite).
   Accumulate samples; recenter.

## Per-rank + NUMA analysis (the variance diagnosis)

The aggregate `bw_mbps` hides structure. Each rank does several saves, and the
first is a warmup. Break the daemon log's per-rank `PERF` lines down by rank
and by save, and compare the ranks on each NUMA socket.

Compare several runs against a stable control workload (gb200 perf). What
to look for and conclude:

- **Dual modality:** per-rank bw partitions along the GPU/socket boundary. On an
  8-GPU 2-NUMA h100 node, ranks 0-3 = NUMA#0, ranks 4-7 = NUMA#1 (confirm from
  the client logs at queue open). If one socket is _consistently_ faster across every run (never
  reversed), that's a systematic asymmetry, not jitter. The aggregate "bimodal"
  mean = a fast-socket + slow-socket population whose mix/amplitude varies.
- **NUMA root cause (confirmed for h100 perf):** The client channel index is
  `(rank % ranksPerPeer) % MaxRanksPerNode`, so 8 ranks share only **4** D2H shm
  staging queues — each pinned buffer (`CudaRegisterHostBuffer` on the queue's
  shm region, `checkpoint.cpp` `InitMsgQueue`) is shared by one NUMA#0 rank and
  one NUMA#1 rank. The server (`server.cpp`, `MsgQueue::Create` loop) first-
  touches the buffers on its own socket, so the 4 ranks on the far socket pay a
  cross-socket D2H penalty (~+17–72% gap) every run. gb200 (4 GPUs/node =
  `MaxRanksPerNode`, single coherent NUMA domain) shows no split and is stable
  (<3%) on the _same code_ — the control that isolates it to host NUMA topology.
- **Confounder to call out:** the dropped warmup save and occasional save spikes
  to 40000+ MB/s look like page-cache (non-durable) writes — separate from NUMA.
- **Fix direction (follow-up, not the baseline task):** set the perf build's
  `MaxRanksPerNode` to the real gpus-per-node (8 for h100) so each rank gets its
  own queue, and/or NUMA-bind/first-touch each queue's shm buffer on its rank's
  local node. Note `D2HBufferSize = TotalShmSize / D2HBufferCount /
MaxRanksPerNode`, so bumping ranks shrinks per-buffer size unless TotalShmSize
  grows.
