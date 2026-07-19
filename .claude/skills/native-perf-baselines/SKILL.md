---
name: native-perf-baselines
description: Reseed the native checkpoint CI performance baselines until the perf/ckptbench jobs go green, and diagnose run-to-run perf variance with per-rank NUMA analysis. Covers the GitLab pipeline access, the baseline-file mechanism, the iterate-by-sampling loop, and the per-rank socket/NUMA breakdown.
when_to_use: A native CI pipeline has yellow/exit-3 perf-drift warnings on gb200/h100/devbox ckptbench or pretrain-native:perf jobs and you want to reseed baselines and open an MR; the user asks to "fix the perf baselines", "reseed baselines", "make the perf jobs green", "update_baselines"; the user asks why a perf number swings run-to-run or wants a per-rank / NUMA breakdown of bw_mbps; mentions gitlab-master pipeline + perf baseline mismatch.
---

# Native Perf Baselines — reseed & NUMA diagnosis

Reseed the `ci/baselines/*.yaml` perf baselines so the
ckptbench / `pretrain-native:perf` jobs stop tripping the non-blocking exit-3
perf-drift check, then open a draft MR. Also: diagnose run-to-run variance by
breaking the aggregate `bw_mbps` into its per-rank samples and per-NUMA-socket
groups.

**Trigger prompt to remember:** _"Reseed the native perf baselines for pipeline
&lt;url&gt; — iterate to green, no widening tolerances, and do the per-rank
NUMA analysis."_ (URL optional; default to the latest pipeline on `ankurv/main`.)

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
3. **Run ONE pipeline at a time.** Always `pl_cancel` the prior pipeline on the
   branch _before_ pushing the next. Overlapping pipelines contend for storage
   IO on the shared runners and produce false slow-mode outliers.
4. Work on a branch off `origin/ankurv/main`; finish with a **draft** MR.

## Setup

```bash
export GITLAB_TOKEN_FILE=/path/to/your/gitlab_token   # or: export GITLAB_TOKEN=glpat-...
source .claude/skills/native-perf-baselines/perf_baseline_tools.sh   # pstatus/pwatch/perrank/numa/...
```

Token: set `GITLAB_TOKEN` (value) or `GITLAB_TOKEN_FILE` (path) before sourcing. Project `ankurv/megatron-lm` on
`gitlab-master.nvidia.com`. The project `ci_config_path` is
`ci/gitlab-ci-native.yml` (no `workflow:` rules), so **any push
to any branch triggers a full pipeline** running the auto (non-manual) jobs.

## The jobs & baseline mechanism

- 4 jobs typically drift: `gb200:ckptbench:1x4`, `gb200:pretrain-native:1x4:perf`,
  `h100:ckptbench:1x8`, `h100:pretrain-native:1x8:perf` (devbox usually fine).
  All are `allow_failure: exit_codes:[3]` → yellow, non-blocking. Deterministic
  jobs gate on md5, not perf; leave them unless they drift.
- Baseline file = `ci/baselines/<platform>_<job>[_<flavor>]_<config>.yaml`.
  Platform is the runner namespace (h100 runner uses `h100_*`, not `h200_*`).
  Metrics: `bw_mbps` (higher better), `wr_tnsr_us` (lower better; anti-correlated
  with bw), plus exact-match info metrics (`total_written_bytes`, `gpu_peak_bytes`).
- The `*:update_baselines` manual jobs just read the test job's `*_report.json`
  artifact and write the **2-significant-figure mean** (`run_job.py`
  `_round_for_baseline`), then commit `[skip ci]` + push. **Editing the YAML
  directly is equivalent and preferred** — full control, no skip-ci retrigger.

## The loop

1. Read the pipeline: `pstatus <pid>`. For each failing focus job,
   `jverdict <jobid>` to get measured-vs-baseline (`Δ`, the n, and the values).
2. `git checkout -b ankurv/native-ci-perf-baselines origin/ankurv/main` (fetch
   first). Confirm `h100_*` baseline files are present.
3. Edit the drifting metrics to the most-likely mean. Anti-correlated bw/wr:
   center **both** so either extreme of a clean run stays within ±5%. Verify
   every clean historical sample lands inside ±5% of the new center.
4. **`git add` the files**, commit, push. (GOTCHA: `git reset --soft`/`commit`
   without `git add` commits the stale index, not your Edit-tool changes — the
   pipeline then runs the wrong baselines. Always `git add` first; verify with
   `git show HEAD:<file>`.)
5. `pl_cancel` any prior running pipeline, then push. Find the new pipeline
   (`pl_latest <branch>`), grab focus job ids from `pstatus`, run `pwatch` in
   the background.
6. Each pipeline = one clean sample per job. Need more samples cheaply? Retry a
   single job (`_post /jobs/<id>/retry` → new id) — sequential retries on one
   runner are clean (no cross-runner contention). Accumulate samples; recenter.
7. When all focus jobs are green on a clean run, squash to one commit and open a
   **draft** MR (`gh pr create --draft` or GitLab UI) describing the reseed and
   any residual variance.

## Per-rank + NUMA analysis (the variance diagnosis)

The aggregate `bw_mbps` hides structure. Each rank does 4 saves; the report
drops save #1 (warmup), so n = (gpus) × 3. Break it down:

```bash
perrank <jobid> <label>   # socket0(r0-3) vs socket1(r4-7) means, gap%, overall
numa <jobid>              # per-rank NUMA node + shm channel (client logs at queue open)
```

Run `perrank` across several runs and a stable control job (gb200 perf). What
to look for and conclude:

- **Dual modality:** per-rank bw partitions along the GPU/socket boundary. On an
  8-GPU 2-NUMA h100 node, ranks 0-3 = NUMA#0, ranks 4-7 = NUMA#1 (confirm with
  `numa`). If one socket is _consistently_ faster across every run (never
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

## Useful API calls

`GET /jobs/<id>` (name/status/failure_reason/allow_failure) ·
`GET /jobs/<id>/trace` (grep, huge) · `POST /jobs/<id>/retry` ·
`POST /pipelines/<id>/cancel` · `GET /pipelines?ref=<branch>&order_by=id&sort=desc`.
GitLab's `!reference` tags break `yaml.safe_load`; lint via
`POST /ci/lint` instead.
