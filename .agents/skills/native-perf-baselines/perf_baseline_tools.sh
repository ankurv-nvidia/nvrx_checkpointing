#!/bin/bash
# Helper functions for the native-perf-baselines skill.
# Usage:  source skills/native-perf-baselines/perf_baseline_tools.sh
#
# GitLab PRIVATE-TOKEN, resolved in order:
#   1. $GITLAB_TOKEN       — the token value itself
#   2. $GITLAB_TOKEN_FILE  — path to a file containing it
# Set one of these before sourcing. Project: ankurv/megatron-lm on
# https://gitlab-master.nvidia.com
PROJ="ankurv%2Fmegatron-lm"
GLAB="https://gitlab-master.nvidia.com/api/v4/projects/$PROJ"
_tok() {
    if [ -n "${GITLAB_TOKEN:-}" ]; then
        printf '%s' "$GITLAB_TOKEN"
    elif [ -n "${GITLAB_TOKEN_FILE:-}" ] && [ -r "${GITLAB_TOKEN_FILE}" ]; then
        cat "$GITLAB_TOKEN_FILE"
    else
        echo "perf_baseline_tools: set GITLAB_TOKEN or GITLAB_TOKEN_FILE (GitLab PRIVATE-TOKEN)" >&2
        return 1
    fi
}
_api() { curl -s --header "PRIVATE-TOKEN: $(_tok)" "$GLAB$1"; }
_post() { curl -s --request POST --header "PRIVATE-TOKEN: $(_tok)" "$GLAB$1"; }

# Latest pipeline id for a branch:  pl_latest <branch>
pl_latest() { _api "/pipelines?ref=$1&order_by=id&sort=desc&per_page=1" | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['id'])"; }

# Job table for a pipeline (focus = ckptbench/perf, non-update/nvrx/16x marked *): pstatus <pid>
pstatus() {
  _api "/pipelines/$1/jobs?per_page=100" | python3 -c "
import json,sys
o={'failed':0,'running':1,'pending':2,'created':3,'manual':4,'success':5,'skipped':6,'canceled':7}
for j in sorted(json.load(sys.stdin),key=lambda x:(o.get(x['status'],9),x['name'])):
    nm=j['name']; f=('ckptbench' in nm or 'perf' in nm) and not any(k in nm for k in('update','nvrx','16x'))
    print(('*' if f else ' ')+f\"{j['id']:>9} {j['status']:>9} {nm}\")"
}

# Cancel a pipeline only when the user explicitly requests it.
pl_cancel() { _post "/pipelines/$1/cancel" >/dev/null && echo "cancelled $1"; }

# Perf PASS/FAIL summary for a job:  jverdict <jobid>
jverdict() { _api "/jobs/$1/trace" | grep -E "\[(PASS|FAIL)\]|Overall:|exit status" | tail -10; }

# Background watcher: exit when all listed job ids are terminal:  pwatch <pid> <jobid...>
pwatch() {
  local pid=$1; shift; local focus="$*"; local term='success|failed|canceled|skipped'
  for i in $(seq 1 120); do
    local data; data=$(_api "/pipelines/$pid/jobs?per_page=100")
    echo "[$i] $(date +%H:%M:%S)"
    echo "$data" | python3 -c "
import json,sys; j={str(x['id']):x for x in json.load(sys.stdin)}
for f in '''$focus'''.split():
    x=j.get(f);  print('  ',f,x['status'],x['name']) if x else None"
    echo "$data" | python3 -c "
import json,sys,re; j={str(x['id']):x['status'] for x in json.load(sys.stdin)}
print('DONE' if all(re.match(r'($term)\$',j.get(f,'?')) for f in '''$focus'''.split()) else 'WAIT')" | grep -q DONE && { echo DONE; break; }
    sleep 30
  done
}

# Per-rank bw analysis grouped by NUMA socket (drops save #1 warmup):  perrank <jobid> [label]
# Socket0 = ranks 0..3, Socket1 = ranks 4..7 on an 8-GPU 2-NUMA node.
perrank() {
  _api "/jobs/$1/trace" \
   | grep -oE "\[native_ckpt:cli\]\[[0-9]+\]\[INFO\]\[perf\] Finished bw_mbps=[0-9.]+" \
   | python3 -c "
import sys,re,statistics
from collections import defaultdict
d=defaultdict(list)
for ln in sys.stdin:
    m=re.search(r'cli\]\[(\d+)\]\[INFO\]\[perf\] Finished bw_mbps=([0-9.]+)',ln)
    if m: d[int(m.group(1))].append(float(m.group(2)))
meas={r:v[1:] for r,v in d.items()}              # drop warmup save
g=lambda rs:(statistics.mean([x for r in rs for x in meas.get(r,[])]) if any(r in meas for r in rs) else 0)
s0,s1,al=g(range(0,4)),g(range(4,8)),g(range(0,8))
print('${2:-job-$1}  ranks',sorted(d))
print(f'  socket0(r0-3)={s0:8.0f}  socket1(r4-7)={s1:8.0f}  gap={100*(s1-s0)/s0 if s0 else 0:+5.1f}%  overall={al:8.0f}')"
}

# Per-rank NUMA node + shm channel (the bindings the client logs at queue open):  numa <jobid>
numa() {
  _api "/jobs/$1/trace" \
   | grep -oE "\[native_ckpt:cli\]\[[0-9]+\]\[INFO\]\[perf\]\[NUMA#[0-9]+\] Using channel=[A-Za-z0-9_]+" \
   | python3 -c "
import sys,re
seen={}
for ln in sys.stdin:
    m=re.search(r'cli\]\[(\d+)\]\[INFO\]\[perf\]\[NUMA#(\d+)\] Using channel=(\S+)',ln)
    if m: seen[int(m.group(1))]=(m.group(2),m.group(3))
for r in sorted(seen): print(f'  rank {r}: NUMA#{seen[r][0]}  channel={seen[r][1]}')"
}
