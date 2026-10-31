#!/usr/bin/env bash
# JET steps for the self-hosted `jet` runners. Each job submits one JET image
# build or workload, streams its log, and turns the verdict into the job status.
# The job environment selects the platform, the workload, and the built image.
#
#   jet.sh select | setup | image | workload | cancel
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

venv=${RUNNER_TEMP:-${repo_root}/build-ci}/jet-client
python=${venv}/bin/python
soft_failure=3

# Lists the workloads ticked on a manual run's form as a JSON array.
select_workloads() {
    python3 - <<'PY'
import json
import os

ticked = [name for name, chosen in json.loads(os.environ["JET_INPUTS"]).items() if chosen is True]
if not ticked:
    raise SystemExit("::error::Tick at least one workload.")
with open(os.environ["GITHUB_OUTPUT"], "a") as output:
    output.write(f"workloads={json.dumps(ticked)}\n")
with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
    summary.write("### Selected workloads\n\n" + "".join(f"- `{name}`\n" for name in ticked))
print("\n".join(ticked))
PY
}

setup() {
    python3 -m venv "${venv}"
    "${venv}/bin/pip" install --quiet --no-cache-dir --retries 10 PyYAML==6.0.2 jet-client==4.58.4 \
        --index-url "${JET_CLIENT_INDEX_URL:?set the JET_CLIENT_INDEX_URL repository variable}"
}

image() {
    setup
    "${python}" ci/jet/submit_workload.py --platform "${JET_PLATFORM:?}" --image
    sed -n "s/^JET_IMAGE_TAG_${JET_PLATFORM^^}=/tag=/p" jet-image.env >> "${GITHUB_OUTPUT:?}"
}

workload() {
    local rc=0
    export "JET_IMAGE_TAG_${JET_PLATFORM^^}=${JET_IMAGE_TAG:?}"
    setup
    "${python}" ci/jet/submit_workload.py --platform "${JET_PLATFORM}" --workload "${JET_WORKLOAD:?}" || rc=$?
    python3 ci/github/perf_report.py workload "${JET_WORKLOAD}" jet-results "${rc}" jet-report \
        || echo "::warning::could not write the ${JET_WORKLOAD} report"
    # A missed baseline fails the run's perf report job instead of this job.
    if (( rc == soft_failure )); then
        rc=0
    fi
    return "${rc}"
}

# A follower that stopped early leaves its JET pipeline id behind.
cancel() {
    [[ -s .jet-pipeline ]] || return 0
    [[ -x ${python} ]] || setup
    "${python}" ci/jet/submit_workload.py --cancel "$(cat .jet-pipeline)"
}

case "${1:-}" in
    select) select_workloads ;;
    setup) setup ;;
    image) image ;;
    workload) workload ;;
    cancel) cancel ;;
    *) echo "usage: jet.sh select|setup|image|workload|cancel" >&2; exit 2 ;;
esac
