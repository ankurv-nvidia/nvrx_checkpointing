#!/usr/bin/env bash
# Source this file before repository Python entry points in every JET step.
set -Eeuo pipefail

runtime_archive=${JET_RUNTIME_ARCHIVE:?}
local_runtime_root=${JET_LOCAL_RUNTIME_ROOT:?}
[[ -f "${runtime_archive}" ]] || {
    echo "JET native runtime archive is missing: ${runtime_archive}" >&2
    return 2 2>/dev/null || exit 2
}

archive_state=$(stat -c '%s:%Y' "${runtime_archive}")
state_marker="${local_runtime_root}/.archive-state"
mkdir -p "$(dirname "${local_runtime_root}")"
exec 8>"${local_runtime_root}.lock"
flock 8

if [[ ! -f "${state_marker}" ]] || [[ $(<"${state_marker}") != "${archive_state}" ]]; then
    staging="${local_runtime_root}.extracting-$$"
    rm -rf "${staging}"
    mkdir -p "${staging}"
    echo "[jet] restoring native runtime ${runtime_archive} on $(hostname)"
    tar -xf "${runtime_archive}" -C "${staging}"
    printf '%s\n' "${archive_state}" > "${staging}/.archive-state"
    rm -rf "${local_runtime_root}"
    mv "${staging}" "${local_runtime_root}"
fi

flock -u 8
exec 8>&-
export NVRX_VENV="${local_runtime_root}/venv"
export VIRTUAL_ENV="${NVRX_VENV}"
export PATH="${NVRX_VENV}/bin:${PATH}"
source "${NVRX_VENV}/bin/activate"
hash -r
