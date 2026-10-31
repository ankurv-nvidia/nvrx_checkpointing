#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(realpath "$(dirname "$0")/../../..")
runtime_root=${JET_RUNTIME_ROOT:?}
local_runtime_root=${JET_LOCAL_RUNTIME_ROOT:?}
runtime_archive=${JET_RUNTIME_ARCHIVE:?}
runtime_key=${JET_RUNTIME_KEY:?}
architecture=$(uname -m)
compute_capability=${NVRX_COMPUTE_CAPABILITY:?}
wheel_mode=${JET_WHEEL_MODE:?}
prepared_marker="${runtime_root}/.prepared"
local_prepared_marker="${local_runtime_root}/.prepared"
expected_state="${runtime_key} ${architecture} ${compute_capability} ${wheel_mode} local-bundle-v11"

[[ "${wheel_mode}" == "prebuilt" ]] || {
    echo "JET_WHEEL_MODE must be prebuilt, found ${wheel_mode}" >&2
    exit 2
}

# A configured cache must be the launcher's mount, never a directory inside the container.
if [[ -n "${JET_RUNTIME_CACHE_DIR:-}" ]] && ! mountpoint -q "${JET_RUNTIME_CACHE_DIR}"; then
    echo "JET runtime cache ${JET_RUNTIME_CACHE_DIR} is not mounted" >&2
    exit 2
fi

mkdir -p "${runtime_root}" "$(dirname "${local_runtime_root}")"
exec 9>"${runtime_root}.lock"
flock 9

if [[ -f "${prepared_marker}" ]] \
    && [[ $(<"${prepared_marker}") == "${expected_state}" ]] \
    && [[ -f "${runtime_archive}" ]]; then
    echo "[jet] reusing native runtime ${runtime_root}"
    exit 0
fi

# The node-local lock protects the node's runtime cache.
exec 8>"${local_runtime_root}.lock"
flock 8

archive_local_runtime() {
    local archive_state archive_tmp
    archive_tmp="${runtime_archive}.writing-$$"
    rm -f "${archive_tmp}"
    trap 'rm -f "${archive_tmp}"' EXIT
    echo "[jet] archiving native runtime ${runtime_archive}"
    tar -C "${local_runtime_root}" -cf "${archive_tmp}" venv
    mv -f "${archive_tmp}" "${runtime_archive}"
    trap - EXIT
    archive_state=$(stat -c '%s:%Y' "${runtime_archive}")
    printf '%s\n' "${archive_state}" > "${local_runtime_root}/.archive-state"
    printf '%s\n' "${expected_state}" > "${local_prepared_marker}"
    printf '%s\n' "${expected_state}" > "${prepared_marker}"
}

local_cache_usable=true
[[ -f "${local_prepared_marker}" ]] || local_cache_usable=false
[[ "${local_cache_usable}" == false ]] \
    || [[ $(<"${local_prepared_marker}") == "${expected_state}" ]] \
    || local_cache_usable=false
[[ -x "${local_runtime_root}/venv/bin/python" ]] || local_cache_usable=false
if [[ "${local_cache_usable}" == true ]]; then
    echo "[jet] reusing node-local native runtime ${local_runtime_root}"
    archive_local_runtime
    echo "[jet] prepared native runtime ${runtime_root}"
    exit 0
fi

rm -f "${prepared_marker}"
rm -rf "${local_runtime_root}"
mkdir -p "${local_runtime_root}"
# Preserve the NGC image's NumPy version in the system-site-packages venv.
base_numpy_version=$(python3 -c 'import numpy; print(numpy.__version__)')
export NVRX_BASE_NUMPY_VERSION="${base_numpy_version}"
export NVRX_VENV="${local_runtime_root}/venv"
export VIRTUAL_ENV="${NVRX_VENV}"
export PATH="${NVRX_VENV}/bin:${PATH}"

# Build the runtime once on node-local storage.
echo "[jet] verifying pinned Megatron checkout ${MEGATRON_ROOT:?}"
bash "${repo_root}/scripts/prepare.sh" megatron_root "${MEGATRON_ROOT:?}"
echo "[jet] configuring image-owned Megatron dependency sources"
bash "${repo_root}/ci/jet/image/install_megatron_source_deps.sh" --configure-only
echo "[jet] preparing native Megatron dependencies"
bash "${repo_root}/ci/jet/image/prepare_megatron_runtime.sh"
git -C "${MEGATRON_ROOT}" checkout --quiet -- pyproject.toml
source "${NVRX_VENV}/bin/activate"
uv pip install --python "${NVRX_VENV}/bin/python" --no-cache \
    "numpy==${base_numpy_version}"
python -c 'import numpy, os; expected = os.environ["NVRX_BASE_NUMPY_VERSION"]; assert numpy.__version__ == expected, (numpy.__version__, expected)'
echo "[jet] retained base-image NumPy ${base_numpy_version}"
# Install the pinned NVRX package without changing the image's Torch stack.
echo "[jet] installing nvidia-resiliency-ext ${NVRX_RESILIENCY_EXT_VERSION:?}"
base_torch_version=$(python -c 'import torch; print(torch.__version__)')
uv pip install --python "${NVRX_VENV}/bin/python" --no-cache --no-deps \
    "nvidia-resiliency-ext==${NVRX_RESILIENCY_EXT_VERSION}"
python -c 'import importlib.metadata as m, os; expected = os.environ["NVRX_RESILIENCY_EXT_VERSION"]; actual = m.version("nvidia-resiliency-ext"); assert actual == expected, (actual, expected); print(f"Using nvidia-resiliency-ext {actual}")'
actual_torch_version=$(python -c 'import torch; print(torch.__version__)')
if [[ "${actual_torch_version}" != "${base_torch_version}" ]]; then
    echo "nvidia-resiliency-ext changed Torch from ${base_torch_version} to ${actual_torch_version}" >&2
    exit 2
fi
echo "[jet] retained base-image Torch ${actual_torch_version}"
echo "[jet] retaining exact-SHA prebuilt nvrx wheel path"

archive_local_runtime
echo "[jet] prepared native runtime ${runtime_root}"
