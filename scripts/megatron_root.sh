#!/usr/bin/env bash
set -Eeou pipefail

COMMIT_HASH=${MEGATRON_COMMIT:-354b5d2d06ebf2c529688db69ce5773932226381}
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

MEGATRON_REPO_URL=${MEGATRON_REPO_URL:-https://github.com/NVIDIA/Megatron-LM.git}
echo "MEGATRON_REPO_URL=${MEGATRON_REPO_URL}" >&2

REPO_ROOT=${SCRIPT_DIR}/..
MEGATRON_ROOT=${1:-${MEGATRON_ROOT:-${REPO_ROOT}/../nvrx-megatron-lm}}

# Serialize concurrent srun ranks: only the lock holder clones/checks out the pinned
# commit; the others block here and then observe the ready clone.
mkdir -p "$(dirname "${MEGATRON_ROOT}")"
exec 9>"${MEGATRON_ROOT}.lock"
flock 9

if [[ ! -d ${MEGATRON_ROOT} ]]; then
    git -c protocol.version=1 clone --quiet "${MEGATRON_REPO_URL}" "${MEGATRON_ROOT}"
fi

# Existing CI caches may have been cloned from the historical private fork.
# Always point the retained checkout at the public repository before fetching.
git -C "${MEGATRON_ROOT}" remote set-url origin "${MEGATRON_REPO_URL}"

if [[ "${COMMIT_HASH}" != $(git -C "${MEGATRON_ROOT}" rev-parse HEAD) ]]; then
    git -C "${MEGATRON_ROOT}" checkout --quiet -- .
    git -C "${MEGATRON_ROOT}" clean --quiet -f -d -x .
    git -C "${MEGATRON_ROOT}" -c protocol.version=1 -c remote.neworigin.url="${MEGATRON_REPO_URL}" fetch neworigin --quiet "${COMMIT_HASH}"
    git -C "${MEGATRON_ROOT}" checkout --quiet "${COMMIT_HASH}"
fi

if ! git -C "${MEGATRON_ROOT}" diff --quiet "${COMMIT_HASH}" -- .; then
    git -C "${MEGATRON_ROOT}" checkout --quiet "${COMMIT_HASH}" -- .
fi

echo "${MEGATRON_ROOT}"
