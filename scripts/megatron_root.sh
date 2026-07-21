#!/usr/bin/env bash
set -Eeou pipefail

COMMIT_HASH=1aa880d0ea1dfddef567785ebc9a384f2a600d18
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

MEGATRON_REPO_URL=${MEGATRON_REPO_URL:-https://github.com/NVIDIA/Megatron-LM.git}
echo "MEGATRON_REPO_URL=${MEGATRON_REPO_URL}" >&2

REPO_ROOT=${SCRIPT_DIR}/..
MEGATRON_ROOT=${1:-${REPO_ROOT}/../nvrx-megatron-lm}

# Serialize concurrent srun ranks: only the lock holder clones/checks out the pinned
# commit; the others block here and then observe the ready clone.
mkdir -p "$(dirname ${MEGATRON_ROOT})"
exec 9>"${MEGATRON_ROOT}.lock"
flock 9

if [[ ! -d ${MEGATRON_ROOT} ]]; then
    git clone --quiet ${MEGATRON_REPO_URL} ${MEGATRON_ROOT}
fi

# Existing CI caches may have been cloned from the historical private fork.
# Always point the retained checkout at the public repository before fetching.
git -C ${MEGATRON_ROOT} remote set-url origin ${MEGATRON_REPO_URL}

if [[ "${COMMIT_HASH}" != $(git -C ${MEGATRON_ROOT} rev-parse HEAD) ]]; then
    git -C ${MEGATRON_ROOT} checkout --quiet -- .
    git -C ${MEGATRON_ROOT} clean --quiet -f -d -x .
    git -C ${MEGATRON_ROOT} -c remote.neworigin.url=${MEGATRON_REPO_URL} fetch neworigin --quiet ${COMMIT_HASH}
    git -C ${MEGATRON_ROOT} checkout --quiet ${COMMIT_HASH}
fi

echo ${MEGATRON_ROOT}
