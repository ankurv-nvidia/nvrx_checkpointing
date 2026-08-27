#!/usr/bin/env bash
set -Eeou pipefail

COMMIT_HASH=${MEGATRON_COMMIT:-aac9b2db7f0d203940f6c0a7f2852a094c18c12e}
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

MEGATRON_REPO_URL=${MEGATRON_REPO_URL:-https://github.com/NVIDIA/Megatron-LM.git}
echo "MEGATRON_REPO_URL=${MEGATRON_REPO_URL}" >&2

REPO_ROOT=${SCRIPT_DIR}/..
MEGATRON_ROOT=${1:-${MEGATRON_ROOT:-${REPO_ROOT}/../nvrx-megatron-lm}}

# HEAD preserves the existing checkout and local modifications without repository mutation.
if [[ "${COMMIT_HASH}" == HEAD ]]; then
    if [[ ! -d ${MEGATRON_ROOT} ]] || ! git -C "${MEGATRON_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "MEGATRON_COMMIT=HEAD requires an existing Git checkout at MEGATRON_ROOT=${MEGATRON_ROOT}" >&2
        exit 1
    fi
    realpath "${MEGATRON_ROOT}"
    exit 0
fi

function git_retry {
    local attempt=1
    while ! "$@"; do
        if (( attempt >= 3 )); then
            return 1
        fi
        echo "git command failed (attempt ${attempt}/3); retrying" >&2
        sleep $((attempt * 2))
        ((attempt += 1))
    done
}

function clone_megatron {
    local staging="${MEGATRON_ROOT}.cloning-$$"
    rm -rf -- "${staging}"
    if ! git_retry git -c protocol.version=1 clone --quiet "${MEGATRON_REPO_URL}" "${staging}"; then
        rm -rf -- "${staging}"
        return 1
    fi
    mv -- "${staging}" "${MEGATRON_ROOT}"
}

# Serialize concurrent srun ranks: only the lock holder clones/checks out the pinned
# commit; the others block here and then observe the ready clone.
mkdir -p "$(dirname "${MEGATRON_ROOT}")"
exec 9>"${MEGATRON_ROOT}.lock"
flock 9

if [[ -e ${MEGATRON_ROOT} ]] && ! git -C "${MEGATRON_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "MEGATRON_ROOT exists but is not a Git checkout: ${MEGATRON_ROOT}" >&2
    exit 1
fi
if [[ ! -d ${MEGATRON_ROOT} ]]; then
    clone_megatron
fi

# Existing CI caches may have been cloned from the historical private fork.
# Always point the retained checkout at the public repository before fetching.
git -C "${MEGATRON_ROOT}" remote set-url origin "${MEGATRON_REPO_URL}"

if [[ "${COMMIT_HASH}" != $(git -C "${MEGATRON_ROOT}" rev-parse HEAD) ]]; then
    git -C "${MEGATRON_ROOT}" checkout --quiet -- .
    git -C "${MEGATRON_ROOT}" clean --quiet -f -d -x .
    git_retry git -C "${MEGATRON_ROOT}" -c protocol.version=1 \
        -c remote.neworigin.url="${MEGATRON_REPO_URL}" \
        fetch neworigin --quiet "${COMMIT_HASH}"
    git -C "${MEGATRON_ROOT}" checkout --quiet "${COMMIT_HASH}"
fi

if ! git -C "${MEGATRON_ROOT}" diff --quiet "${COMMIT_HASH}" -- .; then
    git -C "${MEGATRON_ROOT}" checkout --quiet "${COMMIT_HASH}" -- .
fi

echo "${MEGATRON_ROOT}"
