#!/usr/bin/env bash
set -Eeoux pipefail

COMMIT_HASH=773f875607507d4e917502dd99aaf41421312c5d
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

cloneurl=$(git -C ${SCRIPT_DIR} config --get remote.origin.url)
if [[ -n "${MEGATRON_CI_TOKEN:-}" ]]; then
    # Dedicated access token (PAT / project / group) scoped to megatron-lm.
    # oauth2 works as the username for any token type.
    MEGATRON_REPO_URL=https://oauth2:${MEGATRON_CI_TOKEN}@gitlab-master.nvidia.com/ankurv/megatron-lm.git
elif [[ -n "${CI_JOB_TOKEN:-}" ]]; then
    MEGATRON_REPO_URL=https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab-master.nvidia.com/ankurv/megatron-lm.git
elif [[ ${cloneurl} == ssh://git@* ]]; then
    # if clone url is ssh then use ssh url
    MEGATRON_REPO_URL=ssh://git@gitlab-master.nvidia.com:12051/ankurv/megatron-lm.git
else
    MEGATRON_REPO_URL=https://gitlab-master.nvidia.com/ankurv/megatron-lm.git
fi
# print cloneurl to stderr
echo "MEGATRON_REPO_URL=${MEGATRON_REPO_URL}" >&2
echo "cloneurl=${cloneurl}" >&2

REPO_ROOT=${SCRIPT_DIR}/..
MEGATRON_ROOT=${1:-${REPO_ROOT}/../nvrx-megatron-lm}
if [[ ! -d ${MEGATRON_ROOT} ]]; then
    git clone --quiet ${MEGATRON_REPO_URL} ${MEGATRON_ROOT}
fi

if [[ "${COMMIT_HASH}" == $(git -C ${MEGATRON_ROOT} rev-parse HEAD) ]]; then
    echo ${MEGATRON_ROOT}
    exit 0
fi

git -C ${MEGATRON_ROOT} checkout --quiet -- .
git -C ${MEGATRON_ROOT} clean --quiet -f -d -x . 
git -C ${MEGATRON_ROOT} -c remote.neworigin.url=${MEGATRON_REPO_URL} fetch neworigin --quiet ${COMMIT_HASH}
git -C ${MEGATRON_ROOT} checkout --quiet ${COMMIT_HASH}
echo ${MEGATRON_ROOT}