#!/usr/bin/env bash
set -Eeuo pipefail

deepep_commit=10d4dd7377d5bce900fbb4b80cce863764892b95
nccl_commit=1e0c869c39bb33f1034cb9920bd2a8a8406f04a3
nccl_sha256=3994fd525c52c3d717bc36edd142796a1a0f12daa25e3e4a355f47b07359770e
deepep_root=${NVRX_DEEPEP_ROOT:-/opt/deepep-src}
deepep_url=https://github.com/deepseek-ai/DeepEP.git
nccl_url="https://github.com/NVIDIA/nccl/archive/${nccl_commit}.tar.gz"
nccl_archive=$(mktemp)

cleanup() {
    rm -f "${nccl_archive}"
}
trap cleanup EXIT

curl --fail --location --silent --show-error "${nccl_url}" --output "${nccl_archive}"
echo "${nccl_sha256}  ${nccl_archive}" | sha256sum --check -

rm -rf "${deepep_root}"
mkdir -p "${deepep_root}"
git -C "${deepep_root}" init
git -C "${deepep_root}" remote add origin "${deepep_url}"
git -C "${deepep_root}" fetch --depth 1 origin "${deepep_commit}"
git -C "${deepep_root}" checkout --detach FETCH_HEAD
[[ $(git -C "${deepep_root}" rev-parse HEAD) == "${deepep_commit}" ]]

# Preserve DeepEP metadata and its pinned NCCL gitlink source.
mkdir -p "${deepep_root}/third-party/nccl"
tar \
    --extract \
    --gzip \
    --file "${nccl_archive}" \
    --directory "${deepep_root}/third-party/nccl" \
    --strip-components=1

test -f "${deepep_root}/setup.py"
test -f "${deepep_root}/third-party/nccl/Makefile"
test -d "${deepep_root}/third-party/nccl/src/transport/net_ib/gdaki/doca-gpunetio"
echo "Prepared pinned DeepEP ${deepep_commit} with NCCL ${nccl_commit} at ${deepep_root}"
