#!/usr/bin/env bash
set -Eeuo pipefail

megatron_root=${MEGATRON_ROOT:?}
deps_root=${NVRX_MEGATRON_DEPS_ROOT:-/opt/nvrx-megatron-source-deps}
script_root=$(realpath "$(dirname "$0")")
transformer_engine_root="${deps_root}/TransformerEngine"
transformer_engine_url=https://github.com/NVIDIA/TransformerEngine.git
transformer_engine_commit=5e52befd5262c06289106338c308079d6adb391f
mamba_root="${deps_root}/mamba"
mamba_url=https://github.com/state-spaces/mamba.git
mamba_commit=0048fbf2e7b2f214dcbe703ea3dec2b9647595e1
causal_conv_root="${deps_root}/causal-conv1d"
causal_conv_url=https://github.com/Dao-AILab/causal-conv1d.git
causal_conv_commit=cd81f0413cad2fc1e6f17e785ac39f59aae690cd
fast_hadamard_root="${deps_root}/fast-hadamard-transform"
fast_hadamard_url=https://github.com/Dao-AILab/fast-hadamard-transform.git
fast_hadamard_commit=f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
emerging_optimizers_root="${deps_root}/Emerging-Optimizers"
emerging_optimizers_url=https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
emerging_optimizers_commit=b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16

configure_local_sources() {
    local pyproject="${megatron_root}/pyproject.toml"
    local git_source local_source mamba_git_source mamba_local_source
    local causal_conv_local_source fast_hadamard_git_source
    local fast_hadamard_local_source emerging_optimizers_git_source
    local emerging_optimizers_local_source

    git_source="transformer-engine = { git = \"${transformer_engine_url}\", rev = \"${transformer_engine_commit}\" }"
    local_source="transformer-engine = { path = \"${transformer_engine_root}\" }"
    if grep -Fqx "${git_source}" "${pyproject}"; then
        sed -i "s|^${git_source}$|${local_source}|" "${pyproject}"
    fi
    grep -Fqx "${local_source}" "${pyproject}"

    mamba_git_source="mamba-ssm = { git = \"${mamba_url}\", rev = \"${mamba_commit}\" }"
    mamba_local_source="mamba-ssm = { path = \"${mamba_root}\" }"
    if grep -Fqx "${mamba_git_source}" "${pyproject}"; then
        sed -i "s|^${mamba_git_source}$|${mamba_local_source}|" "${pyproject}"
    fi
    grep -Fqx "${mamba_local_source}" "${pyproject}"

    causal_conv_local_source="causal-conv1d = { path = \"${causal_conv_root}\" }"
    grep -Fqx "${causal_conv_local_source}" "${pyproject}" \
        || sed -i "/^\[tool\.uv\.sources\]$/a ${causal_conv_local_source}" "${pyproject}"
    grep -Fqx "${causal_conv_local_source}" "${pyproject}"

    fast_hadamard_git_source="fast-hadamard-transform = { git = \"${fast_hadamard_url}\", rev = \"${fast_hadamard_commit}\" }"
    fast_hadamard_local_source="fast-hadamard-transform = { path = \"${fast_hadamard_root}\" }"
    if grep -Fqx "${fast_hadamard_git_source}" "${pyproject}"; then
        sed -i "s|^${fast_hadamard_git_source}$|${fast_hadamard_local_source}|" "${pyproject}"
    fi
    grep -Fqx "${fast_hadamard_local_source}" "${pyproject}"

    emerging_optimizers_git_source='emerging_optimizers = { git = "https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git", rev = "v0.3.0" }'
    emerging_optimizers_local_source="emerging_optimizers = { path = \"${emerging_optimizers_root}\" }"
    if grep -Fqx "${emerging_optimizers_git_source}" "${pyproject}"; then
        sed -i "s|^${emerging_optimizers_git_source}$|${emerging_optimizers_local_source}|" "${pyproject}"
    fi
    grep -Fqx "${emerging_optimizers_local_source}" "${pyproject}"
}

if [[ "${1:-}" == "--configure-only" ]]; then
    [[ -d "${transformer_engine_root}/.git" ]]
    [[ -d "${mamba_root}/.git" ]]
    [[ -d "${causal_conv_root}/.git" ]]
    [[ -d "${fast_hadamard_root}/.git" ]]
    [[ -d "${emerging_optimizers_root}/.git" ]]
    configure_local_sources
    echo "Configured Megatron to use image-owned native dependency sources"
    exit 0
fi

rm -rf "${transformer_engine_root}"
mkdir -p "${transformer_engine_root}"
git -C "${transformer_engine_root}" init
git -C "${transformer_engine_root}" remote add origin "${transformer_engine_url}"
git -C "${transformer_engine_root}" fetch --depth 1 origin "${transformer_engine_commit}"
git -C "${transformer_engine_root}" checkout --detach FETCH_HEAD
git -C "${transformer_engine_root}" submodule update --init --recursive --depth 1

[[ $(git -C "${transformer_engine_root}" rev-parse HEAD) == "${transformer_engine_commit}" ]]
if git -C "${transformer_engine_root}" submodule status --recursive | grep -Eq '^[+-U]'; then
    echo "TransformerEngine submodules do not match the pinned root commit" >&2
    exit 1
fi

rm -rf "${mamba_root}"
mkdir -p "${mamba_root}"
git -C "${mamba_root}" init
git -C "${mamba_root}" remote add origin "${mamba_url}"
git -C "${mamba_root}" fetch --depth 1 origin "${mamba_commit}"
git -C "${mamba_root}" checkout --detach FETCH_HEAD
[[ $(git -C "${mamba_root}" rev-parse HEAD) == "${mamba_commit}" ]]

rm -rf "${causal_conv_root}"
mkdir -p "${causal_conv_root}"
git -C "${causal_conv_root}" init
git -C "${causal_conv_root}" remote add origin "${causal_conv_url}"
git -C "${causal_conv_root}" fetch --depth 1 origin "${causal_conv_commit}"
git -C "${causal_conv_root}" checkout --detach FETCH_HEAD
[[ $(git -C "${causal_conv_root}" rev-parse HEAD) == "${causal_conv_commit}" ]]

rm -rf "${fast_hadamard_root}"
mkdir -p "${fast_hadamard_root}"
git -C "${fast_hadamard_root}" init
git -C "${fast_hadamard_root}" remote add origin "${fast_hadamard_url}"
git -C "${fast_hadamard_root}" fetch --depth 1 origin "${fast_hadamard_commit}"
git -C "${fast_hadamard_root}" checkout --detach FETCH_HEAD
[[ $(git -C "${fast_hadamard_root}" rev-parse HEAD) == "${fast_hadamard_commit}" ]]

rm -rf "${emerging_optimizers_root}"
mkdir -p "${emerging_optimizers_root}"
git -C "${emerging_optimizers_root}" init
git -C "${emerging_optimizers_root}" remote add origin "${emerging_optimizers_url}"
git -C "${emerging_optimizers_root}" fetch --depth 1 origin "${emerging_optimizers_commit}"
git -C "${emerging_optimizers_root}" checkout --detach FETCH_HEAD
[[ $(git -C "${emerging_optimizers_root}" rev-parse HEAD) == "${emerging_optimizers_commit}" ]]

# The base image's torch requires C++20, and the pinned sources otherwise match upstream.
for source_root in "${mamba_root}" "${causal_conv_root}"; do
    sed -i 's/-std=c++17/-std=c++20/g' "${source_root}/setup.py"
    grep -q -- '-std=c++20' "${source_root}/setup.py"
    [[ $(git -C "${source_root}" status --short) == " M setup.py" ]]
done

configure_local_sources

echo "Pinned TransformerEngine source ${transformer_engine_commit} at ${transformer_engine_root}"
echo "Pinned Mamba source ${mamba_commit} at ${mamba_root}"
echo "Pinned causal-conv1d ${causal_conv_commit} at ${causal_conv_root}"
echo "Pinned fast-hadamard-transform ${fast_hadamard_commit} at ${fast_hadamard_root}"
echo "Pinned Emerging-Optimizers ${emerging_optimizers_commit} at ${emerging_optimizers_root}"
