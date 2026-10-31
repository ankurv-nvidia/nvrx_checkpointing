#!/usr/bin/env bash
set -Eeuo pipefail

# Build the Megatron runtime overlay on native JET hardware.
repo_root=$(realpath "$(dirname "$0")/../../..")
megatron_root=${MEGATRON_ROOT:?}
venv=${NVRX_VENV:?}
constraints=${NVRX_MEGATRON_CONSTRAINTS:?}
compute_capability=${NVRX_COMPUTE_CAPABILITY:?}
extras=${NVRX_MEGATRON_EXTRAS:-training,dev,otel,te,ssm}

[[ -f "${megatron_root}/pyproject.toml" ]]
[[ -f "${constraints}" ]]
[[ "${compute_capability}" =~ ^[0-9]+\.[0-9]+$ ]]
[[ "${extras}" =~ ^[a-z0-9_-]+(,[a-z0-9_-]+)*$ ]]
[[ "${NVRX_RDMA_PREREQS_READY:-0}" == 1 ]]
[[ -d /opt/rdma-core/build/include && -d /opt/rdma-core/build/lib && -f /usr/include/nvml.h ]]
command -v uv >/dev/null
nvcc --version

# Preserve the architecture-specific CUTLASS runtime from the NGC image.
base_cutlass_dsl_version=$(python3 -c 'import importlib.metadata as m; print(m.version("nvidia-cutlass-dsl"))')

export MAX_JOBS=${NVRX_MAX_JOBS:-4}
export CC=gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST=${compute_capability}
case "${compute_capability}" in
    10.*) export NVTE_CUDA_ARCHS=100 ;;
    12.*) export NVTE_CUDA_ARCHS=120 ;;
    *) export NVTE_CUDA_ARCHS=${compute_capability/./} ;;
esac
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export NVTE_FRAMEWORK=pytorch
export HYBRID_EP_MULTINODE=1
export RDMA_CORE_HOME=/opt/rdma-core/build
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${venv}/bin/python" ]]; then
    uv venv --system-site-packages "${venv}"
fi
source "${venv}/bin/activate"
python -c 'import torch; print(f"Inherited PyTorch {torch.__version__} from {torch.__file__}")'

uv_cache=$(mktemp -d)
deepep_build=$(mktemp -d)
cleanup() {
    rm -rf "${uv_cache}" "${deepep_build}"
}
trap cleanup EXIT
export UV_CACHE_DIR=${uv_cache}
# uv pip does not read pyproject dependency overrides, so Megatron's list is passed explicitly.
overrides="${uv_cache}/megatron-overrides.txt"
python -c 'import sys, tomllib; print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["tool"]["uv"].get("override-dependencies", [])))' \
    "${megatron_root}/pyproject.toml" > "${overrides}"
# Keep the NGC image's quack-kernels, which its FlashAttention-4 requires.
printf '%s\n' "quack-kernels; sys_platform == 'never'" >> "${overrides}"
cat "${overrides}"
uv_args=(--constraint "${constraints}" --override "${overrides}" --no-cache)

uv_megatron() {
    (
        cd "${megatron_root}"
        uv "$@"
    )
}

# Install metadata dependencies without replacing the image's CUTLASS runtime.
uv_megatron pip install "${uv_args[@]}" --no-deps "nvidia-cudnn-frontend==1.28.0"
uv_megatron pip install "${uv_args[@]}" --no-build-isolation -e ".[${extras}]"
# Reconcile the CUTLASS overlay with the image version after dependency resolution.
uv_megatron pip install "${uv_args[@]}" --upgrade \
    "nvidia-cutlass-dsl[cu13]==${base_cutlass_dsl_version}"
# Reinstall the CUDA library wheel as the deterministic namespace owner.
uv_megatron pip install "${uv_args[@]}" --no-deps --reinstall \
    "nvidia-cutlass-dsl-libs-cu13==${base_cutlass_dsl_version}"

python - "${base_cutlass_dsl_version}" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
actual = importlib.metadata.version("nvidia-cutlass-dsl")
if actual != expected:
    raise SystemExit(
        f"CUTLASS DSL changed from base-image {expected} to overlay {actual}",
    )
import cutlass.cutlass_dsl as cutlass_dsl

if not hasattr(cutlass_dsl, "CuteExperimentalDSL"):
    raise SystemExit(
        f"CUTLASS DSL {actual} lacks CuteExperimentalDSL required by FlashInfer",
    )
import flashinfer

flashinfer_version = importlib.metadata.version("flashinfer-python")
print(f"Using base-image CUTLASS DSL {actual} with FlashInfer {flashinfer_version}")
PY

python - "${constraints}" <<'PY'
import importlib.metadata
import pathlib
import sys

for raw_line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==", 1)
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise SystemExit(f"constraint mismatch for {name}: expected {expected}, found {actual}")
print("JET Megatron overlay matches the regular-CI package constraints")
PY

python -c 'import transformer_engine; import transformer_engine.pytorch; print(f"Using TransformerEngine {transformer_engine.__version__}")'
if [[ ",${extras}," == *,ssm,* ]]; then
    python -c 'import causal_conv1d, mamba_ssm; print(f"Using causal-conv1d {causal_conv1d.__version__} and mamba-ssm {mamba_ssm.__version__}")'
fi
if [[ "${NVRX_USE_IMAGE_FLASH_ATTN:-0}" == 1 ]]; then
    python -c 'from packaging.version import Version; import flash_attn; version = Version(flash_attn.__version__); assert Version("2.1.1") <= version < Version("2.8.4"), version; print(f"Reusing image FlashAttention {version}")'
else
    uv_megatron pip install "${uv_args[@]}" --no-build-isolation flash-attn
fi

uv_megatron pip install "${uv_args[@]}" nvidia-nvshmem-cu13==3.7.2
nvshmem_lib_dir=$(python -c 'from pathlib import Path; import nvidia.nvshmem; print(Path(next(iter(nvidia.nvshmem.__path__))) / "lib")')
ln -sfn libnvshmem_host.so.3 "${nvshmem_lib_dir}/libnvshmem_host.so"

# Apply the DeepEP patch on a writable copy of the embedded source.
cp -a "${NVRX_DEEPEP_ROOT:?}/." "${deepep_build}/"
patch -d "${deepep_build}" -p1 < "${repo_root}/scripts/deepep.patch"
patch -d "${deepep_build}" -p1 < "${repo_root}/scripts/deepep-nvshmem.patch"
(
    export TORCH_CUDA_ARCH_LIST=${NVRX_DEEPEP_CUDA_ARCH_LIST:-9.0 10.0 12.0}
    # DeepEP links only NCCL's GPUNetIO objects, so only the SendRecv kernels are generated.
    export ONLY_FUNCS=SendRecv
    cd "${megatron_root}"
    uv pip install "${uv_args[@]}" --no-build-isolation -v "${deepep_build}/."
)

unset CC CXX
pip list
nvidia-smi
