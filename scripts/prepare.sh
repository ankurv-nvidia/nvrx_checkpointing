#!/bin/bash
set -Eeoux pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(realpath ${SCRIPT_DIR}/..)
ARCH=$(uname -m)
# Python packages go into the active virtual environment, or else into the system Python.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
else
    export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1
fi

# The CI build scripts, the JET image and the Slurm runner read these versions from here.
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.11-py3
# The image's PyTorch series, which bases without PyTorch install so that Megatron-LM's pins hold.
PYTORCH_SERIES=2.15
CUDA_VERSION=13.5
CLANG_VERSION=23

# Resolved on demand: only the container/megatron functions need the
# megatron-lm checkout, and resolving it clones the repo — toolchain-only
# invocations (e.g. the CI runner setup running `prepare.sh ubuntu2404`) run
# without git or repo credentials.
function resolve_megatron_root {
    MEGATRON_ROOT=${MEGATRON_ROOT:-$(megatron_root)}
}

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

# Checks out the pinned Megatron-LM commit in the given directory and prints its path.
function megatron_root {
    set +x
    COMMIT_HASH=${MEGATRON_COMMIT:-16251ac1221f4814cd39350def05cefa3c36734a}
    MEGATRON_REPO_URL=${MEGATRON_REPO_URL:-https://github.com/NVIDIA/Megatron-LM.git}
    echo "MEGATRON_REPO_URL=${MEGATRON_REPO_URL}" >&2
    MEGATRON_ROOT=${1:-${MEGATRON_ROOT:-${REPO_ROOT}/../nvrx-megatron-lm}}

    # HEAD preserves the existing checkout and local modifications without repository mutation.
    if [[ "${COMMIT_HASH}" == HEAD ]]; then
        if [[ ! -d ${MEGATRON_ROOT} ]] || ! git -C "${MEGATRON_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
            echo "MEGATRON_COMMIT=HEAD requires an existing Git checkout at MEGATRON_ROOT=${MEGATRON_ROOT}" >&2
            return 1
        fi
        realpath "${MEGATRON_ROOT}"
        return 0
    fi

    # Serialize concurrent srun ranks: only the lock holder clones/checks out the pinned
    # commit; the others block here and then observe the ready clone.
    mkdir -p "$(dirname "${MEGATRON_ROOT}")"
    exec 9>"${MEGATRON_ROOT}.lock"
    flock 9

    if [[ -e ${MEGATRON_ROOT} ]] && ! git -C "${MEGATRON_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "MEGATRON_ROOT exists but is not a Git checkout: ${MEGATRON_ROOT}" >&2
        return 1
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
}

declare -A CUDA_REPOS=()
CUDA_REPOS["noble_x86_64"]="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_REPOS["noble_aarch64"]="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb"

declare -A DOCKER_IMAGES=()
DOCKER_IMAGES["ubuntu2404"]="docker.io/library/ubuntu:24.04"
DOCKER_IMAGES["fedora44"]="docker.io/library/fedora:44"
DOCKER_IMAGES["pytorch2412"]="nvcr.io/nvidia/pytorch:24.12-py3"
DOCKER_IMAGES["pytorch"]="${PYTORCH_IMAGE}"

function fedora {
    # Fedora 44 provides LLVM 22; container compilation uses the Ubuntu path below.
    GCC_VERSION=16
    CLANG_VERSION=22
    export TZ=America/Los_Angeles
    packages=()
    packages+=(vim gnupg wget curl ca-certificates)
    packages+=(
        g++
        gdb
        clang
        clang-tools-extra
        clangd
        lldb-${CLANG_VERSION}
        binutils
        cmake
        mold
        ninja-build
        curl
        wget
        #python3.12-venv
        #python3.12-dev
        python3-pip
        #python3-venv
    )
    # vcpkg
    packages+=(
        zip
        xz
        bzip2k
        tar
        unzip
        cmake
        python3
        git
        pkg-config
    )
    # openssl
    packages+=(
        #libfindbin-libs-perl
        #libipc-run-perl
        #libfile-copy-link-perl
        #libfile-copy-recursive-perl
        #libtime-piece-mysql-perl
        perl-FindBin
        perl-File-Compare
        perl-IPC-Cmd
        perl-File-Copy
        perl-Time-Piece
    )
    dnf -y install "${packages[@]}"

    update-alternatives --install /usr/bin/cc cc /usr/bin/clang-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/lldb-dap lldb-dap /usr/bin/lldb-dap-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.12-config 1

    # Print out tool version info
    for tool in cmake cc gcc g++ clang-${CLANG_VERSION}; do
        echo .
        echo "-- ${tool} version"
        command -v ${tool}
        ${tool} --version
    done
    python3-config --extension-suffix
}

function test {
    bash ${SCRIPT_DIR}/run.sh
}

function pytorch {
    $*
}

function ubuntu2404 {
    GCC_VERSION=13
    UBUNTU_CODENAME=noble
    UBUNTU_VERSION=24.04
    ARCH=$(uname -m)
    export DEBIAN_FRONTEND=noninteractive
    export TZ=America/Los_Angeles
    rm -f /etc/apt/apt.conf.d/docker-clean
    apt update && apt-get install --yes --no-install-recommends wget build-essential ca-certificates curl
    # clang/llvm
    wget https://apt.llvm.org/llvm-snapshot.gpg.key -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    wget https://apt.kitware.com/keys/kitware-archive-latest.asc -O /etc/apt/trusted.gpg.d/kitware-archive-latest.asc
    echo "deb http://apt.llvm.org/${UBUNTU_CODENAME}/ llvm-toolchain-${UBUNTU_CODENAME}-${CLANG_VERSION} main" > /etc/apt/sources.list.d/llvm.list
    echo "deb https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/cmake.list
    packages=()

    packages+=(gnupg build-essential wget curl ca-certificates)

    packages+=(
        g++-${GCC_VERSION}
        clang-${CLANG_VERSION}
        clang-format-${CLANG_VERSION}
        clangd-${CLANG_VERSION}
        llvm-${CLANG_VERSION}
        gdb
        strace
        cmake
        mold
        ninja-build
        curl
        wget
        python3.12-venv
        python3.12-dev
        python3-pip
    )
    # vcpkg
    packages+=(
        zip xz-utils bzip2 tar unzip cmake python3 git git-lfs openssh-client
        pkg-config patchelf valgrind
    )
    packages+=(
        # openssl
        libfindbin-libs-perl
        #perl-File-Compare
        libipc-run-perl
        #libfile-copy-link-perl
        #libfile-copy-recursive-perl
        #libtime-piece-mysql-perl

        # hwloc
        autoconf
        automake
        libtool
    )

    apt update

    apt-get install --yes --no-install-recommends "${packages[@]}"
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/usr/local/bin" UV_NO_MODIFY_PATH=1 sh

    update-alternatives --install /usr/bin/cc cc /usr/bin/clang-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.12-config 1

    # Remove cmake /usr/local/bin that conflicts with /usr/bin/cmake installed by apt
    rm -f /usr/local/bin/cmake /usr/local/bin/ctest /usr/local/bin/cpack
    # Print out tool version info
    for tool in cmake cc gcc g++ clang-${CLANG_VERSION} uv; do
    echo .
    echo "-- ${tool} version"
    command -v ${tool}
    ${tool} --version
    done
    $*
}

function container {
    resolve_megatron_root
    PLAT_SUFFIX=$1
    BASE_IMAGE=${BASE_IMAGE:-${DOCKER_IMAGES["${PLAT_SUFFIX}"]}}
    CONTAINER_NAME=${CONTAINER_NAME:-mlm}
    BUILD_CONTAINER_NAME=${BUILD_CONTAINER_NAME:-${CONTAINER_NAME}-build}
    CACHE_PATH=${CACHE_PATH:-~/cache}

    mkdir -p ${CACHE_PATH}/pip_cache_${PLAT_SUFFIX} ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}
    PODMAN_DEVICES_ARG=$([[ -e /var/run/cdi/nvidia.yaml ]] && echo "--device nvidia.com/gpu=all" || echo "")
    podman run --rm \
        --name ${CONTAINER_NAME}-build \
        --shm-size=34g \
        --cap-add=SYS_PTRACE \
        --replace ${PODMAN_DEVICES_ARG} \
        --security-opt=label=disable \
        -v ${CACHE_PATH}/pip_cache_${PLAT_SUFFIX}:/root/.cache/pip \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/cache/apt \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/cache/dnf \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/lib/dnf \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/lib/apt \
        -v ${MEGATRON_ROOT}:/root/nvrx-megatron-lm \
        -v ${REPO_ROOT}:/root/nvrx_checkpointing \
        -d -it ${BASE_IMAGE} \
        bash

    podman exec -it ${MAX_JOBS:+--env MAX_JOBS=${MAX_JOBS}} ${CONTAINER_NAME}-build bash /root/nvrx_checkpointing/scripts/prepare.sh $*

    podman commit ${CONTAINER_NAME}-build ${CONTAINER_NAME}-image
    podman run \
        --name ${CONTAINER_NAME} \
        --shm-size=34g \
        --cap-add=SYS_PTRACE \
        --replace ${PODMAN_DEVICES_ARG} \
        --security-opt=label=disable \
        -v ${REPO_ROOT}/..:/root \
        -d -it ${CONTAINER_NAME}-image bash
    podman stop ${CONTAINER_NAME}-build
}

function megatron {
    resolve_megatron_root
    _NVCC_VERSION=$(nvcc --version)
    printf '%s\n' "${_NVCC_VERSION}"
    if [[ "${_NVCC_VERSION}" =~ release[[:space:]]+([0-9]+)\. ]]; then
        export TMS_CUDA_MAJOR="${BASH_REMATCH[1]}"
    else
        echo "Unable to determine the CUDA major version from nvcc" >&2
        return 1
    fi
    export MAX_JOBS=${MAX_JOBS:-4}
    export CC=gcc
    export CXX=g++
    BASE_TRANSFORMER_ENGINE_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("transformer-engine"))' 2>/dev/null || true)
    # nvidia-smi prints one line per GPU; capture the full output and slice the first
    # line in-shell. A truncating reader (head) closes the pipe early and SIGPIPEs
    # nvidia-smi, which pipefail would turn into a fatal exit on multi-GPU nodes.
    _COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
    _COMPUTE_CAP=${_COMPUTE_CAP%%$'\n'*}
    export TORCH_CUDA_ARCH_LIST=${_COMPUTE_CAP}
    NVTE_CUDA_ARCHS=$(nvte_cuda_archs "${_COMPUTE_CAP}")
    export NVTE_CUDA_ARCHS
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    if [ -f "$HOME/.local/bin/env" ]; then
        source $HOME/.local/bin/env
    fi
    # A base without PyTorch, such as plain Ubuntu, gets the image's series built for the driver's CUDA.
    python -c 'import torch' 2>/dev/null || uv pip install "torch==${PYTORCH_SERIES}.*" --torch-backend=auto
    if [[ -n "${PREPARE_WHEEL_CACHE:-}" ]]; then
        # Every revision of this script, toolchain and GPU architecture keeps caches of its own.
        PREPARE_WHEEL_CACHE="${PREPARE_WHEEL_CACHE}/$({
            sha256sum < "${SCRIPT_DIR}/prepare.sh"
            uname -m
            nvcc --version
            python -c 'import sys, torch; print(sys.version, torch.__version__)'
            echo "${TORCH_CUDA_ARCH_LIST}"
        } | sha256sum | cut -c1-16)"
        export UV_CACHE_DIR="${PREPARE_WHEEL_CACHE}/uv"
    else
        # By default ${HOME} is mounted at root in enroot; use a temporary directory for the
        # uv cache to avoid collision.
        export UV_CACHE_DIR="$(mktemp -d)" UV_NO_CACHE=1
    fi
    # Installs keep a base image's NumPy, which packages such as its Numba cap.
    export UV_CONSTRAINT=$(mktemp)
    python -c 'import importlib.metadata as m; print("numpy==" + m.version("numpy"))' 2>/dev/null > "${UV_CONSTRAINT}" || true
    printf '%s\n' 'tilelang==0.1.15' 'apache-tvm-ffi==0.1.12' >> "${UV_CONSTRAINT}"
    uv pip install "nvidia-cudnn-frontend[cutedsl]==1.32.0"
    # uv pip does not read pyproject dependency overrides, so Megatron's list is passed explicitly.
    UV_OVERRIDES=$(mktemp)
    python - "${MEGATRON_ROOT}/pyproject.toml" > "${UV_OVERRIDES}" <<'PY'
import re
import sys
import tomllib

replaced = {"transformer-engine", "nvidia-cudnn-frontend", "tilelang", "apache-tvm-ffi"}
with open(sys.argv[1], "rb") as stream:
    requirements = tomllib.load(stream)["tool"]["uv"].get("override-dependencies", [])
for requirement in requirements:
    name = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip()).group()
    if re.sub(r"[-_.]+", "-", name).lower() not in replaced:
        print(requirement)
PY
    if [[ -n "${BASE_TRANSFORMER_ENGINE_VERSION}" ]]; then
        printf '%s\n' "transformer-engine; sys_platform == 'never'" >> "${UV_OVERRIDES}"
    else
        printf '%s\n' 'transformer-engine==2.21.0' >> "${UV_OVERRIDES}"
    fi
    printf '%s\n' "nvidia-cudnn-frontend[cutedsl]==1.32.0" >> "${UV_OVERRIDES}"
    printf '%s\n' 'tilelang==0.1.15' 'apache-tvm-ffi==0.1.12' >> "${UV_OVERRIDES}"
    # A base image's quack-kernels stays in use; its FlashAttention-4 needs that release.
    python -c 'import importlib.metadata as m; m.version("quack-kernels")' 2>/dev/null && echo "quack-kernels; sys_platform == 'never'" >> "${UV_OVERRIDES}"
    cat "${UV_OVERRIDES}"

    # The base's torch requires C++20, so Mamba, causal-conv1d and a missing flash-attn are built from source with it.
    read -r MAMBA_GIT MAMBA_REV <<< "$(python -c 'import sys, tomllib; s = tomllib.load(open(sys.argv[1], "rb"))["tool"]["uv"]["sources"]["mamba-ssm"]; print(s["git"], s["rev"])' "${MEGATRON_ROOT}/pyproject.toml")"
    export MAMBA_FORCE_BUILD=TRUE
    specs=("causal-conv1d https://github.com/Dao-AILab/causal-conv1d.git v1.7.0" "mamba ${MAMBA_GIT} ${MAMBA_REV}")
    python -c 'import flash_attn' 2>/dev/null || specs+=("flash-attention https://github.com/Dao-AILab/flash-attention.git v2.8.3.post2")
    for spec in "${specs[@]}"; do
        read -r name url ref <<< "${spec}"
        git clone --quiet "${url}" "${name}"
        git -C "${name}" checkout --quiet "${ref}"
        sed -i 's/-std=c++17/-std=c++20/g' "${name}/setup.py"
        grep -q -- '-std=c++20' "${name}/setup.py"
        FLASH_ATTN_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" wheel_install "${name}" --override "${UV_OVERRIDES}"
        rm -rf "${name}"
    done
    printf '%s\n' "causal-conv1d; sys_platform == 'never'" "mamba-ssm; sys_platform == 'never'" >> "${UV_OVERRIDES}"

    # Megatron-LM builds without isolation, so a base without pybind11 first gets its declared build requirements.
    python -c 'import pybind11' 2>/dev/null || uv pip install -r <(python -c 'import sys, tomllib; print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["build-system"]["requires"]))' "${MEGATRON_ROOT}/pyproject.toml")
    uv pip install --no-build-isolation --override "${UV_OVERRIDES}" -e "${MEGATRON_ROOT}[training,dev,otel,te,ssm]"
    if [[ -n "${BASE_TRANSFORMER_ENGINE_VERSION}" ]]; then
        python -c 'import importlib.metadata as m, sys; actual = m.version("transformer-engine"); assert actual == sys.argv[1], (actual, sys.argv[1]); import transformer_engine.pytorch' \
            "${BASE_TRANSFORMER_ENGINE_VERSION}"
    fi
    # The CUTLASS DSL that Megatron-LM's packages bring must still serve FlashInfer.
    python -c 'import cutlass.cutlass_dsl as dsl, flashinfer, tilelang; dsl.CuteExperimentalDSL'
    # Error :  TypeError: make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'
    # uv pip install --system --no-cache --break-system-packages --no-build-isolation "flash-attn-4>=4.0.0b11" nvidia-cutlass-dsl[cu13]

    # Keep the DeepEP HybridEP installation symmetric with Megatron-LM's CI image.
    DEEPEP_COMMIT=10d4dd7377d5bce900fbb4b80cce863764892b95
    export HYBRID_EP_MULTINODE=1
    export RDMA_CORE_HOME=/opt/rdma-core/build
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH:-}"

    uv pip install nvidia-nvshmem-cu13==3.8.0
    NVSHMEM_LIB_DIR=$(python -c 'from pathlib import Path; import nvidia.nvshmem; print(Path(next(iter(nvidia.nvshmem.__path__))) / "lib")')
    pushd "${NVSHMEM_LIB_DIR}"
        ln -sf libnvshmem_host.so.3 libnvshmem_host.so
    popd

    git clone https://github.com/deepseek-ai/DeepEP.git
    pushd DeepEP
        git fetch origin "${DEEPEP_COMMIT}"
        git checkout FETCH_HEAD
        patch -p1 < "${SCRIPT_DIR}/deepep.patch"
        # Backport deepseek-ai/DeepEP#564 for NVSHMEM 3.5 and newer.
        patch -p1 < "${SCRIPT_DIR}/deepep-nvshmem.patch"
    popd

    # DeepEP links only NCCL's GPUNetIO objects, so only the SendRecv kernels are generated.
    ONLY_FUNCS="SendRecv" wheel_install DeepEP
    rm -rf DeepEP

    unset CC
    unset CXX
    python -c 'import megatron.core, transformer_engine.pytorch, causal_conv1d, mamba_ssm, flash_attn, deep_ep'
    #pip install -e "${MEGATRON_ROOT}" --no-build-isolation --break-system-packages || true
    uv pip list
    #pip install  --no-build-isolation --break-system-packages
    # if nvidia gpus found then run run.sh
    nvidia-smi || echo "No nvidia gpus found"
    $*
}

# Transformer Engine expands only the Blackwell family heads into the feature-suffixed targets its kernels need.
function nvte_cuda_archs {
    case "$1" in
        10.*) echo 100 ;;
        12.*) echo 120 ;;
        *) echo "${1/./}" ;;
    esac
}

# Installs a source tree's wheel with the given uv pip arguments, reusing a cached wheel of the same tree and GPU architectures.
function wheel_install {
    local source=$1 wheels="" key build
    shift
    if [[ -n "${PREPARE_WHEEL_CACHE:-}" ]]; then
        mkdir -p "${PREPARE_WHEEL_CACHE}"
        git -C "${source}" add --all
        key=$({
            git -C "${source}" rev-parse HEAD
            git -C "${source}" diff --cached --binary HEAD
            echo "${TORCH_CUDA_ARCH_LIST:-}"
        } | sha256sum | cut -c1-16)
        wheels="${PREPARE_WHEEL_CACHE}/$(basename "$(realpath "${source}")")-${key}"
    fi
    if [[ -z "${wheels}" ]] || ! compgen -G "${wheels}/*.whl" > /dev/null; then
        build=$(mktemp -d "${PREPARE_WHEEL_CACHE:-${TMPDIR:-/tmp}}/.wheel-build.XXXXXX")
        uv build --wheel --no-build-isolation --python "$(command -v python)" --out-dir "${build}" "${source}"
        if [[ -z "${wheels}" ]]; then
            wheels=${build}
        else
            # A concurrent run that stored the same wheels first keeps its copy.
            mv -T "${build}" "${wheels}" || rm -rf "${build}"
        fi
    fi
    uv pip install "$@" "${wheels}"/*.whl
    [[ -n "${PREPARE_WHEEL_CACHE:-}" ]] || rm -rf "${wheels}"
}

# Installs the RDMA, InfiniBand verbs, mlx5 and NVML development files that GPU networking packages compile against.
function gpu_networking {
    apt-get update
    apt-get install --yes --allow-change-held-packages rdma-core libibverbs-dev
    apt-get clean
    local arch_lib nvml_dir
    arch_lib=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
    if [[ ! -f "/usr/lib/${arch_lib}/libmlx5.so" ]]; then
        ln -sf "/usr/lib/${arch_lib}/libmlx5.so.1" "/usr/lib/${arch_lib}/libmlx5.so"
    fi
    mkdir -p /opt/rdma-core/build
    ln -sfn /usr/include /opt/rdma-core/build/include
    ln -sfn "/usr/lib/${arch_lib}" /opt/rdma-core/build/lib
    # Only the payload is unpacked because installing the package pulls in driver packages that overrun container overlays.
    nvml_dir=$(mktemp -d)
    (cd "${nvml_dir}" && apt-get update && apt-get download libnvidia-ml-dev && dpkg-deb -x libnvidia-ml-dev_*.deb /)
    rm -rf "${nvml_dir}" /var/lib/apt/lists/*
    $*
}

function cuda13 {
    . /etc/os-release
    # CUDA stuff
    packages=(
        pciutils
        cuda-toolkit-${CUDA_VERSION/./-}
        libcudnn9-cuda-${CUDA_VERSION%%.*}
        libcutlass-dev
        cudnn9-cuda-${CUDA_VERSION%%.*}
        libnccl-dev
        libtime-piece-mysql-perl
    )

    # cuda keyring
    wget ${CUDA_REPOS["${UBUNTU_CODENAME}_${ARCH}"]} -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb

    apt update && apt-get install --yes --no-install-recommends "${packages[@]}"

    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-${CUDA_VERSION}}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
    $*
}

# Cross-compiles aarch64 (Grace) wheels on an x86_64 host that has run cuda13.
function cross_arm64 {
    # NVIDIA signs the sbsa cross repository with the key the CUDA keyring installs.
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/ /" \
        > /etc/apt/sources.list.d/cuda-ubuntu2404-cross-linux-sbsa.list
    apt update && apt-get install --yes --no-install-recommends crossbuild-essential-arm64 "cuda-cross-sbsa-${CUDA_VERSION/./-}"
    $*
}

if [ "$#" -eq 0 ]; then
    container ubuntu2404 cuda13 gpu_networking megatron test
else
    $*
fi
