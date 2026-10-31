#!/bin/bash
set -Eeoux pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(realpath ${SCRIPT_DIR}/..)
ARCH=$(uname -m)

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
DOCKER_IMAGES["pytorch"]="nvcr.io/nvidia/pytorch:26.09-py3"

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
    CLANG_VERSION=23
    GCC_VERSION=13
    UBUNTU_CODENAME=noble
    UBUNTU_VERSION=24.04
    ARCH=$(uname -m)
    export DEBIAN_FRONTEND=noninteractive
    export TZ=America/Los_Angeles
    rm -f /etc/apt/apt.conf.d/docker-clean
    apt update && apt-get install --yes --no-install-recommends wget software-properties-common build-essential ca-certificates curl
    # clang/llvm
    wget https://apt.llvm.org/llvm-snapshot.gpg.key -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    wget https://apt.kitware.com/keys/kitware-archive-latest.asc -O /etc/apt/trusted.gpg.d/kitware-archive-latest.asc
    echo "deb http://apt.llvm.org/${UBUNTU_CODENAME}/ llvm-toolchain-${UBUNTU_CODENAME}-${CLANG_VERSION} main" > /etc/apt/sources.list.d/llvm.list
    echo "deb https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/cmake.list
    packages=()

    packages+=(gnupg build-essential software-properties-common wget curl ca-certificates)

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

    podman exec -it ${CONTAINER_NAME}-build bash /root/nvrx_checkpointing/scripts/prepare.sh $*

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
    git config --system protocol.version 1
    resolve_megatron_root
    nvcc --version || cuda13
    _NVCC_VERSION=$(nvcc --version)
    printf '%s\n' "${_NVCC_VERSION}"
    if [[ "${_NVCC_VERSION}" =~ release[[:space:]]+([0-9]+)\. ]]; then
        export TMS_CUDA_MAJOR="${BASH_REMATCH[1]}"
    else
        echo "Unable to determine the CUDA major version from nvcc" >&2
        return 1
    fi
    export MAX_JOBS=4
    export CC=gcc
    export CXX=g++
    # TransformerEngine's CMake recognizes only Blackwell family heads (100/120) and
    # expands them into the CUDA-13 feature-suffixed gencode targets its kernels require
    # (input 100 -> generic sm_100 plus arch-specific sm_100a and sm_103a). Map the live
    # compute capability to the family head TE can expand; torch's arch list takes the raw cc.
    # nvidia-smi prints one line per GPU; capture the full output and slice the first
    # line in-shell. A truncating reader (head) closes the pipe early and SIGPIPEs
    # nvidia-smi, which pipefail would turn into a fatal exit on multi-GPU nodes.
    _COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
    _COMPUTE_CAP=${_COMPUTE_CAP%%$'\n'*}
    export TORCH_CUDA_ARCH_LIST=${_COMPUTE_CAP}
    case "${_COMPUTE_CAP}" in
        10.*) export NVTE_CUDA_ARCHS=100 ;;   # Blackwell datacenter (sm_100/sm_103)
        12.*) export NVTE_CUDA_ARCHS=120 ;;   # Blackwell consumer (sm_120)
        *)    export NVTE_CUDA_ARCHS=$(echo "${_COMPUTE_CAP}" | sed 's/\.//') ;;  # pre-Blackwell: bare arch (9.0 -> 90)
    esac
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    if [ -f "$HOME/.local/bin/env" ]; then
        source $HOME/.local/bin/env
    fi
    # By default ${HOME} is mounted at root in enroot; use a temporary directory for the
    # uv cache to avoid collision.
    export UV_CACHE_DIR="$(mktemp -d)"
    # The pinned Transformer Engine build needs its locked cuDNN frontend during metadata resolution.
    uv pip install --system --no-cache --break-system-packages "nvidia-cudnn-frontend[cutedsl]==1.28.0"
    # uv pip does not read pyproject dependency overrides, so Megatron's list is passed explicitly.
    UV_OVERRIDES="${UV_CACHE_DIR}/megatron-overrides.txt"
    python -c 'import sys, tomllib; print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["tool"]["uv"].get("override-dependencies", [])))' \
        "${MEGATRON_ROOT}/pyproject.toml" > "${UV_OVERRIDES}"
    cat "${UV_OVERRIDES}"

    # The image's torch requires C++20, so Mamba and causal-conv1d are built from source with it.
    read -r MAMBA_GIT MAMBA_REV <<< "$(python -c 'import sys, tomllib; s = tomllib.load(open(sys.argv[1], "rb"))["tool"]["uv"]["sources"]["mamba-ssm"]; print(s["git"], s["rev"])' "${MEGATRON_ROOT}/pyproject.toml")"
    export MAMBA_FORCE_BUILD=TRUE
    for spec in "causal-conv1d https://github.com/Dao-AILab/causal-conv1d.git v1.7.0" "mamba ${MAMBA_GIT} ${MAMBA_REV}"; do
        read -r name url ref <<< "${spec}"
        git clone --quiet "${url}" "${name}"
        git -C "${name}" checkout --quiet "${ref}"
        sed -i 's/-std=c++17/-std=c++20/g' "${name}/setup.py"
        grep -q -- '-std=c++20' "${name}/setup.py"
        uv pip install --system --no-cache --break-system-packages --no-build-isolation --override "${UV_OVERRIDES}" "./${name}"
        rm -rf "${name}"
    done
    printf '%s\n' "causal-conv1d; sys_platform == 'never'" "mamba-ssm; sys_platform == 'never'" >> "${UV_OVERRIDES}"

    uv pip install --system --no-cache --break-system-packages --no-build-isolation --override "${UV_OVERRIDES}" -e "${MEGATRON_ROOT}[training,dev,otel,te,ssm]"
    uv pip install --system --no-cache --break-system-packages --no-build-isolation flash-attn
    # Error :  TypeError: make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'
    # uv pip install --system --no-cache --break-system-packages --no-build-isolation "flash-attn-4>=4.0.0b11" nvidia-cutlass-dsl[cu13]

    # Keep the DeepEP HybridEP installation symmetric with Megatron-LM's CI image.
    DEEPEP_COMMIT=10d4dd7377d5bce900fbb4b80cce863764892b95
    export HYBRID_EP_MULTINODE=1
    export RDMA_CORE_HOME=/opt/rdma-core/build
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH:-}"

    apt-get update
    apt-get install --yes --allow-change-held-packages rdma-core libibverbs-dev
    apt-get clean
    ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
    if [[ ! -f "/usr/lib/${ARCH_LIB}/libmlx5.so" ]]; then
        ln -sf "/usr/lib/${ARCH_LIB}/libmlx5.so.1" "/usr/lib/${ARCH_LIB}/libmlx5.so"
    fi
    mkdir -p "${RDMA_CORE_HOME}"
    ln -sfn /usr/include "${RDMA_CORE_HOME}/include"
    ln -sfn "/usr/lib/${ARCH_LIB}" "${RDMA_CORE_HOME}/lib"

    # Megatron requires NVSHMEM >=3.6.5: older releases can crash while initializing
    # split NVLink-domain PE layouts. Keep the image reproducible at the current cu13 release.
    uv pip install --system --no-cache --break-system-packages nvidia-nvshmem-cu13==3.7.2
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

    # DeepEP compiles against nvml.h from libnvidia-ml-dev. Installing that package
    # via apt pulls nvidia-firmware-580 (76 MB) and libnvidia-compute as hard
    # dependencies whose dpkg unpack overruns the enroot writable overlay and aborts
    # with a broken pipe. Fetch the dev package alone and unpack just its payload
    # (nvml.h and the libnvidia-ml.so dev symlink); libnvidia-ml.so.1 itself is
    # already present from the injected driver that nvidia-smi above relies on.
    apt-get update
    apt-get download libnvidia-ml-dev
    dpkg-deb -x libnvidia-ml-dev_*.deb /
    rm -f libnvidia-ml-dev_*.deb
    # DeepEP links only NCCL's GPUNetIO objects, so only the SendRecv kernels are generated.
    ONLY_FUNCS="SendRecv" TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" uv pip install --system --no-cache --break-system-packages --no-build-isolation -v DeepEP/.
    rm -rf /var/lib/apt/lists/*
    rm -rf DeepEP

    unset CC
    unset CXX
    #pip install -e "${MEGATRON_ROOT}" --no-build-isolation --break-system-packages || true
    pip list
    #pip install  --no-build-isolation --break-system-packages
    # if nvidia gpus found then run run.sh
    nvidia-smi || echo "No nvidia gpus found"
    $*
}

function cuda13 {
    . /etc/os-release
    # CUDA stuff
    packages=(
        pciutils
        cuda-toolkit-13-4
        libcudnn9-cuda-13
        libcutlass-dev
        cudnn9-cuda-13
        libnccl-dev
        libtime-piece-mysql-perl
    )

    # cuda keyring
    wget ${CUDA_REPOS["${UBUNTU_CODENAME}_${ARCH}"]} -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb

    apt update && apt-get install --yes --no-install-recommends "${packages[@]}"

    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
}

# Cross-compiles aarch64 (Grace) wheels on an x86_64 host that has run cuda13.
function cross_arm64 {
    # NVIDIA signs the sbsa cross repository with the key the CUDA keyring installs.
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/ /" \
        > /etc/apt/sources.list.d/cuda-ubuntu2404-cross-linux-sbsa.list
    apt update && apt-get install --yes --no-install-recommends crossbuild-essential-arm64 cuda-cross-sbsa-13-4
    $*
}

if [ "$#" -eq 0 ]; then
    container ubuntu2404 megatron test
else
    $*
fi
