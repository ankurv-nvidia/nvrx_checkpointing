#!/bin/bash
set -Eeoux pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
MEGATRON_ROOT=${SCRIPT_DIR}/../../..
ARCH=$(uname -m)

declare -A CUDA_REPOS=()
CUDA_REPOS["noble_x86_64"]="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_REPOS["noble_aarch64"]="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb"

declare -A DOCKER_IMAGES=()
DOCKER_IMAGES["ubuntu2404"]="docker.io/library/ubuntu:24.04"
DOCKER_IMAGES["fedora44"]="docker.io/library/fedora:44"
DOCKER_IMAGES["pytorch2412"]="nvcr.io/nvidia/pytorch:24.12-py3"
DOCKER_IMAGES["pytorch"]="nvcr.io/nvidia/pytorch:26.04-py3"

function fedora {
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
    bash ${MEGATRON_ROOT}/megatron/native/scripts/run.sh
}

function pytorch {
    $*
}

function ubuntu2404 {
    CLANG_VERSION=22
    GCC_VERSION=13
    UBUNTU_CODENAME=noble
    UBUNTU_VERSION=24.04
    ARCH=$(uname -m)
    export DEBIAN_FRONTEND=noninteractive
    export TZ=America/Los_Angeles
    rm /etc/apt/apt.conf.d/docker-clean
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

    update-alternatives --install /usr/bin/cc cc /usr/bin/clang-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-${CLANG_VERSION} 100
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.12-config 1

    # Print out tool version info
    for tool in cmake cc gcc g++ clang-${CLANG_VERSION}; do
    echo .
    echo "-- ${tool} version"
    command -v ${tool}
    ${tool} --version
    done
    $*
}

function container {
    PLAT_SUFFIX=$1
    BASE_IMAGE=${BASE_IMAGE:-${DOCKER_IMAGES["${PLAT_SUFFIX}"]}}
    CONTAINER_NAME=${CONTAINER_NAME:-mlm}
    BUILD_CONTAINER_NAME=${BUILD_CONTAINER_NAME:-${CONTAINER_NAME}-build}
    CACHE_PATH=${CACHE_PATH:-~/cache}

    mkdir -p ${CACHE_PATH}/pip_cache_${PLAT_SUFFIX} ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}
    PODMAN_DEVICES_ARG=$([[ -e /var/run/cdi/nvidia.yaml ]] && echo "--device nvidia.com/gpu=all" || echo "")
    podman run --rm \
        --name ${CONTAINER_NAME}-build \
        --cap-add=SYS_PTRACE \
        --replace ${PODMAN_DEVICES_ARG} \
        --security-opt=label=disable \
        -v ${CACHE_PATH}/pip_cache_${PLAT_SUFFIX}:/root/.cache/pip \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/cache/apt \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/cache/dnf \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/lib/dnf \
        -v ${CACHE_PATH}/pkg_cache_${PLAT_SUFFIX}:/var/lib/apt \
        -v ${MEGATRON_ROOT}/..:/root \
        -d -it ${BASE_IMAGE} \
        bash

    podman exec -it ${CONTAINER_NAME}-build bash /root/megatron-lm/megatron/native/scripts/prepare.sh $*

    podman commit ${CONTAINER_NAME}-build ${CONTAINER_NAME}-image
    podman run --name ${CONTAINER_NAME} --shm-size=34g --cap-add=SYS_PTRACE --replace ${PODMAN_DEVICES_ARG} --security-opt=label=disable -v ${MEGATRON_ROOT}/..:/root -d -it ${CONTAINER_NAME}-image bash
    podman stop ${CONTAINER_NAME}-build
}

function megatron {
    nvcc --version || cuda13
    nvcc --version
    export MAX_JOBS=4
    export CC=gcc
    export CXX=g++
    export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    export NVTE_CUDA_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//')
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv cache clear
    uv pip install --system --break-system-packages pybind11 poetry poetry_dynamic_versioning grpcio-tools 
    uv pip install --system --break-system-packages --no-build-isolation -e "${MEGATRON_ROOT}[build,training,dev]"
    uv pip install --system --break-system-packages --no-build-isolation -e "${MEGATRON_ROOT}[te,ssm]"
    uv pip install --system --break-system-packages --no-build-isolation tiktoken flash-attn # Needed for running models
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
        cuda-toolkit-13-1
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

    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
}

if [ "$#" -eq 0 ]; then
    container ubuntu2404 megatron test
else
    $*
fi
