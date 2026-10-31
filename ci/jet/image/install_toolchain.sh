#!/usr/bin/env bash
set -Eeuo pipefail

clang_version=22
gcc_version=13
ubuntu_codename=noble

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

rm -f /etc/apt/apt.conf.d/docker-clean
apt-get update
apt-get install --yes --no-install-recommends wget build-essential ca-certificates curl

wget https://apt.llvm.org/llvm-snapshot.gpg.key -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc
wget https://apt.kitware.com/keys/kitware-archive-latest.asc -O /etc/apt/trusted.gpg.d/kitware-archive-latest.asc
echo "deb http://apt.llvm.org/${ubuntu_codename}/ llvm-toolchain-${ubuntu_codename}-${clang_version} main" \
    > /etc/apt/sources.list.d/llvm.list
echo "deb https://apt.kitware.com/ubuntu/ ${ubuntu_codename} main" \
    > /etc/apt/sources.list.d/cmake.list

packages=(
    gnupg
    build-essential
    wget
    curl
    ca-certificates
    "g++-${gcc_version}"
    "clang-${clang_version}"
    "clang-format-${clang_version}"
    "clangd-${clang_version}"
    "llvm-${clang_version}"
    gdb
    strace
    cmake
    mold
    ninja-build
    python3.12-venv
    python3.12-dev
    python3-pip
    zip
    xz-utils
    bzip2
    tar
    unzip
    python3
    git
    git-lfs
    openssh-client
    pkg-config
    patchelf
    valgrind
    libfindbin-libs-perl
    libipc-run-perl
    autoconf
    automake
    libtool
    rdma-core
    libibverbs-dev
)

apt-get update
apt-get install --yes --no-install-recommends "${packages[@]}"

# DeepEP consumes this RDMA and NVML development layout.
architecture_lib=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
if [[ ! -f "/usr/lib/${architecture_lib}/libmlx5.so" ]]; then
    ln -sf "/usr/lib/${architecture_lib}/libmlx5.so.1" "/usr/lib/${architecture_lib}/libmlx5.so"
fi
mkdir -p /opt/rdma-core/build
ln -sfn /usr/include /opt/rdma-core/build/include
ln -sfn "/usr/lib/${architecture_lib}" /opt/rdma-core/build/lib

# Extract only the NVML development payload; JET injects the runtime driver.
nvml_download_dir=$(mktemp -d)
(
    cd "${nvml_download_dir}"
    apt-get download libnvidia-ml-dev
    dpkg-deb -x libnvidia-ml-dev_*.deb /
)
rm -rf "${nvml_download_dir}"

update-alternatives --install /usr/bin/cc cc "/usr/bin/clang-${clang_version}" 100
update-alternatives --install /usr/bin/c++ c++ "/usr/bin/clang++-${clang_version}" 100
update-alternatives --install /usr/bin/gcc gcc "/usr/bin/gcc-${gcc_version}" 100
update-alternatives --install /usr/bin/g++ g++ "/usr/bin/g++-${gcc_version}" 100
update-alternatives --install /usr/bin/clangd clangd "/usr/bin/clangd-${clang_version}" 100
update-alternatives --install /usr/bin/clang-format clang-format "/usr/bin/clang-format-${clang_version}" 100
update-alternatives --install /usr/bin/python python /usr/bin/python3 1
update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.12-config 1

# The base image's /usr/local CMake shadows the apt-managed version.
rm -f /usr/local/bin/cmake /usr/local/bin/ctest /usr/local/bin/cpack

for tool in cmake cc gcc g++ "clang-${clang_version}"; do
    echo "-- ${tool} version"
    command -v "${tool}"
    "${tool}" --version
done
