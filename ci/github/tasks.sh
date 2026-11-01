#!/usr/bin/env bash
# CI tasks for the self-hosted build runner, which carries the toolchain from
# `scripts/prepare.sh ubuntu2404 cuda13`.
#
#   tasks.sh wheel | wheel-aarch64 | pytest | ctest
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

cache=${NVRX_CI_CACHE:-${HOME}/.cache/nvrx-ci}
scratch=${RUNNER_TEMP:-${repo_root}/build-ci}
venv=${scratch}/venv
python=${venv}/bin/python

source <(grep -E '^(CUDA|CLANG)_VERSION=' scripts/prepare.sh)
export CC=clang-${CLANG_VERSION} CXX=clang++-${CLANG_VERSION}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-${CUDA_VERSION}}
export PATH=${CUDA_HOME}/bin:${PATH}
# vcpkg lives in the job's scratch space; only its binary and download caches persist.
export VCPKG_ROOT=${scratch}/vcpkg
export VCPKG_BINARY_SOURCES="clear;files,${cache}/vcpkg-binary,readwrite"
export VCPKG_DOWNLOADS=${cache}/vcpkg-downloads
# Package downloads ride out brief network outages.
export UV_HTTP_RETRIES=10
mkdir -p "${cache}/vcpkg-binary" "${VCPKG_DOWNLOADS}"

make_venv() {
    uv venv --quiet --seed --python /usr/bin/python3.12 "${venv}"
    uv pip install --quiet --python "${python}" nanobind 'cmake-pywhl>=0.0.4' pytest pytest-cmake hypothesis numpy pyyaml
}

configure() {
    local build_dir=$1
    shift
    cmake -B "${build_dir}" -S . -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DPython3_EXECUTABLE="${python}" \
        -DBUILDENV_DISABLE_MARCH_NATIVE=ON \
        "$@"
}

wheel_version() {
    echo "${PHOENIX_WHEEL_VERSION:-0.0.1.dev${GITHUB_RUN_NUMBER:-0}+g$(git rev-parse --short=8 HEAD)}"
}

wheel() {
    make_venv
    configure build-wheel \
        -DPHOENIX_BUILD_WHEEL=ON \
        -DPHOENIX_WHEEL_VERSION="$(wheel_version)" \
        -DPHOENIX_REQUIRE_CUDA=ON \
        -DPHOENIX_BUILD_BENCHMARK=OFF \
        -DPHOENIX_BUILD_TESTS=OFF
    cmake --build build-wheel -j"$(nproc)" --target nvrx_checkpointing_whl
    configure build-dev \
        -DPHOENIX_BUILD_WHEEL=OFF \
        -DPHOENIX_BUILD_BENCHMARK=OFF \
        -DPHOENIX_BUILD_TESTS=ON \
        -DPHOENIX_BUILD_CUDA=OFF
    cmake --build build-dev -j"$(nproc)" --target ckpttest _distcp_shim
    mkdir -p dist dev_binaries
    cp build-wheel/*-abi3-*.whl dist/
    cp build-dev/ckpttest build-dev/_distcp_shim*.so dev_binaries/
    ls -la dist dev_binaries
}

# The aarch64 wheel links against a downloaded aarch64 CPython's headers and library.
wheel_aarch64() {
    local python_root=${cache}/python-aarch64/cpython-3.12-linux-aarch64-gnu
    # vcpkg's make-based ports prefer an exported compiler over the triplet's cross compiler.
    unset CC CXX
    make_venv
    uv python install --quiet --no-bin --install-dir "${cache}/python-aarch64" cpython-3.12-linux-aarch64-gnu
    configure build-wheel-aarch64 \
        -DCMAKE_TOOLCHAIN_FILE="${repo_root}/cmake/toolchains/aarch64-neoverse-v2.cmake" \
        -DPython3_INCLUDE_DIR="${python_root}/include/python3.12" \
        -DPython3_LIBRARY="${python_root}/lib/libpython3.12.so" \
        -DPython_EXECUTABLE="${python}" \
        -DPython_INCLUDE_DIR="${python_root}/include/python3.12" \
        -DPython_LIBRARY="${python_root}/lib/libpython3.12.so" \
        -DPHOENIX_BUILD_WHEEL=ON \
        -DPHOENIX_WHEEL_VERSION="$(wheel_version)" \
        -DPHOENIX_REQUIRE_CUDA=ON \
        -DPHOENIX_BUILD_BENCHMARK=OFF \
        -DPHOENIX_BUILD_TESTS=OFF
    cmake --build build-wheel-aarch64 -j"$(nproc)" --target nvrx_checkpointing_whl
    mkdir -p dist
    cp build-wheel-aarch64/*-abi3-linux_aarch64.whl dist/
    ls -la dist
}

wheel_pytest() {
    make_venv
    uv pip install --quiet --python "${python}" torch --index-url https://download.pytorch.org/whl/cpu
    "${python}" ci/scripts/wheel_install.py --from-dist dist
    mkdir -p ci/ci-results
    export PYTHONPATH=${repo_root}/dev_binaries
    "${python}" -m pip show nvrx_checkpointing
    "${python}" -c "import nvrx_checkpointing as m; print(m.SHLIB_PATH, m.SERVER_PATH)"
    "${python}" -c "import nvrx_checkpointing.debug"
    "${python}" -m nvrx_checkpointing.metadata_compare --help >/dev/null
    "${python}" -m pytest tests -v --junitxml=ci/ci-results/pytest.xml
    chmod +x dev_binaries/ckpttest
    ./dev_binaries/ckpttest
}

ctest_gate() {
    make_venv
    "${python}" ci/scripts/wheel_install.py --from-dist dist
    configure build-test \
        -DPHOENIX_BUILD_WHEEL=OFF \
        -DPHOENIX_BUILD_BENCHMARK=ON \
        -DPHOENIX_BUILD_TESTS=ON \
        -DPHOENIX_BUILD_CUDA=OFF
    cmake --build build-test -j"$(nproc)" --target ckpttest _distcp_shim ckpt_stress_test dcp_load distcp_roundtrip quantized_snapshot
    mkdir -p ci/ci-results
    export MEGATRON_NATIVE_BUILD_DIR=${repo_root}/build-test
    export PYTHONPATH=${repo_root}/build-test
    "${python}" -c "import torch; print(torch.__version__, torch.__file__)"
    ctest --test-dir build-test --output-on-failure --output-junit "${repo_root}/ci/ci-results/ctest.xml"
    "${python}" ci/scripts/ckpt_stress.py --gpus-per-node 2
}

case "${1:-}" in
    wheel) wheel ;;
    wheel-aarch64) wheel_aarch64 ;;
    pytest) wheel_pytest ;;
    ctest) ctest_gate ;;
    *) echo "usage: tasks.sh wheel|wheel-aarch64|pytest|ctest" >&2; exit 2 ;;
esac
