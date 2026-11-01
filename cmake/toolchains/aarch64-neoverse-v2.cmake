# Cross-compiles for aarch64 Grace (Neoverse V2) on an x86_64 host prepared with `scripts/prepare.sh cross_arm64`.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../scripts/prepare.sh" _nvrx_versions REGEX "^(CLANG|CUDA)_VERSION=")
foreach(_nvrx_version IN LISTS _nvrx_versions)
    string(REGEX MATCH "^([A-Z]+)_VERSION=(.+)$" _nvrx_match "${_nvrx_version}")
    set(_nvrx_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
endforeach()

set(CMAKE_C_COMPILER clang-${_nvrx_CLANG})
set(CMAKE_CXX_COMPILER clang++-${_nvrx_CLANG})
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_C_FLAGS_INIT "-mcpu=neoverse-v2")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=neoverse-v2")

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /usr/local/cuda-${_nvrx_CUDA}/targets/sbsa-linux)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
