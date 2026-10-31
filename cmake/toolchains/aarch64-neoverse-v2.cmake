# Cross-compiles for aarch64 Grace (Neoverse V2) on an x86_64 host prepared with `scripts/prepare.sh cross_arm64`.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_C_FLAGS_INIT "-mcpu=neoverse-v2")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=neoverse-v2")

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /usr/local/cuda-13.4/targets/sbsa-linux)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
