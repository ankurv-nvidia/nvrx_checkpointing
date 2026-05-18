cmake_minimum_required(VERSION 3.28)
project(phoenix)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
set(VCPKG_AUTOCLONE_COMMIT "2026.02.27" CACHE STRING "The commit to clone vcpkg at if auto-cloning is enabled")
set(VCPKG_ROOT_FALLBACK_TO_BINARY_DIR ON CACHE BOOL "Allow vcpkg to be found in the binary directory if not found in the source directory")

include(CMakeDependentOption)

# Valgrind build mode: triggered by env var PHOENIX_BUILD_FOR_VALGRIND=1
# (set by ci/scripts/run_job.py for the `valgrind` job). Two effects:
#   1. Force CPPFORGE_DISABLE_MARCH_NATIVE so BuildEnv.cmake omits -march=native
#      (valgrind doesn't model host-specific instructions and will SIGILL).
#   2. Define PHOENIX_BUILD_FOR_VALGRIND=1 globally so headers like
#      src/checkpoint/messages.h can shrink ring-buffer sizes and thread counts
#      to keep valgrind's address-space and shadow-memory footprint manageable.
if (DEFINED ENV{PHOENIX_BUILD_FOR_VALGRIND} AND NOT "$ENV{PHOENIX_BUILD_FOR_VALGRIND}" STREQUAL "" AND NOT "$ENV{PHOENIX_BUILD_FOR_VALGRIND}" STREQUAL "0")
    message(STATUS "PHOENIX_BUILD_FOR_VALGRIND is set: disabling -march=native and shrinking D2H buffer sizes")
    set(CPPFORGE_DISABLE_MARCH_NATIVE ON)
    add_compile_definitions(PHOENIX_BUILD_FOR_VALGRIND=1)
endif()

include(BuildEnv)
include(vcpkg)
include(CheckLanguage)
check_language(CUDA)


vcpkg_init()

find_package(CUDAToolkit QUIET)
# find METAL if on macOS
set(METAL_FOUND FALSE)
if (APPLE)
    find_library(METAL_LIBRARY Metal)
    if (METAL_LIBRARY)
        message(STATUS "Found Metal library at ${METAL_LIBRARY}")
        set(METAL_FOUND TRUE)
    endif()
endif()

option(PHOENIX_BUILD_PYTHON_BINDINGS    "Build Python Bindings"         ON)
option(PHOENIX_BUILD_TORCH_BINDINGS     "Build PyTorch Bindings"        ON)
option(PHOENIX_BUILD_BENCHMARK          "Build benchmark tools"         ON)
option(PHOENIX_BUILD_TESTS              "Build Catch2 test executable"  OFF)

# The pure-C++ distcp writer no longer requires Python; the Python bindings
# block stays optional. Disabling it skips `_distcp_shim` and the pytest-driven
# parity/round-trip tests but keeps the server + ckpttest build working.

# set dependent default to ON if cuda is found
cmake_dependent_option(PHOENIX_BUILD_CUDA   "Build CUDA Support"    ON CUDAToolkit_FOUND    OFF)
cmake_dependent_option(PHOENIX_BUILD_METAL  "Build Metal Support"   ON METAL_FOUND          OFF)

set(PYTHON_VENV_USE_SYSTEM_PACKAGES ON)

set(CMAKE_CXX_STANDARD 26)
EnableStrictCompilation()
InitClangDIntellisense()

include_directories(include)

set(BOOST_INCLUDE_LIBRARIES interprocess process dll asio program_options url beast iostreams)

include(fmt)
include(Boost)
include(hwloc)
include(miniz)

#find_package(Boost CONFIG REQUIRED OPTIONAL_COMPONENTS interprocess)

add_library(phoenix_checkpoint OBJECT src/checkpoint/checkpoint.cpp)

function(target_is_python_extension target)
    if (APPLE)
        find_package(Python3 COMPONENTS Development.Embed Development.SABIModule REQUIRED QUIET)
        message(VERBOSE "Linking with Python Libraries to ensure no undefined symbols remain ...
            Python3_LIBRARIES=${Python3_LIBRARIES}
            Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}
            Python3_SABI_LIBRARIES=${Python3_SABI_LIBRARIES}
            Python3_SABI_LIBRARY_DIRS=${Python3_SABI_LIBRARY_DIRS}
            Python3_SABI_INCLUDE_DIRS=${Python3_SABI_INCLUDE_DIRS}
            Python3_SABI_LIBRARY=${Python3_SABI_LIBRARY}")
        target_link_libraries(${target} PRIVATE Python3::Python Python3::SABIModule)
    elseif (UNIX)
        find_program(PATCHELF patchelf REQUIRED)
        find_package(Python3 COMPONENTS Development.Embed Development.SABIModule REQUIRED QUIET)
        target_link_libraries(${target} PRIVATE Python3::Python Python3::SABIModule)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${PATCHELF} --remove-needed libpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.so.1.0 $<TARGET_FILE:${target}>
            COMMENT "Removing libpython dependency with patchelf..."
        )
    else()
        message(FATAL_ERROR "Unsupported platform ${CMAKE_SYSTEM_NAME} for target_is_python_extension")
    endif()
endfunction()

target_include_directories(phoenix_checkpoint PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/checkpoint
)

target_link_libraries(phoenix_checkpoint PUBLIC
    fmt::fmt-header-only
    Boost::interprocess Boost::process Boost::dll Boost::asio Boost::beast Boost::url
    hwloc::hwloc
)

# target_compile_definitions(phoenix_checkpoint PUBLIC SHM_RING_BUFFER_ENABLE_PERF_METRICS=1)
#target_compile_definitions(phoenix_checkpoint PUBLIC $<$<CONFIG:Debug>:SHM_RING_BUFFER_ENABLE_PERF_METRICS>)

if (PHOENIX_BUILD_CUDA)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDAToolkit:
            CUDAToolkit_VERSION=${CUDAToolkit_VERSION}
            CUDAToolkit_BIN_DIR=${CUDAToolkit_BIN_DIR}
            CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}
            CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}
            CUDAToolkit_LIBRARY_ROOT=${CUDAToolkit_LIBRARY_ROOT}
            CUDAToolkit_TARGET_DIR=${CUDAToolkit_TARGET_DIR}
            CUDAToolkit_NVCC_EXECUTABLE=${CUDAToolkit_NVCC_EXECUTABLE}")
    set(CMAKE_CUDA_FLAGS -allow-unsupported-compiler)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    check_language(CUDA)
    enable_language(CUDA)
    target_sources(phoenix_checkpoint PRIVATE src/checkpoint/d2h_cuda.cpp)
    target_compile_definitions(phoenix_checkpoint PRIVATE PHOENIX_HAVE_CUDA=1)
    target_link_libraries(phoenix_checkpoint PUBLIC CUDA::cudart_static)
endif()

if (PHOENIX_BUILD_METAL)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    message(STATUS "Linking with Metal library at ${METAL_LIBRARY}")
    target_sources(phoenix_checkpoint PRIVATE src/checkpoint/d2h_mtl.mm)
    target_compile_definitions(phoenix_checkpoint PRIVATE PHOENIX_HAVE_METAL=1)
    target_link_libraries(phoenix_checkpoint PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
endif()

target_sources(phoenix_checkpoint PRIVATE src/checkpoint/d2h_cpu.cpp)
target_compile_definitions(phoenix_checkpoint PRIVATE PHOENIX_HAVE_CPU=1)

# D2H Bandwidth Benchmark Tool
if (PHOENIX_BUILD_CUDA)
    if (PHOENIX_BUILD_BENCHMARK)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler" CACHE STRING "" FORCE)
        add_executable(d2h_benchmark src/benchmark/d2h_bandwidth_benchmark.cu)
        set_target_properties(d2h_benchmark PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES 80
        )
        target_link_libraries(d2h_benchmark PRIVATE CUDA::cudart)
    endif()
endif()

if (PHOENIX_BUILD_BENCHMARK OR PHOENIX_BUILD_TESTS)
    add_library(phoenix_checkpoint_fixture STATIC src/benchmark/ckpt_fixture.cpp)
    # phoenix_checkpoint_distcp is listed alongside phoenix_checkpoint_server because
    # CMake only propagates OBJECT-library object files one consumer hop deep — the
    # transitive path fixture -> server -> distcp would otherwise leave distcp.cpp.o
    # out of this static archive.
    target_link_libraries(phoenix_checkpoint_fixture PUBLIC
        phoenix_checkpoint
        phoenix_checkpoint_server
        phoenix_checkpoint_distcp
        fmt::fmt-header-only
        Boost::interprocess
        Boost::process
        Boost::program_options
    )
endif()

if (PHOENIX_BUILD_BENCHMARK)
    include(Benchmark)
    SupressWarningForTarget(benchmark)
    add_executable(ckptbench src/benchmark/ckptbench.cpp)
    target_link_libraries(ckptbench PRIVATE phoenix_checkpoint_fixture benchmark::benchmark)

    add_executable(shm_ipc_benchmark src/benchmark/shm_ipc_benchmark.cpp)
    target_include_directories(shm_ipc_benchmark PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/src/checkpoint)
    target_link_libraries(shm_ipc_benchmark PRIVATE fmt::fmt-header-only Boost::interprocess Boost::process Boost::program_options)
endif()

if (PHOENIX_BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
    include(PythonEnv)
    pip_install(nanobind)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
        OUTPUT_VARIABLE nanobind_ROOT
    )
    find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(checkpoint
        FREE_THREADED STABLE_ABI NB_DOMAIN checkpoint
        src/python/bindings.cpp)
    SupressWarningForTarget(checkpoint)
    SupressWarningForTarget(nanobind-static)
    SupressWarningForTarget(nanobind-static-abi3)
    target_link_libraries(checkpoint PRIVATE phoenix_checkpoint fmt::fmt-header-only)
    target_is_python_extension(checkpoint)

    if (PHOENIX_BUILD_TORCH_BINDINGS)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import os; import torch; print(os.path.dirname(torch.__file__) + '/share/cmake/Torch')"
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE Torch_DIR
        )
        find_package(Torch CONFIG QUIET)
        if (NOT Torch_FOUND)
            if (PHOENIX_BUILD_CUDA)
                set(pytorch_index_url "https://download.pytorch.org/whl/cu${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}")
                message(STATUS "Installing CUDA-enabled PyTorch from ${pytorch_index_url} into venv for Python bindings...")
                pip_install(torch --index-url ${pytorch_index_url})
                set(TORCH_CUDA_ARCH_LIST "8.0")
            else()
                pip_install(torch --index-url https://download.pytorch.org/whl/cpu)
            endif()
        endif()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import os; import torch; print(os.path.dirname(torch.__file__) + '/share/cmake/Torch')"
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE Torch_DIR
        )
        find_package(Torch CONFIG REQUIRED)
        if (Torch_FOUND)
            find_library(torch_python_LIB NAMES torch_python REQUIRED PATHS ${Torch_DIR}/../../../lib)
            target_sources(checkpoint PRIVATE src/python/torch_bindings.cpp)
            target_compile_definitions(checkpoint PRIVATE HAVE_TORCH=1)
            target_include_directories(checkpoint PRIVATE ${TORCH_INCLUDE_DIRS})
            target_link_libraries(checkpoint PRIVATE ${TORCH_LIBRARIES} ${torch_python_LIB})
            target_compile_definitions(checkpoint PRIVATE ${TORCH_CXX_FLAGS})
        endif()
    endif()

    add_custom_command(TARGET checkpoint POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove $<PATH:APPEND,${CMAKE_CURRENT_LIST_DIR},$<TARGET_FILE_NAME:checkpoint>>
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:checkpoint>
            $<PATH:APPEND,${CMAKE_CURRENT_LIST_DIR},$<TARGET_FILE_NAME:checkpoint>>
        COMMENT "Creating symbolic link..."
    )
endif()

if (COMMAND vcpkg_install)
    vcpkg_install(openssl)
endif()

find_package(OpenSSL CONFIG REQUIRED)

# Writer library — pure-C++ ZIP+pickle emitter (no Python, no nanobind, no
# libtorch). Both the production server and the parity-test shim link against
# this. Format spec lives in the header comment at the top of
# `src/checkpoint/distcp.cpp`.
#
# NOTE: OBJECT library so the same .o files feed both `checkpoint_server` and
# `_distcp_shim`. CMake propagates OBJECT-lib object files one consumer hop
# deep, so STATIC/EXECUTABLE consumers more than one level removed must list
# this target directly in their `target_link_libraries` (see
# `phoenix_checkpoint_fixture` and `checkpoint_server` below). Promoting to
# STATIC + POSITION_INDEPENDENT_CODE=ON would let transitive PUBLIC
# propagation handle this naturally; defer that until a third consumer hits
# the same wall.
add_library(phoenix_checkpoint_distcp OBJECT src/checkpoint/distcp.cpp)
target_link_libraries(phoenix_checkpoint_distcp PUBLIC
    phoenix_checkpoint
    fmt::fmt-header-only
    miniz::miniz)

add_library(phoenix_checkpoint_server OBJECT src/checkpoint/server.cpp)
target_link_libraries(phoenix_checkpoint_server PUBLIC phoenix_checkpoint phoenix_checkpoint_distcp fmt::fmt-header-only Boost::interprocess Boost::iostreams Boost::url OpenSSL::Crypto)

add_executable(checkpoint_server src/checkpoint/server_main.cpp)
# phoenix_checkpoint_distcp is listed directly because CMake only propagates
# OBJECT-lib object files one consumer hop — without this, distcp.cpp.o is
# missing from the executable even though server.cpp references it.
target_link_libraries(checkpoint_server PUBLIC phoenix_checkpoint_server phoenix_checkpoint_distcp)

add_custom_command(TARGET checkpoint_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove $<PATH:APPEND,${CMAKE_CURRENT_LIST_DIR},$<TARGET_FILE_NAME:checkpoint_server>>
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE:checkpoint_server>
        $<PATH:APPEND,${CMAKE_CURRENT_LIST_DIR},$<TARGET_FILE_NAME:checkpoint_server>>
    COMMENT "Creating symbolic link..."
)

if (PHOENIX_BUILD_TESTS)
    include(Catch2)
    enable_testing()
    include(Catch)

    add_executable(ckpttest src/benchmark/ckpttest.cpp)
    # Plain `Catch2::Catch2` (not WithMain) — ckpttest.cpp owns main() so it
    # can register `--registry` on the Catch2 CLI.
    target_link_libraries(ckpttest PRIVATE phoenix_checkpoint_fixture Catch2::Catch2)
    # When the venv python is available, ckpttest can shell out to it for the
    # `.metadata` unpickle smoke test. Without the venv, those tests are
    # silently skipped at compile time.
    if (PHOENIX_BUILD_PYTHON_BINDINGS AND PYTHON_VIRTUAL_ENV_PATH)
        target_compile_definitions(ckpttest PRIVATE
            PHOENIX_VENV_PYTHON="${PYTHON_VIRTUAL_ENV_PATH}/bin/python"
        )
    endif()
    # The `[round-trip]` test reads its state registry from `--registry`. ctest
    # gets it via EXTRA_ARGS; hand-runs need `ckpttest --registry <path>`.
    catch_discover_tests(ckpttest
        EXTRA_ARGS
            --registry "${CMAKE_CURRENT_SOURCE_DIR}/metadata/1x2_checkpoints_iter_0000250/registry.bin"
    )

    if (PHOENIX_BUILD_PYTHON_BINDINGS)
        # Parity-test shim: a separate nanobind module that exposes
        # `WriteRankShard` to Python with no server / IPC machinery. Backs
        # tests/test_distcp_roundtrip.py.
        nanobind_add_module(_distcp_shim
            FREE_THREADED STABLE_ABI NB_DOMAIN _distcp_shim
            src/python/distcp_shim.cpp)
        SupressWarningForTarget(_distcp_shim)
        target_link_libraries(_distcp_shim PRIVATE
            phoenix_checkpoint_distcp fmt::fmt-header-only)
        target_is_python_extension(_distcp_shim)

        add_custom_command(TARGET _distcp_shim POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove $<PATH:APPEND,${CMAKE_CURRENT_LIST_DIR},$<TARGET_FILE_NAME:_distcp_shim>>
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE:_distcp_shim>
                $<PATH:APPEND,${CMAKE_CURRENT_LIST_DIR},$<TARGET_FILE_NAME:_distcp_shim>>
            COMMENT "Creating symbolic link for _distcp_shim..."
        )

        # Wire Python tests into ctest via pytest-cmake.
        pip_install(pytest-cmake hypothesis)
        set(Pytest_ROOT      "${PYTHON_VIRTUAL_ENV_PATH}/share/Pytest")
        # FORCE because find_program may have cached NOTFOUND on an earlier
        # configure run before the venv had pytest installed.
        set(PYTEST_EXECUTABLE "${PYTHON_VIRTUAL_ENV_PATH}/bin/pytest" CACHE FILEPATH "Path to pytest" FORCE)
        find_package(Pytest REQUIRED)

        # DCP load round-trip via the shim — gold-standard format-compat test.
        # Writes a checkpoint dir end-to-end (WriteRankShard + WriteMetadata),
        # then reads it back through `torch.distributed.checkpoint.load`.
        pytest_discover_tests(dcp_load
            TEST_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dcp_load.py
            PYTHON_PATH_PREPEND ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS _distcp_shim
            ENVIRONMENT
                MEGATRON_NATIVE_BUILD_DIR=/tmp/megatron-native-autobuild-disabled
        )

        # Per-tensor `torch.load` round-trip via the shim — independent check
        # that each per-tensor ZIP slice decodes correctly.
        pytest_discover_tests(distcp_roundtrip
            TEST_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_distcp_roundtrip.py
            PYTHON_PATH_PREPEND ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS _distcp_shim
            ENVIRONMENT
                MEGATRON_NATIVE_BUILD_DIR=/tmp/megatron-native-autobuild-disabled
        )
    endif()
endif()
