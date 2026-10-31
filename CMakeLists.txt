cmake_minimum_required(VERSION 3.31)
project(phoenix)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg-commit.txt" VCPKG_AUTOCLONE_COMMIT LIMIT_COUNT 1)
set(VCPKG_ROOT_FALLBACK_TO_BINARY_DIR ON CACHE BOOL "Allow vcpkg  be found in the binary directory if not found in the source directory")

include(CMakeDependentOption)

include(BuildEnv)
include(vcpkg)
include(CheckLanguage)
check_language(CUDA)

foreach(overlay ports triplets)
    string(TOUPPER "VCPKG_OVERLAY_${overlay}" overlay_env)
    cmake_path(CONVERT "$ENV{${overlay_env}}" TO_CMAKE_PATH_LIST overlay_dirs)
    list(PREPEND overlay_dirs "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg-${overlay}")
    list(REMOVE_DUPLICATES overlay_dirs)
    cmake_path(CONVERT "${overlay_dirs}" TO_NATIVE_PATH_LIST overlay_dirs)
    set(ENV{${overlay_env}} "${overlay_dirs}")
endforeach()
set(ENV{VCPKG_C_COMPILER} "${CMAKE_C_COMPILER}")
set(ENV{VCPKG_CXX_COMPILER} "${CMAKE_CXX_COMPILER}")

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

option(PHOENIX_BUILD_PYTHON_BINDINGS    "Build Python Bindings"      ON)
option(PHOENIX_BUILD_WHEEL              "Build Python Wheel"         ON)
set(PHOENIX_WHEEL_VERSION "0.0.1" CACHE STRING "Python wheel version")

option(PHOENIX_BUILD_BENCHMARK          "Build benchmark tools"             ON)
option(PHOENIX_BUILD_TESTS              "Build Catch2 test executable"      OFF)
option(PHOENIX_REQUIRE_CUDA             "Fail if CUDA is unavailable"       OFF)

if (PHOENIX_BUILD_WHEEL AND NOT PHOENIX_BUILD_PYTHON_BINDINGS)
    message(FATAL_ERROR "PHOENIX_BUILD_WHEEL requires PHOENIX_BUILD_PYTHON_BINDINGS")
endif()

# The pure-C++ distcp writer no longer requires Python; the Python bindings
# block stays optional. Disabling it skips `_distcp_shim` and the pytest-driven
# parity/round-trip tests but keeps the server + ckpttest build working.

# set dependent default to ON if cuda is found
cmake_dependent_option(PHOENIX_BUILD_CUDA   "Build CUDA Support"    ON CUDAToolkit_FOUND    OFF)
cmake_dependent_option(PHOENIX_BUILD_METAL  "Build Metal Support"   ON METAL_FOUND          OFF)

# CUDA is autodetected, so a broken toolchain degrades to a CPU-only build
# silently. Builds that ship to GPU machines (the CI wheels) must fail the
# configure instead: a CPU-only DeviceContext maps device pointers into
# CPUContext::AddD2H and crashes at runtime.
if (PHOENIX_REQUIRE_CUDA AND NOT PHOENIX_BUILD_CUDA)
    message(FATAL_ERROR "PHOENIX_REQUIRE_CUDA is set but the CUDA toolkit was not found")
endif()

set(PYTHON_VENV_USE_SYSTEM_PACKAGES ON)

set(CMAKE_CXX_STANDARD 26)
EnableStrictCompilation()
InitClangDIntellisense()

include_directories(include)

set(BOOST_INCLUDE_LIBRARIES interprocess process dll asio program_options url beast iostreams)

include(Boost)
include(fmt)
include(hwloc)
include(miniz)
include(OpenTelemetry)
include(spdlog)
include(zlib)

#find_package(Boost CONFIG REQUIRED OPTIONAL_COMPONENTS interprocess)

add_library(phoenix_checkpoint OBJECT
    src/checkpoint/checkpoint.cpp
    src/checkpoint/logtel.cpp
    src/checkpoint/quant.cpp
)

target_include_directories(phoenix_checkpoint PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/checkpoint
)

target_link_libraries(phoenix_checkpoint PUBLIC
    Boost::asio
    Boost::beast
    Boost::container
    Boost::dll
    Boost::interprocess
    Boost::process
    Boost::url
    fmt::fmt-header-only
    hwloc::hwloc
    opentelemetry-cpp::api
    opentelemetry-cpp::sdk
    opentelemetry-cpp::common
    opentelemetry-cpp::resources
    opentelemetry-cpp::trace
    opentelemetry-cpp::metrics
    opentelemetry-cpp::logs
    opentelemetry-cpp::otlp_recordable
    opentelemetry-cpp::otlp_file_client
    opentelemetry-cpp::otlp_file_exporter
    opentelemetry-cpp::otlp_file_metric_exporter
    opentelemetry-cpp::otlp_file_log_record_exporter
    opentelemetry-cpp::proto
    spdlog::spdlog_header_only
    ZLIB::ZLIB
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
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 23)
        target_compile_options(phoenix_checkpoint PRIVATE -Wno-allocator-wrappers)
    endif()
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
    # distcpio is listed alongside daemon because
    # CMake only propagates OBJECT-library object files one consumer hop deep — the
    # transitive path fixture -> server -> distcp would otherwise leave distcp.cpp.o
    # out of this static archive.
    target_link_libraries(phoenix_checkpoint_fixture PUBLIC
        phoenix_checkpoint
        daemon
        distcpio
        fmt::fmt-header-only
        Boost::interprocess
        Boost::process
        Boost::program_options
    )
endif()

if (PHOENIX_BUILD_BENCHMARK)
    include(Benchmark)
    SupressWarningForTarget(benchmark)
    SupressLintingForTarget(benchmark)
    add_executable(ckptbench src/benchmark/ckptbench.cpp)
    target_link_libraries(ckptbench PRIVATE phoenix_checkpoint_fixture benchmark::benchmark)

    add_executable(shm_ipc_benchmark src/benchmark/shm_ipc_benchmark.cpp)
    target_include_directories(shm_ipc_benchmark PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/src/checkpoint)
    target_link_libraries(shm_ipc_benchmark PRIVATE fmt::fmt-header-only Boost::interprocess Boost::process Boost::program_options)

    # Connection-layer scale test for the checkpoint web service. Reuses the real
    # in-process server (via the fixture -> server object files) and drives it with
    # a Boost.Beast async connect storm.
    add_executable(ckpt_stress_test src/benchmark/ckpt_stress_test.cpp)
    target_link_libraries(ckpt_stress_test PRIVATE
        phoenix_checkpoint_fixture
        Boost::asio Boost::beast Boost::program_options
        fmt::fmt-header-only)
endif()

if (PHOENIX_BUILD_PYTHON_BINDINGS)
    include(DLPack)

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
        FREE_THREADED STABLE_ABI NB_DOMAIN LTO NB_SUPPRESS_WARNINGS
        src/python/bindings.cpp
        src/python/torch_bindings.cpp)
    SupressWarningForTarget(checkpoint)
    SupressWarningForTarget(nanobind-static)
    SupressWarningForTarget(nanobind-static-abi3)
    SupressLintingForTarget(nanobind-static)
    SupressLintingForTarget(nanobind-static-abi3)
    target_link_libraries(checkpoint PRIVATE phoenix_checkpoint fmt::fmt-header-only dlpack::dlpack )
    target_is_python_extension(checkpoint)
    if (PHOENIX_BUILD_TESTS)
        if (PHOENIX_BUILD_CUDA)
            set(pytorch_index_url "https://download.pytorch.org/whl/cu${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}")
            message(STATUS "Installing CUDA-enabled PyTorch from ${pytorch_index_url} into venv for Python bindings...")
            pip_install(torch --index-url ${pytorch_index_url})
            set(TORCH_CUDA_ARCH_LIST "8.0")
        else()
            pip_install(torch --index-url https://download.pytorch.org/whl/cpu)
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
    vcpkg_install(cpptrace)
endif()

find_package(OpenSSL CONFIG REQUIRED)
# cpptrace symbolizes the checkpoint_server crash backtrace (demangled, file:line).
find_package(cpptrace CONFIG REQUIRED)

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
add_library(distcpio OBJECT
    src/checkpoint/distcp.cpp
    src/checkpoint/host_dequant.cpp)
target_link_libraries(distcpio PUBLIC
    phoenix_checkpoint
    fmt::fmt-header-only
    miniz::miniz
    ZLIB::ZLIB)

add_library(daemon OBJECT
    src/checkpoint/server.cpp)
# miniz is pulled in transitively via distcpio (PUBLIC above).
target_link_libraries(daemon PUBLIC phoenix_checkpoint distcpio fmt::fmt-header-only Boost::interprocess Boost::iostreams Boost::url OpenSSL::Crypto)

add_executable(checkpoint_server src/checkpoint/server_main.cpp)
# distcpio is listed directly because CMake only propagates
# OBJECT-lib object files one consumer hop — without this, distcp.cpp.o is
# missing from the executable even though server.cpp references it.
target_link_libraries(checkpoint_server PUBLIC phoenix_checkpoint daemon distcpio cpptrace::cpptrace)

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
    catch_discover_tests(ckpttest)

    if (PHOENIX_BUILD_PYTHON_BINDINGS)
        # Parity-test shim: a separate nanobind module that exposes
        # `WriteRankShard` to Python with no server / IPC machinery. Backs
        # tests/test_distcp_roundtrip.py.
        nanobind_add_module(_distcp_shim
            FREE_THREADED STABLE_ABI NB_DOMAIN _distcp_shim
            src/python/distcp_shim.cpp)
        SupressWarningForTarget(_distcp_shim)
        target_link_libraries(_distcp_shim PRIVATE
            distcpio fmt::fmt-header-only)
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

        pytest_discover_tests(quantized_snapshot
            TEST_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantized_snapshot.py
            PYTHON_PATH_PREPEND ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS checkpoint checkpoint_server
            ENVIRONMENT
                MEGATRON_NATIVE_BUILD_DIR=/tmp/megatron-native-autobuild-disabled
        )

    endif()
endif()

if (PHOENIX_BUILD_WHEEL AND PHOENIX_BUILD_PYTHON_BINDINGS)
    pip_install(cmake-pywhl)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import cmake_pywhl.build,pathlib; print(pathlib.Path(cmake_pywhl.build.__file__).parent)"
        OUTPUT_VARIABLE PyWhl_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )

    find_package(PyWhl CONFIG REQUIRED)
    add_pywhl_module(nvrx_checkpointing
        TARGETS
            checkpoint_server
            checkpoint
        SCRIPTS
            __init__.py
            ckpt_diag.py
            debug.py
            megatron@megatron
            megatron_stubs@megatron_stubs
            quantized_snapshot.py
            metadata_compare.py
            peers.py
            telemetry.py
    )

    add_pywhl_package(nvrx_checkpointing
        VERSION "${PHOENIX_WHEEL_VERSION}"
        MODULES nvrx_checkpointing
        LICENSE_FILE LICENSE
        WHL_FILE_OUTVAR PHOENIX_WHEEL_FILE
    )
    # Buildable wheel target for CI. The pywhl `nvrx_checkpointing` target
    # names its ninja depfile as a target-level dependency, which a fresh
    # Ninja tree rejects (nothing produces the depfile until the packaging
    # command has run once); `all_whl` additionally editable-installs into
    # the configuring interpreter. Depending on the wheel file itself avoids
    # both.
    add_custom_target(nvrx_checkpointing_whl DEPENDS "${PHOENIX_WHEEL_FILE}")
endif()
