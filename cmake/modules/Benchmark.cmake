cmake_policy(SET CMP0168 NEW)
include_guard()
include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_INSTALL_DOCS OFF)
find_package(PFM QUIET)
if (PFM_FOUND)
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    SOURCE_SUBDIR  .
    GIT_PROGRESS FALSE GIT_SHALLOW ON SYSTEM
    FIND_PACKAGE_ARGS NAMES benchmark
    ${BENCHMARK_FETCHCONTENT_OPTIONS}
)
if (COMMAND vcpkg_install)
    vcpkg_install(benchmark)
endif()

FetchContent_MakeAvailable(benchmark)
