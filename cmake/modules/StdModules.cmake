include_guard(GLOBAL)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "`import std` needs clang with libc++, but the C++ compiler is ${CMAKE_CXX_COMPILER_ID}")
endif()
if (NOT CMAKE_GENERATOR MATCHES "^Ninja")
    message(FATAL_ERROR "C++ modules need a Ninja generator; reconfigure with -G Ninja")
endif()

string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
# CUDA-linked targets keep libstdc++; static libc++ keeps the other binaries free of a libc++ runtime dependency.
add_link_options("$<$<LINK_LANGUAGE:CXX>:-stdlib=libc++;-static-libstdc++>")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

set(_std_modules_query "${CMAKE_CXX_COMPILER}" -stdlib=libc++)
if (CMAKE_CXX_COMPILER_TARGET)
    list(APPEND _std_modules_query "${CMAKE_CXX_COMPILE_OPTIONS_TARGET}${CMAKE_CXX_COMPILER_TARGET}")
endif()
if (CMAKE_SYSROOT)
    list(APPEND _std_modules_query "${CMAKE_CXX_COMPILE_OPTIONS_SYSROOT}${CMAKE_SYSROOT}")
endif()
execute_process(
    COMMAND ${_std_modules_query} -print-file-name=libc++.modules.json
    OUTPUT_VARIABLE _std_modules_manifest
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
if (NOT IS_ABSOLUTE "${_std_modules_manifest}")
    message(FATAL_ERROR "libc++.modules.json not found; install libc++-${CMAKE_CXX_COMPILER_VERSION_MAJOR}-dev and libc++abi-${CMAKE_CXX_COMPILER_VERSION_MAJOR}-dev (scripts/prepare.sh)")
endif()

file(READ "${_std_modules_manifest}" _std_modules_json)
cmake_path(GET _std_modules_manifest PARENT_PATH _std_modules_manifest_dir)
string(JSON _std_modules_count LENGTH "${_std_modules_json}" modules)
math(EXPR _std_modules_last "${_std_modules_count} - 1")
foreach (_std_module_index RANGE ${_std_modules_last})
    string(JSON _std_module_source GET "${_std_modules_json}" modules ${_std_module_index} source-path)
    cmake_path(ABSOLUTE_PATH _std_module_source BASE_DIRECTORY "${_std_modules_manifest_dir}" NORMALIZE)
    cmake_path(GET _std_module_source PARENT_PATH _std_module_source_dir)
    list(APPEND _std_modules_sources "${_std_module_source}")
    list(APPEND _std_modules_base_dirs "${_std_module_source_dir}")
    string(JSON _std_module_include_dirs ERROR_VARIABLE _std_module_include_dirs_missing
        GET "${_std_modules_json}" modules ${_std_module_index} local-arguments system-include-directories)
    if (NOT _std_module_include_dirs_missing)
        string(JSON _std_module_include_dirs_count LENGTH "${_std_module_include_dirs}")
        math(EXPR _std_module_include_dirs_last "${_std_module_include_dirs_count} - 1")
        foreach (_std_module_include_dir_index RANGE ${_std_module_include_dirs_last})
            string(JSON _std_module_include_dir GET "${_std_module_include_dirs}" ${_std_module_include_dir_index})
            cmake_path(ABSOLUTE_PATH _std_module_include_dir BASE_DIRECTORY "${_std_modules_manifest_dir}" NORMALIZE)
            list(APPEND _std_modules_include_dirs "${_std_module_include_dir}")
        endforeach()
    endif()
endforeach()
list(REMOVE_DUPLICATES _std_modules_base_dirs)
list(REMOVE_DUPLICATES _std_modules_include_dirs)
message(STATUS "C++ standard library modules: ${_std_modules_sources}")

add_library(phoenix_std STATIC)
target_sources(phoenix_std PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS ${_std_modules_base_dirs}
    FILES ${_std_modules_sources}
)
target_include_directories(phoenix_std SYSTEM PRIVATE ${_std_modules_include_dirs})
target_compile_options(phoenix_std PRIVATE -Wno-reserved-module-identifier)
SupressWarningForTarget(phoenix_std)
SupressLintingForTarget(phoenix_std)
