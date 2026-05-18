include_guard()

if (COMMAND vcpkg_install)
    vcpkg_install(hwloc)
endif()

find_path(hwloc_INCLUDE_DIR NAMES hwloc.h REQUIRED)
find_library(hwloc_LIBRARY  NAMES hwloc   REQUIRED)

add_library(hwloc::hwloc INTERFACE IMPORTED)
target_include_directories(hwloc::hwloc INTERFACE "${hwloc_INCLUDE_DIR}")
target_link_libraries(hwloc::hwloc INTERFACE "${hwloc_LIBRARY}")

if (APPLE)
    # hwloc uses IOKit and CoreFoundation on macOS for hardware topology discovery
    target_link_libraries(hwloc::hwloc INTERFACE "-framework IOKit" "-framework CoreFoundation")
endif()
