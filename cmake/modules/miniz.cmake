include_guard()
include(FetchContent)

FetchContent_Declare(
    miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz
    GIT_TAG        3.1.1
    SOURCE_SUBDIR  .
    GIT_PROGRESS FALSE GIT_SHALLOW ON SYSTEM
    FIND_PACKAGE_ARGS NAMES miniz
)

if (COMMAND vcpkg_install)
    vcpkg_install(miniz)
endif()

FetchContent_MakeAvailable(miniz)
if (COMMAND SupressWarningForTarget)
    SupressWarningForTarget(miniz)
endif()
