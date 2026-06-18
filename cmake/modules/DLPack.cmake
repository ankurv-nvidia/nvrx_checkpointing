include_guard()
include(FetchContent)

# DLPack's upstream CMake declares `project(dlpack VERSION 0.6)` regardless
# of the actual release tag, so the installed ConfigVersionFile reports 0.6
# even for the v1.3 release. Passing a version to FIND_PACKAGE_ARGS would
# trip CMake's version-compatibility check and force a FetchContent fall-
# through; the version pin we actually rely on lives in `vcpkg.json`.
FetchContent_Declare(
    dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack
    GIT_TAG        v1.3
    SOURCE_SUBDIR  .
    GIT_PROGRESS FALSE GIT_SHALLOW ON SYSTEM
    FIND_PACKAGE_ARGS NAMES dlpack
)

if (COMMAND vcpkg_install)
    vcpkg_install(dlpack)
endif()

FetchContent_MakeAvailable(dlpack)


