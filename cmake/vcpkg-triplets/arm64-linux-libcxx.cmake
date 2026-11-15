set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

set(VCPKG_C_FLAGS "")
set(VCPKG_CXX_FLAGS "-stdlib=libc++")
set(VCPKG_LINKER_FLAGS "-stdlib=libc++")
# The explicit target serves native Grace builds and x86_64-hosted cross builds alike.
set(VCPKG_CMAKE_CONFIGURE_OPTIONS
    "-DCMAKE_C_COMPILER=$ENV{VCPKG_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=$ENV{VCPKG_CXX_COMPILER}"
    "-DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu"
    "-DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu"
)
set(VCPKG_MAKE_BUILD_TRIPLET "--host=aarch64-linux-gnu")
set(ENV{CC} "$ENV{VCPKG_C_COMPILER}")
set(ENV{CXX} "$ENV{VCPKG_CXX_COMPILER}")
