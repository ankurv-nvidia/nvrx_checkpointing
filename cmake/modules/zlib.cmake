include_guard()

if (COMMAND vcpkg_install)
    vcpkg_install(zlib)
endif()

find_package(ZLIB REQUIRED)
