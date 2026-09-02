include_guard()

if (COMMAND vcpkg_install)
    vcpkg_install("spdlog")
endif()

find_package(spdlog CONFIG REQUIRED)
