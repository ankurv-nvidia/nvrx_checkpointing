include_guard()

if (COMMAND vcpkg_install)
    # Limit cold dependency builds to avoid exhausting CI memory.
    if (NOT DEFINED ENV{VCPKG_MAX_CONCURRENCY})
        set(ENV{VCPKG_MAX_CONCURRENCY} "4")
    endif()
    _vcpkg_is_packages_applicable(legacy_otel_installed "opentelemetry-cpp:${VCPKG_TARGET_TRIPLET}")
    if (legacy_otel_installed)
        execute_process(
            COMMAND_ERROR_IS_FATAL ANY
            WORKING_DIRECTORY "${VCPKG_ROOT}"
            COMMAND "${VCPKG_EXE}" remove "opentelemetry-cpp:${VCPKG_TARGET_TRIPLET}" --recurse
        )
        _vcpkg_refresh_available_ports()
    endif()
    vcpkg_install("opentelemetry-cpp23[otlp-file]")
endif()

find_package(opentelemetry-cpp CONFIG REQUIRED)
