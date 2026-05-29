include_guard()

if (DEFINED Python3_EXECUTABLE AND NOT EXISTS "${Python3_EXECUTABLE}")
    message(WARNING "Python executable ${Python3_EXECUTABLE} does not exist")
    set(Python3_VENV_EXECUTABLE "${Python3_EXECUTABLE}")
    get_filename_component(python_bin_path "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(PYTHON_VIRTUAL_ENV_PATH "${python_bin_path}" DIRECTORY)
    unset(Python3_EXECUTABLE)
    unset(Python3_EXECUTABLE CACHE)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
# This is needed to get Python::Module and so on ...
# find_package(Python3 ...) will defined Python3::Module
find_package(Python COMPONENTS Interpreter REQUIRED)

option(PYTHON_VENV_USE_SYSTEM_PACKAGES "Use system packages in the python virtual environment" ON)

if(PYTHON_VENV_USE_SYSTEM_PACKAGES)
  set(PYTHON_VIRTUAL_ENV_ARGS --system-site-packages)
endif()

if (NOT DEFINED PYTHON_VERSION)
    if (NOT DEFINED Python3_VERSION)
        message(FATAL_ERROR "Cannot determine python version")
    else()
        set(PYTHON_VERSION ${Python3_VERSION})
    endif()
endif()

set(PYTHON_VERSION_FULL ${PYTHON_VERSION})
string(REGEX MATCH "(3.[0-9]+)" PYTHON_VERSION_MAJOR_MINOR ${PYTHON_VERSION})
string(REPLACE "." "" PYTHON_VERSION_NO_DOTS ${PYTHON_VERSION_MAJOR_MINOR})

macro(_PythonGetPrefixPath)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE PYTHON_VIRTUAL_ENV_PATH
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(STRIP "${PYTHON_VIRTUAL_ENV_PATH}" PYTHON_VIRTUAL_ENV_PATH)
    file(TO_CMAKE_PATH "${PYTHON_VIRTUAL_ENV_PATH}" PYTHON_VIRTUAL_ENV_PATH)
endmacro()

macro(_PythonGetSitePackagesPath)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_PATH
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(STRIP "${PYTHON_SITE_PACKAGES_PATH}" PYTHON_SITE_PACKAGES_PATH)
    file(TO_CMAKE_PATH "${PYTHON_SITE_PACKAGES_PATH}" PYTHON_SITE_PACKAGES_PATH)
endmacro()

macro(_PythonGetBasePrefixPath)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.base_prefix)"
        OUTPUT_VARIABLE PYTHON_PREFIX_PATH
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(STRIP "${PYTHON_PREFIX_PATH}" PYTHON_PREFIX_PATH)
    file(TO_CMAKE_PATH "${PYTHON_PREFIX_PATH}" PYTHON_PREFIX_PATH)
endmacro()

macro(_FindPythonFromVENV)
    find_program(Python3_VENV_EXECUTABLE
        NAMES python3 python
        NO_DEFAULT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        HINTS
            ${PYTHON_VIRTUAL_ENV_PATH}/Scripts
            ${PYTHON_VIRTUAL_ENV_PATH}/bin)
endmacro()
find_program(UV_EXECUTABLE NAMES uv)
if (NOT EXISTS "${UV_EXECUTABLE}")
    message(STATUS "uv not found, using pip for python package management")
    set(PYTHON_USE_UV OFF)
else()
    message(STATUS "uv found, using uv for python package management")
    set(PYTHON_USE_UV ON)
endif()
if(NOT "${Python3_EXECUTABLE}" STREQUAL "${Python3_VENV_EXECUTABLE}")
    _PythonGetBasePrefixPath()
    _PythonGetPrefixPath()
    if ("${PYTHON_VIRTUAL_ENV_PATH}" STREQUAL "${PYTHON_PREFIX_PATH}")
        set(PYTHON_VIRTUAL_ENV_PATH "${PROJECT_BINARY_DIR}/python_venv")
        unset(Python3_EXECUTABLE      CACHE)
        unset(Python3_VENV_EXECUTABLE CACHE)
        _FindPythonFromVENV()
        if (NOT EXISTS "${Python3_VENV_EXECUTABLE}")
            message(STATUS "Setting up Python virtual environment at ${PYTHON_VIRTUAL_ENV_PATH}")
            file(GLOB_RECURSE venvfiles "${PYTHON_VIRTUAL_ENV_PATH}" *)
            message(STATUS "${PYTHON_VIRTUAL_ENV_PATH}:: ${venvfile}")

            if(PYTHON_USE_UV)
                set(PYTHON_ENV_SETUP_COMMAND uv venv)
            else()
                set(PYTHON_ENV_SETUP_COMMAND "${Python3_EXECUTABLE}" -m venv)
            endif()
            execute_process(
                COMMAND ${PYTHON_ENV_SETUP_COMMAND} ${PYTHON_VIRTUAL_ENV_ARGS} "${PYTHON_VIRTUAL_ENV_PATH}"
                COMMAND_ERROR_IS_FATAL ANY
            )

            unset(Python3_EXECUTABLE)
            unset(Python3_EXECUTABLE CACHE)
            _FindPythonFromVENV()
            if (NOT EXISTS "${Python3_VENV_EXECUTABLE}")
                message(FATAL_ERROR "Cannot find python venv ${Python3_VENV_EXECUTABLE}. Something went wrong")
            endif()
        endif()

        message(STATUS "Found Virtual Env Python: ${Python3_VENV_EXECUTABLE}")
        unset(Python3_EXECUTABLE)
        unset(Python3_EXECUTABLE CACHE)
        set(Python3_EXECUTABLE      "${Python3_VENV_EXECUTABLE}" CACHE PATH "Location of Python virtual environment exe" FORCE)
        set(Python3_VENV_EXECUTABLE "${Python3_VENV_EXECUTABLE}" CACHE PATH "Location of Python virtual environment exe" FORCE)
        _PythonGetPrefixPath()
        if ("${PYTHON_VIRTUAL_ENV_PATH}" STREQUAL "${PYTHON_PREFIX_PATH}")
            message(FATAL_ERROR "Unable to set up python virtual environment ${PYTHON_VIRTUAL_ENV_PATH} != ${PYTHON_PREFIX_PATH}")
        endif()
    endif()
else()
    # Incremental builds will hit this else() block
    _PythonGetPrefixPath()
endif()

_PythonGetSitePackagesPath()

message(STATUS "Using Python:
        FULL_VERSION=${PYTHON_VERSION_FULL} SHORT_VERSION=${PYTHON_VERSION_MAJOR_MINOR} VERSION_NO_DOTS=${PYTHON_VERSION_NO_DOTS}
        PYTHON_VIRTUAL_ENV_PATH=${PYTHON_VIRTUAL_ENV_PATH}
        Python3_EXECUTABLE=${Python3_EXECUTABLE}
        Python_SOSABI=${Python3_SOSABI}
        Python_SOABI=${Python3_SOABI}
        Python_SITELIB=${Python3_SITELIB}
        Python_SITEARCH=${Python3_SITEARCH}
        PYTHON_SITE_PACKAGES_PATH=${PYTHON_SITE_PACKAGES_PATH}")

function(pip_install_requirements req_file)
    pip_install("--upgrade" "-r" "${req_file}")
endfunction()

function(pip_install)
    if (PYTHON_USE_UV)
        uv_pip_install(${ARGN})
    else()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pip install ${ARGN}
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
endfunction()

function(uv_pip_install)
    execute_process(
        COMMAND ${UV_EXECUTABLE} pip install --python ${Python3_EXECUTABLE} ${ARGN}
        COMMAND_ERROR_IS_FATAL ANY
    )
endfunction()
