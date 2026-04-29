# Helper macros & functions for the LucretiaTT CMake build.
# Adapted from ImpactX (BSD-3-Clause-LBNL); renamed and trimmed for our use.


# Set C++17 globally for the superbuild (AMReX + openPMD-api need it).
macro(set_cxx17_superbuild)
    if(NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
        set(CMAKE_CXX_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CUDA_EXTENSIONS)
        set(CMAKE_CUDA_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CUDA_STANDARD_REQUIRED)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endmacro()


# Use ccache if available.
macro(set_ccache)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        if(LucretiaTT_COMPUTE STREQUAL CUDA)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        endif()
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
    else()
        message(STATUS "Could NOT find CCache")
    endif()
    mark_as_advanced(CCACHE_PROGRAM)
endmacro()


# Default build directories: ${CMAKE_BINARY_DIR}/{lib,bin}.
macro(luctt_set_default_build_dirs)
    if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for archives")
        mark_as_advanced(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for libraries")
        mark_as_advanced(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                CACHE PATH "Build directory for binaries")
        mark_as_advanced(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    endif()
endmacro()


# Default install directories.
macro(luctt_set_default_install_dirs)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        include(GNUInstallDirs)
        if(NOT CMAKE_INSTALL_CMAKEDIR)
            set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake"
                    CACHE PATH "CMake config package location for installed targets")
            mark_as_advanced(CMAKE_INSTALL_CMAKEDIR)
        endif()
    endif()
endmacro()


# Default to Release build.
macro(set_default_build_type default_build_type)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE ${default_build_type}
                CACHE STRING
                "Choose the build type: Release, Debug, RelWithDebInfo, MinSizeRel" FORCE)
            set_property(CACHE CMAKE_BUILD_TYPE
                PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
        endif()
        if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
            message(WARNING "CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}' is not one of "
                    "${CMAKE_CONFIGURATION_TYPES}. Typo?")
        endif()
    endif()
endmacro()


# Compile warnings.
function(luctt_set_compile_warnings tgt)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND NOT WIN32)
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wshadow -Woverloaded-virtual -Wunreachable-code -Wno-array-bounds)
        if(NOT LucretiaTT_COMPUTE STREQUAL CUDA)
            target_compile_options(${tgt} PRIVATE -Wpedantic)
        endif()
    endif()
endfunction()


# IPO / LTO.
function(luctt_enable_IPO tgt)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT is_IPO_available)
    if(is_IPO_available)
        set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(FATAL_ERROR "Interprocedural optimization is not available, set LucretiaTT_IPO=OFF")
    endif()
endfunction()


# Source version (from git describe).
function(get_source_version NAME SOURCE_DIR)
    find_package(Git QUIET)
    set(_tmp "")
    if(EXISTS ${SOURCE_DIR}/.git AND ${GIT_FOUND})
        execute_process(COMMAND git describe --abbrev=12 --dirty --always --tags
            WORKING_DIRECTORY ${SOURCE_DIR}
            OUTPUT_VARIABLE _tmp)
        string(STRIP "${_tmp}" _tmp)
    endif()
    if(NOT _tmp AND ${NAME}_VERSION)
        set(_tmp "${${NAME}_VERSION}-nogit")
    endif()
    set(${NAME}_GIT_VERSION "${_tmp}" CACHE INTERNAL "")
endfunction()


# Print build summary.
function(luctt_print_summary)
    message("")
    message("LucretiaTT build configuration:")
    message("  Version:    ${LucretiaTT_VERSION} (${LucretiaTT_GIT_VERSION})")
    message("  C++:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message("                ${CMAKE_CXX_COMPILER}")
    message("  Build type: ${CMAKE_BUILD_TYPE}")
    message("  Options:")
    message("    COMPUTE:  ${LucretiaTT_COMPUTE}")
    message("    PRECISION: ${LucretiaTT_PRECISION}")
    message("    MPI:      ${LucretiaTT_MPI}")
    message("    FFT:      ${LucretiaTT_FFT}")
    message("    OPENPMD:  ${LucretiaTT_OPENPMD}")
    message("    FASTMATH: ${LucretiaTT_FASTMATH}")
    message("    IPO:      ${LucretiaTT_IPO}")
    message("")
endfunction()
