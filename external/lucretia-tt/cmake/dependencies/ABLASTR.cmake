# Fetch and configure ABLASTR (lives in the WarpX repo as a build-only sub-library).
# ABLASTR transitively pulls AMReX and openPMD-api.
#
# Adapted from ImpactX (BSD-3-Clause-LBNL); renamed for LucretiaTT and stripped
# of the pyAMReX/Python paths that we don't use.

macro(find_ablastr)
    if(LucretiaTT_ablastr_src)
        message(STATUS "Compiling local ABLASTR ...")
        message(STATUS "ABLASTR source path: ${LucretiaTT_ablastr_src}")
        if(NOT IS_DIRECTORY ${LucretiaTT_ablastr_src})
            message(FATAL_ERROR "Specified directory LucretiaTT_ablastr_src='${LucretiaTT_ablastr_src}' does not exist!")
        endif()
    elseif(LucretiaTT_ablastr_internal)
        message(STATUS "Downloading ABLASTR ...")
        message(STATUS "ABLASTR repository: ${LucretiaTT_ablastr_repo} (${LucretiaTT_ablastr_branch})")
        include(FetchContent)
    endif()

    # Transitive control: AMReX
    set(WarpX_amrex_internal ${LucretiaTT_amrex_internal} CACHE BOOL
        "Download & build AMReX" FORCE)
    if(LucretiaTT_amrex_src)
        set(WarpX_amrex_src ${LucretiaTT_amrex_src} CACHE PATH
            "Local path to AMReX source directory (preferred if set)" FORCE)
        list(APPEND CMAKE_MODULE_PATH "${WarpX_amrex_src}/Tools/CMake")
    elseif(LucretiaTT_amrex_internal)
        if(LucretiaTT_amrex_repo)
            set(WarpX_amrex_repo ${LucretiaTT_amrex_repo} CACHE STRING
                "Repository URI to pull and build AMReX from if(LucretiaTT_amrex_internal)" FORCE)
        endif()
        if(LucretiaTT_amrex_branch)
            set(WarpX_amrex_branch ${LucretiaTT_amrex_branch} CACHE STRING
                "Repository branch for LucretiaTT_amrex_repo if(LucretiaTT_amrex_internal)" FORCE)
        endif()
    endif()

    # Transitive control: openPMD-api
    if(LucretiaTT_openpmd_src)
        set(WarpX_openpmd_src ${LucretiaTT_openpmd_src} CACHE PATH
            "Local path to openPMD-api source directory (preferred if set)" FORCE)
    elseif(LucretiaTT_openpmd_internal)
        if(LucretiaTT_openpmd_repo)
            set(WarpX_openpmd_repo ${LucretiaTT_openpmd_repo} CACHE STRING
                "Repository URI to pull and build openPMD-api from if(LucretiaTT_openpmd_internal)" FORCE)
        endif()
        if(LucretiaTT_openpmd_branch)
            set(WarpX_openpmd_branch ${LucretiaTT_openpmd_branch} CACHE STRING
                "Repository branch for LucretiaTT_openpmd_repo if(LucretiaTT_openpmd_internal)" FORCE)
        endif()
    else()
        set(WarpX_openpmd_internal ${LucretiaTT_openpmd_internal} CACHE STRING
            "Download & build openPMD-api" FORCE)
    endif()

    # openPMD-api / parallel-HDF5 toggle is handled by the
    # disable_openpmd_mpi.cmake patch script (PATCH_COMMAND on the
    # FetchContent_Declare for ABLASTR below). The patch reads
    # LUCTT_OPENPMD_MPI from its own command line and either edits
    # WarpX/cmake/dependencies/openPMD.cmake to force openPMD_USE_MPI=OFF
    # (the default; works without parallel HDF5 installed) or no-ops
    # (when LucretiaTT_OPENPMD_MPI=ON; requires hdf5-mpi).

    # Transitive control: ABLASTR / WarpX (we only build the ABLASTR sublib).
    if(LucretiaTT_ablastr_internal OR LucretiaTT_ablastr_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        set(ABLASTR_FASTMATH ${LucretiaTT_FASTMATH} CACHE BOOL "" FORCE)
        set(AMReX_FASTMATH   ${LucretiaTT_FASTMATH} CACHE BOOL "" FORCE)
        set(ABLASTR_FFT      ${LucretiaTT_FFT}      CACHE BOOL "" FORCE)
        set(AMReX_FFT        ${LucretiaTT_FFT}      CACHE BOOL "" FORCE)

        # Disable WarpX's own targets — we only want the ABLASTR shared kernels.
        set(WarpX_APP    OFF                          CACHE BOOL     "" FORCE)
        set(WarpX_LIB    OFF                          CACHE BOOL     "" FORCE)
        set(WarpX_QED    OFF                          CACHE BOOL     "" FORCE)
        set(WarpX_COMPUTE              ${LucretiaTT_COMPUTE}              CACHE INTERNAL "" FORCE)
        set(WarpX_DIMS                 3                                  CACHE INTERNAL "" FORCE)
        set(WarpX_FFT                  ${LucretiaTT_FFT}                  CACHE BOOL     "" FORCE)
        set(WarpX_OPENPMD              ${LucretiaTT_OPENPMD}              CACHE INTERNAL "" FORCE)
        set(WarpX_PRECISION            ${LucretiaTT_PRECISION}            CACHE INTERNAL "" FORCE)
        set(WarpX_MPI                  ${LucretiaTT_MPI}                  CACHE INTERNAL "" FORCE)
        set(WarpX_MPI_THREAD_MULTIPLE  ${LucretiaTT_MPI_THREAD_MULTIPLE}  CACHE INTERNAL "" FORCE)
        set(WarpX_IPO                  ${LucretiaTT_IPO}                  CACHE INTERNAL "" FORCE)

        if(LucretiaTT_ablastr_src)
            if(LucretiaTT_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
            endif()
            add_subdirectory(${LucretiaTT_ablastr_src} _deps/localablastr-build/)
            if(DEFINED AMReX_DIR)
                list(APPEND CMAKE_MODULE_PATH "${AMReX_DIR}/AMReXCMakeModules")
            else()
                list(APPEND CMAKE_MODULE_PATH "${FETCHCONTENT_BASE_DIR}/fetchedamrex-src/Tools/CMake")
            endif()
        else()
            if(LucretiaTT_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
            endif()
            FetchContent_Declare(fetchedablastr
                GIT_REPOSITORY ${LucretiaTT_ablastr_repo}
                GIT_TAG        ${LucretiaTT_ablastr_branch}
                BUILD_IN_SOURCE 0
                PATCH_COMMAND  ${CMAKE_COMMAND}
                               -D LUCTT_OPENPMD_MPI=${LucretiaTT_OPENPMD_MPI}
                               -P ${LucretiaTT_SOURCE_DIR}/cmake/patches/disable_openpmd_mpi.cmake
                UPDATE_DISCONNECTED 1
            )
            FetchContent_MakeAvailable(fetchedablastr)
            if(DEFINED AMReX_DIR)
                list(APPEND CMAKE_MODULE_PATH "${AMReX_DIR}/AMReXCMakeModules")
            else()
                list(APPEND CMAKE_MODULE_PATH "${FETCHCONTENT_BASE_DIR}/fetchedamrex-src/Tools/CMake")
            endif()

            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDABLASTR)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDABLASTR)
        endif()

        mark_as_advanced(AMREX_BUILD_DATETIME)

        message(STATUS "ABLASTR: Using version '${WarpX_VERSION}' (${WarpX_GIT_VERSION})")
    else()
        message(FATAL_ERROR "Pre-installed ABLASTR not yet supported. Set LucretiaTT_ablastr_internal=ON.")
    endif()

    # AMReX CMake helper scripts (e.g. setup_target_for_cuda_compilation).
    list(APPEND CMAKE_MODULE_PATH "${AMReX_DIR}/AMReXCMakeModules")

    if(NOT TARGET AMReX::amrex)
        find_package(AMReX CONFIG REQUIRED)
    endif()

    # openPMD: external find_package fallback if neither src nor internal was set
    if(NOT LucretiaTT_openpmd_src AND NOT LucretiaTT_openpmd_internal)
        if(LucretiaTT_MPI)
            set(COMPONENT_WMPI MPI)
        else()
            set(COMPONENT_WMPI NOMPI)
        endif()
        find_package(openPMD 0.17.0 CONFIG REQUIRED COMPONENTS ${COMPONENT_WMPI})
        message(STATUS "openPMD-api: Found version '${openPMD_VERSION}'")
    endif()
endmacro()


# Local source-tree options ###################################################
set(LucretiaTT_amrex_src   "" CACHE PATH "Local path to AMReX source directory (preferred if set)")
set(LucretiaTT_ablastr_src "" CACHE PATH "Local path to ABLASTR source directory (preferred if set)")
set(LucretiaTT_openpmd_src "" CACHE PATH "Local path to openPMD-api source directory (preferred if set)")

# Git fetch options ###########################################################
# ABLASTR lives inside the WarpX repository.
set(LucretiaTT_ablastr_repo "https://github.com/BLAST-WarpX/warpx.git"
    CACHE STRING "Repository URI to pull and build ABLASTR from")
set(LucretiaTT_ablastr_branch "26.04"
    CACHE STRING "Repository branch (tag) for LucretiaTT_ablastr_repo")

set(LucretiaTT_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING "Repository URI to pull and build AMReX from")
set(LucretiaTT_amrex_branch "26.04"
    CACHE STRING "Repository branch (tag) for LucretiaTT_amrex_repo")

set(LucretiaTT_openpmd_repo "https://github.com/openPMD/openPMD-api.git"
    CACHE STRING "Repository URI to pull and build openPMD-api from")
set(LucretiaTT_openpmd_branch "0.17.0"
    CACHE STRING "Repository branch (tag) for LucretiaTT_openpmd_repo")

find_ablastr()
