# Patch WarpX/cmake/dependencies/openPMD.cmake so that openPMD-api is
# always built without MPI support, even when WarpX/AMReX are built with
# MPI. This lets the rest of the application use MPI for compute while
# openPMD-api uses serial HDF5 (rank 0 writes after an MPI gather).
#
# Reason: openPMD-api built with MPI requires *parallel* HDF5
# (HDF5_IS_PARALLEL=TRUE). On macOS, Homebrew's serial `hdf5` formula
# conflicts with `hdf5-mpi`, so installing parallel HDF5 means giving up
# the serial one (which other code relies on). Decoupling openPMD's MPI
# from WarpX's MPI sidesteps the dependency conflict at zero physics
# cost (BeamIO already gathers to rank 0 before writing).
#
# Run with:
#   cmake -P disable_openpmd_mpi.cmake
# from the WarpX source root (FetchContent_Declare PATCH_COMMAND).

set(_target "cmake/dependencies/openPMD.cmake")
if(NOT EXISTS "${_target}")
    message(STATUS "[lucretia-tt patch] ${_target} not found; nothing to do")
    return()
endif()

file(READ "${_target}" _contents)

set(_old "set(openPMD_USE_MPI         \${WarpX_MPI}  CACHE INTERNAL \"\")")
set(_new "set(openPMD_USE_MPI         OFF           CACHE INTERNAL \"\")  # patched by lucretia-tt: always serial openPMD")

string(FIND "${_contents}" "${_old}" _pos)
if(_pos LESS 0)
    string(FIND "${_contents}" "patched by lucretia-tt" _already)
    if(_already GREATER_EQUAL 0)
        message(STATUS "[lucretia-tt patch] ${_target} already patched")
    else()
        message(WARNING "[lucretia-tt patch] expected line not found in ${_target}; "
                        "WarpX upstream may have changed openPMD.cmake")
    endif()
    return()
endif()

string(REPLACE "${_old}" "${_new}" _patched "${_contents}")
file(WRITE "${_target}" "${_patched}")
message(STATUS "[lucretia-tt patch] disabled openPMD_USE_MPI in ${_target}")
