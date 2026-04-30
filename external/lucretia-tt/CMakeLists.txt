cmake_minimum_required(VERSION 3.24.0)
project(LucretiaTT VERSION 0.1.0 LANGUAGES CXX)

include(${LucretiaTT_SOURCE_DIR}/cmake/LucretiaTTFunctions.cmake)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported. "
            "Create a build directory: cmake -S . -B build")
endif()

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

set_cxx17_superbuild()


# CCache ######################################################################
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(LucretiaTT_CCACHE_DEFAULT ON)
else()
    set(LucretiaTT_CCACHE_DEFAULT OFF)
endif()
option(LucretiaTT_CCACHE "Enable ccache for faster rebuilds" ${LucretiaTT_CCACHE_DEFAULT})
if(LucretiaTT_CCACHE)
    set_ccache()
endif()


# Build directories ###########################################################
luctt_set_default_build_dirs()
luctt_set_default_install_dirs()


# Options #####################################################################
set(LucretiaTT_COMPUTE_VALUES NOACC OMP CUDA SYCL HIP)
set(LucretiaTT_COMPUTE OMP CACHE STRING
    "On-node accelerated computing backend (NOACC/OMP/CUDA/SYCL/HIP)")
set_property(CACHE LucretiaTT_COMPUTE PROPERTY STRINGS ${LucretiaTT_COMPUTE_VALUES})
if(NOT LucretiaTT_COMPUTE IN_LIST LucretiaTT_COMPUTE_VALUES)
    message(FATAL_ERROR "LucretiaTT_COMPUTE (${LucretiaTT_COMPUTE}) must be one of ${LucretiaTT_COMPUTE_VALUES}")
endif()

set(LucretiaTT_PRECISION_VALUES SINGLE DOUBLE)
set(LucretiaTT_PRECISION DOUBLE CACHE STRING "Floating point precision (SINGLE/DOUBLE)")
set_property(CACHE LucretiaTT_PRECISION PROPERTY STRINGS ${LucretiaTT_PRECISION_VALUES})
if(NOT LucretiaTT_PRECISION IN_LIST LucretiaTT_PRECISION_VALUES)
    message(FATAL_ERROR "LucretiaTT_PRECISION (${LucretiaTT_PRECISION}) must be one of ${LucretiaTT_PRECISION_VALUES}")
endif()

option(LucretiaTT_MPI                 "Multi-node MPI support"           ON)
option(LucretiaTT_MPI_THREAD_MULTIPLE "MPI thread-multiple (async I/O)"  ON)
option(LucretiaTT_FFT                 "FFT-based Poisson (IGF)"          ON)
option(LucretiaTT_OPENPMD             "openPMD I/O (HDF5/ADIOS)"         ON)
option(LucretiaTT_OPENPMD_MPI         "Build openPMD-api with MPI / parallel HDF5 (requires hdf5-mpi). Default OFF: BeamIO MPI-gathers to rank 0 then writes serially, which works with serial HDF5 on every platform."
                                                                         OFF)
option(LucretiaTT_FASTMATH            "Enable -ffast-math optimisations" OFF)
option(LucretiaTT_IPO                 "Interprocedural / link-time optimization" OFF)

mark_as_advanced(LucretiaTT_MPI_THREAD_MULTIPLE)

option(LucretiaTT_ablastr_internal  "Download & build ABLASTR"   ON)
option(LucretiaTT_amrex_internal    "Download & build AMReX"     ON)
option(LucretiaTT_openpmd_internal  "Download & build openPMD-api" ON)

set_default_build_type("Release")


# Dependencies ################################################################
include(${LucretiaTT_SOURCE_DIR}/cmake/dependencies/ABLASTR.cmake)


# Targets #####################################################################
add_executable(app)
add_executable(LucretiaTT::app ALIAS app)
set_target_properties(app PROPERTIES
    OUTPUT_NAME lucretia-tt
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(app PRIVATE
    $<BUILD_INTERFACE:${LucretiaTT_SOURCE_DIR}/src>
)

# AMReX/ABLASTR linkage
target_link_libraries(app PRIVATE WarpX::ablastr_3d)

if(LucretiaTT_OPENPMD)
    target_link_libraries(app PRIVATE openPMD::openPMD)
    target_compile_definitions(app PUBLIC LUCRETIATT_USE_OPENPMD)
endif()

# Our field-map I/O code (Solenoid3D, FieldMap3D, BeamMonitor scratch reads,
# main.cpp parmparse-from-h5) calls the HDF5 C API directly. In serial-openPMD
# builds the include path leaks through openPMD's INTERFACE properties, but
# that breaks under openPMD+MPI when openPMD is wired against parallel HDF5
# but our own files want serial HDF5 headers. Find HDF5 explicitly so we
# always have a known include path for our direct uses.
find_package(HDF5 REQUIRED COMPONENTS C)
target_link_libraries(app PRIVATE HDF5::HDF5)

if(LucretiaTT_FFT)
    target_compile_definitions(app PUBLIC LUCRETIATT_USE_FFT)
endif()

if(LucretiaTT_FASTMATH)
    target_link_libraries(app PRIVATE AMReX::Flags_FASTMATH)
endif()

# Sources -- adds main.cpp and (later) subdirectories
add_subdirectory(src)


# C++17 (matches AMReX requirement) ###########################################
target_compile_features(app PUBLIC cxx_std_17)
set_target_properties(app PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

if(LucretiaTT_IPO)
    luctt_enable_IPO(app)
endif()

# CUDA-specific target setup
if(LucretiaTT_COMPUTE STREQUAL CUDA)
    setup_target_for_cuda_compilation(app)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        target_compile_features(app PUBLIC cuda_std_17)
        set_target_properties(app PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON
        )
    endif()
endif()


# Compile warnings ############################################################
luctt_set_compile_warnings(app)


# Tests #######################################################################
include(CTest)
if(BUILD_TESTING)
    enable_testing()
    if(EXISTS ${LucretiaTT_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    endif()
    if(EXISTS ${LucretiaTT_SOURCE_DIR}/examples/CMakeLists.txt)
        add_subdirectory(examples)
    endif()
endif()


# Summary #####################################################################
get_source_version(LucretiaTT ${LucretiaTT_SOURCE_DIR})
luctt_print_summary()
