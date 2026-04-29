# lucretia-tt

Time-based 3D particle-in-cell tracker for the Lucretia accelerator-physics
codebase. Standalone C++/CUDA executable invoked from Lucretia's MATLAB layer
via `mpirun`. Built on AMReX + ABLASTR (lifted as BSD-licensed dependencies
from BLAST-LBNL).

Designed for photoinjector and capture-section simulation:
- Cathode emission with thermal MTE, image charge
- 3D space charge via FFT-based open-BC Poisson (ABLASTR `OpenBCSolver`)
- Analytic + 3D-field-map elements: SW gun, TW structure, solenoid, quadrupole
- Boris pusher (with Vay/Higuera-Cary as alternatives)
- openPMD HDF5 I/O, freeze-plane handoff to Lucretia's s-based `TrackThru`

Target scale: 10^7 macroparticles routine, 10^8 occasional, on a single
8x A100 workstation; CPU-MPI build also supported for clusters.

See `~/.claude/plans/polymorphic-exploring-squid.md` for the full architecture
plan.

---

## Phase 0 — environment setup

Before building lucretia-tt, install the dependency chain and verify it works
by building **ImpactX** (whose dependency tree is a strict superset of ours).
This validates the toolchain on your hardware before we write code that
depends on it.

### Linux (target deployment, 8x A100 box)

Recommended via [Spack](https://spack.io):

```bash
# install Spack itself
git clone --depth=1 https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh

# create a Spack environment matching the WarpX-recommended stack
spack env create luctt
spack env activate luctt
spack add cmake@3.24:
spack add openmpi+cuda fabrics=ucx
spack add ucx+cuda+gdrcopy+thread_multiple
spack add hdf5@1.14: +mpi +cxx
spack add openpmd-api +mpi +hdf5
spack add cuda@12: cuda_arch=80
spack add ccache
spack install -j 32
```

Activate the env in any shell where you build:

```bash
. ~/spack/share/spack/setup-env.sh
spack env activate luctt
```

Verify CUDA-aware MPI:
```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
# expected: mca:mpi:base:param:mpi_built_with_cuda_support:value:true
```

### macOS (dev box)

Single-rank CPU-OMP only — no MPI, no CUDA. Useful for code authoring,
not for production runs.

```bash
brew install cmake hdf5 ccache libomp
brew install open-mpi      # optional, for testing CPU-MPI builds
```

(openPMD-api will be FetchContent-built by lucretia-tt — no Homebrew package
needed.)

### Phase 0 verification — build ImpactX

```bash
git clone --depth 1 https://github.com/BLAST-ImpactX/impactx.git /tmp/impactx
cmake -S /tmp/impactx -B /tmp/impactx/build \
      -DImpactX_COMPUTE=CUDA \
      -DImpactX_MPI=ON \
      -DImpactX_FFT=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/impactx/build -j 16

# run a 10000-particle FODO test on 4 GPUs
cd /tmp/impactx/build
mpirun -np 4 ./bin/impactx ../examples/fodo/input_fodo.in
```

If this completes without error and produces openPMD output, the dependency
chain is healthy and we can proceed to Phase 1. If it fails, capture the
error message and we triage before writing lucretia-tt code.

---

## Building lucretia-tt

After Phase 0 is verified:

```bash
cd /path/to/Lucretia/external/lucretia-tt
cmake -S . -B build \
      -DLucretiaTT_COMPUTE=CUDA \
      -DLucretiaTT_MPI=ON \
      -DLucretiaTT_FFT=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16

# binary lands at build/bin/lucretia-tt
./build/bin/lucretia-tt --help
```

Build flavours:

| Flavour | CMake flags |
|---|---|
| Single-node CPU OMP (dev) | `-DLucretiaTT_COMPUTE=OMP -DLucretiaTT_MPI=OFF` |
| CPU MPI (cluster) | `-DLucretiaTT_COMPUTE=OMP -DLucretiaTT_MPI=ON` |
| Single GPU | `-DLucretiaTT_COMPUTE=CUDA -DLucretiaTT_MPI=OFF` |
| 8x A100 multi-GPU | `-DLucretiaTT_COMPUTE=CUDA -DLucretiaTT_MPI=ON -DCMAKE_CUDA_ARCHITECTURES=80` |

---

## Repository layout

```
external/lucretia-tt/
├── CMakeLists.txt                 top-level superbuild
├── cmake/
│   ├── LucretiaTTFunctions.cmake  helper macros
│   └── dependencies/
│       └── ABLASTR.cmake          FetchContent ABLASTR (pulls AMReX + openPMD transitively)
├── src/
│   ├── main.cpp                   entry point
│   ├── particles/                 TimeBunch (extends amrex::ParticleContainerPureSoA)
│   ├── elements/                  Drift3D, SWCavity, Solenoid3D, etc.
│   │   └── mixin/                 Named, Aperture, Position3D, TimeWindow, FieldSource
│   ├── tracking/                  Boris/Vay/HC pusher, time-stepping loop
│   ├── spacecharge/               wrappers around ablastr::fields::computePhi + image charge
│   └── io/                        openPMD field-map reader + beam dumps
├── tests/                         per-phase validation cases (ctest)
├── examples/                      runnable simulation inputs
└── tools/                         field-map import scripts (Python)
```
