#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParticleContainer.H>

#include <string>


namespace lucretiatt {
namespace particles {

/** Per-particle SoA real attribute layout.
 *
 *  In ParticleContainerPureSoA the first AMREX_SPACEDIM real attributes
 *  serve as positions (used for spatial sorting); the rest are physics
 *  attributes specific to lucretia-tt. */
struct RealSoA
{
    enum Comp {
        x = 0,
        y,
        z,
        px,
        py,
        pz,
        q,        // charge (C)
        qm,       // q/m  (C/kg)
        w,        // numerical weight (real particles per macroparticle)
        t_birth,  // emission time (s)
        nattribs
    };
};

/** Per-particle SoA int attribute layout. */
struct IntSoA
{
    enum Comp {
        alive = 0,  // 1 = alive, 0 = absorbed/lost
        nattribs
    };
};

using TimeBunchBase =
    amrex::ParticleContainerPureSoA<RealSoA::nattribs, IntSoA::nattribs>;


/** Particle container for the time-based 3D PIC tracker.
 *
 *  Lab-frame coordinates (x, y, z) with t as the independent variable.
 *  Per-particle:
 *    - px, py, pz : proper velocity u = gamma * v in m/s (WarpX convention)
 *    - q          : charge in C
 *    - qm         : charge-to-mass q/m in C/kg
 *    - w          : numerical weight (real particles per macroparticle)
 *    - t_birth    : emission time in s
 *    - alive      : 1 = active, 0 = absorbed/lost */
class TimeBunch
    : public TimeBunchBase
{
public:
    TimeBunch (
        const amrex::Geometry& geom,
        const amrex::DistributionMapping& dm,
        const amrex::BoxArray& ba);

    /** Add `n` electrons all at (x, y, z) with proper velocity (ux, uy, uz).
     *
     *  Charge and qm default to electron values. t_birth = 0.
     *  Emitted on the IO rank then Redistribute()'d. */
    void AddParticles (
        int                 n,
        amrex::ParticleReal x,  amrex::ParticleReal y,  amrex::ParticleReal z,
        amrex::ParticleReal ux, amrex::ParticleReal uy, amrex::ParticleReal uz,
        amrex::ParticleReal weight = amrex::ParticleReal(1.0));

    /** Add `n` electrons with per-particle (x,y,z,ux,uy,uz) sourced from
     *  caller-owned arrays. All `n` particles share the same weight and
     *  birth time t_birth. Used by CathodeSource emission. */
    void AddParticlesFromArrays (
        int                        n,
        const amrex::ParticleReal* xs,  const amrex::ParticleReal* ys,  const amrex::ParticleReal* zs,
        const amrex::ParticleReal* uxs, const amrex::ParticleReal* uys, const amrex::ParticleReal* uzs,
        amrex::ParticleReal        weight,
        amrex::ParticleReal        t_birth);

    /** Print total particle count, plus per-rank breakdown when MPI > 1. */
    void PrintSummary (const std::string& tag);
};

} // namespace particles
} // namespace lucretiatt
