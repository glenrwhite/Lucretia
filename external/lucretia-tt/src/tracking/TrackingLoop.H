#pragma once

#include "elements/All.H"
#include "particles/TimeBunch.H"

#include <AMReX_REAL.H>


namespace lucretiatt {

namespace spacecharge { class SpaceCharge; }

namespace tracking {

/** Time-stepping orchestrator.
 *
 *  Phase 4A: gathers E,B from each lattice element via std::visit, adds
 *  optional space-charge E from a SpaceCharge solver, and calls Boris
 *  per particle. Image-charge contribution from CathodeSource elements
 *  is also added inline. */
class TrackingLoop
{
public:
    /** Advance `bunch` by one timestep dt at simulation time t.
     *
     *  When `sc` is non-null, a Poisson solve is run once at the start
     *  of the step and per-particle E_sc is gathered and added to the
     *  external-field E before the Boris call. */
    void step (
        particles::TimeBunch&                 bunch,
        const elements::Lattice&              lattice,
        amrex::Real                           t,
        amrex::Real                           dt,
        spacecharge::SpaceCharge*             sc = nullptr);
};

} // namespace tracking
} // namespace lucretiatt
