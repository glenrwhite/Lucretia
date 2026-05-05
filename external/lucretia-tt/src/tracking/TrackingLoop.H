#pragma once

#include "elements/All.H"
#include "particles/TimeBunch.H"

#include <AMReX_REAL.H>


namespace lucretiatt {

namespace spacecharge {
    class SpaceCharge;
    class SpaceChargeSlice;
}

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
     *  external-field E before the Boris call.
     *
     *  When `slice_sc` is non-null, the 1D longitudinal slice SC is
     *  computed and its E_z REPLACES the longitudinal component of the
     *  3D mesh SC contribution (mesh handles transverse only, slice
     *  handles longitudinal). The slice model is mesh-free in z and
     *  stays resolved even when the bunch compresses below one mesh
     *  cell — which is the normal case once a photoinjector bunch
     *  crosses the gun. */
    void step (
        particles::TimeBunch&                 bunch,
        const elements::Lattice&              lattice,
        amrex::Real                           t,
        amrex::Real                           dt,
        spacecharge::SpaceCharge*             sc       = nullptr,
        spacecharge::SpaceChargeSlice*        slice_sc = nullptr);

    /** Enable ImpactT-style behind-cathode drift. Particles with
     *  z < cathode_z advance at the universal speed `betazini * c`
     *  (no field gather, no Boris push, no SC contribution). Once a
     *  particle's z >= cathode_z, normal tracking takes over.
     *
     *  This synchronizes the cross-cathode arrival of particles emitted
     *  with thermal pz spread: instead of dispersing over ~30 ps (and
     *  ~30° of RF phase) the way per-particle drift would, all particles
     *  arrive at z=0 within ~ z_init_spread / (betazini*c) of each other
     *  -- typically ~1-2 ps for the LCLS injector setup -- which matches
     *  ImpactT's `driftemission_BeamBunch` behaviour.
     *
     *  Default off (cathode_z = -inf, betazini = 0). */
    void set_behind_cathode_drift (amrex::Real cathode_z,
                                   amrex::Real betazini) noexcept
    {
        m_cathode_z                = cathode_z;
        m_behind_cathode_betazini  = betazini;
        m_behind_cathode_drift_on  = true;
    }
    bool behind_cathode_drift_enabled () const noexcept { return m_behind_cathode_drift_on; }

    /** Use ImpactT-style centroid-z reference time for external-field
     *  gather. ImpactT line 1436 of AccSimulator.f90 sets
     *  refptcl(5) = sgcenter(5) (bunch z-centroid in normalized units),
     *  then field gather uses tg = refptcl(5). The effective field
     *  formula becomes `cos(omega * z_centroid/c + phase)` -- a SPATIAL
     *  phase tied to the bunch's progress through the lattice, NOT
     *  wallclock time. For relativistic bunches z_centroid/c == t (modulo
     *  offset) so both conventions agree. During the cathode region
     *  (beta<1), they diverge by tens of degrees of RF phase across the
     *  emission window, which appears to be the source of the chirp-
     *  sign reversal seen in the no-SC sanity check vs ImpactT.
     *
     *  When enabled, TrackingLoop computes a per-step centroid z and
     *  uses t_eff = z_centroid/c + t_offset for the field gather time
     *  (replacing wallclock t). t_offset is set so t_eff matches
     *  wallclock t at the moment of bunch emission start (typically
     *  Tini), preserving the RF phase reference. */
    void set_centroid_field_phase (bool on, amrex::Real t_offset = amrex::Real(0.0)) noexcept
    {
        m_use_centroid_phase = on;
        m_centroid_t_offset  = t_offset;
    }
    bool centroid_field_phase_enabled () const noexcept { return m_use_centroid_phase; }

private:
    bool        m_behind_cathode_drift_on = false;
    amrex::Real m_cathode_z               = 0.0;
    amrex::Real m_behind_cathode_betazini = 0.0;
    bool        m_use_centroid_phase      = false;
    amrex::Real m_centroid_t_offset       = 0.0;
};

} // namespace tracking
} // namespace lucretiatt
