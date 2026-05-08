#pragma once

#include "mixin/Named.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>


namespace lucretiatt {
namespace particles {
    class TimeBunch;
}

namespace elements {

/** Aperture collimator (mirror of ImpactT type-(-11)).
 *
 *  Two modes:
 *
 *  1. **fire_once mode** (default for ImpactT compat): the collimator
 *     fires ONCE at the integration step when the bunch CENTROID first
 *     crosses z=m_z_target. At that step, ALL alive particles whose
 *     (x, y) lies outside the aperture are killed regardless of each
 *     particle's individual z. This mirrors IMPACT-T's lostXY_BeamBunch:
 *     the trigger is `if(distance.le.tcol .and. (distance+dzz).ge.tcol)`
 *     in AccSimulator.f90:1669, and the kill check itself ignores the
 *     particle's z (BeamBunch.f90:1304-1314). Kept across calls via the
 *     mutable m_fired latch.
 *
 *  2. **z-range mode** (legacy / stop_bkw): the collimator is active
 *     across [m_z_start, m_z_end] and checks each alive particle whose
 *     z falls in that range. Used for backward-particle filters
 *     (kill_backward=true) where the kill semantics are "anywhere
 *     past z_start".
 *
 *  Inherits TimeWindow + provides a no-op gather_E_B so it satisfies
 *  the variant's lambda contract for the field-gather sweep. */
struct Collimator : public Named, public TimeWindow
{
    Collimator () = default;

    /** Legacy z-range constructor (used by stop_bkw and the old API). */
    Collimator (std::string name,
                amrex::Real z_start, amrex::Real z_end,
                amrex::Real xmin, amrex::Real xmax,
                amrex::Real ymin, amrex::Real ymax,
                bool kill_backward = false)
        : Named(std::move(name))
        , TimeWindow()
        , m_z_start(z_start)
        , m_z_end(z_end)
        , m_xmin(xmin)
        , m_xmax(xmax)
        , m_ymin(ymin)
        , m_ymax(ymax)
        , m_kill_backward(kill_backward)
        , m_fire_once(false)
    {}

    /** No external field contribution. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
                     amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;
    }

    /** Apply the aperture: in fire_once mode, kill all alive particles
     *  outside the aperture at the step when the bunch centroid first
     *  crosses m_z_target; in z-range mode, kill alive particles whose
     *  individual z is in [m_z_start, m_z_end] and outside the aperture. */
    void apply (particles::TimeBunch& bunch) const;

    amrex::Real m_z_start       = 0.0;
    amrex::Real m_z_end         = 0.0;
    amrex::Real m_xmin          = 0.0;
    amrex::Real m_xmax          = 0.0;
    amrex::Real m_ymin          = 0.0;
    amrex::Real m_ymax          = 0.0;
    bool        m_kill_backward = false;

    // fire_once mode: when set, ignore [m_z_start, m_z_end] and instead
    // fire when the bunch CENTROID crosses m_z_target. Mirrors IMPACT-T's
    // lostXY trigger semantics. m_fired is mutable so that const apply()
    // can latch the "already fired" state.
    bool         m_fire_once = false;
    amrex::Real  m_z_target  = 0.0;
    mutable bool m_fired     = false;
};

} // namespace elements
} // namespace lucretiatt
