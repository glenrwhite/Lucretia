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

/** Rectangular aperture collimator (mirror of ImpactT type-(-11)).
 *
 *  Active over the lab z range [z_start, z_end]. Particles inside this z
 *  range with x outside [xmin, xmax] OR y outside [ymin, ymax] are
 *  KILLED (alive=0, parked at z=-1e9 to keep them out of bunch stats).
 *
 *  Optional pz<0 filter: with kill_backward = true, particles with
 *  uz<0 anywhere in [z_start, z_end] are also killed. This mirrors
 *  ImpactT's `stop_bkw` flag (negative dz in the type-(-11) line).
 *
 *  Applied at the end of every integration step in TrackingLoop, after
 *  the second half-drift / first-order emission update. Cheap: a single
 *  per-particle bounds check. Particles already dead skip the test.
 *
 *  Inherits TimeWindow + provides a no-op gather_E_B so it satisfies
 *  the variant's lambda contract for the field-gather sweep (it adds
 *  nothing to the field). */
struct Collimator : public Named, public TimeWindow
{
    Collimator () = default;
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

    /** Apply the aperture to every alive particle in the bunch. */
    void apply (particles::TimeBunch& bunch) const;

    amrex::Real m_z_start       = 0.0;
    amrex::Real m_z_end         = 0.0;
    amrex::Real m_xmin          = 0.0;
    amrex::Real m_xmax          = 0.0;
    amrex::Real m_ymin          = 0.0;
    amrex::Real m_ymax          = 0.0;
    bool        m_kill_backward = false;
};

} // namespace elements
} // namespace lucretiatt
