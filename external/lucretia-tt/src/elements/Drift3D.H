#pragma once

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>


namespace lucretiatt {
namespace elements {

/** Vacuum drift over [m_z, m_z + m_length] x [m_t_start, m_t_stop].
 *
 *  In time-based PIC the drift contributes no external field; particles
 *  evolve under their own space-charge and any other active elements'
 *  fields. Aperture/diagnostic checks may layer on in later phases. */
struct Drift3D
    : public Named
    , public Position3D
    , public TimeWindow
{
    Drift3D () = default;
    Drift3D (std::string n,
             amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real length)
        : Named(std::move(n))
        , Position3D(x, y, z, length)
        , TimeWindow()
    {}

    /** External-field contribution at (x,y,z,t). Drift contributes zero. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
                     amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;
    }
};

} // namespace elements
} // namespace lucretiatt
