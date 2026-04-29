#pragma once

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>


namespace lucretiatt {
namespace elements {

/** Uniform magnetostatic field over its spatial + temporal extent.
 *
 *  Useful as a test element (cyclotron benchmark) and as the simplest
 *  approximation to a long solenoid. Position3D defaults make it
 *  cover the entire domain; TimeWindow defaults make it always-on. */
struct UniformB
    : public Named
    , public Position3D
    , public TimeWindow
{
    UniformB () = default;
    UniformB (std::string n, amrex::Real Bx, amrex::Real By, amrex::Real Bz)
        : Named(std::move(n))
        , Position3D()
        , TimeWindow()
        , m_Bx(Bx), m_By(By), m_Bz(Bz)
    {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
                     amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0;  Ey = 0.0;  Ez = 0.0;
        Bx = m_Bx; By = m_By; Bz = m_Bz;
    }

    amrex::Real m_Bx = 0.0;
    amrex::Real m_By = 0.0;
    amrex::Real m_Bz = 0.0;
};

} // namespace elements
} // namespace lucretiatt
