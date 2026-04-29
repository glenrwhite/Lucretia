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

/** Hard-edge analytic magnetic quadrupole.
 *
 *  Field within [m_z, m_z + m_length]:
 *    B_x = G * y
 *    B_y = G * x
 *    B_z = 0
 *  Outside this z-range, all components = 0.
 *
 *  G is the gradient in T/m (positive = horizontally focusing for
 *  positive charges; for electrons, sign flips). Phase 2A uses a strict
 *  hard edge -- no fringe-field model. An Enge-function fringe extension
 *  can be layered on as a sibling element in Phase 2B without touching
 *  this struct. */
struct QuadAnalytic
    : public Named
    , public Position3D
    , public TimeWindow
{
    QuadAnalytic () = default;

    QuadAnalytic (
        std::string n,
        amrex::Real z0, amrex::Real length,
        amrex::Real gradient_T_per_m)
        : Named(std::move(n))
        , Position3D(0.0, 0.0, z0, length)
        , TimeWindow()
        , m_gradient(gradient_T_per_m)
    {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        const amrex::Real z_local = z - m_z;
        if (z_local < amrex::Real(0.0) || z_local > m_length) {
            Ex = 0.0; Ey = 0.0; Ez = 0.0;
            Bx = 0.0; By = 0.0; Bz = 0.0;
            return;
        }
        Ex = 0.0;       Ey = 0.0;       Ez = 0.0;
        Bx = m_gradient * y;
        By = m_gradient * x;
        Bz = 0.0;
    }

    amrex::Real m_gradient = 0.0;  // T/m
};

} // namespace elements
} // namespace lucretiatt
