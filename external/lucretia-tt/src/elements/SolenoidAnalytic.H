#pragma once

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>
#include <utility>


namespace lucretiatt {
namespace elements {

/** Analytic solenoid: Gaussian on-axis B_z profile, paraxial B_r.
 *
 *  On-axis (analytic stand-in for a POISSON map):
 *    B_z(0,0,z) = peak_Bz * exp(-((z_local - z_peak) / sigma)^2)
 *  where z_local = z - m_z, z_peak is measured from the element start.
 *  All other components are derived from Maxwell axisymmetric expansion:
 *    B_r = -(r/2) dB_z/dz   (paraxial)
 *  In Cartesian:  B_x = -(x/2) dB_z/dz,  B_y = -(y/2) dB_z/dz
 *  (the 1/r and r factors cancel since B_r r_hat = (B_r/r)*(x x_hat + y y_hat).)
 *
 *  Spatial gating: outside [m_z, m_z + m_length] all components are 0.
 *  No E field, no time dependence.
 *
 *  Phase 2B uses an analytic Gaussian profile so we can validate Larmor
 *  rotation without needing a POISSON file. Phase 2C will replace this
 *  with a file-loaded 2D B_z(r,z) reader (Solenoid3D). */
struct SolenoidAnalytic
    : public Named
    , public Position3D
    , public TimeWindow
{
    SolenoidAnalytic () = default;

    SolenoidAnalytic (
        std::string n,
        amrex::Real z0, amrex::Real length,
        amrex::Real peak_Bz, amrex::Real sigma, amrex::Real z_peak)
        : Named(std::move(n))
        , Position3D(0.0, 0.0, z0, length)
        , TimeWindow()
        , m_peak_Bz(peak_Bz)
        , m_sigma(sigma)
        , m_z_peak(z_peak)
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

        const amrex::Real u  = (z_local - m_z_peak) / m_sigma;
        const amrex::Real e  = std::exp(-u * u);
        const amrex::Real Bz_axis = m_peak_Bz * e;
        // d/dz exp(-u^2) = (-2u/sigma) exp(-u^2)
        const amrex::Real dBz_dz  = -m_peak_Bz * amrex::Real(2.0) * u / m_sigma * e;

        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = amrex::Real(-0.5) * x * dBz_dz;
        By = amrex::Real(-0.5) * y * dBz_dz;
        Bz = Bz_axis;
    }

    amrex::Real m_peak_Bz = 0.0;   // T  (peak on-axis field)
    amrex::Real m_sigma   = 0.0;   // m  (Gaussian RMS width)
    amrex::Real m_z_peak  = 0.0;   // m  (location of peak, relative to m_z)
};

} // namespace elements
} // namespace lucretiatt
