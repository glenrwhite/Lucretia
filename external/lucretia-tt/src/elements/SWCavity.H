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

/** N-cell pi-mode standing-wave cavity, analytic on-axis profile.
 *
 *  On-axis longitudinal field, sine profile (zero at endplates):
 *    E_z(0,0,z,t) = peak_field * sin(N pi z_local / L) * cos(omega t + phi)
 *  where z_local = z - z0, L = m_length, omega = 2 pi f_RF.
 *
 *  Paraxial extension off-axis (Maxwell, axisymmetric TM-like):
 *    E_r   = -(r/2) dE_z/dz
 *    B_phi =  (r / 2 c^2) dE_z/dt
 *  The (1/r) factors cancel cleanly when projecting back to Cartesian,
 *  so the kernel below works for r = 0 too.
 *
 *  Spatial gating is built in: outside [m_z, m_z + m_length] all
 *  components return 0. Time gating is left to the visit-side
 *  active(t) check. Phase 0 means peak axial field at t = 0. */
struct SWCavity
    : public Named
    , public Position3D
    , public TimeWindow
{
    SWCavity () = default;

    SWCavity (
        std::string n,
        amrex::Real z0, amrex::Real length,
        int          n_cells,
        amrex::Real  f_RF,
        amrex::Real  peak_field,
        amrex::Real  phase)
        : Named(std::move(n))
        , Position3D(0.0, 0.0, z0, length)
        , TimeWindow()
        , m_n_cells(n_cells)
        , m_f_RF(f_RF)
        , m_peak_field(peak_field)
        , m_phase(phase)
    {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        const amrex::Real z_local = z - m_z;
        if (z_local < amrex::Real(0.0) || z_local > m_length) {
            Ex = 0.0; Ey = 0.0; Ez = 0.0;
            Bx = 0.0; By = 0.0; Bz = 0.0;
            return;
        }

        constexpr amrex::Real pi = amrex::Real(3.14159265358979323846);
        constexpr amrex::Real c  = amrex::Real(299792458.0);
        constexpr amrex::Real inv_2c2 = amrex::Real(1.0) / (amrex::Real(2.0) * c * c);

        const amrex::Real omega = amrex::Real(2.0) * pi * m_f_RF;
        const amrex::Real kz    = amrex::Real(m_n_cells) * pi / m_length;

        const amrex::Real cos_t = std::cos(omega * t + m_phase);
        const amrex::Real sin_t = std::sin(omega * t + m_phase);
        const amrex::Real sin_z = std::sin(kz * z_local);
        const amrex::Real cos_z = std::cos(kz * z_local);

        const amrex::Real Ez_axis = m_peak_field * sin_z * cos_t;
        const amrex::Real dEz_dz  = m_peak_field * kz   * cos_z * cos_t;
        const amrex::Real dEz_dt  = -m_peak_field * sin_z * omega * sin_t;

        Ez = Ez_axis;
        Ex = amrex::Real(-0.5) * x * dEz_dz;
        Ey = amrex::Real(-0.5) * y * dEz_dz;

        const amrex::Real fB = inv_2c2 * dEz_dt;
        Bx = -y * fB;
        By =  x * fB;
        Bz = 0.0;
    }

    int         m_n_cells    = 1;
    amrex::Real m_f_RF       = 2.856e9;   // S-band default
    amrex::Real m_peak_field = 0.0;       // V/m (on-axis crest amplitude)
    amrex::Real m_phase      = 0.0;       // rad
};

} // namespace elements
} // namespace lucretiatt
