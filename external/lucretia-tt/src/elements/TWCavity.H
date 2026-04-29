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

/** Forward TM traveling-wave cavity, constant amplitude.
 *
 *  On-axis longitudinal field:
 *    E_z(0,0,z,t) = peak_field * cos(omega t - kz z_local + phi)
 *  where omega = 2 pi f_RF and kz = omega / v_phase. v_phase = c gives
 *  a synchronous wave for relativistic particles.
 *
 *  Paraxial extension (Maxwell, axisymmetric TM):
 *    E_r   = -(r/2) dE_z/dz
 *    B_phi =  (r / 2 c^2) dE_z/dt
 *  Same projection to Cartesian as SWCavity.
 *
 *  Spatial gating: outside [m_z, m_z + m_length] all components return 0.
 *
 *  Phase 2B simplification: constant amplitude. Constant-gradient or
 *  constant-impedance attenuation profiles can layer on by multiplying
 *  E_0 by an envelope and adjusting dE_z/dz to include the envelope
 *  derivative -- planned for Phase 2C. */
struct TWCavity
    : public Named
    , public Position3D
    , public TimeWindow
{
    TWCavity () = default;

    TWCavity (
        std::string n,
        amrex::Real z0, amrex::Real length,
        amrex::Real f_RF,
        amrex::Real peak_field,
        amrex::Real phase,
        amrex::Real v_phase = amrex::Real(299792458.0))
        : Named(std::move(n))
        , Position3D(0.0, 0.0, z0, length)
        , TimeWindow()
        , m_f_RF(f_RF)
        , m_peak_field(peak_field)
        , m_phase(phase)
        , m_v_phase(v_phase)
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
        const amrex::Real kz    = omega / m_v_phase;

        const amrex::Real arg   = omega * t - kz * z_local + m_phase;
        const amrex::Real cos_a = std::cos(arg);
        const amrex::Real sin_a = std::sin(arg);

        const amrex::Real Ez_axis = m_peak_field * cos_a;
        // d/dz of cos(omega*t - kz*z + phi) = +kz * sin(...)
        const amrex::Real dEz_dz  = m_peak_field * kz    * sin_a;
        // d/dt = -omega * sin(...)
        const amrex::Real dEz_dt  = -m_peak_field * omega * sin_a;

        Ez = Ez_axis;
        Ex = amrex::Real(-0.5) * x * dEz_dz;
        Ey = amrex::Real(-0.5) * y * dEz_dz;

        const amrex::Real fB = inv_2c2 * dEz_dt;
        Bx = -y * fB;
        By =  x * fB;
        Bz = 0.0;
    }

    amrex::Real m_f_RF       = 2.856e9;
    amrex::Real m_peak_field = 0.0;
    amrex::Real m_phase      = 0.0;
    amrex::Real m_v_phase    = 299792458.0;
};

} // namespace elements
} // namespace lucretiatt
