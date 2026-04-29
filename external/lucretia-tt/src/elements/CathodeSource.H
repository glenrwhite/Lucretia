#pragma once

#include "particles/TimeBunch.H"

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>


namespace lucretiatt {
namespace elements {

/** Particle source: emits macroparticles over time at a planar cathode.
 *
 *  Two responsibilities:
 *
 *  1. Emission. Each timestep, emit_new_particles(bunch, t, dt) computes
 *     the per-step emission count from the difference of the pulse-shape
 *     CDF at t and t+dt, samples the per-particle (x, y, ux, uy)
 *     distributions, and adds the particles to bunch via
 *     TimeBunch::AddParticlesFromArrays.
 *
 *     The total over the run is approximately n_macroparticles_total
 *     (modulo per-step rounding). Per-particle weight is set so the sum
 *     |q| * w over emitted particles equals total_charge.
 *
 *     All emission happens on the IO rank in Phase 3. Rank-distributed
 *     emission is a Phase 6 enhancement.
 *
 *  2. Image charge plane. When image_charge_enabled is true, the
 *     TrackingLoop applies a self-image E_z = -q / (16 pi eps0 z_above^2)
 *     to every particle whose z is above z_cathode. The kernel handles
 *     this -- the element only stores the plane location and toggle.
 *
 *  No external EM field contribution: gather_E_B returns zero. */
struct CathodeSource
    : public Named
    , public Position3D
    , public TimeWindow
{
    enum class PulseShape       { Gaussian, FlatTop };
    enum class TransverseProfile { Gaussian, UniformDisk };

    CathodeSource () = default;

    CathodeSource (
        std::string  n,
        amrex::Real  z_cathode,
        bool         image_charge_enabled,
        int          n_macroparticles_total,
        amrex::Real  total_charge,
        PulseShape   pulse_shape,
        amrex::Real  pulse_duration,
        amrex::Real  pulse_t0,
        TransverseProfile transverse_profile,
        amrex::Real  spot_size,
        amrex::Real  mte)
        : Named(std::move(n))
        // Position3D extends from z_cathode upstream-edge by pulse_duration*c worth
        // of margin; however emission positions are set explicitly so this is just
        // for bookkeeping. Use a generous length.
        , Position3D(0.0, 0.0, z_cathode, amrex::Real(1.0))
        , TimeWindow()
        , m_z_cathode(z_cathode)
        , m_image_charge_enabled(image_charge_enabled)
        , m_n_macroparticles_total(n_macroparticles_total)
        , m_total_charge(total_charge)
        , m_pulse_shape(pulse_shape)
        , m_pulse_duration(pulse_duration)
        , m_pulse_t0(pulse_t0)
        , m_transverse_profile(transverse_profile)
        , m_spot_size(spot_size)
        , m_mte(mte)
    {}

    /** Emit particles for the time window [t, t + dt]. Returns count. */
    int emit_new_particles (particles::TimeBunch& bunch,
                            amrex::Real t, amrex::Real dt) const;

    /** Cumulative fraction of total particles emitted by time `t`. */
    amrex::Real pulse_cdf (amrex::Real t) const;

    /** No external field contribution. Image-charge handled by TrackingLoop. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
                     amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;
    }

    amrex::Real m_z_cathode             = 0.0;
    bool        m_image_charge_enabled  = true;

    int                m_n_macroparticles_total = 0;
    amrex::Real        m_total_charge           = 0.0;   // C
    PulseShape         m_pulse_shape            = PulseShape::Gaussian;
    amrex::Real        m_pulse_duration         = 0.0;   // s (FWHM Gaussian, full flat-top)
    amrex::Real        m_pulse_t0               = 0.0;   // s (centre Gaussian, start flat-top)
    TransverseProfile  m_transverse_profile     = TransverseProfile::Gaussian;
    amrex::Real        m_spot_size              = 0.0;   // m (RMS Gaussian, radius disk)
    amrex::Real        m_mte                    = 0.0;   // eV (mean transverse energy)
};

} // namespace elements
} // namespace lucretiatt
