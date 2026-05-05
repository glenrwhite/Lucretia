#pragma once

#include "particles/TimeBunch.H"

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>
#include <vector>


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
    enum class PulseShape       { Gaussian, FlatTop, SuperGaussian };
    enum class TransverseProfile { Gaussian, UniformDisk, SuperGaussian };

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
        amrex::Real  mte,
        // SuperGaussian-only extras (ignored unless the corresponding
        // shape is SuperGaussian). alpha matches the distgen "deformable"
        // convention: alpha=1 -> Gaussian, alpha->0 -> uniform-with-edges.
        amrex::Real  pulse_alpha          = amrex::Real(1.0),
        amrex::Real  pulse_slope          = amrex::Real(0.0),
        amrex::Real  transverse_alpha     = amrex::Real(1.0),
        amrex::Real  transverse_truncate  = amrex::Real(0.0),   // 0 -> 4*sigma fallback
        // ImpactT-style emission knobs (default off so existing runs are
        // unchanged). When `longitudinal_thermal` is true, uz is sampled
        // from a half-Maxwell with sigma = sqrt(MTE * |e| / m_e) (matches
        // distgen's MTE convention). When `z_init_spread` > 0, each
        // emitted particle's initial z is uniformly distributed in
        // [cathode_z - z_init_spread, cathode_z]; combined with
        // tracking.behind_cathode_betazini and tracking.behind_cathode_z
        // in the TrackingLoop, this reproduces ImpactT's universal-
        // betazini drift behind the cathode (so all particles cross z=0
        // within a tight window of ~ z_init_spread / (betazini*c) and
        // see the same RF phase, instead of dispersing over 30+ ps the
        // way per-particle thermal-pz drift would).
        bool         longitudinal_thermal = false,
        amrex::Real  z_init_spread        = amrex::Real(0.0))
        : Named(std::move(n))
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
        , m_pulse_alpha(pulse_alpha)
        , m_pulse_slope(pulse_slope)
        , m_transverse_alpha(transverse_alpha)
        , m_transverse_truncate(transverse_truncate)
        , m_longitudinal_thermal(longitudinal_thermal)
        , m_z_init_spread(z_init_spread)
    {
        build_lookup_tables();
    }

    /** Emit particles for the time window [t, t + dt]. Returns count. */
    int emit_new_particles (particles::TimeBunch& bunch,
                            amrex::Real t, amrex::Real dt) const;

    /** Cumulative fraction of total particles emitted by time `t`. */
    amrex::Real pulse_cdf (amrex::Real t) const;

    /** Sample a transverse position (x, y) consistent with the chosen
     *  TransverseProfile. Used internally by emit_new_particles; exposed
     *  here for tests. */
    void sample_xy (amrex::Real& x, amrex::Real& y) const;

    /** Build the tabulated CDFs used by SuperGaussian sampling. Called
     *  by the constructor; safe to re-call (rebuilds from scratch). */
    void build_lookup_tables ();

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
    amrex::Real        m_spot_size              = 0.0;   // m (RMS Gaussian, radius disk, sigma_xy SuperGaussian)
    amrex::Real        m_mte                    = 0.0;   // eV (mean transverse energy)

    // SuperGaussian-only shape parameters.
    //
    // Temporal: pdf(t) ~ exp(-(|t-t0|/(sqrt(2)*sigma))^p) * max(1+slope*t/T_max, 0)
    //   pulse_duration is interpreted as sigma (NOT FWHM) when shape is
    //   SuperGaussian. p = 1/alpha. T_max = 4*sigma is the support.
    // Transverse: pdf(r) ~ r * exp(-(r/(sqrt(2)*sigma))^p)
    //   spot_size is sigma. p = 1/alpha. R_max = transverse_truncate * sigma
    //   (or 4*sigma when transverse_truncate <= 0).
    amrex::Real m_pulse_alpha          = 1.0;
    amrex::Real m_pulse_slope          = 0.0;
    amrex::Real m_transverse_alpha     = 1.0;
    amrex::Real m_transverse_truncate  = 0.0;

    // Tabulated cdfs for SuperGaussian. Built by build_lookup_tables().
    // 4096 points each, matching the MATLAB CathodeLaserDist sampler so
    // the two produce statistically-identical distributions.
    std::vector<amrex::Real> m_pulse_cdf_table;        // cdf at uniform t in [t0-T_max, t0+T_max]
    amrex::Real              m_pulse_cdf_t_lo = 0.0;
    amrex::Real              m_pulse_cdf_t_hi = 0.0;
    std::vector<amrex::Real> m_radial_inv_cdf_table;   // r at uniform u in [0,1]

    // ImpactT-style emission options (off by default).
    bool        m_longitudinal_thermal = false;
    amrex::Real m_z_init_spread        = 0.0;     // m: initial z in [z_cathode - spread, z_cathode]
};

} // namespace elements
} // namespace lucretiatt
