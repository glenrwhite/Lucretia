#include "CathodeSource.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>

#include <cmath>
#include <vector>


namespace lucretiatt {
namespace elements {

namespace {

constexpr amrex::Real kElectronCharge = -1.602176634e-19;  // C
constexpr amrex::Real kElectronMass   =  9.1093837015e-31; // kg
constexpr amrex::Real kPi             =  3.14159265358979323846;
constexpr amrex::Real kSqrt2          =  1.41421356237309504880;
constexpr amrex::Real kFwhmToSigma    =  1.0 / (2.0 * 1.17741002251547469);

constexpr int kSGTableN = 4096;     // matches the MATLAB CathodeLaserDist sampler

} // anonymous


void CathodeSource::build_lookup_tables ()
{
    using amrex::Real;

    // ---- Temporal SuperGaussian: forward CDF on a uniform t-grid ----
    if (m_pulse_shape == PulseShape::SuperGaussian && m_pulse_duration > 0.0) {
        const Real sigma = m_pulse_duration;          // sigma (not FWHM) for SG
        const Real alpha = std::max(m_pulse_alpha, Real(1e-3));
        const Real p     = Real(1.0) / alpha;
        const Real T_max = Real(4.0) * sigma;

        m_pulse_cdf_t_lo = m_pulse_t0 - T_max;
        m_pulse_cdf_t_hi = m_pulse_t0 + T_max;
        m_pulse_cdf_table.assign(kSGTableN, Real(0.0));

        // Compute pdf on the uniform grid, then cumsum & normalise.
        std::vector<Real> pdf(kSGTableN);
        for (int i = 0; i < kSGTableN; ++i) {
            const Real t_grid = m_pulse_cdf_t_lo
                + (m_pulse_cdf_t_hi - m_pulse_cdf_t_lo) * Real(i) / Real(kSGTableN - 1);
            const Real arg    = std::abs(t_grid - m_pulse_t0) / (kSqrt2 * sigma);
            const Real env    = std::exp(-std::pow(arg, p));
            const Real ramp   = Real(1.0) + m_pulse_slope * (t_grid - m_pulse_t0) / T_max;
            pdf[i] = env * std::max(ramp, Real(0.0));
        }
        Real sum = 0.0;
        for (int i = 0; i < kSGTableN; ++i) {
            sum += pdf[i];
            m_pulse_cdf_table[i] = sum;
        }
        if (sum > 0.0) {
            for (int i = 0; i < kSGTableN; ++i) m_pulse_cdf_table[i] /= sum;
        }
    } else {
        m_pulse_cdf_table.clear();
    }

    // ---- Transverse SuperGaussian: inverse CDF table (r at uniform u) ----
    if (m_transverse_profile == TransverseProfile::SuperGaussian && m_spot_size > 0.0) {
        const Real sigma = m_spot_size;
        const Real alpha = std::max(m_transverse_alpha, Real(1e-3));
        const Real p     = Real(1.0) / alpha;
        const Real R_max = (m_transverse_truncate > 0.0)
                           ? (m_transverse_truncate * sigma)
                           : (Real(4.0) * sigma);

        // Build pdf(r) ~ r * exp(-(r/(sqrt(2)*sigma))^p), integrate to a CDF
        // on the uniform r-grid, then invert to get r(u) at uniform u in [0,1].
        std::vector<Real> r_grid(kSGTableN), pdf(kSGTableN), cdf(kSGTableN);
        for (int i = 0; i < kSGTableN; ++i) {
            r_grid[i] = R_max * Real(i) / Real(kSGTableN - 1);
            const Real arg = r_grid[i] / (kSqrt2 * sigma);
            pdf[i] = r_grid[i] * std::exp(-std::pow(arg, p));
        }
        Real sum = 0.0;
        for (int i = 0; i < kSGTableN; ++i) { sum += pdf[i]; cdf[i] = sum; }
        if (sum > 0.0) for (int i = 0; i < kSGTableN; ++i) cdf[i] /= sum;

        // Tabulated inverse: r at uniform u_grid in [0, 1].
        m_radial_inv_cdf_table.assign(kSGTableN, Real(0.0));
        int j = 0;
        for (int i = 0; i < kSGTableN; ++i) {
            const Real u = Real(i) / Real(kSGTableN - 1);
            while (j + 1 < kSGTableN && cdf[j + 1] < u) ++j;
            if (j + 1 >= kSGTableN || cdf[j + 1] <= cdf[j]) {
                m_radial_inv_cdf_table[i] = r_grid[kSGTableN - 1];
            } else {
                const Real frac = (u - cdf[j]) / (cdf[j + 1] - cdf[j]);
                m_radial_inv_cdf_table[i] = r_grid[j] + frac * (r_grid[j + 1] - r_grid[j]);
            }
        }
    } else {
        m_radial_inv_cdf_table.clear();
    }
}


void CathodeSource::sample_xy (amrex::Real& x, amrex::Real& y) const
{
    using amrex::Real;
    switch (m_transverse_profile) {
    case TransverseProfile::Gaussian: {
        x = m_spot_size * Real(amrex::RandomNormal(0.0, 1.0));
        y = m_spot_size * Real(amrex::RandomNormal(0.0, 1.0));
        break;
    }
    case TransverseProfile::UniformDisk: {
        const Real r  = m_spot_size * std::sqrt(Real(amrex::Random()));
        const Real th = Real(2.0) * kPi * Real(amrex::Random());
        x = r * std::cos(th);
        y = r * std::sin(th);
        break;
    }
    case TransverseProfile::SuperGaussian: {
        // Inverse-CDF table lookup
        const Real u  = Real(amrex::Random());
        const Real fi = u * Real(kSGTableN - 1);
        int        i  = int(fi);
        if (i < 0)               i = 0;
        if (i >= kSGTableN - 1)  i = kSGTableN - 2;
        const Real frac = fi - Real(i);
        const Real r    = m_radial_inv_cdf_table.empty()
            ? Real(0.0)
            : m_radial_inv_cdf_table[i]
              + frac * (m_radial_inv_cdf_table[i + 1] - m_radial_inv_cdf_table[i]);
        const Real th = Real(2.0) * kPi * Real(amrex::Random());
        x = r * std::cos(th);
        y = r * std::sin(th);
        break;
    }
    }
}


amrex::Real CathodeSource::pulse_cdf (amrex::Real t) const
{
    switch (m_pulse_shape) {
    case PulseShape::Gaussian: {
        if (m_pulse_duration <= 0.0) { return (t >= m_pulse_t0) ? 1.0 : 0.0; }
        const amrex::Real sigma = m_pulse_duration * kFwhmToSigma;
        const amrex::Real arg   = (t - m_pulse_t0) / (kSqrt2 * sigma);
        return amrex::Real(0.5) * (amrex::Real(1.0) + std::erf(arg));
    }
    case PulseShape::FlatTop: {
        if (m_pulse_duration <= 0.0) { return (t >= m_pulse_t0) ? 1.0 : 0.0; }
        const amrex::Real f = (t - m_pulse_t0) / m_pulse_duration;
        if (f <= 0.0) { return 0.0; }
        if (f >= 1.0) { return 1.0; }
        return f;
    }
    case PulseShape::SuperGaussian: {
        if (m_pulse_cdf_table.empty()) {
            return (t >= m_pulse_t0) ? amrex::Real(1.0) : amrex::Real(0.0);
        }
        if (t <= m_pulse_cdf_t_lo) return amrex::Real(0.0);
        if (t >= m_pulse_cdf_t_hi) return amrex::Real(1.0);
        const amrex::Real fi = (t - m_pulse_cdf_t_lo)
            / (m_pulse_cdf_t_hi - m_pulse_cdf_t_lo) * amrex::Real(kSGTableN - 1);
        const int i = std::min(int(fi), kSGTableN - 2);
        const amrex::Real frac = fi - amrex::Real(i);
        return m_pulse_cdf_table[i]
             + frac * (m_pulse_cdf_table[i + 1] - m_pulse_cdf_table[i]);
    }
    }
    return 0.0;
}


int CathodeSource::emit_new_particles (
    particles::TimeBunch& bunch,
    amrex::Real t, amrex::Real dt) const
{
    if (m_n_macroparticles_total <= 0 || dt <= 0.0) { return 0; }

    const amrex::Real cdf_t  = pulse_cdf(t);
    const amrex::Real cdf_tp = pulse_cdf(t + dt);
    const amrex::Real n_real = amrex::Real(m_n_macroparticles_total) * (cdf_tp - cdf_t);
    int n_emit = int(std::round(n_real));
    if (n_emit <= 0) { return 0; }

    // Per-particle weight: total_charge in macroparticle units of |e|.
    // Each macroparticle has q = -|e| (electron) and weight w. The
    // physical charge contribution is q * w. For the bunch sum to equal
    // -total_charge (electrons): sum(w) * |e| = |total_charge|, so
    // each macroparticle gets w = |total_charge| / (N_total * |e|).
    const amrex::ParticleReal weight =
        amrex::ParticleReal(std::abs(m_total_charge)
                            / (m_n_macroparticles_total * std::abs(kElectronCharge)));

    // Thermal momentum spread from MTE.
    // Per the MTE convention: mean transverse energy per particle = MTE
    // (counting both x and y), so per-direction KE = MTE/2.
    // Non-relativistic: p_T = sqrt(2 m KE_T), per direction
    //   sigma_p_per_dim = sqrt(m * MTE * |e|)
    // We store proper velocity u = p/m, so:
    //   sigma_u_per_dim = sqrt(MTE * |e| / m_e)
    const amrex::Real mte_J = m_mte * std::abs(kElectronCharge);
    const amrex::Real sigma_u = std::sqrt(mte_J / kElectronMass);

    // Sample on rank 0 only. Multi-rank emission is Phase 6.
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        bunch.Redistribute();
        return 0;
    }

    // Initial offset above the cathode. 1 nm gives unphysically huge
    // image-charge force at the first step; 1 um is a reasonable
    // approximation to the photoemission depth and keeps the image-force
    // bounded enough for Boris to integrate cleanly.
    //
    // ImpactT-mode: when m_z_init_spread > 0, particles are seeded
    // BEHIND the cathode (z in [z_cathode - z_init_spread, z_cathode]).
    // The TrackingLoop's behind_cathode_drift advances them at universal
    // betazini until they cross z=0, then normal Boris kicks in. This
    // mirrors ImpactT's partcl.data + driftemission_BeamBunch model and
    // avoids the per-particle thermal-pz dispersion that explodes
    // sigma_z when the particles don't share a synchronized drift.
    constexpr amrex::Real z_offset = amrex::Real(1.0e-6);

    std::vector<amrex::ParticleReal> xs(n_emit), ys(n_emit), zs(n_emit);
    std::vector<amrex::ParticleReal> uxs(n_emit), uys(n_emit), uzs(n_emit);

    for (int i = 0; i < n_emit; ++i)
    {
        // Transverse position via the configured profile (Gaussian,
        // UniformDisk, or SuperGaussian via inverse-CDF table).
        amrex::Real xi = 0.0, yi = 0.0;
        sample_xy(xi, yi);
        xs[i] = amrex::ParticleReal(xi);
        ys[i] = amrex::ParticleReal(yi);
        if (m_z_init_spread > amrex::Real(0.0)) {
            // Uniform random behind cathode. Combined with the universal-
            // betazini drift in TrackingLoop, this gives a uniform spread
            // in cross-cathode times of width z_init_spread/(betazini*c).
            zs[i] = m_z_cathode
                  - m_z_init_spread * amrex::ParticleReal(amrex::Random());
        } else {
            zs[i] = m_z_cathode + z_offset;
        }

        // Thermal momentum (proper-velocity units)
        uxs[i] = sigma_u * amrex::ParticleReal(amrex::RandomNormal(0.0, 1.0));
        uys[i] = sigma_u * amrex::ParticleReal(amrex::RandomNormal(0.0, 1.0));
        if (m_longitudinal_thermal) {
            // Half-Maxwell on uz: |Normal(0, sigma_u)|. Per-particle
            // thermal longitudinal momentum matching the MTE = m_mte
            // convention (mean uz = sigma_u * sqrt(2/pi), std uz =
            // sigma_u * sqrt(1 - 2/pi)). With m_mte = 414 meV this gives
            // mean uz ~ 2.16e5 m/s, std uz ~ 1.63e5 m/s -- numerically
            // identical to ImpactT's distgen-generated partcl.data
            // distribution (verified 2026-05-04).
            uzs[i] = sigma_u
                * amrex::ParticleReal(std::abs(amrex::RandomNormal(0.0, 1.0)));
        } else {
            uzs[i] = 0.0;  // legacy: no longitudinal kick at emission
        }
    }

    // Birth time: sample UNIFORMLY in [t, t+dt] per particle.
    //
    // Previously all particles in an emission step got the same
    // mid-step t_birth = t + dt/2, then were pushed by full dt with
    // identical initial conditions (z = z_cathode + 1 um, uz = 0,
    // thermal MTE on ux,uy). Result: particles in one emission step
    // saw IDENTICAL field history -> identical final energy ->
    // discrete energy clusters (one spike per emission step) in the
    // exit beam. With 1M macros over ~20 emission steps this gave
    // P-spikes 5x the median bin count, visible in fort.18-style
    // histograms.
    //
    // Sampling t_birth uniformly within [t, t+dt] per particle gives
    // each particle a different effective alive-duration on its first
    // step. The TrackingLoop kernel uses dt_eff = min(dt, t+dt-t_birth)
    // to accumulate field kicks over only that fraction of dt for
    // freshly-emitted particles. This breaks the synchronization that
    // produced the spikes without changing the bunch's mean dynamics.
    std::vector<amrex::ParticleReal> t_births(n_emit);
    for (int i = 0; i < n_emit; ++i) {
        t_births[i] = amrex::ParticleReal(
            t + dt * amrex::Real(amrex::Random()));
    }

    bunch.AddParticlesFromArrays(
        n_emit,
        xs.data(), ys.data(), zs.data(),
        uxs.data(), uys.data(), uzs.data(),
        weight, t_births.data());

    return n_emit;
}

} // namespace elements
} // namespace lucretiatt
