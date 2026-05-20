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

// van der Corput sequence in base b. Bit-reverses the integer i in base b.
amrex::Real van_der_corput (int i, int b)
{
    amrex::Real x = 0.0;
    amrex::Real f = 1.0 / amrex::Real(b);
    int k = i;
    while (k > 0) {
        x += f * amrex::Real(k % b);
        k /= b;
        f /= amrex::Real(b);
    }
    return x;
}

// Acklam's rational approximation to the inverse standard-Normal CDF.
// Accurate to ~1e-9 over (1e-15, 1 - 1e-15). Used to convert Hammersley
// uniforms to Normal samples (replaces amrex::RandomNormal).
amrex::Real normal_inv_cdf (amrex::Real p)
{
    static constexpr amrex::Real a[6] = {
        -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00};
    static constexpr amrex::Real b[5] = {
        -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01};
    static constexpr amrex::Real c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00};
    static constexpr amrex::Real d[4] = {
         7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00};
    if (p < 1e-15)        p = 1e-15;
    else if (p > 1 - 1e-15) p = 1 - 1e-15;
    amrex::Real q, r, x;
    if (p < 0.02425) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p > 0.97575) {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    return x;
}

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


void CathodeSource::sample_xy (amrex::Real u_r, amrex::Real u_phi,
                               amrex::Real& x, amrex::Real& y) const
{
    using amrex::Real;
    // Clamp u_r away from 1.0 to keep log() finite (Gaussian-radial path).
    if (u_r < 1e-15)        u_r = 1e-15;
    else if (u_r > 1 - 1e-15) u_r = 1 - 1e-15;
    const Real th = Real(2.0) * kPi * u_phi;
    switch (m_transverse_profile) {
    case TransverseProfile::Gaussian: {
        // Radial Gaussian inverse CDF: r = sigma * sqrt(-2 ln(1 - u)).
        // Matches CathodeLaserDist's r_type='gaussian' sampling so the
        // joint (r, phi) realization is the same dimensionality and
        // ordering as imp's partcl.data.
        const Real r = m_spot_size * std::sqrt(-2.0 * std::log(1.0 - u_r));
        x = r * std::cos(th);
        y = r * std::sin(th);
        break;
    }
    case TransverseProfile::UniformDisk: {
        const Real r = m_spot_size * std::sqrt(u_r);
        x = r * std::cos(th);
        y = r * std::sin(th);
        break;
    }
    case TransverseProfile::SuperGaussian: {
        // Inverse-CDF table lookup
        const Real fi = u_r * Real(kSGTableN - 1);
        int        i  = int(fi);
        if (i < 0)               i = 0;
        if (i >= kSGTableN - 1)  i = kSGTableN - 2;
        const Real frac = fi - Real(i);
        const Real r    = m_radial_inv_cdf_table.empty()
            ? Real(0.0)
            : m_radial_inv_cdf_table[i]
              + frac * (m_radial_inv_cdf_table[i + 1] - m_radial_inv_cdf_table[i]);
        x = r * std::cos(th);
        y = r * std::sin(th);
        break;
    }
    }
}


void CathodeSource::build_hammersley_table () const
{
    if (m_n_macroparticles_total <= 0) { m_hammersley_built = true; return; }
    const int N = m_n_macroparticles_total;
    m_hammersley.assign(std::size_t(N) * kHammersleyDims, amrex::Real(0.0));
    // Dim 0: i/N (best uniformity -- maps onto t_birth fraction)
    // Dims 1..5: van der Corput in primes 2, 3, 5, 7, 11
    static constexpr int primes[5] = {2, 3, 5, 7, 11};
    for (int i = 0; i < N; ++i) {
        m_hammersley[std::size_t(i) * kHammersleyDims + 0] =
            (amrex::Real(i) + amrex::Real(0.5)) / amrex::Real(N);
        for (int d = 1; d < kHammersleyDims; ++d) {
            m_hammersley[std::size_t(i) * kHammersleyDims + d] =
                van_der_corput(i + 1, primes[d - 1]);
        }
    }
    m_hammersley_built = true;
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

    // Lazily build the Hammersley low-discrepancy table on first emission.
    // All per-particle uniforms below are read from this table by global
    // index (m_n_emitted_count + i), so the marginal AND joint distributions
    // match the MATLAB CathodeLaserDist sampler that produces ImpactT's
    // partcl.data -- this drops per-slice sampling noise vs the prior
    // amrex::Random/RandomNormal path.
    if (!m_hammersley_built) { build_hammersley_table(); }

    // Cap n_emit at the remaining table budget. The CDF-based n_emit estimate
    // can over-shoot the total in the final emission step due to round().
    if (m_n_emitted_count + n_emit > m_n_macroparticles_total) {
        n_emit = m_n_macroparticles_total - m_n_emitted_count;
    }
    if (n_emit <= 0) { return 0; }

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
    std::vector<amrex::ParticleReal> t_births(n_emit);

    for (int i = 0; i < n_emit; ++i)
    {
        const std::size_t row = std::size_t(m_n_emitted_count + i)
                              * kHammersleyDims;
        const amrex::Real u_t   = m_hammersley[row + 0];
        const amrex::Real u_r   = m_hammersley[row + 1];
        const amrex::Real u_phi = m_hammersley[row + 2];
        const amrex::Real u_ux  = m_hammersley[row + 3];
        const amrex::Real u_uy  = m_hammersley[row + 4];
        const amrex::Real u_z   = m_hammersley[row + 5];   // shared u_uz / z_init slot

        // Transverse position via the configured profile.
        amrex::Real xi = 0.0, yi = 0.0;
        sample_xy(u_r, u_phi, xi, yi);
        xs[i] = amrex::ParticleReal(xi);
        ys[i] = amrex::ParticleReal(yi);

        if (m_z_init_spread > amrex::Real(0.0)) {
            zs[i] = m_z_cathode
                  - m_z_init_spread * amrex::ParticleReal(u_z);
        } else {
            zs[i] = m_z_cathode + z_offset;
        }

        // Thermal momentum: invert standard-Normal CDF on Hammersley
        // uniforms. Matches CathodeLaserDist::generate's gauss_inv path.
        uxs[i] = sigma_u * amrex::ParticleReal(normal_inv_cdf(u_ux));
        uys[i] = sigma_u * amrex::ParticleReal(normal_inv_cdf(u_uy));
        if (m_longitudinal_thermal) {
            // Half-Maxwell on uz: |Normal(0, sigma_u)|. Re-uses the same
            // Hammersley dim that drives z_init_spread (the two ImpactT-
            // style options aren't used together in practice).
            uzs[i] = sigma_u * amrex::ParticleReal(
                std::abs(normal_inv_cdf(u_z)));
        } else {
            uzs[i] = 0.0;
        }

        // Birth time: spread within [t, t+dt] using the Hammersley
        // best-uniformity dim (i/N). Per-particle stagger breaks the
        // emission-step P-spike pattern (see prior comment) without
        // injecting RNG noise.
        t_births[i] = amrex::ParticleReal(t + dt * u_t);
    }

    bunch.AddParticlesFromArrays(
        n_emit,
        xs.data(), ys.data(), zs.data(),
        uxs.data(), uys.data(), uzs.data(),
        weight, t_births.data());

    m_n_emitted_count += n_emit;
    return n_emit;
}

} // namespace elements
} // namespace lucretiatt
