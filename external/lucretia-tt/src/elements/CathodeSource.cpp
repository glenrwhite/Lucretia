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
constexpr amrex::Real kFwhmToSigma    =  1.0 / (2.0 * 1.17741002251547469);  // 1/(2 sqrt(2 ln 2))

// Mid-point-rule estimate for arbitrary cdfs. We use closed forms below
// for the supported pulse shapes, but this is the fallback.

} // anonymous


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
    constexpr amrex::Real z_offset = amrex::Real(1.0e-6);

    std::vector<amrex::ParticleReal> xs(n_emit), ys(n_emit), zs(n_emit);
    std::vector<amrex::ParticleReal> uxs(n_emit), uys(n_emit), uzs(n_emit);

    for (int i = 0; i < n_emit; ++i)
    {
        // Transverse position
        amrex::ParticleReal xi = 0.0, yi = 0.0;
        switch (m_transverse_profile) {
        case TransverseProfile::Gaussian: {
            xi = m_spot_size * amrex::ParticleReal(amrex::RandomNormal(0.0, 1.0));
            yi = m_spot_size * amrex::ParticleReal(amrex::RandomNormal(0.0, 1.0));
            break;
        }
        case TransverseProfile::UniformDisk: {
            // Uniform on disk of radius m_spot_size: r = R*sqrt(rand), theta = 2*pi*rand
            const amrex::Real r = m_spot_size * std::sqrt(amrex::Random());
            const amrex::Real th = amrex::Real(2.0) * kPi * amrex::Random();
            xi = r * std::cos(th);
            yi = r * std::sin(th);
            break;
        }
        }
        xs[i] = xi;
        ys[i] = yi;
        zs[i] = m_z_cathode + z_offset;

        // Thermal momentum (proper-velocity units)
        uxs[i] = sigma_u * amrex::ParticleReal(amrex::RandomNormal(0.0, 1.0));
        uys[i] = sigma_u * amrex::ParticleReal(amrex::RandomNormal(0.0, 1.0));
        uzs[i] = 0.0;  // no longitudinal kick at emission
    }

    // Birth time: assign mid-window time so particles inherit a
    // consistent timestamp with their emission.
    const amrex::Real t_birth = t + amrex::Real(0.5) * dt;

    bunch.AddParticlesFromArrays(
        n_emit,
        xs.data(), ys.data(), zs.data(),
        uxs.data(), uys.data(), uzs.data(),
        weight, t_birth);

    return n_emit;
}

} // namespace elements
} // namespace lucretiatt
