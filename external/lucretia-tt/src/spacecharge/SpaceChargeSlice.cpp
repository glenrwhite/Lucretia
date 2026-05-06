#include "SpaceChargeSlice.H"

#include <AMReX_OpenMP.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>


namespace lucretiatt {
namespace spacecharge {

namespace {
constexpr amrex::Real kPi             = amrex::Real(3.14159265358979323846);
constexpr amrex::Real kEps0           = amrex::Real(8.8541878128e-12);
constexpr amrex::Real kSpeedOfLight   = amrex::Real(299792458.0);
} // anonymous


SpaceChargeSlice::SpaceChargeSlice (int n_slices, amrex::Real radius_factor)
    : m_n_slices(n_slices)
    , m_radius_factor(radius_factor)
{
    if (m_n_slices < 4) { m_n_slices = 4; }
    m_Q_slice.assign(m_n_slices, amrex::Real(0.0));
    m_Ez_slice.assign(m_n_slices, amrex::Real(0.0));
}


void SpaceChargeSlice::compute (particles::TimeBunch& bunch)
{
    using namespace amrex;
    using namespace particles;

    // ---- Pass 1: bunch statistics (z range, mean x/y, sigma_xy from central
    //              second moment, mean beta_z) ----
    Real z_min   =  std::numeric_limits<Real>::infinity();
    Real z_max   = -std::numeric_limits<Real>::infinity();
    Real sum_x   = 0.0;
    Real sum_y   = 0.0;
    Real sum_x2  = 0.0;
    Real sum_y2  = 0.0;
    Real sum_bz  = 0.0;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    long n_alive = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& xs  = soa.GetRealData(RealSoA::x);
        const auto& ys  = soa.GetRealData(RealSoA::y);
        const auto& zs  = soa.GetRealData(RealSoA::z);
        const auto& uxs = soa.GetRealData(RealSoA::px);
        const auto& uys = soa.GetRealData(RealSoA::py);
        const auto& uzs = soa.GetRealData(RealSoA::pz);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            if (alives[i] == 0) { continue; }
            const Real zp = zs[i];
            if (zp < z_min) z_min = zp;
            if (zp > z_max) z_max = zp;
            sum_x  += xs[i];
            sum_y  += ys[i];
            sum_x2 += xs[i] * xs[i];
            sum_y2 += ys[i] * ys[i];
            const Real ux = uxs[i], uy = uys[i], uz = uzs[i];
            const Real gamma = std::sqrt(Real(1.0) +
                (ux*ux + uy*uy + uz*uz) / (kSpeedOfLight * kSpeedOfLight));
            sum_bz += (uz / kSpeedOfLight) / gamma;
            ++n_alive;
        }
    }

    // Reduce n_alive across ranks so all ranks use the same denominator
    // for the second moments.
    long n_alive_g = n_alive;
    ParallelDescriptor::ReduceLongSum(n_alive_g);
    if (n_alive_g == 0) {
        std::fill(m_Q_slice.begin(),  m_Q_slice.end(),  Real(0.0));
        std::fill(m_Ez_slice.begin(), m_Ez_slice.end(), Real(0.0));
        m_total_Q = 0.0;
        return;
    }

    Real z_min_g = z_min, z_max_g = z_max;
    Real sum_x_g  = sum_x,  sum_y_g  = sum_y;
    Real sum_x2_g = sum_x2, sum_y2_g = sum_y2, sum_bz_g = sum_bz;
    ParallelDescriptor::ReduceRealMin(z_min_g);
    ParallelDescriptor::ReduceRealMax(z_max_g);
    ParallelDescriptor::ReduceRealSum(sum_x_g);
    ParallelDescriptor::ReduceRealSum(sum_y_g);
    ParallelDescriptor::ReduceRealSum(sum_x2_g);
    ParallelDescriptor::ReduceRealSum(sum_y2_g);
    ParallelDescriptor::ReduceRealSum(sum_bz_g);

    // Central second moments using only alive particles.
    const Real x_mean = sum_x_g / Real(n_alive_g);
    const Real y_mean = sum_y_g / Real(n_alive_g);
    const Real var_x  = std::max(sum_x2_g / Real(n_alive_g) - x_mean * x_mean, Real(0.0));
    const Real var_y  = std::max(sum_y2_g / Real(n_alive_g) - y_mean * y_mean, Real(0.0));
    const Real sigma_x  = std::sqrt(var_x);
    const Real sigma_y  = std::sqrt(var_y);
    const Real sigma_xy = Real(0.5) * (sigma_x + sigma_y);

    // Bunch radius for the disk-stack formula. Standard photoinjector
    // practice: a = factor * sigma_xy with factor in [1.5, 2.5]. We use
    // 2.0 by default. For a Gaussian transverse, this gives ~85% of the
    // beam charge inside `a`; the formula is approximate anyway.
    m_a = m_radius_factor * sigma_xy;
    if (m_a < Real(1.0e-9)) { m_a = Real(1.0e-9); }      // absolute floor

    const Real beta_z = sum_bz_g / Real(n_alive_g);
    const Real one_minus_b2 = Real(1.0) - beta_z * beta_z;
    m_gamma = (one_minus_b2 > Real(0.0))
              ? Real(1.0) / std::sqrt(one_minus_b2)
              : Real(1.0);

    // ---- Slice grid: span [z_min, z_max] with small pad ----
    const Real z_span = z_max_g - z_min_g;
    if (z_span < Real(1.0e-12)) {
        // Bunch has effectively zero z-extent (e.g. all particles
        // emitted at one instant). No slice gradient → no E_z. Leave
        // Q_slice and Ez_slice at zero from the previous step.
        std::fill(m_Q_slice.begin(),  m_Q_slice.end(),  Real(0.0));
        std::fill(m_Ez_slice.begin(), m_Ez_slice.end(), Real(0.0));
        m_total_Q = 0.0;
        return;
    }
    const Real z_pad = Real(0.02) * z_span;
    Real z_lo_raw = z_min_g - z_pad;
    Real z_hi_raw = z_max_g + z_pad;
    Real dz_raw   = (z_hi_raw - z_lo_raw) / Real(m_n_slices);

    // CRITICAL: the disk-stack formula
    //   E_z(d) ∝ Q * sign(d) * (1 - g|d|/sqrt(g²d² + a²)) / a²
    // saturates to ±σ_s/(2ε₀) at d -> 0±, i.e. it gives the MAXIMUM
    // possible disk-edge field at small |d|. That's physically correct
    // for a true uniform disk of finite radius a — but if the slice
    // spacing dz is << a/γ (as happens whenever the bunch is shorter
    // than its rest-frame transverse radius), every slice ends up
    // exchanging this peak-field force with its neighbours, producing
    // a wildly unphysical longitudinal explosion.
    //
    // Standard photoinjector codes side-step this by holding the slice
    // grid coarser than the natural rest-frame radius. We do the same:
    // floor dz so that the grid spacing in the bunch rest frame
    // (= γ * dz_lab) is at least 0.5 * a. With γ from the mean beta_z,
    // this gives well-conditioned forces from the cathode (γ=1) all
    // the way through L0A (γ ~ 100). Slices get COARSER than the bunch
    // when needed; the bunch then occupies fewer slices but the
    // disk-stack formula stays in its valid regime.
    const Real dz_min = Real(0.1) * m_a / std::max(m_gamma, Real(1.0));
    if (dz_raw < dz_min) {
        const Real z_centre = Real(0.5) * (z_lo_raw + z_hi_raw);
        const Real half_extent = Real(0.5) * dz_min * Real(m_n_slices);
        z_lo_raw = z_centre - half_extent;
        z_hi_raw = z_centre + half_extent;
        dz_raw   = dz_min;
    }
    m_z_lo = z_lo_raw;
    m_z_hi = z_hi_raw;
    m_dz   = dz_raw;

    // ---- Pass 2: CIC-deposit physical charge into z-slices ----
    std::fill(m_Q_slice.begin(), m_Q_slice.end(), Real(0.0));
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& qs = soa.GetRealData(RealSoA::q);
        const auto& ws = soa.GetRealData(RealSoA::w);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            if (alives[i] == 0) { continue; }
            const Real zp = zs[i];
            const Real qw = qs[i] * ws[i];               // physical charge of macroparticle
            const Real f  = (zp - m_z_lo) / m_dz - Real(0.5);
            int idx0 = int(std::floor(f));
            Real frac = f - Real(idx0);
            // Clamp so CIC stays within [0, m_n_slices-1]
            if (idx0 < 0) {
                idx0 = 0; frac = Real(0.0);
            } else if (idx0 + 1 >= m_n_slices) {
                idx0 = m_n_slices - 2; frac = Real(1.0);
            }
            m_Q_slice[idx0]     += qw * (Real(1.0) - frac);
            m_Q_slice[idx0 + 1] += qw * frac;
        }
    }
    ParallelDescriptor::ReduceRealSum(m_Q_slice.data(), m_n_slices);

    // total bunch charge (diagnostic)
    m_total_Q = Real(0.0);
    for (Real q : m_Q_slice) { m_total_Q += q; }

    // ---- Pass 3: E_z at each slice centre via disk-stack sum ----
    //
    // Standard 1D longitudinal SC kernel for a uniform-radius round
    // bunch (Lawson-Greenstein-ish), Lorentz-boosted via mean gamma:
    //   E_z(z_i) = (1/(2 pi eps0 a^2)) * Σ_j Q_j * sign(d) *
    //                 (1 - g|d| / sqrt(g²d² + a²))
    // d = z_i - z_j;  g = m_gamma. The formula is the rest-frame
    // disk-stack with the substitution z_rest = g * z_lab.
    //
    // For self (i==j) the "1 - g|d|/..." factor → 1, but sign(d) is 0
    // by our convention (regularize with sign(0)=0), so the diagonal
    // contributes nothing. Physically, an infinitely thin disk
    // contributes 0 to its own on-axis field.
    const Real prefactor = Real(1.0) / (Real(2.0) * kPi * kEps0 * m_a * m_a);
    const Real g2        = m_gamma * m_gamma;

#ifdef AMREX_USE_OMP
    #pragma omp parallel for schedule(static) if (m_n_slices > 32)
#endif
    for (int i = 0; i < m_n_slices; ++i) {
        const Real z_i = m_z_lo + (Real(i) + Real(0.5)) * m_dz;
        Real Ez = Real(0.0);
        for (int j = 0; j < m_n_slices; ++j) {
            const Real Qj = m_Q_slice[j];
            if (Qj == Real(0.0)) { continue; }
            const Real z_j = m_z_lo + (Real(j) + Real(0.5)) * m_dz;
            const Real d   = z_i - z_j;
            if (d == Real(0.0)) { continue; }            // self → 0
            const Real ad   = std::abs(d);
            const Real root = std::sqrt(g2 * ad * ad + m_a * m_a);
            const Real factor = (Real(1.0) - m_gamma * ad / root);
            // sign(d) * factor = (d>0 ? +1 : -1) * factor
            Ez += Qj * (d > Real(0.0) ? factor : -factor);
        }
        m_Ez_slice[i] = prefactor * Ez;
    }
}


amrex::Real SpaceChargeSlice::E_z_at (amrex::Real z) const noexcept
{
    if (m_dz <= amrex::Real(0.0)) { return amrex::Real(0.0); }
    if (m_gamma >= m_gamma_off)   { return amrex::Real(0.0); }
    const amrex::Real f = (z - m_z_lo) / m_dz - amrex::Real(0.5);
    if (f <= amrex::Real(0.0))               { return m_Ez_slice.front(); }
    if (f >= amrex::Real(m_n_slices - 1))    { return m_Ez_slice.back();  }
    const int  i    = int(f);
    const amrex::Real frac = f - amrex::Real(i);
    return (amrex::Real(1.0) - frac) * m_Ez_slice[i]
         +                  frac     * m_Ez_slice[i + 1];
}

} // namespace spacecharge
} // namespace lucretiatt
