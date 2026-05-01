#include "WakeField.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParIter.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <vector>


namespace lucretiatt {
namespace elements {

namespace {

constexpr amrex::Real kZ0     = amrex::Real(376.730313667);    // free-space impedance, Ohm
constexpr amrex::Real kCLight = amrex::Real(299792458.0);      // m/s
constexpr amrex::Real kPi     = amrex::Real(3.14159265358979323846);

inline amrex::Real bane_w_long (amrex::Real s,
                                amrex::Real a, amrex::Real g, amrex::Real L)
{
    if (s < amrex::Real(0.0)) return amrex::Real(0.0);
    const amrex::Real s_z0 = amrex::Real(0.41) * std::pow(a, amrex::Real(1.8))
                           * std::pow(g, amrex::Real(1.6))
                           / std::pow(L, amrex::Real(2.4));
    return (kZ0 * kCLight / (kPi * a * a)) * std::exp(-std::sqrt(s / s_z0));
}

inline amrex::Real bane_w_trans (amrex::Real s,
                                 amrex::Real a, amrex::Real g, amrex::Real L)
{
    if (s < amrex::Real(0.0)) return amrex::Real(0.0);
    const amrex::Real s_t0 = amrex::Real(0.169) * std::pow(a, amrex::Real(1.79))
                           * std::pow(g, amrex::Real(0.38))
                           / std::pow(L, amrex::Real(1.17));
    const amrex::Real x = std::sqrt(s / s_t0);
    return (amrex::Real(4.0) * kZ0 * kCLight * s_t0 / (kPi * std::pow(a, amrex::Real(4))))
         * (amrex::Real(1.0) - (amrex::Real(1.0) + x) * std::exp(-x));
}

} // anonymous


void WakeField::apply_wake (particles::TimeBunch& bunch, amrex::Real dt) const
{
    using namespace amrex;
    using namespace particles;

    if (m_n_slices < 2 || dt <= Real(0.0)) return;
    if (m_iris_a <= Real(0.0) || m_gap_g <= Real(0.0) || m_period_L <= Real(0.0)) return;
    if (m_z_end <= m_z_start) return;

    const Real z_lo = m_z_start;
    const Real z_hi = m_z_end;
    const Real ds   = (z_hi - z_lo) / Real(m_n_slices);

    std::vector<Real> qslice (m_n_slices, Real(0.0));
    std::vector<Real> qxslice(m_n_slices, Real(0.0));
    std::vector<Real> qyslice(m_n_slices, Real(0.0));

    // ---- Pass 1: bin local particles into slices ----
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    long n_in_range_local = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& xs = soa.GetRealData(RealSoA::x);
        const auto& ys = soa.GetRealData(RealSoA::y);
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& qs = soa.GetRealData(RealSoA::q);
        const auto& ws = soa.GetRealData(RealSoA::w);
        for (int p = 0; p < np; ++p) {
            const Real z = zs[p];
            if (z < z_lo || z >= z_hi) continue;
            const int i = std::min(int((z - z_lo) / ds), m_n_slices - 1);
            const Real qw = qs[p] * ws[p];
            qslice [i] += qw;
            qxslice[i] += qw * xs[p];
            qyslice[i] += qw * ys[p];
            ++n_in_range_local;
        }
    }

    // Skip everything if no particles are in the wake's z range on
    // ANY rank (saves the ALLREDUCE cost when the bunch is upstream).
    long n_in_range_global = n_in_range_local;
    ParallelDescriptor::ReduceLongSum(n_in_range_global);
    if (n_in_range_global == 0) return;

    // ---- ALLREDUCE slice profiles across MPI ranks ----
    if (ParallelDescriptor::NProcs() > 1) {
        ParallelDescriptor::ReduceRealSum(qslice .data(), m_n_slices);
        ParallelDescriptor::ReduceRealSum(qxslice.data(), m_n_slices);
        ParallelDescriptor::ReduceRealSum(qyslice.data(), m_n_slices);
    }

    // ---- Pass 2: convolve to get per-slice wake fields ----
    // Convention: head of bunch is at largest z; slice i is kicked by
    // slices j >= i (slices ahead in z, which the test particle
    // traversed earlier). The j == i term is taken at half weight
    // (in-slice average; trailing half of slice charge contributes,
    // leading half doesn't).
    std::vector<Real> Ez(m_n_slices, Real(0.0));
    std::vector<Real> Ex(m_n_slices, Real(0.0));
    std::vector<Real> Ey(m_n_slices, Real(0.0));
    for (int i = 0; i < m_n_slices; ++i) {
        Real ez = 0.0, ex = 0.0, ey = 0.0;
        for (int j = i; j < m_n_slices; ++j) {
            const Real s = (Real(j) - Real(i)) * ds;
            const Real wself = (j == i) ? Real(0.5) : Real(1.0);
            if (m_long_on) {
                ez += wself * qslice[j] * bane_w_long(s, m_iris_a, m_gap_g, m_period_L);
            }
            if (m_trans_on) {
                const Real wt = bane_w_trans(s, m_iris_a, m_gap_g, m_period_L);
                ex += wself * qxslice[j] * wt;
                ey += wself * qyslice[j] * wt;
            }
        }
        // Sign: wake decelerates same-sign trailing charges. The Bane
        // w_z is positive; the resulting field on a unit-positive test
        // charge from a positive source charge is NEGATIVE (opposing
        // motion). For a generic q_test the kick is q_test * E_z; the
        // negative sign here ensures the standard convention.
        Ez[i] = -ez;
        Ex[i] = -ex;
        Ey[i] = -ey;
    }

    // ---- Pass 3: apply per-particle kick ----
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
        for (int p = 0; p < np; ++p) {
            const Real z = ptd.rdata(RealSoA::z)[p];
            if (z < z_lo || z >= z_hi) continue;
            const int i = std::min(int((z - z_lo) / ds), m_n_slices - 1);
            const Real qm = ptd.rdata(RealSoA::qm)[p];
            ptd.rdata(RealSoA::px)[p] += qm * Ex[i] * dt;
            ptd.rdata(RealSoA::py)[p] += qm * Ey[i] * dt;
            ptd.rdata(RealSoA::pz)[p] += qm * Ez[i] * dt;
        }
    }
}

} // namespace elements
} // namespace lucretiatt
