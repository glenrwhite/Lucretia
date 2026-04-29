#include "SpaceCharge.H"

#include <ablastr/fields/IntegratedGreenFunctionSolver.H>

#include <AMReX_MFIter.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>


namespace lucretiatt {
namespace spacecharge {

namespace {
constexpr amrex::Real kSpeedOfLight = amrex::Real(299792458.0);
}


SpaceCharge::SpaceCharge (
    const amrex::Geometry&            geom,
    const amrex::BoxArray&            cell_ba,
    const amrex::DistributionMapping& /*dm*/)
    : m_geom(geom)
    // Force a single box covering the whole domain so deposit + IGF FFT
    // see one contiguous mesh. Phase 6 will revisit for multi-rank.
    , m_ba_nodal(amrex::convert(amrex::BoxArray{cell_ba.minimalBox()},
                                 amrex::IntVect{1, 1, 1}))
    , m_dm(amrex::DistributionMapping{m_ba_nodal})
{
    constexpr int kNComp = 1;
    const amrex::IntVect ng_phi(1);
    const amrex::IntVect ng_E(1);

    // rho needs no ghosts in the IGF solver path (the solver wraps it
    // up internally). phi and E need a ghost layer for centered diff
    // and CIC gather across box boundaries.
    m_rho.define(m_ba_nodal, m_dm, kNComp, 0);
    m_phi.define(m_ba_nodal, m_dm, kNComp, ng_phi);
    m_Ex.define (m_ba_nodal, m_dm, kNComp, ng_E);
    m_Ey.define (m_ba_nodal, m_dm, kNComp, ng_E);
    m_Ez.define (m_ba_nodal, m_dm, kNComp, ng_E);

    m_rho.setVal(0.0);
    m_phi.setVal(0.0);
    m_Ex.setVal(0.0);
    m_Ey.setVal(0.0);
    m_Ez.setVal(0.0);
}


amrex::Real SpaceCharge::compute_mean_beta_z (particles::TimeBunch& bunch) const
{
    using namespace amrex;
    using namespace particles;

    long const n_local = bunch.NumberOfParticlesAtLevel(
        0, /*only_valid*/ true, /*only_local*/ true);
    long n_total = bunch.TotalNumberOfParticles(true, false);
    if (n_total == 0) { return 0.0; }

    // Compute sum of beta_z = (uz/c) / sqrt(1 + (u/c)^2)
    Real sum_beta_z = 0.0;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& uxs = soa.GetRealData(RealSoA::px);
        const auto& uys = soa.GetRealData(RealSoA::py);
        const auto& uzs = soa.GetRealData(RealSoA::pz);
        for (int i = 0; i < np; ++i) {
            const Real ux = uxs[i];
            const Real uy = uys[i];
            const Real uz = uzs[i];
            const Real gamma = std::sqrt(Real(1.0) +
                (ux*ux + uy*uy + uz*uz) / (kSpeedOfLight * kSpeedOfLight));
            sum_beta_z += (uz / kSpeedOfLight) / gamma;
        }
    }
    ParallelDescriptor::ReduceRealSum(sum_beta_z);
    amrex::ignore_unused(n_local);

    return sum_beta_z / Real(n_total);
}


void SpaceCharge::deposit_charge (particles::TimeBunch& bunch)
{
    using namespace amrex;
    using namespace particles;

    m_rho.setVal(0.0);

    auto const* dx = m_geom.CellSize();
    auto const* lo = m_geom.ProbLo();
    const Real inv_dV = Real(1.0) / (dx[0] * dx[1] * dx[2]);
    const Real inv_dx = Real(1.0) / dx[0];
    const Real inv_dy = Real(1.0) / dx[1];
    const Real inv_dz = Real(1.0) / dx[2];

    // Single-box single-rank: deposit into the FAB on rank 0.
    if (!ParallelDescriptor::IOProcessor()) { return; }

    auto& rho_fab = m_rho[0];
    auto rho_arr  = rho_fab.array();
    Box const& vbox = rho_fab.box();
    auto const& vlo = vbox.smallEnd();

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& xs = soa.GetRealData(RealSoA::x);
        const auto& ys = soa.GetRealData(RealSoA::y);
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& qs = soa.GetRealData(RealSoA::q);
        const auto& ws = soa.GetRealData(RealSoA::w);

        for (int p = 0; p < np; ++p) {
            const Real fx = (xs[p] - lo[0]) * inv_dx;
            const Real fy = (ys[p] - lo[1]) * inv_dy;
            const Real fz = (zs[p] - lo[2]) * inv_dz;
            const int  ix = int(std::floor(fx));
            const int  iy = int(std::floor(fy));
            const int  iz = int(std::floor(fz));
            if (ix     < vlo[0] || ix + 1 > vbox.bigEnd(0) ||
                iy     < vlo[1] || iy + 1 > vbox.bigEnd(1) ||
                iz     < vlo[2] || iz + 1 > vbox.bigEnd(2))
            {
                continue;  // off-grid particle: skip
            }
            const Real wx1 = fx - Real(ix);   const Real wx0 = Real(1.0) - wx1;
            const Real wy1 = fy - Real(iy);   const Real wy0 = Real(1.0) - wy1;
            const Real wz1 = fz - Real(iz);   const Real wz0 = Real(1.0) - wz1;

            const Real qw = qs[p] * ws[p] * inv_dV;
            rho_arr(ix  , iy  , iz  ) += qw * wx0 * wy0 * wz0;
            rho_arr(ix+1, iy  , iz  ) += qw * wx1 * wy0 * wz0;
            rho_arr(ix  , iy+1, iz  ) += qw * wx0 * wy1 * wz0;
            rho_arr(ix+1, iy+1, iz  ) += qw * wx1 * wy1 * wz0;
            rho_arr(ix  , iy  , iz+1) += qw * wx0 * wy0 * wz1;
            rho_arr(ix+1, iy  , iz+1) += qw * wx1 * wy0 * wz1;
            rho_arr(ix  , iy+1, iz+1) += qw * wx0 * wy1 * wz1;
            rho_arr(ix+1, iy+1, iz+1) += qw * wx1 * wy1 * wz1;
        }
    }
}


void SpaceCharge::compute_E_from_phi ()
{
    using namespace amrex;

    m_phi.FillBoundary(m_geom.periodicity());

    auto const* dx = m_geom.CellSize();
    const Real inv_2dx = Real(1.0) / (Real(2.0) * dx[0]);
    const Real inv_2dy = Real(1.0) / (Real(2.0) * dx[1]);
    const Real inv_2dz = Real(1.0) / (Real(2.0) * dx[2]);

    for (MFIter mfi(m_phi, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& vbox = mfi.validbox();
        auto const& phi = m_phi.const_array(mfi);
        auto Ex = m_Ex.array(mfi);
        auto Ey = m_Ey.array(mfi);
        auto Ez = m_Ez.array(mfi);

        ParallelFor(vbox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            Ex(i, j, k) = -(phi(i+1, j, k) - phi(i-1, j, k)) * inv_2dx;
            Ey(i, j, k) = -(phi(i, j+1, k) - phi(i, j-1, k)) * inv_2dy;
            Ez(i, j, k) = -(phi(i, j, k+1) - phi(i, j, k-1)) * inv_2dz;
        });
    }
    m_Ex.FillBoundary(m_geom.periodicity());
    m_Ey.FillBoundary(m_geom.periodicity());
    m_Ez.FillBoundary(m_geom.periodicity());
}


void SpaceCharge::solve (particles::TimeBunch& bunch)
{
    using namespace amrex;

    m_last_beta_z = compute_mean_beta_z(bunch);

    deposit_charge(bunch);

    auto const* dx = m_geom.CellSize();
    const Real inv_g = std::sqrt(Real(1.0) - m_last_beta_z * m_last_beta_z);
    const Real dz_scaled = (inv_g > Real(0.0)) ? dx[2] / inv_g : dx[2];
    std::array<Real, 3> cell_size = {dx[0], dx[1], dz_scaled};

    ablastr::fields::computePhiIGF(m_rho, m_phi, cell_size, /*is_2d_slices*/ false);

    compute_E_from_phi();

    int verbose = 0;
    amrex::ParmParse("space_charge").query("verbose", verbose);
    if (verbose > 0) {
        Real total_q = m_rho.sum(0) * dx[0] * dx[1] * dx[2];
        Real phi_max = m_phi.norm0(0, 0);
        Real Ex_max = m_Ex.norm0(0, 0);
        Real Ey_max = m_Ey.norm0(0, 0);
        Real Ez_max = m_Ez.norm0(0, 0);
        amrex::Print() << "[SpaceCharge] beta_z=" << m_last_beta_z
                       << "  sum(rho)*dV=" << total_q
                       << "  |phi|_max=" << phi_max
                       << "  |E|_max=(" << Ex_max << "," << Ey_max << "," << Ez_max << ")\n";
    }
}

} // namespace spacecharge
} // namespace lucretiatt
