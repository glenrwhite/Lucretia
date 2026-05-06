#include "SpaceCharge.H"

#include <ablastr/fields/IntegratedGreenFunctionSolver.H>

#include <AMReX_FFT_OpenBCSolver.H>
#include <AMReX_MFIter.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <memory>


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
    // Force m_rho to live on a SINGLE BoxArray box (the full nodal domain),
    // independent of the multi-box decomposition the caller may have built
    // for the particle container. The reason: deposit_charge needs to write
    // to nodes that may not belong to the current particle tile (since the
    // bunch can drift across the SC mesh as the SC mesh recenters but the
    // particle tile assignment is fixed-at-construction in TimeBunch).
    // With a single FAB, deposit can index arbitrary positions without a
    // tile lookup; the FFT runs single-rank (fine for Mac dev / single-rank
    // photoinjector runs at 48^3).
    , m_ba_nodal(amrex::convert(amrex::BoxArray(cell_ba.minimalBox()),
                                amrex::IntVect{1, 1, 1}))
    , m_dm(amrex::DistributionMapping(amrex::BoxArray(cell_ba.minimalBox())))
{
    constexpr int kNComp = 1;
    const amrex::IntVect ng_rho(1);  // CIC deposit crosses tile edges
    const amrex::IntVect ng_phi(1);
    const amrex::IntVect ng_E(1);

    // rho needs 1 ghost so CIC deposits that straddle a tile boundary
    // can be summed back via SumBoundary. phi and E need a ghost layer
    // for centered diff and CIC gather across box boundaries.
    m_rho.define(m_ba_nodal, m_dm, kNComp, ng_rho);
    m_phi.define(m_ba_nodal, m_dm, kNComp, ng_phi);
    m_Ex.define (m_ba_nodal, m_dm, kNComp, ng_E);
    m_Ey.define (m_ba_nodal, m_dm, kNComp, ng_E);
    m_Ez.define (m_ba_nodal, m_dm, kNComp, ng_E);

    m_rho.setVal(0.0);
    m_phi.setVal(0.0);
    m_Ex.setVal(0.0);
    m_Ey.setVal(0.0);
    m_Ez.setVal(0.0);

    build_self_force_lut();
}


void SpaceCharge::build_self_force_lut ()
{
    using namespace amrex;

    auto const* dx_arr = m_geom.CellSize();
    const Real dx = dx_arr[0];
    const Real dy = dx_arr[1];
    const Real dz = dx_arr[2];
    const Real eps2 = Real(1.0e-6) * (dx * dx);   // matches TrackingLoop softening

    constexpr int N = kSelfForceLutN;
    const int stride1 = N + 1;
    const int stride2 = stride1 * stride1;
    const int total   = stride1 * stride1 * stride1;

    for (int c = 0; c < 3; ++c) {
        m_self_force_lut[c].assign(total, 0.0);
    }

    for (int ix = 0; ix <= N; ++ix) {
        const Real wx_frac = Real(ix) / Real(N);
        for (int iy = 0; iy <= N; ++iy) {
            const Real wy_frac = Real(iy) / Real(N);
            for (int iz = 0; iz <= N; ++iz) {
                const Real wz_frac = Real(iz) / Real(N);

                Real Ex_unit = 0.0, Ey_unit = 0.0, Ez_unit = 0.0;
                for (int a = 0; a < 2; ++a) {
                    const Real wa   = (a == 0) ? (Real(1.0) - wx_frac) : wx_frac;
                    const Real rx_v = (wx_frac - Real(a)) * dx;
                    for (int b = 0; b < 2; ++b) {
                        const Real wb   = (b == 0) ? (Real(1.0) - wy_frac) : wy_frac;
                        const Real ry_v = (wy_frac - Real(b)) * dy;
                        for (int cc = 0; cc < 2; ++cc) {
                            const Real wc   = (cc == 0) ? (Real(1.0) - wz_frac) : wz_frac;
                            const Real rz_v = (wz_frac - Real(cc)) * dz;
                            const Real r2 = rx_v*rx_v + ry_v*ry_v + rz_v*rz_v + eps2;
                            const Real r3_inv = Real(1.0) / (r2 * std::sqrt(r2));
                            const Real factor = wa * wb * wc * r3_inv;
                            Ex_unit += factor * rx_v;
                            Ey_unit += factor * ry_v;
                            Ez_unit += factor * rz_v;
                        }
                    }
                }

                const int idx = ix * stride2 + iy * stride1 + iz;
                m_self_force_lut[0][idx] = Ex_unit;
                m_self_force_lut[1][idx] = Ey_unit;
                m_self_force_lut[2][idx] = Ez_unit;
            }
        }
    }
}


amrex::Real SpaceCharge::compute_mean_z (particles::TimeBunch& bunch) const
{
    using namespace amrex;
    using namespace particles;

    Real sum_z = 0.0;
    long n_alive = 0;
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            if (alives[i] == 0) { continue; }
            sum_z += zs[i];
            ++n_alive;
        }
    }
    ParallelDescriptor::ReduceRealSum(sum_z);
    ParallelDescriptor::ReduceLongSum(n_alive);
    if (n_alive == 0) { return 0.0; }
    return sum_z / Real(n_alive);
}


bool SpaceCharge::compute_bunch_range (
    particles::TimeBunch& bunch,
    amrex::Real& xmin, amrex::Real& xmax,
    amrex::Real& ymin, amrex::Real& ymax,
    amrex::Real& zmin, amrex::Real& zmax) const
{
    using namespace amrex;
    using namespace particles;

    Real lo_x =  std::numeric_limits<Real>::max();
    Real hi_x = -std::numeric_limits<Real>::max();
    Real lo_y =  std::numeric_limits<Real>::max();
    Real hi_y = -std::numeric_limits<Real>::max();
    Real lo_z =  std::numeric_limits<Real>::max();
    Real hi_z = -std::numeric_limits<Real>::max();
    long n_alive = 0;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& xs = soa.GetRealData(RealSoA::x);
        const auto& ys = soa.GetRealData(RealSoA::y);
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            if (alives[i] == 0) { continue; }   // skip parked dead particles
            lo_x = std::min(lo_x, xs[i]);  hi_x = std::max(hi_x, xs[i]);
            lo_y = std::min(lo_y, ys[i]);  hi_y = std::max(hi_y, ys[i]);
            lo_z = std::min(lo_z, zs[i]);  hi_z = std::max(hi_z, zs[i]);
            ++n_alive;
        }
    }
    ParallelDescriptor::ReduceRealMin(lo_x);  ParallelDescriptor::ReduceRealMax(hi_x);
    ParallelDescriptor::ReduceRealMin(lo_y);  ParallelDescriptor::ReduceRealMax(hi_y);
    ParallelDescriptor::ReduceRealMin(lo_z);  ParallelDescriptor::ReduceRealMax(hi_z);
    ParallelDescriptor::ReduceLongSum(n_alive);
    if (n_alive == 0) { return false; }
    xmin = lo_x;  xmax = hi_x;
    ymin = lo_y;  ymax = hi_y;
    zmin = lo_z;  zmax = hi_z;
    return true;
}


bool SpaceCharge::compute_bunch_stats (
    particles::TimeBunch& bunch,
    amrex::Real& x_c,     amrex::Real& y_c,     amrex::Real& z_c,
    amrex::Real& sigma_x, amrex::Real& sigma_y, amrex::Real& sigma_z) const
{
    using namespace amrex;
    using namespace particles;

    Real sum_x = 0.0,  sum_y = 0.0,  sum_z = 0.0;
    Real sum_x2 = 0.0, sum_y2 = 0.0, sum_z2 = 0.0;
    long n_alive = 0;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& xs = soa.GetRealData(RealSoA::x);
        const auto& ys = soa.GetRealData(RealSoA::y);
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            // Skip dead particles (e.g. cathode re-cross kills, parked
            // at z=-1e9 by TrackingLoop). Including them would skew
            // the centroid and sigmas catastrophically and trigger
            // the adaptive mesh to enclose a 10^9 m extent.
            if (alives[i] == 0) { continue; }
            sum_x  += xs[i];        sum_y  += ys[i];        sum_z  += zs[i];
            sum_x2 += xs[i]*xs[i];  sum_y2 += ys[i]*ys[i];  sum_z2 += zs[i]*zs[i];
            ++n_alive;
        }
    }
    ParallelDescriptor::ReduceRealSum(sum_x);  ParallelDescriptor::ReduceRealSum(sum_y);  ParallelDescriptor::ReduceRealSum(sum_z);
    ParallelDescriptor::ReduceRealSum(sum_x2); ParallelDescriptor::ReduceRealSum(sum_y2); ParallelDescriptor::ReduceRealSum(sum_z2);
    ParallelDescriptor::ReduceLongSum(n_alive);
    if (n_alive == 0) { return false; }

    const Real inv_N = Real(1.0) / Real(n_alive);
    x_c = sum_x * inv_N;
    y_c = sum_y * inv_N;
    z_c = sum_z * inv_N;
    sigma_x = std::sqrt(std::max(sum_x2 * inv_N - x_c * x_c, Real(0.0)));
    sigma_y = std::sqrt(std::max(sum_y2 * inv_N - y_c * y_c, Real(0.0)));
    sigma_z = std::sqrt(std::max(sum_z2 * inv_N - z_c * z_c, Real(0.0)));
    return true;
}


void SpaceCharge::recenter (amrex::Real z_new)
{
    using namespace amrex;
    auto const* lo = m_geom.ProbLo();
    auto const* hi = m_geom.ProbHi();
    const Real lz_half = Real(0.5) * (hi[2] - lo[2]);

    // Keep transverse extent + integer Box; only shift z.
    RealBox rb({lo[0], lo[1], z_new - lz_half},
               {hi[0], hi[1], z_new + lz_half});
    Array<int, AMREX_SPACEDIM> is_per{0, 0, 0};
    m_geom.define(m_geom.Domain(), rb, CoordSys::cartesian, is_per);

    // Old field data refers to the old physical extent and is now
    // meaningless. Reset; the next solve() will repopulate.
    m_rho.setVal(0.0);
    m_phi.setVal(0.0);
    m_Ex.setVal(0.0);
    m_Ey.setVal(0.0);
    m_Ez.setVal(0.0);

    ++m_n_recenters;
    // Recenter (z-only) keeps cell SIZES unchanged, so the self-force
    // LUT (which depends only on the cell aspect ratio) stays valid.

    // Recenter events used to print one line each, but with a co-moving
    // mesh through a 5 m injector + small box you get hundreds of them
    // and they swamp the verbose output. Gate behind space_charge.verbose
    // so it's still available for debugging when explicitly requested.
    int verbose = 0;
    amrex::ParmParse("space_charge").query("verbose", verbose);
    if (verbose > 0) {
        amrex::Print() << "[SpaceCharge] recenter: z -> " << z_new
                       << " m (recenter #" << m_n_recenters << ")\n";
    }
}


void SpaceCharge::resize (
    amrex::Real x_centroid, amrex::Real y_centroid, amrex::Real z_centroid,
    amrex::Real half_x,     amrex::Real half_y,     amrex::Real half_z)
{
    using namespace amrex;

    // Old cell aspect ratio (relative to dx) — used to decide if we need
    // to rebuild the self-force LUT. The LUT is built from a Coulomb
    // sum over the 8 surrounding nodes weighted by CIC fractions and
    // depends ONLY on the cell shape (dx:dy:dz), not its absolute size.
    auto const* dx_old = m_geom.CellSize();
    const Real ar_y_old = dx_old[1] / dx_old[0];
    const Real ar_z_old = dx_old[2] / dx_old[0];

    RealBox rb({x_centroid - half_x, y_centroid - half_y, z_centroid - half_z},
               {x_centroid + half_x, y_centroid + half_y, z_centroid + half_z});
    Array<int, AMREX_SPACEDIM> is_per{0, 0, 0};
    m_geom.define(m_geom.Domain(), rb, CoordSys::cartesian, is_per);

    // Reset all field MultiFabs (old data refers to old extent).
    m_rho.setVal(0.0);
    m_phi.setVal(0.0);
    m_Ex.setVal(0.0);
    m_Ey.setVal(0.0);
    m_Ez.setVal(0.0);

    ++m_n_recenters;

    // Rebuild self-force LUT only if cell aspect ratio shifted by > 5%
    // (LUT depends on shape, not size). Threshold is empirical;
    // tighter values rebuild more often (slow), looser values let the
    // LUT drift out of calibration. 5% gives ~few percent error in the
    // self-force subtraction, which is below the IGF discretization
    // error anyway.
    auto const* dx_new = m_geom.CellSize();
    const Real ar_y_new = dx_new[1] / dx_new[0];
    const Real ar_z_new = dx_new[2] / dx_new[0];
    const Real rel_dy   = std::abs(ar_y_new - ar_y_old) / std::max(ar_y_old, Real(1.0e-30));
    const Real rel_dz   = std::abs(ar_z_new - ar_z_old) / std::max(ar_z_old, Real(1.0e-30));
    if (rel_dy > Real(0.05) || rel_dz > Real(0.05)) {
        build_self_force_lut();
        ++m_n_lut_rebuilds;
    }

    int verbose = 0;
    amrex::ParmParse("space_charge").query("verbose", verbose);
    if (verbose > 0) {
        amrex::Print() << "[SpaceCharge] resize: centroid=("
                       << x_centroid << "," << y_centroid << "," << z_centroid
                       << ") half=(" << half_x << "," << half_y << "," << half_z
                       << ") cell=(" << dx_new[0] << "," << dx_new[1] << "," << dx_new[2]
                       << ")  recenter #" << m_n_recenters
                       << "  LUT rebuilds=" << m_n_lut_rebuilds << "\n";
    }
}


amrex::Real SpaceCharge::compute_mean_beta_z (particles::TimeBunch& bunch) const
{
    using namespace amrex;
    using namespace particles;

    Real sum_beta_z = 0.0;
    long n_alive = 0;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& uxs = soa.GetRealData(RealSoA::px);
        const auto& uys = soa.GetRealData(RealSoA::py);
        const auto& uzs = soa.GetRealData(RealSoA::pz);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            if (alives[i] == 0) { continue; }
            const Real ux = uxs[i];
            const Real uy = uys[i];
            const Real uz = uzs[i];
            const Real gamma = std::sqrt(Real(1.0) +
                (ux*ux + uy*uy + uz*uz) / (kSpeedOfLight * kSpeedOfLight));
            sum_beta_z += (uz / kSpeedOfLight) / gamma;
            ++n_alive;
        }
    }
    ParallelDescriptor::ReduceRealSum(sum_beta_z);
    ParallelDescriptor::ReduceLongSum(n_alive);
    if (n_alive == 0) { return 0.0; }

    return sum_beta_z / Real(n_alive);
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

    // Position-based deposit: with the single-FAB m_rho (forced in the
    // SC constructor regardless of the cell_ba subdivision the caller
    // passed in), we iterate particles via TimeBunch's PIter (whatever
    // tile structure that uses) and deposit directly into m_rho's only
    // FAB. The check is purely "does the particle's deposit footprint
    // fall inside m_rho's nodal box" -- there's no tile-correspondence
    // requirement between the particle iterator and m_rho.
    //
    // Single-rank assumption: m_rho has exactly one FAB. Get it via the
    // first MFIter step. (For multi-rank we'd need either ParallelCopy
    // from a per-rank scratch FAB, or per-particle box-ownership lookup.)
    Box      gbox;
    Array4<Real> rho_arr;
    {
        MFIter mfi(m_rho);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mfi.isValid(),
            "SpaceCharge::deposit_charge: m_rho has no local FAB on this rank "
            "(single-FAB assumption violated -- only single-rank runs supported here)");
        gbox    = m_rho[mfi].box();
        rho_arr = m_rho.array(mfi);
    }
    auto const& glo = gbox.smallEnd();
    auto const& ghi = gbox.bigEnd();

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    long total_deposited = 0;
    long total_outside   = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto& soa = pti.GetStructOfArrays();
        const int np = pti.numParticles();
        const auto& xs = soa.GetRealData(RealSoA::x);
        const auto& ys = soa.GetRealData(RealSoA::y);
        const auto& zs = soa.GetRealData(RealSoA::z);
        const auto& qs = soa.GetRealData(RealSoA::q);
        const auto& ws = soa.GetRealData(RealSoA::w);
        const auto& alives = soa.GetIntData(IntSoA::alive);

        for (int p = 0; p < np; ++p) {
            // Skip dead particles (e.g. cathode re-cross kills): they
            // remain in the bunch container for ID stability, but they
            // don't contribute to the SC charge density.
            if (alives[p] == 0) { continue; }

            const Real fx = (xs[p] - lo[0]) * inv_dx;
            const Real fy = (ys[p] - lo[1]) * inv_dy;
            const Real fz = (zs[p] - lo[2]) * inv_dz;
            const int  ix = int(std::floor(fx));
            const int  iy = int(std::floor(fy));
            const int  iz = int(std::floor(fz));
            const Real wx1 = fx - Real(ix);   const Real wx0 = Real(1.0) - wx1;
            const Real wy1 = fy - Real(iy);   const Real wy0 = Real(1.0) - wy1;
            const Real wz1 = fz - Real(iz);   const Real wz0 = Real(1.0) - wz1;
            const Real qw = qs[p] * ws[p] * inv_dV;

            // Direct deposit (real charge at z_p). Image charges, when
            // enabled, are NOT deposited here -- they're computed via a
            // second IGF solve with a z-shifted Green function (see
            // SpaceCharge::solve).
            //
            // Particles outside m_rho's footprint don't contribute to SC.
            // (They still get tracked through the lattice, just without
            // SC defocus from this solve. This matches IMPACT-T's
            // behavior of dropping particles from SC outside the mesh.)
            if (ix     >= glo[0] && ix + 1 <= ghi[0] &&
                iy     >= glo[1] && iy + 1 <= ghi[1] &&
                iz     >= glo[2] && iz + 1 <= ghi[2])
            {
                rho_arr(ix  , iy  , iz  ) += qw * wx0 * wy0 * wz0;
                rho_arr(ix+1, iy  , iz  ) += qw * wx1 * wy0 * wz0;
                rho_arr(ix  , iy+1, iz  ) += qw * wx0 * wy1 * wz0;
                rho_arr(ix+1, iy+1, iz  ) += qw * wx1 * wy1 * wz0;
                rho_arr(ix  , iy  , iz+1) += qw * wx0 * wy0 * wz1;
                rho_arr(ix+1, iy  , iz+1) += qw * wx1 * wy0 * wz1;
                rho_arr(ix  , iy+1, iz+1) += qw * wx0 * wy1 * wz1;
                rho_arr(ix+1, iy+1, iz+1) += qw * wx1 * wy1 * wz1;
                ++total_deposited;
            } else {
                ++total_outside;
            }
        }
    }
    // SumBoundary not needed -- single-FAB m_rho has no internal tile
    // boundaries. Cross-rank reduction would be needed for multi-rank,
    // but that's gated by the AMREX_ALWAYS_ASSERT above.

    // DEBUG: simplified diagnostic for the position-based deposit.
    // Prints how many particles deposited successfully vs how many fell
    // outside the SC mesh's nodal box.
    int verbose_dbg = 0;
    amrex::ParmParse("space_charge").query("verbose", verbose_dbg);
    if (verbose_dbg > 0) {
        amrex::Print() << "[SCdbg] mesh xrange=[" << lo[0] << "," << lo[0]+dx[0]*(m_geom.Domain().bigEnd(0)+1)
                       << "] yrange=[" << lo[1] << "," << lo[1]+dx[1]*(m_geom.Domain().bigEnd(1)+1)
                       << "] zrange=[" << lo[2] << "," << lo[2]+dx[2]*(m_geom.Domain().bigEnd(2)+1)
                       << "] dx=(" << dx[0] << "," << dx[1] << "," << dx[2] << ")"
                       << " gbox=[" << glo[0] << ".." << ghi[0]
                       << "]x[" << glo[1] << ".." << ghi[1]
                       << "]x[" << glo[2] << ".." << ghi[2] << "]"
                       << " deposited=" << total_deposited
                       << " outside=" << total_outside << "\n";
    }
}


void SpaceCharge::smooth_rho ()
{
    using namespace amrex;
    if (m_rho_smooth_passes <= 0) { return; }
    int verbose = 0;
    amrex::ParmParse("space_charge").query("verbose", verbose);
    if (verbose > 0) {
        amrex::Print() << "[SpaceCharge] smooth_rho: " << m_rho_smooth_passes << " passes\n";
    }

    // Scratch with same layout + ghosts as m_rho. Reused across passes.
    MultiFab scratch(m_rho.boxArray(), m_rho.DistributionMap(),
                     m_rho.nComp(), m_rho.nGrowVect());

    constexpr Real inv64 = Real(1.0) / Real(64.0);
    for (int pass = 0; pass < m_rho_smooth_passes; ++pass) {
        // Copy current rho (incl. ghosts) into scratch and refresh
        // ghost cells from neighbours. With OPEN BC the ghost values
        // are 0 (FillBoundary on a periodicity()=none geometry leaves
        // them untouched after the explicit setVal we do at deposit).
        MultiFab::Copy(scratch, m_rho, 0, 0, m_rho.nComp(), m_rho.nGrowVect());
        scratch.FillBoundary(m_geom.periodicity());

        for (MFIter mfi(m_rho, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& vbox = mfi.validbox();
            auto const& src = scratch.const_array(mfi);
            auto       dst  = m_rho.array(mfi);

            ParallelFor(vbox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Separable 3D binomial (1,2,1)/4 in each axis.
                // Outer-product weights: w(dx,dy,dz) = wx*wy*wz where
                // w_a = 1 if |a|=1, 2 if a=0. Sum over 3^3 cube,
                // normalised by 4^3 = 64.
                Real s = Real(0.0);
                for (int dz = -1; dz <= 1; ++dz) {
                    const Real wz = (dz == 0) ? Real(2.0) : Real(1.0);
                    for (int dy = -1; dy <= 1; ++dy) {
                        const Real wy = (dy == 0) ? Real(2.0) : Real(1.0);
                        for (int dx = -1; dx <= 1; ++dx) {
                            const Real wx = (dx == 0) ? Real(2.0) : Real(1.0);
                            s += wx * wy * wz * src(i + dx, j + dy, k + dz);
                        }
                    }
                }
                dst(i, j, k) = s * inv64;
            });
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

    // Exact-bunch-range adaptive (ImpactT-style): mesh resized every step
    // to EXACTLY the alive-particle min/max in each axis, with NO padding
    // and no hysteresis. Mirrors AccSimulator.f90:1724-1733
    // (zadjmax=0.0d0; update_CompDom(grange,...)). Cell density per cell
    // matches ImpactT, fixing the 4× sigma_gamma growth in L0A acceleration
    // that the padded adaptive mode exhibited.
    bool geom_changed = false;
    if (m_exact_range) {
        Real xmin, xmax, ymin, ymax, zmin, zmax;
        if (compute_bunch_range(bunch, xmin, xmax, ymin, ymax, zmin, zmax)) {
            // Guard against degenerate ranges (single particle, zero spread).
            constexpr Real kMinHalfExtent = Real(1.0e-9);   // 1 nm floor
            const Real x_c    = Real(0.5) * (xmin + xmax);
            const Real y_c    = Real(0.5) * (ymin + ymax);
            const Real z_c    = Real(0.5) * (zmin + zmax);
            const Real half_x = std::max(Real(0.5) * (xmax - xmin), kMinHalfExtent);
            const Real half_y = std::max(Real(0.5) * (ymax - ymin), kMinHalfExtent);
            const Real half_z = std::max(Real(0.5) * (zmax - zmin), kMinHalfExtent);
            resize(x_c, y_c, z_c, half_x, half_y, half_z);
            geom_changed = true;
        }
    }
    // Hybrid mode: STATIC xy + ADAPTIVE z. Keeps the well-conditioned
    // transverse cell aspect (per task #24's IGF-discretization
    // observation that high cell aspect degrades transverse SC accuracy)
    // while letting z resolve compressed bunches. Higher priority than
    // pure adaptive but lower than exact_range.
    else if (m_hybrid_z) {
        Real x_c, y_c, z_c, sx, sy, sz;
        if (compute_bunch_stats(bunch, x_c, y_c, z_c, sx, sy, sz)) {
            const Real pad_z = std::max(m_pad_factor * sz, m_min_pad_z);

            auto const* lo_p = m_geom.ProbLo();
            auto const* hi_p = m_geom.ProbHi();
            const Real cur_xy_x_c  = Real(0.5) * (hi_p[0] + lo_p[0]);
            const Real cur_xy_y_c  = Real(0.5) * (hi_p[1] + lo_p[1]);
            const Real cur_half_x  = Real(0.5) * (hi_p[0] - lo_p[0]);
            const Real cur_half_y  = Real(0.5) * (hi_p[1] - lo_p[1]);
            const Real cur_half_z  = Real(0.5) * (hi_p[2] - lo_p[2]);
            const Real cur_z_c     = Real(0.5) * (hi_p[2] + lo_p[2]);

            // Trigger only on z change: 2x size hysteresis or > 50% half-extent
            // centroid drift in z. xy half-extent + xy centroid stay frozen.
            const bool first_call   = (m_n_recenters == 0);
            const bool z_size_drift = pad_z > Real(2.0) * cur_half_z ||
                                      pad_z < Real(0.5) * cur_half_z;
            const bool z_cent_drift = std::abs(z_c - cur_z_c) > Real(0.5) * cur_half_z;

            if (first_call || z_size_drift || z_cent_drift) {
                resize(cur_xy_x_c, cur_xy_y_c, z_c, cur_half_x, cur_half_y, pad_z);
                geom_changed = true;
            }
        }
    }
    // Adaptive resize: track the bunch's CURRENT sigmas and resize the
    // mesh to enclose ~pad_factor * sigma in each direction. Takes
    // priority over the simpler comoving recenter (and supersedes it
    // — adaptive does both).
    else if (m_adaptive) {
        Real x_c, y_c, z_c, sx, sy, sz;
        if (compute_bunch_stats(bunch, x_c, y_c, z_c, sx, sy, sz)) {
            const Real pad_x = std::max(m_pad_factor * sx, m_min_pad_xy);
            const Real pad_y = std::max(m_pad_factor * sy, m_min_pad_xy);
            const Real pad_z = std::max(m_pad_factor * sz, m_min_pad_z);

            // Trigger criteria: first solve, sigma drift > 50%, or
            // centroid drift past 25% of current half-extent in any
            // direction. Holds the LUT rebuild rate to a few per run
            // for typical photoinjector cases.
            auto const* lo_p = m_geom.ProbLo();
            auto const* hi_p = m_geom.ProbHi();
            const Real cur_half_x = Real(0.5) * (hi_p[0] - lo_p[0]);
            const Real cur_half_y = Real(0.5) * (hi_p[1] - lo_p[1]);
            const Real cur_half_z = Real(0.5) * (hi_p[2] - lo_p[2]);
            const Real cur_x_c    = Real(0.5) * (hi_p[0] + lo_p[0]);
            const Real cur_y_c    = Real(0.5) * (hi_p[1] + lo_p[1]);
            const Real cur_z_c    = Real(0.5) * (hi_p[2] + lo_p[2]);

            // Trigger sparingly: only resize when sigmas have at least
            // doubled or halved (factor-2 hysteresis) or the centroid
            // has drifted past 50% of current half-extent. Tighter
            // triggers cause LUT-rebuild thrashing every step, which
            // empirically degrades transverse-SC accuracy in
            // photoinjector runs (the per-step LUT noise overwhelms
            // the bunch's own SC field).
            const bool first_call = (m_n_recenters == 0);
            const bool size_drift = pad_x > Real(2.0) * cur_half_x || pad_x < Real(0.5) * cur_half_x ||
                                    pad_y > Real(2.0) * cur_half_y || pad_y < Real(0.5) * cur_half_y ||
                                    pad_z > Real(2.0) * cur_half_z || pad_z < Real(0.5) * cur_half_z;
            const bool cent_drift = std::abs(x_c - cur_x_c) > Real(0.5) * cur_half_x ||
                                    std::abs(y_c - cur_y_c) > Real(0.5) * cur_half_y ||
                                    std::abs(z_c - cur_z_c) > Real(0.5) * cur_half_z;

            if (first_call || size_drift || cent_drift) {
                resize(x_c, y_c, z_c, pad_x, pad_y, pad_z);
                geom_changed = true;
            }
        }
    }
    // Co-moving recenter check (z-only): shift the mesh in z if the
    // bunch has drifted into the outer 25% of the box. Skipped when
    // adaptive is on (adaptive already does both centroid and size).
    else if (m_comoving) {
        const long n = bunch.TotalNumberOfParticles(true, false);
        if (n > 0) {
            const Real z_centroid = compute_mean_z(bunch);
            auto const* lo_p = m_geom.ProbLo();
            auto const* hi_p = m_geom.ProbHi();
            const Real margin = Real(0.25) * (hi_p[2] - lo_p[2]);
            if (z_centroid < lo_p[2] + margin || z_centroid > hi_p[2] - margin) {
                recenter(z_centroid);
                geom_changed = true;
            }
        }
    }

    // No SetParticleGeometry / Redistribute is needed here: deposit_charge
    // is position-based and writes to the single-FAB m_rho directly,
    // independent of the particle tile assignment in TimeBunch. (The
    // geom_changed flag is preserved above in case a future MPI extension
    // wants to react to mesh resize events; suppress the unused warning.)
    (void)geom_changed;

    m_last_beta_z = compute_mean_beta_z(bunch);

    deposit_charge(bunch);
    smooth_rho();

    auto const* dx = m_geom.CellSize();
    const Real inv_g = std::sqrt(Real(1.0) - m_last_beta_z * m_last_beta_z);
    const Real gamma = (inv_g > Real(0.0)) ? Real(1.0) / inv_g : Real(1.0);
    const Real dz_scaled = (inv_g > Real(0.0)) ? dx[2] / inv_g : dx[2];
    std::array<Real, 3> cell_size = {dx[0], dx[1], dz_scaled};

    // Direct (free-space) Poisson solve.
    ablastr::fields::computePhiIGF(m_rho, m_phi, cell_size, /*is_2d_slices*/ false);

    // Cathode image-charge solve via z-shifted Green function
    // (Qiang/IMPACT-T method). Adds the contribution from images of
    // the bunch's particles, mirrored about the cathode plane, into
    // m_phi without needing to extend the mesh below the cathode.
    if (m_image_plane_enabled) {
        // Make sure m_phi_image exists and has the same layout as m_phi.
        if (!m_phi_image.ok() ||
            m_phi_image.boxArray() != m_phi.boxArray() ||
            m_phi_image.DistributionMap() != m_phi.DistributionMap())
        {
            m_phi_image.define(m_phi.boxArray(), m_phi.DistributionMap(),
                               m_phi.nComp(), m_phi.nGrowVect());
        }
        m_phi_image.setVal(0.0);

        // Compute z_centroid in the rest-frame-stretched coordinates
        // used by the IGF (z_solver = gamma * z_lab). Shift the Green
        // function by -2 * z_centroid_above_cathode in solver coords —
        // this places the image of the bunch into the FFT domain.
        const Real z_centroid_lab = compute_mean_z(bunch);
        const Real z_above_lab    = z_centroid_lab - m_image_plane_z;

        // Optional cutoff: skip the image solve when the bunch is far
        // enough above the cathode that image effects are negligible.
        // Matches IMPACT-T's `Zimage` screen.
        if (z_above_lab > Real(0.0) && z_above_lab <= m_image_cutoff) {
            const Real z_shift = -Real(2.0) * gamma * z_above_lab;
            compute_phi_IGF_shifted(m_rho, m_phi_image, cell_size, z_shift);

            // phi_total = phi_direct - phi_image (image source has
            // opposite sign to the real bunch — the negation is folded
            // into the subtraction here rather than into rho).
            MultiFab::Subtract(m_phi, m_phi_image, 0, 0, m_phi.nComp(), m_phi.nGrowVect());
        }
    }

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


void SpaceCharge::compute_phi_IGF_shifted (
    amrex::MultiFab const&            rho,
    amrex::MultiFab&                  phi,
    std::array<amrex::Real, 3> const& cell_size,
    amrex::Real                       z_shift) const
{
    using namespace amrex;
    using namespace amrex::literals;

    // Mirror of ablastr::fields::computePhiIGF, with `+ z_shift` added
    // to the z coordinate passed to SumOfIntegratedPotential3D. The
    // z-shifted Green function is the Qiang/IMPACT-T trick: we leave
    // rho on the bunch mesh and shift the kernel so the convolution
    // gives the image-charge contribution at observation points within
    // the same mesh — no need for an extended/mirror mesh in z.
    //
    // The shift is exact only for sources at the centroid; for sources
    // off-centroid there's a small phase error that's bounded by
    // ~(sigma_z / z_centroid_above_cathode)^2. Same approximation
    // IMPACT-T makes (zshift = -2*z_centroid in their notation).

    Box domain = rho.boxArray().minimalBox();
    domain.grow(phi.nGrowVect());

    int nprocs = ParallelDescriptor::NProcs();
    {
        ParmParse pp("ablastr");
        pp.queryAdd("nprocs_igf_fft", nprocs);
        nprocs = std::max(1, std::min(nprocs, ParallelDescriptor::NProcs()));
    }

    static std::unique_ptr<FFT::OpenBCSolver<Real>> obc_solver_image;
    if (!obc_solver_image) {
        ExecOnFinalize([&] () { obc_solver_image.reset(); });
    }
    if (!obc_solver_image || obc_solver_image->Domain() != domain) {
        FFT::Info info{};
        info.setNumProcs(nprocs);
        obc_solver_image = std::make_unique<FFT::OpenBCSolver<Real>>(domain, info);
    }

    auto const& lo = domain.smallEnd();
    Real const dx = cell_size[0];
    Real const dy = cell_size[1];
    Real const dz = cell_size[2];

    obc_solver_image->setGreensFunction(
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Real
        {
            int const i0 = i - lo[0];
            int const j0 = j - lo[1];
            int const k0 = k - lo[2];
            Real const x = i0 * dx;
            Real const y = j0 * dy;
            Real const z = k0 * dz + z_shift;     // <-- the shift
            return ablastr::fields::SumOfIntegratedPotential3D(x, y, z, dx, dy, dz);
        });

    obc_solver_image->solve(phi, rho);
}

} // namespace spacecharge
} // namespace lucretiatt
