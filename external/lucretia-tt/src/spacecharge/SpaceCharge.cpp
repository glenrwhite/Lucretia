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
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace lucretiatt {
namespace spacecharge {

namespace {
constexpr amrex::Real kSpeedOfLight = amrex::Real(299792458.0);

// TSC (triangle-shape-cloud / quadratic spline) shape function weights.
// Particle is at fractional position p in [-0.5, 0.5) relative to the
// NEAREST node. Returns weights for the three surrounding nodes
// (node-1, node, node+1) which sum to 1.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tsc_weights (amrex::Real p, amrex::Real& w_lo, amrex::Real& w_mid, amrex::Real& w_hi) noexcept
{
    using amrex::Real;
    const Real half_minus_p = Real(0.5) - p;
    const Real half_plus_p  = Real(0.5) + p;
    w_lo  = Real(0.5) * half_minus_p * half_minus_p;
    w_mid = Real(0.75) - p * p;
    w_hi  = Real(0.5) * half_plus_p  * half_plus_p;
}
}


SpaceCharge::SpaceCharge (
    const amrex::Geometry&            geom,
    const amrex::BoxArray&            cell_ba,
    const amrex::DistributionMapping& dm)
    : m_geom(geom)
    // Multi-FAB: use the caller's BoxArray + DistributionMapping. The
    // cell_ba is converted to nodal index type so the IGF solver gets
    // node-centered fields.
    //
    // Single-rank with multi-FAB: each FAB is owned by this rank; deposit
    // iterates MFIter over all FABs, SumBoundary merges ghost contributions
    // back into owning FABs. Gather walks the FAB list per particle (see
    // GatherFABs in SpaceCharge.H).
    //
    // Multi-rank: each rank owns a slab of FABs. Particles whose stencil
    // crosses a rank boundary need their deposit ghost-cell-summed via
    // SumBoundary, AND particles whose owning FAB is on another rank need
    // their data communicated. Currently TrackingLoop iterates particles
    // locally so the latter is not yet handled -- multi-rank will need
    // particle redistribution after each SC mesh recenter (TODO).
    , m_ba_nodal(amrex::convert(cell_ba, amrex::IntVect{1, 1, 1}))
    , m_dm(dm)
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

    if (m_shape_order == 1) {
        // CIC: 8-node cube. LUT indexed by wx_frac in [0,1) (fractional
        // position within cell whose lo-corner is the deposit base node).
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
    } else {
        // TSC: 27-node cube around NEAREST node. LUT indexed by p in
        // [-0.5, +0.5). The N+1 samples span [-0.5, +0.5] inclusive
        // (LUT_idx = (p + 0.5) * N), so trilinear interp in TrackingLoop
        // uses the same indexing pattern.
        Real wxs[3], wys[3], wzs[3];
        for (int ix = 0; ix <= N; ++ix) {
            const Real px = Real(ix) / Real(N) - Real(0.5);
            tsc_weights(px, wxs[0], wxs[1], wxs[2]);
            for (int iy = 0; iy <= N; ++iy) {
                const Real py = Real(iy) / Real(N) - Real(0.5);
                tsc_weights(py, wys[0], wys[1], wys[2]);
                for (int iz = 0; iz <= N; ++iz) {
                    const Real pz = Real(iz) / Real(N) - Real(0.5);
                    tsc_weights(pz, wzs[0], wzs[1], wzs[2]);

                    Real Ex_unit = 0.0, Ey_unit = 0.0, Ez_unit = 0.0;
                    for (int kk = 0; kk < 3; ++kk) {
                        const Real wk   = wzs[kk];
                        const Real rz_v = (pz - Real(kk - 1)) * dz;   // (kk-1) ∈ {-1,0,1}
                        for (int jj = 0; jj < 3; ++jj) {
                            const Real wj   = wys[jj];
                            const Real ry_v = (py - Real(jj - 1)) * dy;
                            for (int ii = 0; ii < 3; ++ii) {
                                const Real wi   = wxs[ii];
                                const Real rx_v = (px - Real(ii - 1)) * dx;
                                const Real r2 = rx_v*rx_v + ry_v*ry_v + rz_v*rz_v + eps2;
                                const Real r3_inv = Real(1.0) / (r2 * std::sqrt(r2));
                                const Real factor = wi * wj * wk * r3_inv;
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
            if (m_z_filter_min_active && zs[i] < m_z_filter_min) { continue; }
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
            if (alives[i] == 0) { continue; }
            // Optional z-filter: skip particles below cathode plane.
            // Matches ImpactT's flagpos=1 mode during emission, which
            // confines the SC mesh to z>=cathode_z so behind-cathode
            // drifting particles don't extend the mesh below the plane.
            if (m_z_filter_min_active && zs[i] < m_z_filter_min) { continue; }
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
            // Skip dead particles (parked at z=-1e9). Including them would
            // skew the centroid catastrophically.
            if (alives[i] == 0) { continue; }
            // Optional z-filter: skip particles below cathode plane.
            if (m_z_filter_min_active && zs[i] < m_z_filter_min) { continue; }
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
        const auto& zs  = soa.GetRealData(RealSoA::z);
        const auto& uxs = soa.GetRealData(RealSoA::px);
        const auto& uys = soa.GetRealData(RealSoA::py);
        const auto& uzs = soa.GetRealData(RealSoA::pz);
        const auto& alives = soa.GetIntData(IntSoA::alive);
        for (int i = 0; i < np; ++i) {
            if (alives[i] == 0) { continue; }
            if (m_z_filter_min_active && zs[i] < m_z_filter_min) { continue; }
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

    // Position-based deposit, multi-FAB compatible:
    //   For each FAB on this rank (MFIter), iterate ALL particles and
    //   deposit those whose footprint overlaps the FAB's grown box (interior
    //   + 1 ghost cell). After the loop, SumBoundary merges ghost-cell
    //   contributions across rank/box boundaries into the interior cells of
    //   the owning FAB.
    //
    // Cost: O(N_particles * N_FABs_local). For typical 48^3 mesh + 27 boxes
    // (max_grid_size=16), the per-particle bounds check is ~10 cycles ×
    // 27 FABs = ~270 cycles. The actual deposit (8 corner adds) only fires
    // for the 1 (interior) or 2-4 (near boundary) FABs that overlap.
    //
    // Particles fully outside the SC mesh footprint are skipped silently
    // (they still get external-field gather + Boris in TrackingLoop, just
    // no SC). Matches the previous single-FAB behavior.

    // Gather ALL particle data into flat vectors first (avoids nested
    // MFIter: PIter internally uses MFIter on the particle container, and
    // AMReX disallows nested MFIters by default).
    //
    // Multi-rank: TimeBunch's PIter only sees LOCAL particles, but the SC
    // mesh is distributed differently from the particles. Particles whose
    // owning FAB is on a DIFFERENT rank wouldn't deposit anywhere if we
    // only used local particles. Workaround: AllGather all particles to
    // every rank, then each rank's deposit only touches its own FABs (per
    // valid-box ownership rule). Brute-force but correct; for true
    // production scaling, the right fix is to redistribute particles to
    // match the SC mesh's BoxArray after each recenter.
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    Vector<Real> all_x, all_y, all_z, all_q, all_w;
    Vector<int>  all_alive;
    {
        // Local particle count (this rank only).
        std::size_t local_np = 0;
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            local_np += pti.numParticles();
        }
        Vector<Real> loc_x, loc_y, loc_z, loc_q, loc_w;
        Vector<int>  loc_alive;
        loc_x.reserve(local_np); loc_y.reserve(local_np); loc_z.reserve(local_np);
        loc_q.reserve(local_np); loc_w.reserve(local_np); loc_alive.reserve(local_np);
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            auto& soa = pti.GetStructOfArrays();
            const int np = pti.numParticles();
            const auto& xs_p = soa.GetRealData(RealSoA::x);
            const auto& ys_p = soa.GetRealData(RealSoA::y);
            const auto& zs_p = soa.GetRealData(RealSoA::z);
            const auto& qs_p = soa.GetRealData(RealSoA::q);
            const auto& ws_p = soa.GetRealData(RealSoA::w);
            const auto& al_p = soa.GetIntData(IntSoA::alive);
            for (int i = 0; i < np; ++i) {
                loc_x.push_back(xs_p[i]); loc_y.push_back(ys_p[i]); loc_z.push_back(zs_p[i]);
                loc_q.push_back(qs_p[i]); loc_w.push_back(ws_p[i]);
                loc_alive.push_back(al_p[i]);
            }
        }
        // AllGather across ranks. With single rank the local arrays become
        // the global arrays (no-op).
        const int nprocs = ParallelDescriptor::NProcs();
        if (nprocs == 1) {
            all_x = std::move(loc_x); all_y = std::move(loc_y); all_z = std::move(loc_z);
            all_q = std::move(loc_q); all_w = std::move(loc_w);
            all_alive = std::move(loc_alive);
        } else {
#ifdef AMREX_USE_MPI
            // Counts and displacements for MPI_Allgatherv.
            const int my_count = int(loc_x.size());
            Vector<int> counts(nprocs);
            MPI_Allgather(&my_count, 1, MPI_INT,
                          counts.data(), 1, MPI_INT,
                          ParallelDescriptor::Communicator());
            Vector<int> disps(nprocs, 0);
            for (int r = 1; r < nprocs; ++r) disps[r] = disps[r-1] + counts[r-1];
            const std::size_t total = std::size_t(disps.back()) + std::size_t(counts.back());
            all_x.resize(total);    all_y.resize(total);    all_z.resize(total);
            all_q.resize(total);    all_w.resize(total);    all_alive.resize(total);

            // Pick the right MPI types for Real (float or double) and int.
            MPI_Datatype real_t = (sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
            MPI_Allgatherv(loc_x.data(), my_count, real_t,
                           all_x.data(), counts.data(), disps.data(),
                           real_t, ParallelDescriptor::Communicator());
            MPI_Allgatherv(loc_y.data(), my_count, real_t,
                           all_y.data(), counts.data(), disps.data(),
                           real_t, ParallelDescriptor::Communicator());
            MPI_Allgatherv(loc_z.data(), my_count, real_t,
                           all_z.data(), counts.data(), disps.data(),
                           real_t, ParallelDescriptor::Communicator());
            MPI_Allgatherv(loc_q.data(), my_count, real_t,
                           all_q.data(), counts.data(), disps.data(),
                           real_t, ParallelDescriptor::Communicator());
            MPI_Allgatherv(loc_w.data(), my_count, real_t,
                           all_w.data(), counts.data(), disps.data(),
                           real_t, ParallelDescriptor::Communicator());
            MPI_Allgatherv(loc_alive.data(), my_count, MPI_INT,
                           all_alive.data(), counts.data(), disps.data(),
                           MPI_INT, ParallelDescriptor::Communicator());
#else
            // No MPI build: shouldn't happen since nprocs > 1 implies MPI.
            all_x = std::move(loc_x); all_y = std::move(loc_y); all_z = std::move(loc_z);
            all_q = std::move(loc_q); all_w = std::move(loc_w);
            all_alive = std::move(loc_alive);
#endif
        }
    }
    const std::size_t np_total = all_x.size();

    long total_deposited = 0;

    // Domain hi (nodal): m_geom.Domain() is cell-centered (size = ncell), so
    // the nodal domain hi is bigEnd + 1 in each direction. Used to detect FABs
    // whose vhi sits on the global boundary — they own their high-side nodes
    // (since no neighbor exists to claim them).
    Box const& dom_cc = m_geom.Domain();
    const int dhi_x = dom_cc.bigEnd(0) + 1;
    const int dhi_y = dom_cc.bigEnd(1) + 1;
    const int dhi_z = dom_cc.bigEnd(2) + 1;

    for (MFIter mfi(m_rho); mfi.isValid(); ++mfi) {
        Box const& vbox    = mfi.validbox();        // interior only -- ownership
        Box const& gbox    = m_rho[mfi].box();      // valid + 1 ghost (deposit footprint)
        Array4<Real> rho_arr = m_rho.array(mfi);
        auto const& vlo = vbox.smallEnd();
        auto const& vhi = vbox.bigEnd();
        auto const& glo = gbox.smallEnd();
        auto const& ghi = gbox.bigEnd();
        // For nodal MultiFabs, the validboxes of adjacent FABs OVERLAP at
        // their shared boundary nodes (e.g., max_grid_size=16: FAB1 vhi=16,
        // FAB2 vlo=16; node 16 is in BOTH validboxes). Without explicit
        // exclusion, particles whose base node lands on a shared boundary
        // get deposited TWICE -- once by each adjacent FAB. The exclusion
        // rule: a FAB owns base node ix iff (ix < vhi) OR (vhi == domain_hi).
        // The "domain_hi" exception ensures the rightmost FAB still owns
        // its outermost node, since no FAB exists to its right. (Task #53.)
        const bool right_x = (vhi[0] == dhi_x);
        const bool right_y = (vhi[1] == dhi_y);
        const bool right_z = (vhi[2] == dhi_z);
        const int  vhi_eff_x = right_x ? vhi[0] : (vhi[0] - 1);
        const int  vhi_eff_y = right_y ? vhi[1] : (vhi[1] - 1);
        const int  vhi_eff_z = right_z ? vhi[2] : (vhi[2] - 1);

        // Use the gathered flat arrays so we don't nest a PIter (= MFIter).
        const auto& xs = all_x;
        const auto& ys = all_y;
        const auto& zs = all_z;
        const auto& qs = all_q;
        const auto& ws = all_w;
        const auto& alives = all_alive;
        const int   np = int(np_total);
        {
            if (m_shape_order == 1) {
                for (int p = 0; p < np; ++p) {
                    if (alives[p] == 0) { continue; }
                    if (m_z_filter_min_active && zs[p] < m_z_filter_min) { continue; }

                    const Real fx = (xs[p] - lo[0]) * inv_dx;
                    const Real fy = (ys[p] - lo[1]) * inv_dy;
                    const Real fz = (zs[p] - lo[2]) * inv_dz;
                    const int  ix = int(std::floor(fx));
                    const int  iy = int(std::floor(fy));
                    const int  iz = int(std::floor(fz));
                    // Ownership: this FAB owns the particle iff its base node
                    // (ix, iy, iz) is inside the FAB's VALID box, with the
                    // high-side shared-node exclusion (see comments above
                    // the MFIter loop for the nodal-overlap fix).
                    if (!(ix >= vlo[0] && ix <= vhi_eff_x &&
                          iy >= vlo[1] && iy <= vhi_eff_y &&
                          iz >= vlo[2] && iz <= vhi_eff_z))
                    { continue; }
                    const Real wx1 = fx - Real(ix);   const Real wx0 = Real(1.0) - wx1;
                    const Real wy1 = fy - Real(iy);   const Real wy0 = Real(1.0) - wy1;
                    const Real wz1 = fz - Real(iz);   const Real wz0 = Real(1.0) - wz1;
                    const Real qw = qs[p] * ws[p] * inv_dV;

                    // Stencil neighbors (ix+1, iy+1, iz+1) may fall in this FAB's
                    // ghost layer -- write there too; SumBoundary merges to the
                    // owning FAB (which will then have their interior contribution
                    // SUMMED with our ghost-layer contribution).
                    if (ix + 1 <= ghi[0] && iy + 1 <= ghi[1] && iz + 1 <= ghi[2])
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
                    }
                }
            } else {
                Real wx[3], wy[3], wz[3];
                for (int p = 0; p < np; ++p) {
                    if (alives[p] == 0) { continue; }
                    if (m_z_filter_min_active && zs[p] < m_z_filter_min) { continue; }

                    const Real fx = (xs[p] - lo[0]) * inv_dx;
                    const Real fy = (ys[p] - lo[1]) * inv_dy;
                    const Real fz = (zs[p] - lo[2]) * inv_dz;
                    const int  ix = int(std::round(fx));
                    const int  iy = int(std::round(fy));
                    const int  iz = int(std::round(fz));
                    // TSC ownership: the central node (ix, iy, iz) is the
                    // base. Use the same vhi_eff exclusion as CIC.
                    if (!(ix >= vlo[0] && ix <= vhi_eff_x &&
                          iy >= vlo[1] && iy <= vhi_eff_y &&
                          iz >= vlo[2] && iz <= vhi_eff_z))
                    { continue; }
                    const Real px = fx - Real(ix);
                    const Real py = fy - Real(iy);
                    const Real pz = fz - Real(iz);
                    tsc_weights(px, wx[0], wx[1], wx[2]);
                    tsc_weights(py, wy[0], wy[1], wy[2]);
                    tsc_weights(pz, wz[0], wz[1], wz[2]);
                    const Real qw = qs[p] * ws[p] * inv_dV;

                    if (ix - 1 >= glo[0] && ix + 1 <= ghi[0] &&
                        iy - 1 >= glo[1] && iy + 1 <= ghi[1] &&
                        iz - 1 >= glo[2] && iz + 1 <= ghi[2])
                    {
                        for (int kk = 0; kk < 3; ++kk) {
                            const Real wzz = wz[kk];
                            const int  zk  = iz - 1 + kk;
                            for (int jj = 0; jj < 3; ++jj) {
                                const Real wyy = wy[jj];
                                const int  yj  = iy - 1 + jj;
                                const Real wyz = wzz * wyy;
                                for (int ii = 0; ii < 3; ++ii) {
                                    rho_arr(ix - 1 + ii, yj, zk) += qw * wx[ii] * wyz;
                                }
                            }
                        }
                        ++total_deposited;
                    }
                }
            }
        }
    }

    // Merge ghost-cell contributions into interior cells of owning FAB.
    // Required for multi-FAB / multi-rank correctness; for single-FAB this
    // is essentially a no-op (no internal box boundaries to merge across).
    m_rho.SumBoundary(m_geom.periodicity());

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
                       << " deposited=" << total_deposited << "\n";
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


void SpaceCharge::replicate_e_fields_for_gather ()
{
    using namespace amrex;

    // Single-rank: clear replicas (gather_fabs falls back to local MFIter).
    if (ParallelDescriptor::NProcs() <= 1) {
        m_Ex_replica.clear();
        m_Ey_replica.clear();
        m_Ez_replica.clear();
        m_replica_boxes.clear();
        m_replica_valid_boxes.clear();
        m_Ex_image_replica.clear();
        m_Ey_image_replica.clear();
        m_Ez_image_replica.clear();
        return;
    }

#ifdef AMREX_USE_MPI
    // Helper: pack one MultiFab's local FABs into flat buffers, AllGather
    // across ranks, and unpack into per-rank FArrayBox vectors.
    auto replicate_one = [&](MultiFab const& src,
                              std::vector<FArrayBox>& dst_fabs,
                              std::vector<Box>* dst_boxes_ghost = nullptr,
                              std::vector<Box>* dst_boxes_valid = nullptr) {
        const int nprocs = ParallelDescriptor::NProcs();

        // Step 1: collect local FAB metadata + data sizes.
        // Each FAB record: 6 ints (ghost lo/hi) + 6 ints (valid lo/hi) +
        //                  1 int (data length), then `data length` doubles.
        constexpr int kIntsPerFab = 13;
        std::vector<int>  local_meta;
        std::vector<Real> local_data;
        for (MFIter mfi(src); mfi.isValid(); ++mfi) {
            const Box gb = src[mfi].box();         // valid + ghost
            const Box vb = mfi.validbox();          // interior
            const int len = int(gb.numPts());
            local_meta.push_back(gb.smallEnd(0)); local_meta.push_back(gb.smallEnd(1));
            local_meta.push_back(gb.smallEnd(2)); local_meta.push_back(gb.bigEnd(0));
            local_meta.push_back(gb.bigEnd(1));   local_meta.push_back(gb.bigEnd(2));
            local_meta.push_back(vb.smallEnd(0)); local_meta.push_back(vb.smallEnd(1));
            local_meta.push_back(vb.smallEnd(2)); local_meta.push_back(vb.bigEnd(0));
            local_meta.push_back(vb.bigEnd(1));   local_meta.push_back(vb.bigEnd(2));
            local_meta.push_back(len);
            Real const* p = src[mfi].dataPtr();
            local_data.insert(local_data.end(), p, p + len);
        }
        const int my_meta_count = int(local_meta.size());
        const int my_data_count = int(local_data.size());

        // Step 2: AllGather counts to compute displacements.
        std::vector<int> meta_counts(nprocs), meta_disps(nprocs, 0);
        std::vector<int> data_counts(nprocs), data_disps(nprocs, 0);
        MPI_Allgather(&my_meta_count, 1, MPI_INT, meta_counts.data(), 1,
                      MPI_INT, ParallelDescriptor::Communicator());
        MPI_Allgather(&my_data_count, 1, MPI_INT, data_counts.data(), 1,
                      MPI_INT, ParallelDescriptor::Communicator());
        for (int r = 1; r < nprocs; ++r) {
            meta_disps[r] = meta_disps[r-1] + meta_counts[r-1];
            data_disps[r] = data_disps[r-1] + data_counts[r-1];
        }
        const std::size_t total_meta = std::size_t(meta_disps.back())
                                     + std::size_t(meta_counts.back());
        const std::size_t total_data = std::size_t(data_disps.back())
                                     + std::size_t(data_counts.back());

        // Step 3: AllGatherv metadata and data.
        std::vector<int>  all_meta(total_meta);
        std::vector<Real> all_data(total_data);
        MPI_Allgatherv(local_meta.data(), my_meta_count, MPI_INT,
                       all_meta.data(), meta_counts.data(), meta_disps.data(),
                       MPI_INT, ParallelDescriptor::Communicator());
        const MPI_Datatype real_t =
            (sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
        MPI_Allgatherv(local_data.data(), my_data_count, real_t,
                       all_data.data(), data_counts.data(), data_disps.data(),
                       real_t, ParallelDescriptor::Communicator());

        // Step 4: unpack into FArrayBox vector.
        const int total_fabs = int(total_meta / kIntsPerFab);
        dst_fabs.clear();
        dst_fabs.reserve(total_fabs);
        if (dst_boxes_ghost) { dst_boxes_ghost->clear(); dst_boxes_ghost->reserve(total_fabs); }
        if (dst_boxes_valid) { dst_boxes_valid->clear(); dst_boxes_valid->reserve(total_fabs); }
        std::size_t data_off = 0;
        for (int f = 0; f < total_fabs; ++f) {
            const int* m = all_meta.data() + f * kIntsPerFab;
            Box gb({m[0], m[1], m[2]}, {m[3], m[4], m[5]}, IndexType::TheNodeType());
            Box vb({m[6], m[7], m[8]}, {m[9], m[10], m[11]}, IndexType::TheNodeType());
            const int len = m[12];
            dst_fabs.emplace_back(gb, 1);
            std::memcpy(dst_fabs.back().dataPtr(), all_data.data() + data_off,
                        std::size_t(len) * sizeof(Real));
            data_off += std::size_t(len);
            if (dst_boxes_ghost) dst_boxes_ghost->push_back(gb);
            if (dst_boxes_valid) dst_boxes_valid->push_back(vb);
        }
    };

    // Direct E (always replicated).
    replicate_one(m_Ex, m_Ex_replica, &m_replica_boxes, &m_replica_valid_boxes);
    replicate_one(m_Ey, m_Ey_replica);
    replicate_one(m_Ez, m_Ez_replica);

    // Image E (only when active this solve).
    if (m_image_active_this_solve && m_Ex_image.ok()) {
        replicate_one(m_Ex_image, m_Ex_image_replica);
        replicate_one(m_Ey_image, m_Ey_image_replica);
        replicate_one(m_Ez_image, m_Ez_image_replica);
    } else {
        m_Ex_image_replica.clear();
        m_Ey_image_replica.clear();
        m_Ez_image_replica.clear();
    }
#endif // AMREX_USE_MPI
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

    // Task #61/#63: also compute IMAGE E from m_phi_image (kept separate
    // from m_phi so its kick can be applied with negated B-field sign,
    // mirroring imp's gradEB_FieldQuant betC sign flip).
    if (m_image_active_this_solve && m_phi_image.ok()) {
        if (!m_Ex_image.ok() ||
            m_Ex_image.boxArray() != m_Ex.boxArray() ||
            m_Ex_image.DistributionMap() != m_Ex.DistributionMap())
        {
            m_Ex_image.define(m_Ex.boxArray(), m_Ex.DistributionMap(),
                              m_Ex.nComp(), m_Ex.nGrowVect());
            m_Ey_image.define(m_Ey.boxArray(), m_Ey.DistributionMap(),
                              m_Ey.nComp(), m_Ey.nGrowVect());
            m_Ez_image.define(m_Ez.boxArray(), m_Ez.DistributionMap(),
                              m_Ez.nComp(), m_Ez.nGrowVect());
        }
        m_phi_image.FillBoundary(m_geom.periodicity());
        for (MFIter mfi(m_phi_image, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const& vbox = mfi.validbox();
            auto const& phi_img = m_phi_image.const_array(mfi);
            auto Ex_img = m_Ex_image.array(mfi);
            auto Ey_img = m_Ey_image.array(mfi);
            auto Ez_img = m_Ez_image.array(mfi);
            ParallelFor(vbox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Ex_img(i, j, k) = -(phi_img(i+1, j, k) - phi_img(i-1, j, k)) * inv_2dx;
                Ey_img(i, j, k) = -(phi_img(i, j+1, k) - phi_img(i, j-1, k)) * inv_2dy;
                Ez_img(i, j, k) = -(phi_img(i, j, k+1) - phi_img(i, j, k-1)) * inv_2dz;
            });
        }
        m_Ex_image.FillBoundary(m_geom.periodicity());
        m_Ey_image.FillBoundary(m_geom.periodicity());
        m_Ez_image.FillBoundary(m_geom.periodicity());
    }
}


void SpaceCharge::solve (particles::TimeBunch& bunch)
{
    using namespace amrex;
    using namespace particles;

    ++m_solve_count;

    // ---- Resize-jump diagnostic: capture E_old at all alive particle
    // positions BEFORE any resize/deposit/solve. Re-gather after the
    // solve to quantify the per-particle field discontinuity caused by
    // a mesh resize. Skipped on the very first call (E_old is zeroed
    // by the constructor — no meaningful "discontinuity" to report).
    std::vector<Real> diag_x, diag_y, diag_z;
    std::vector<Real> diag_Ex_old, diag_Ey_old, diag_Ez_old;
    const bool do_diag = m_diag_resize_jump && m_n_recenters > 0;
    if (do_diag) {
        // Same CIC gather we use in TrackingLoop, kept inline for clarity.
        auto const* lo_p = m_geom.ProbLo();
        auto const* dx   = m_geom.CellSize();
        const Real lo0 = lo_p[0], lo1 = lo_p[1], lo2 = lo_p[2];
        const Real dxi0 = Real(1.0)/dx[0], dxi1 = Real(1.0)/dx[1], dxi2 = Real(1.0)/dx[2];
        const Box mesh_box_old = m_Ex[MFIter(m_Ex)].box();

        Array4<const Real> Ex_old = m_Ex.const_array(MFIter(m_Ex));
        Array4<const Real> Ey_old = m_Ey.const_array(MFIter(m_Ey));
        Array4<const Real> Ez_old = m_Ez.const_array(MFIter(m_Ez));

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
                if (alives[i] == 0) { continue; }
                if (m_z_filter_min_active && zs[i] < m_z_filter_min) { continue; }
                const Real xp = xs[i], yp = ys[i], zp = zs[i];
                const Real fx = (xp - lo0) * dxi0;
                const Real fy = (yp - lo1) * dxi1;
                const Real fz = (zp - lo2) * dxi2;
                const int ix = int(std::floor(fx));
                const int iy = int(std::floor(fy));
                const int iz = int(std::floor(fz));
                // Skip particles outside the OLD mesh footprint -- gather
                // would read garbage / out of array. Only count particles
                // that we can compare cleanly on both old and new meshes.
                if (ix    < mesh_box_old.smallEnd(0) || ix+1 > mesh_box_old.bigEnd(0) ||
                    iy    < mesh_box_old.smallEnd(1) || iy+1 > mesh_box_old.bigEnd(1) ||
                    iz    < mesh_box_old.smallEnd(2) || iz+1 > mesh_box_old.bigEnd(2)) {
                    continue;
                }
                const Real wx = fx - Real(ix), wy = fy - Real(iy), wz = fz - Real(iz);
                const Real omx = Real(1.0) - wx, omy = Real(1.0) - wy, omz = Real(1.0) - wz;
                auto cic = [&](Array4<const Real> const& F) {
                    return omx*omy*omz*F(ix  ,iy  ,iz  ,0)
                         + wx *omy*omz*F(ix+1,iy  ,iz  ,0)
                         + omx*wy *omz*F(ix  ,iy+1,iz  ,0)
                         + wx *wy *omz*F(ix+1,iy+1,iz  ,0)
                         + omx*omy*wz *F(ix  ,iy  ,iz+1,0)
                         + wx *omy*wz *F(ix+1,iy  ,iz+1,0)
                         + omx*wy *wz *F(ix  ,iy+1,iz+1,0)
                         + wx *wy *wz *F(ix+1,iy+1,iz+1,0);
                };
                diag_x.push_back(xp);
                diag_y.push_back(yp);
                diag_z.push_back(zp);
                diag_Ex_old.push_back(cic(Ex_old));
                diag_Ey_old.push_back(cic(Ey_old));
                diag_Ez_old.push_back(cic(Ez_old));
            }
        }
    }

    // Exact-bunch-range adaptive (ImpactT-style): mesh resized every step
    // to EXACTLY the alive-particle min/max in each axis, with NO padding
    // and no hysteresis. Mirrors AccSimulator.f90:1724-1733
    // (zadjmax=0.0d0; update_CompDom(grange,...)). Cell density per cell
    // matches ImpactT, fixing the 4× sigma_gamma growth in L0A acceleration
    // that the padded adaptive mode exhibited.
    bool geom_changed = false;
    // Optional outlier kill: in exact_range mode, a few outlier particles
    // can pull the bunch min/max well past the bulk (e.g., 4-5 particles at
    // 4-5*sigma drag the mesh wider, packing the bulk into fewer cells and
    // inflating per-cell density). Killing them BEFORE the range computation
    // keeps the mesh tight without leaving particles half-gathered outside
    // the mesh footprint. Marks alive=0 so subsequent forces (RF, SOL, slice
    // SC) also skip them — analogous to a soft particle aperture.
    if (m_outlier_kill_sigma > Real(0.0) && m_exact_range) {
        Real x_c, y_c, z_c, sx, sy, sz;
        if (compute_bunch_stats(bunch, x_c, y_c, z_c, sx, sy, sz)) {
            // Use max(N*sigma, abs_floor) so we never clip the bunch when
            // sigma is tiny (e.g., very early emission with near-zero sigma_z).
            // Default floor 5mm transverse, 10mm longitudinal — large enough
            // that legitimate gun-region bunches always sit inside the cap.
            const Real cap_x = std::max(m_outlier_kill_sigma * sx,
                                        m_outlier_kill_floor_xy);
            const Real cap_y = std::max(m_outlier_kill_sigma * sy,
                                        m_outlier_kill_floor_xy);
            const Real cap_z = std::max(m_outlier_kill_sigma * sz,
                                        m_outlier_kill_floor_z);
            using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
            constexpr int lev_kill = 0;
            long n_killed_local = 0;
            for (PIter pti(bunch, lev_kill); pti.isValid(); ++pti) {
                auto ptd = pti.GetParticleTile().getParticleTileData();
                const int np = pti.numParticles();
                for (int p = 0; p < np; ++p) {
                    if (ptd.idata(IntSoA::alive)[p] == 0) continue;
                    const Real xp = ptd.rdata(RealSoA::x)[p] - x_c;
                    const Real yp = ptd.rdata(RealSoA::y)[p] - y_c;
                    const Real zp = ptd.rdata(RealSoA::z)[p] - z_c;
                    if (std::abs(xp) > cap_x || std::abs(yp) > cap_y
                            || std::abs(zp) > cap_z) {
                        ptd.idata(IntSoA::alive)[p] = 0;
                        ptd.rdata(RealSoA::x )[p] = Real(0.0);
                        ptd.rdata(RealSoA::y )[p] = Real(0.0);
                        ptd.rdata(RealSoA::z )[p] = Real(-1.0e9);
                        ptd.rdata(RealSoA::px)[p] = Real(0.0);
                        ptd.rdata(RealSoA::py)[p] = Real(0.0);
                        ptd.rdata(RealSoA::pz)[p] = Real(0.0);
                        ++n_killed_local;
                    }
                }
            }
            ParallelDescriptor::ReduceLongSum(n_killed_local);
            m_n_outliers_killed += n_killed_local;
            if (m_outlier_kill_verbose && n_killed_local > 0) {
                amrex::Print() << "[SC] outlier_kill solve#" << m_solve_count
                               << ": killed " << n_killed_local
                               << " (cap=" << m_outlier_kill_sigma << "*sigma; "
                               << "cum=" << m_n_outliers_killed << ")\n";
            }
        }
    }

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

            // Trigger sparingly. Tighter triggers cause field-discontinuity
            // noise to dominate per-step kicks, inflating eps_nx in the gun
            // region (verified by tasks #33+#34). Factor-3 size hysteresis
            // and 50% centroid threshold gives a few resizes through the
            // gun, enough to track bunch growth but rare enough to avoid
            // resize-noise accumulation. Adjustable via m_resize_hyst.
            const bool first_call = (m_n_recenters == 0);
            const Real f_lo = Real(1.0) / m_resize_hyst;
            const Real f_hi = m_resize_hyst;
            const bool size_drift = pad_x > f_hi * cur_half_x || pad_x < f_lo * cur_half_x ||
                                    pad_y > f_hi * cur_half_y || pad_y < f_lo * cur_half_y ||
                                    pad_z > f_hi * cur_half_z || pad_z < f_lo * cur_half_z;
            const Real cdt = m_cent_drift_threshold;
            const bool cent_drift = std::abs(x_c - cur_x_c) > cdt * cur_half_x ||
                                    std::abs(y_c - cur_y_c) > cdt * cur_half_y ||
                                    std::abs(z_c - cur_z_c) > cdt * cur_half_z;

            if (first_call || size_drift || cent_drift) {
                Real new_x_c = x_c, new_y_c = y_c, new_z_c = z_c;
                Real new_half_x = pad_x, new_half_y = pad_y, new_half_z = pad_z;
                bool took_integer_shift = false;
                // Integer-cell snap: when only the centroid trigger fires
                // (size unchanged), snap the new center to (cur_center + N*dx)
                // and freeze the half-extent at its current value. The
                // resulting mesh has the same cell sizes and cell positions
                // (in absolute terms) as the old one, just relabelled. The
                // particles' fractional positions in their cells are
                // preserved -> same deposit -> same phi -> same gather E.
                // Zero discretization noise from this kind of resize.
                if (m_integer_cell_shift && cent_drift && !size_drift && !first_call) {
                    auto const* dx_cur = m_geom.CellSize();
                    const long n_shift_x = std::lround((x_c - cur_x_c) / dx_cur[0]);
                    const long n_shift_y = std::lround((y_c - cur_y_c) / dx_cur[1]);
                    const long n_shift_z = std::lround((z_c - cur_z_c) / dx_cur[2]);
                    new_x_c = cur_x_c + Real(n_shift_x) * dx_cur[0];
                    new_y_c = cur_y_c + Real(n_shift_y) * dx_cur[1];
                    new_z_c = cur_z_c + Real(n_shift_z) * dx_cur[2];
                    new_half_x = cur_half_x;
                    new_half_y = cur_half_y;
                    new_half_z = cur_half_z;
                    took_integer_shift = true;
                }
                if (m_diag_resize_jump) {
                    amrex::Print() << "[SC.adapt_trigger] first=" << (int)first_call
                                   << " size=" << (int)size_drift
                                   << " cent=" << (int)cent_drift
                                   << " snap=" << (int)took_integer_shift
                                   << "\n";
                }
                resize(new_x_c, new_y_c, new_z_c, new_half_x, new_half_y, new_half_z);
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

    // Note: a Redistribute()-based approach (sync bunch geometry to SC's
    // shrunk mesh) was attempted but discards behind-cathode and dead
    // particles silently — would need a wider bunch geometry separated
    // from SC's adaptive mesh. Instead we replicate the SC field across
    // ranks after solve (see solve() bottom: replicate_e_fields_for_gather
    // is called when nprocs>1). Multi-rank deposit is still position-based
    // via AllGather of particle data (no change here).
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

            // Task #63: apply z-reversal to m_phi_image. Mirrors imp's
            // invfft3d1Img_FFT (FFT.f90:303) which reverses z-order during
            // image inverse FFT. This converts the centroid-shifted-Green
            // result into the proper "extended-source image" form (each
            // particle has its own image at -z, not all sharing -z_centroid).
            {
                const auto lo_dom = m_geom.Domain().smallEnd();
                const int nx_dom = m_geom.Domain().length(0) + 1;
                const int ny_dom = m_geom.Domain().length(1) + 1;
                const int nz_dom = m_geom.Domain().length(2) + 1;
                std::vector<double> img_flat((std::size_t)nx_dom * ny_dom * nz_dom, 0.0);
                for (MFIter mfi(m_phi_image); mfi.isValid(); ++mfi) {
                    Array4<const Real> arr = m_phi_image.const_array(mfi);
                    Box const& vbox = mfi.validbox();
                    auto const lo_v = vbox.smallEnd();
                    auto const hi_v = vbox.bigEnd();
                    for (int k = lo_v[2]; k <= hi_v[2]; ++k)
                    for (int j = lo_v[1]; j <= hi_v[1]; ++j)
                    for (int i = lo_v[0]; i <= hi_v[0]; ++i) {
                        const int gi = i - lo_dom[0];
                        const int gj = j - lo_dom[1];
                        const int gk = k - lo_dom[2];
                        if (gi < 0 || gi >= nx_dom || gj < 0 || gj >= ny_dom ||
                            gk < 0 || gk >= nz_dom) continue;
                        img_flat[(std::size_t)((gk * ny_dom + gj) * nx_dom + gi)] =
                            (double)arr(i, j, k, 0);
                    }
                }
                // Multi-rank: each rank only filled img_flat with its own
                // local FABs above. The z-flip read at gk_flip below may
                // correspond to a DIFFERENT rank's z-slab, so we need a
                // global view. Sum-reduce: validbox values don't overlap
                // (each cell owned by exactly one rank), so summation == union.
                // (Task #54.)
                if (ParallelDescriptor::NProcs() > 1) {
                    ParallelDescriptor::ReduceRealSum(
                        img_flat.data(), int(img_flat.size()));
                }
                for (MFIter mfi(m_phi_image); mfi.isValid(); ++mfi) {
                    Array4<Real> arr = m_phi_image.array(mfi);
                    Box const& vbox = mfi.validbox();
                    auto const lo_v = vbox.smallEnd();
                    auto const hi_v = vbox.bigEnd();
                    for (int k = lo_v[2]; k <= hi_v[2]; ++k)
                    for (int j = lo_v[1]; j <= hi_v[1]; ++j)
                    for (int i = lo_v[0]; i <= hi_v[0]; ++i) {
                        const int gi = i - lo_dom[0];
                        const int gj = j - lo_dom[1];
                        const int gk_orig = k - lo_dom[2];
                        const int gk_flip = nz_dom - 1 - gk_orig;
                        if (gi < 0 || gi >= nx_dom || gj < 0 || gj >= ny_dom ||
                            gk_flip < 0 || gk_flip >= nz_dom) continue;
                        arr(i, j, k, 0) = (Real)img_flat[(std::size_t)((gk_flip * ny_dom + gj) * nx_dom + gi)];
                    }
                }
                m_phi_image.FillBoundary(m_geom.periodicity());
            }

            m_image_active_this_solve = true;
        } else {
            m_image_active_this_solve = false;
            if (m_phi_image.ok()) m_phi_image.setVal(0.0);
        }
    }

    compute_E_from_phi();

    // Multi-rank: replicate the E field across all ranks so gather_fabs()
    // can serve any particle's stencil regardless of mesh-slab ownership.
    // No-op for single-rank. (Task #54.)
    replicate_e_fields_for_gather();

    // ---- Mesh-field dump (cross-code comparison harness) ----
    // When configured, write the entire SC field state to a binary file
    // at this exact solve count. Format documented in SpaceCharge.H.
    if (m_dump_field_at_step > 0 && m_solve_count == m_dump_field_at_step) {
        auto const* lo_p = m_geom.ProbLo();
        auto const* dx_p = m_geom.CellSize();
        const int nx = m_geom.Domain().length(0) + 1;   // nodal counts
        const int ny = m_geom.Domain().length(1) + 1;
        const int nz = m_geom.Domain().length(2) + 1;
        const Real inv_g = std::sqrt(Real(1.0) - m_last_beta_z * m_last_beta_z);
        const Real gamma_b = (inv_g > Real(0.0)) ? Real(1.0) / inv_g : Real(1.0);

        std::string path = m_dump_field_path;
        if (path.empty()) {
            path = std::string("/tmp/sc_field_dump_step")
                 + std::to_string(m_dump_field_at_step) + ".bin";
        }
        FILE* f = std::fopen(path.c_str(), "wb");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f != nullptr,
            "SpaceCharge::solve: failed to open dump_field path for writing");
        std::int32_t hdr[3] = {nx, ny, nz};
        std::fwrite(hdr,  sizeof(std::int32_t), 3, f);
        double lo3[3] = {(double)lo_p[0], (double)lo_p[1], (double)lo_p[2]};
        double dx3[3] = {(double)dx_p[0], (double)dx_p[1], (double)dx_p[2]};
        std::fwrite(lo3,  sizeof(double), 3, f);
        std::fwrite(dx3,  sizeof(double), 3, f);
        double gb_d = (double)gamma_b;
        std::fwrite(&gb_d, sizeof(double), 1, f);

        auto write_fab = [&](MultiFab const& mf) {
            // Multi-FAB compatible. Iterate all FABs on this rank and copy
            // only their VALID-box cells into the global buffer at the
            // correct domain index. Domain-low (lo_dom = m_geom.Domain().smallEnd())
            // gives the offset to subtract from each FAB index.
            std::vector<double> buf((std::size_t)nx * ny * nz, 0.0);
            const auto lo_dom = m_geom.Domain().smallEnd();
            // Nodal index offset: nodal small end = cell small end (== 0 for
            // a 0-based domain).
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                Array4<const Real> arr = mf.const_array(mfi);
                Box const& vbox = mfi.validbox();   // interior cells of this FAB
                auto const lo_v = vbox.smallEnd();
                auto const hi_v = vbox.bigEnd();
                for (int k = lo_v[2]; k <= hi_v[2]; ++k)
                    for (int j = lo_v[1]; j <= hi_v[1]; ++j)
                        for (int i = lo_v[0]; i <= hi_v[0]; ++i) {
                            const int gi = i - lo_dom[0];
                            const int gj = j - lo_dom[1];
                            const int gk = k - lo_dom[2];
                            if (gi < 0 || gi >= nx || gj < 0 || gj >= ny ||
                                gk < 0 || gk >= nz) {
                                continue;
                            }
                            buf[(std::size_t)((gk * ny + gj) * nx + gi)] =
                                (double)arr(i, j, k, 0);
                        }
            }
            std::fwrite(buf.data(), sizeof(double), buf.size(), f);
        };
        write_fab(m_rho);
        write_fab(m_phi);
        write_fab(m_Ex);
        write_fab(m_Ey);
        write_fab(m_Ez);
        std::fclose(f);
        amrex::Print() << "[SC.dump_field] wrote " << path
                       << " at solve_count=" << m_solve_count
                       << " (n=" << nx << "x" << ny << "x" << nz
                       << ", lo=" << lo_p[0] << "," << lo_p[1] << "," << lo_p[2]
                       << ", dx=" << dx_p[0] << "," << dx_p[1] << "," << dx_p[2]
                       << ", gamma_b=" << gamma_b << ")\n";
    }

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

    // ---- Resize-jump diagnostic: re-gather E_new at the saved positions
    // and compare to E_old. Only report if a resize actually fired this
    // call (otherwise the difference is just per-step evolution, which is
    // the integrator's normal smooth path).
    if (do_diag && geom_changed && !diag_x.empty()) {
        auto const* lo_p_new = m_geom.ProbLo();
        auto const* dx_new   = m_geom.CellSize();
        const Real lo0 = lo_p_new[0], lo1 = lo_p_new[1], lo2 = lo_p_new[2];
        const Real dxi0 = Real(1.0)/dx_new[0], dxi1 = Real(1.0)/dx_new[1], dxi2 = Real(1.0)/dx_new[2];
        const Box mesh_box_new = m_Ex[MFIter(m_Ex)].box();
        Array4<const Real> Ex_new = m_Ex.const_array(MFIter(m_Ex));
        Array4<const Real> Ey_new = m_Ey.const_array(MFIter(m_Ey));
        Array4<const Real> Ez_new = m_Ez.const_array(MFIter(m_Ez));

        long n_in   = 0;
        Real sum_dEx = 0, sum_dEy = 0, sum_dEz = 0;
        Real sum_dEx2 = 0, sum_dEy2 = 0, sum_dEz2 = 0;
        Real max_dEx = 0, max_dEy = 0, max_dEz = 0;
        Real sum_Ex_new2 = 0, sum_Ey_new2 = 0, sum_Ez_new2 = 0;

        const long N = (long)diag_x.size();
        for (long k = 0; k < N; ++k) {
            const Real xp = diag_x[k], yp = diag_y[k], zp = diag_z[k];
            const Real fx = (xp - lo0) * dxi0;
            const Real fy = (yp - lo1) * dxi1;
            const Real fz = (zp - lo2) * dxi2;
            const int ix = int(std::floor(fx));
            const int iy = int(std::floor(fy));
            const int iz = int(std::floor(fz));
            if (ix    < mesh_box_new.smallEnd(0) || ix+1 > mesh_box_new.bigEnd(0) ||
                iy    < mesh_box_new.smallEnd(1) || iy+1 > mesh_box_new.bigEnd(1) ||
                iz    < mesh_box_new.smallEnd(2) || iz+1 > mesh_box_new.bigEnd(2)) {
                continue;   // outside new mesh: no clean comparison
            }
            const Real wx = fx - Real(ix), wy = fy - Real(iy), wz = fz - Real(iz);
            const Real omx = Real(1.0) - wx, omy = Real(1.0) - wy, omz = Real(1.0) - wz;
            auto cic = [&](Array4<const Real> const& F) {
                return omx*omy*omz*F(ix  ,iy  ,iz  ,0)
                     + wx *omy*omz*F(ix+1,iy  ,iz  ,0)
                     + omx*wy *omz*F(ix  ,iy+1,iz  ,0)
                     + wx *wy *omz*F(ix+1,iy+1,iz  ,0)
                     + omx*omy*wz *F(ix  ,iy  ,iz+1,0)
                     + wx *omy*wz *F(ix+1,iy  ,iz+1,0)
                     + omx*wy *wz *F(ix  ,iy+1,iz+1,0)
                     + wx *wy *wz *F(ix+1,iy+1,iz+1,0);
            };
            const Real Ex_n = cic(Ex_new);
            const Real Ey_n = cic(Ey_new);
            const Real Ez_n = cic(Ez_new);
            const Real dEx = Ex_n - diag_Ex_old[k];
            const Real dEy = Ey_n - diag_Ey_old[k];
            const Real dEz = Ez_n - diag_Ez_old[k];
            sum_dEx  += dEx;     sum_dEy  += dEy;     sum_dEz  += dEz;
            sum_dEx2 += dEx*dEx; sum_dEy2 += dEy*dEy; sum_dEz2 += dEz*dEz;
            sum_Ex_new2 += Ex_n*Ex_n; sum_Ey_new2 += Ey_n*Ey_n; sum_Ez_new2 += Ez_n*Ez_n;
            if (std::abs(dEx) > max_dEx) max_dEx = std::abs(dEx);
            if (std::abs(dEy) > max_dEy) max_dEy = std::abs(dEy);
            if (std::abs(dEz) > max_dEz) max_dEz = std::abs(dEz);
            ++n_in;
        }

        if (n_in > 0) {
            const Real inv_N = Real(1.0) / Real(n_in);
            const Real mean_dEx = sum_dEx * inv_N;
            const Real mean_dEy = sum_dEy * inv_N;
            const Real mean_dEz = sum_dEz * inv_N;
            const Real std_dEx  = std::sqrt(std::max(sum_dEx2*inv_N - mean_dEx*mean_dEx, Real(0.0)));
            const Real std_dEy  = std::sqrt(std::max(sum_dEy2*inv_N - mean_dEy*mean_dEy, Real(0.0)));
            const Real std_dEz  = std::sqrt(std::max(sum_dEz2*inv_N - mean_dEz*mean_dEz, Real(0.0)));
            const Real rms_Ex_new = std::sqrt(sum_Ex_new2 * inv_N);
            const Real rms_Ey_new = std::sqrt(sum_Ey_new2 * inv_N);
            const Real rms_Ez_new = std::sqrt(sum_Ez_new2 * inv_N);
            ++m_diag_jump_count;
            // qE*dt -> dp / (m_e*c) (normalized momentum) for electrons
            constexpr Real kQe = Real(1.602176634e-19);
            constexpr Real kMec = Real(9.1093837015e-31) * Real(299792458.0);
            const Real dt = m_diag_dt_hint;
            const Real dpx_norm_std = (dt > 0) ? (kQe * std_dEx * dt / kMec) : Real(0.0);
            const Real dpy_norm_std = (dt > 0) ? (kQe * std_dEy * dt / kMec) : Real(0.0);
            const Real dpz_norm_std = (dt > 0) ? (kQe * std_dEz * dt / kMec) : Real(0.0);
            amrex::Print()
                << "[SC.diag.resize_jump #" << m_diag_jump_count
                << "] N=" << n_in
                << " resize#" << m_n_recenters
                << " dx_new=(" << dx_new[0] << "," << dx_new[1] << "," << dx_new[2] << ")\n"
                << "                       mean(dE)=(" << mean_dEx << "," << mean_dEy << "," << mean_dEz << ") V/m"
                << "  std(dE)=(" << std_dEx << "," << std_dEy << "," << std_dEz << ")\n"
                << "                       rms(E_new)=(" << rms_Ex_new << "," << rms_Ey_new << "," << rms_Ez_new << ") V/m"
                << "  max(|dE|)=(" << max_dEx << "," << max_dEy << "," << max_dEz << ")\n"
                << "                       rel std(dE)/rms(E)=(" << std_dEx/std::max(rms_Ex_new,Real(1e-30))
                << "," << std_dEy/std::max(rms_Ey_new,Real(1e-30))
                << "," << std_dEz/std::max(rms_Ez_new,Real(1e-30)) << ")"
                << "  coherent_frac=(" << std::abs(mean_dEx)/std::max(std_dEx,Real(1e-30))
                << "," << std::abs(mean_dEy)/std::max(std_dEy,Real(1e-30))
                << "," << std::abs(mean_dEz)/std::max(std_dEz,Real(1e-30)) << ")\n";
            if (dt > 0) {
                amrex::Print()
                << "                       implied std(d(p/mc))=(" << dpx_norm_std
                << "," << dpy_norm_std
                << "," << dpz_norm_std << ")  (dt=" << dt << "s)\n";
            }
        }
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
