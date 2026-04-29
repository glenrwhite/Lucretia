#pragma once

#include "particles/TimeBunch.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>


namespace lucretiatt {
namespace spacecharge {

/** Single-rank, single-level FFT-IGF space-charge solver.
 *
 *  Pipeline:
 *    1. Reset rho to 0.
 *    2. CIC-deposit per-particle q * w into nodal rho [C/m^3].
 *    3. Solve Poisson via ABLASTR's IGF (open BC, FFT-based) with the
 *       Lorentz boost handled by scaling the z cell size by 1/sqrt(1-beta_z^2).
 *    4. Compute Ex, Ey, Ez = -grad phi via nodal centered difference.
 *    5. FillBoundary on Ex, Ey, Ez so particles near box edges can gather
 *       across periodic / mirrored neighbours (single-rank: no-op).
 *
 *  Use:
 *    SpaceCharge sc(geom, cell_ba, dm);   // construct once
 *    sc.solve(bunch);                     // call per timestep
 *    auto Ex = sc.ExArr();                // hand to particle-loop kernel
 *    ...
 *    auto E = ablastr::particles::doGatherVectorFieldNodal(
 *        x, y, z, Ex, Ey, Ez, sc.dxi(), sc.lo());
 *
 *  Phase 4A simplifications:
 *    - Single rank, single box (no per-tile MFIter on E gather).
 *    - SC-induced B field neglected (O(beta^2) correction; transverse SC
 *      is reduced by 1/gamma^2 as captured implicitly through the boost
 *      stretch and the v x B term vanishes for stationary particles).
 *    - Beta from bunch mean p_z; per-particle beta corrections deferred. */
class SpaceCharge
{
public:
    SpaceCharge (const amrex::Geometry&            geom,
                 const amrex::BoxArray&            cell_ba,
                 const amrex::DistributionMapping& dm);

    /** Per-step entry point. Computes mean beta_z from bunch, deposits,
     *  solves Poisson, computes E. When co-moving is enabled (see
     *  set_comoving), also recenters the SC mesh on the bunch z-centroid
     *  if the centroid has drifted past the central 50% of the mesh. */
    void solve (particles::TimeBunch& bunch);

    /** Toggle co-moving mesh: when on, solve() shifts the Geometry's
     *  RealBox in z so the mesh follows the bunch. The integer BoxArray
     *  stays fixed (so the IGF Green function table is reusable). */
    void set_comoving (bool on) noexcept { m_comoving = on; }
    bool comoving    () const noexcept { return m_comoving; }

    /** Shift the Geometry's RealBox so the mesh is centered on z_new
     *  (longitudinal only; transverse extent unchanged). BoxArray is
     *  preserved so existing MultiFabs stay valid; rho is reset to 0. */
    void recenter (amrex::Real z_new);

    /** Last beta_z used in the solve. */
    amrex::Real last_beta_z () const noexcept { return m_last_beta_z; }
    /** Number of times the mesh has been recentered since construction. */
    int n_recenters () const noexcept { return m_n_recenters; }

    // ---- Access for the particle-loop kernel ----

    amrex::Array4<const amrex::Real> ExArr () const { return m_Ex[0].const_array(); }
    amrex::Array4<const amrex::Real> EyArr () const { return m_Ey[0].const_array(); }
    amrex::Array4<const amrex::Real> EzArr () const { return m_Ez[0].const_array(); }

    /** Valid index range of the SC field arrays (single nodal box). */
    amrex::Box mesh_box () const { return m_Ex[0].box(); }

    amrex::GpuArray<amrex::Real, 3> dxi () const
    {
        auto const* dx = m_geom.CellSize();
        return { amrex::Real(1.0) / dx[0],
                 amrex::Real(1.0) / dx[1],
                 amrex::Real(1.0) / dx[2] };
    }
    amrex::GpuArray<amrex::Real, 3> lo () const
    {
        auto const* lo = m_geom.ProbLo();
        return { lo[0], lo[1], lo[2] };
    }

private:
    amrex::Real compute_mean_beta_z (particles::TimeBunch& bunch) const;
    amrex::Real compute_mean_z      (particles::TimeBunch& bunch) const;
    void deposit_charge          (particles::TimeBunch& bunch);
    void compute_E_from_phi      ();

    amrex::Geometry            m_geom;
    amrex::BoxArray            m_ba_nodal;
    amrex::DistributionMapping m_dm;

    amrex::MultiFab            m_rho;   // C/m^3,  node-centered
    amrex::MultiFab            m_phi;   // V,      node-centered, 1 ghost
    amrex::MultiFab            m_Ex;    // V/m,    node-centered, 1 ghost
    amrex::MultiFab            m_Ey;    // V/m,    node-centered, 1 ghost
    amrex::MultiFab            m_Ez;    // V/m,    node-centered, 1 ghost

    amrex::Real                m_last_beta_z = 0.0;
    bool                       m_comoving    = false;
    int                        m_n_recenters = 0;
};

} // namespace spacecharge
} // namespace lucretiatt
