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

#include <array>
#include <cmath>
#include <vector>


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
 *  Use (per-tile, MFIter-driven):
 *    SpaceCharge sc(geom, cell_ba, dm);   // construct once
 *    sc.solve(bunch);                     // call per timestep
 *    for (PIter pti(bunch, 0); pti.isValid(); ++pti) {
 *        auto Ex = sc.Ex_array(pti);
 *        ...
 *        auto E = ablastr::particles::doGatherVectorFieldNodal(
 *            x, y, z, Ex, Ey, Ez, sc.dxi(), sc.lo());
 *    }
 *
 *  Phase 6 (multi-box) simplifications still in force:
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

    /** Toggle adaptive mesh extent: when on, solve() resizes the mesh
     *  every step to track the bunch's CURRENT sigmas (not just the
     *  centroid). The new pad in each direction is
     *      pad_axis = max(pad_factor * sigma_axis, min_pad_axis)
     *  and the integer BoxArray (ncell) is preserved so the IGF FFT
     *  plan stays valid (only cell sizes change, which the IGF Green
     *  function recomputes per call from cell_size).
     *
     *  The self-force LUT is rebuilt when the cell aspect ratio shifts
     *  by more than 5% — the LUT depends on the cell shape and would
     *  otherwise drift out of calibration as the bunch evolves.
     *
     *  Default: off. Use only when the 3D mesh SC is needed AND the
     *  bunch evolves significantly in size (e.g. emittance growth in
     *  drifts, focus through a quad). For a photoinjector the slice
     *  SC alone (mesh off) gives better results — see SpaceChargeSlice. */
    void set_adaptive (bool on,
                       amrex::Real pad_factor   = amrex::Real(5.0),
                       amrex::Real min_pad_xy   = amrex::Real(1.0e-3),
                       amrex::Real min_pad_z    = amrex::Real(1.0e-3)) noexcept
    {
        m_adaptive    = on;
        m_pad_factor  = pad_factor;
        m_min_pad_xy  = min_pad_xy;
        m_min_pad_z   = min_pad_z;
    }
    bool adaptive    () const noexcept { return m_adaptive; }

    /** Hybrid mesh mode: STATIC transverse (xy frozen at the constructor's
     *  initial RealBox xy extent) + ADAPTIVE longitudinal (z resizes per
     *  step to track the bunch's CURRENT sigma_z).
     *
     *  Motivation: the IGF accuracy degrades when cell aspect ratio in the
     *  z-stretched solver coords gets extreme (1:1:gamma at large gamma).
     *  Pure adaptive lets sigma_z->small drive cells->small in z, and the
     *  IGF then produces noticeable transverse SC error from the high
     *  cell aspect (verified to ~10% under at gamma=12 in test_relativistic_pencil).
     *  Pure static keeps cells coarse but loses longitudinal resolution
     *  for compressed bunches. Hybrid: keep xy resolution (where the IGF
     *  Green function is best behaved) and let z adapt.
     *
     *  When on, supersedes set_adaptive (which adapts all three axes) and
     *  set_comoving (z-shift only). Use set_adaptive xor set_hybrid_z_adaptive
     *  xor set_exact_range; setting more than one yields the highest-priority
     *  mode (exact_range > hybrid_z_adaptive > adaptive > comoving).
     *
     *  Default off. */
    void set_hybrid_z_adaptive (bool on,
                                amrex::Real pad_factor = amrex::Real(5.0),
                                amrex::Real min_pad_z  = amrex::Real(1.0e-3)) noexcept
    {
        m_hybrid_z   = on;
        m_pad_factor = pad_factor;
        m_min_pad_z  = min_pad_z;
    }
    bool hybrid_z_adaptive () const noexcept { return m_hybrid_z; }

    /** Use ImpactT-style "exact bunch range" adaptive mesh: the SC mesh
     *  is shrunk every step to EXACTLY the alive-particle range in each
     *  axis, with NO padding. Mirrors AccSimulator.f90:1724-1733
     *  (zadjmax = 0.0d0; update_CompDom(grange, ...)).
     *
     *  Differences from set_adaptive:
     *   - No pad_factor; mesh extent = (max - min) per axis from particles.
     *   - No hysteresis: resize is computed every step (the mesh tracks
     *     the bunch instantaneously).
     *   - No min_pad floor.
     *
     *  Supersedes set_adaptive when on. Recommended for cross-code
     *  calibration with ImpactT — keeps cell density per cell
     *  consistent between codes. */
    void set_exact_range (bool on) noexcept { m_exact_range = on; }
    bool exact_range () const noexcept { return m_exact_range; }

    bool compute_bunch_range (
        particles::TimeBunch& bunch,
        amrex::Real& xmin, amrex::Real& xmax,
        amrex::Real& ymin, amrex::Real& ymax,
        amrex::Real& zmin, amrex::Real& zmax) const;

    /** Set the number of binomial-smoother passes applied to rho after
     *  deposit and before the IGF Poisson solve. Each pass convolves rho
     *  with the 3x3x3 separable (1,2,1)/4 kernel (volume-preserving;
     *  zero phase shift; isotropic Gaussian-equivalent sigma ~ 0.7*dx
     *  per pass). Useful when the per-cell particle count is low and
     *  CIC deposit is noisy.
     *
     *  Default: 0 (no smoothing). 1 pass cuts per-cell shot noise by
     *  ~2x; 2 passes ~4x; each pass costs one rho copy + one stencil
     *  pass over the mesh (cheap relative to the IGF FFT). */
    void set_rho_smooth_passes (int n) noexcept { m_rho_smooth_passes = std::max(0, n); }
    int  rho_smooth_passes () const noexcept { return m_rho_smooth_passes; }

    /** Particle deposit/gather shape order. 1 = CIC (linear, 8-node cloud,
     *  current default). 2 = TSC (quadratic, 27-node cloud).
     *
     *  TSC has wider deposit footprint, so per-particle field gradients
     *  seen on gather are smoother. Standard photoinjector PIC codes use
     *  TSC by default for emittance studies; CIC is common in low-noise
     *  contexts where per-particle counts are high.
     *
     *  Cost: TSC deposit + gather ~27/8 = 3.4x CIC (per-particle work).
     *  IGF Poisson solve unchanged. Self-force LUT is rebuilt for the
     *  active shape order.
     *
     *  Default: 1 (CIC). */
    void set_shape_order (int order)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(order == 1 || order == 2,
            "SpaceCharge::set_shape_order: only 1 (CIC) or 2 (TSC) supported");
        if (order != m_shape_order) {
            m_shape_order = order;
            build_self_force_lut();
            ++m_n_lut_rebuilds;
        }
    }
    int shape_order () const noexcept { return m_shape_order; }

    /** Enable cathode image-charge handling.
     *
     *  When enabled, deposit_charge also deposits a MIRROR rho with
     *  flipped sign at the mirror positions about the cathode plane:
     *  for each particle at z > z_cath, an image deposit goes to
     *  (x, y, 2*z_cath - z) with -q*w. The IGF Poisson solve over the
     *  combined direct+image distribution gives a phi that satisfies
     *  the conducting-cathode boundary condition (phi = 0 on the
     *  cathode plane). This matches IMPACT-T's "shifted Green function"
     *  approach in the limit where the image positions fit within the
     *  current SC mesh.
     *
     *  Implementation note (vs IMPACT-T): IMPACT-T does a SECOND FFT
     *  with a Green function shifted by 2*z_centroid, so the image
     *  positions are always "in frame" regardless of where the mesh
     *  is in lab z. Our simpler mirror-deposit-into-same-mesh approach
     *  silently drops image deposits whose mirror positions fall
     *  outside the mesh — fine for the early-emission near-cathode
     *  regime (where the comoving mesh straddles the cathode), and
     *  the image effect is anyway negligible (~1/γ²) once the bunch
     *  is relativistic and the mesh has moved away.
     *
     *  Past z = `image_cutoff` the image-charge term is dropped
     *  entirely (matches IMPACT-T's `Zimage` parameter). This is set
     *  per-particle: only particles with z - z_cath < image_cutoff
     *  contribute to the image deposit. */
    void set_image_plane (amrex::Real z_cath,
                          amrex::Real image_cutoff = amrex::Real(0.05)) noexcept
    {
        m_image_plane_enabled = true;
        m_image_plane_z       = z_cath;
        m_image_cutoff        = image_cutoff;
    }
    void unset_image_plane () noexcept { m_image_plane_enabled = false; }
    bool image_plane_enabled () const noexcept { return m_image_plane_enabled; }

    /** Shift the Geometry's RealBox so the mesh is centered on z_new
     *  (longitudinal only; transverse extent unchanged). BoxArray is
     *  preserved so existing MultiFabs stay valid; rho is reset to 0. */
    void recenter (amrex::Real z_new);

    /** Resize the Geometry's RealBox to a new (centroid, half-extent).
     *  BoxArray + ncell are preserved (so cell sizes scale with the
     *  new extent). Self-force LUT is rebuilt if the cell aspect ratio
     *  changed by more than 5%; rho/phi/E are reset. */
    void resize (amrex::Real x_centroid, amrex::Real y_centroid,
                 amrex::Real z_centroid,
                 amrex::Real half_x,     amrex::Real half_y,
                 amrex::Real half_z);

    /** Last beta_z used in the solve. */
    amrex::Real last_beta_z () const noexcept { return m_last_beta_z; }
    /** Last bunch-mean gamma used to stretch the z mesh in the solve. */
    amrex::Real last_gamma  () const noexcept
    { const amrex::Real b = m_last_beta_z;
      const amrex::Real ig = std::sqrt(amrex::Real(1.0) - b*b);
      return (ig > amrex::Real(0.0)) ? amrex::Real(1.0)/ig : amrex::Real(1.0); }
    /** Number of times the mesh has been recentered (or resized) since construction. */
    int n_recenters () const noexcept { return m_n_recenters; }
    /** Number of times the self-force LUT has been rebuilt since construction. */
    int n_lut_rebuilds () const noexcept { return m_n_lut_rebuilds; }

    // ---- Access for the particle-loop kernel ----
    //
    // Single-FAB layout: SC's m_rho/m_phi/m_Ex/m_Ey/m_Ez each live on one
    // BoxArray box covering the full nodal domain (forced in the constructor
    // independent of the cell_ba subdivision the caller may have built for
    // the particle container). The returned Array4 is the only FAB's data.
    // No tile lookup -- particle iteration in TrackingLoop is independent
    // of SC mesh tiling.

    amrex::Array4<const amrex::Real> Ex_array () const
    {
        amrex::MFIter mfi(m_Ex);
        return m_Ex.const_array(mfi);
    }
    amrex::Array4<const amrex::Real> Ey_array () const
    {
        amrex::MFIter mfi(m_Ey);
        return m_Ey.const_array(mfi);
    }
    amrex::Array4<const amrex::Real> Ez_array () const
    {
        amrex::MFIter mfi(m_Ez);
        return m_Ez.const_array(mfi);
    }

    /** Valid index range of the SC field arrays. Includes ghost cells so
     *  CIC gather handles particles right at the box edge. */
    amrex::Box mesh_box () const
    {
        amrex::MFIter mfi(m_Ex);
        return m_Ex[mfi].box();
    }

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

    // ---- Self-force lookup table ----
    //
    // The self-force-subtraction step (see TrackingLoop) computes the
    // Coulomb field at a particle's position from its own deposited CIC
    // cloud (8 weighted point charges at the surrounding nodes). The
    // expensive bit is 8 sqrt + 8 division per particle. The shape of
    // this field depends only on the particle's fractional position in
    // its cell (wx_frac, wy_frac, wz_frac) and on the cell aspect ratio
    // (which is constant for the simulation). The dependence on charge
    // and weight is a single multiplicative factor.
    //
    // We precompute a (N+1)^3 table per E component at construction,
    // then look up + trilinear-interpolate per particle. ~20 cycles of
    // arithmetic + 8 cached reads, vs ~150 cycles for the original 8
    // sqrt+div sum. The table is ~860 KiB at N=32, easily L2-resident.
    //
    // self_force_unit(c, wx, wy, wz) returns E_c per unit (q*w*kCoulomb).
    // The consumer multiplies by qw_kc to get the field, then by the
    // empirical kSelfForceFactor to get the actual subtraction term.
    //
    // N=64 (table size 65^3 * 3 * 8 bytes = 6.6 MiB, fits in L3) gives
    // ~0.025% trilinear-interp error on the self-force shape, which
    // matches the per-particle exact calc to within the empirical fudge
    // factor calibration noise (test_uniform_sphere ~3e-3 rel err).
    static constexpr int kSelfForceLutN = 64;
    int                     lut_n         () const noexcept { return kSelfForceLutN; }
    amrex::Real const*      lut_ptr       (int comp) const noexcept
    { return m_self_force_lut[comp].data(); }

private:
    amrex::Real compute_mean_beta_z (particles::TimeBunch& bunch) const;
    amrex::Real compute_mean_z      (particles::TimeBunch& bunch) const;
    void deposit_charge          (particles::TimeBunch& bunch);
    void smooth_rho              ();
    void compute_E_from_phi      ();
    void build_self_force_lut    ();

    /** ABLASTR-style IGF Poisson solve with a z-SHIFTED Green function.
     *
     *  Solves phi = G_shifted ⊛ rho  where the Green function used in
     *  the convolution has its z-coordinate offset by `z_shift`. For
     *  cathode image charges (Qiang/IMPACT-T method), z_shift =
     *  -2 * (z_centroid - z_cath) * gamma — places the bunch's image
     *  in the FFT domain so the convolution gives the image-charge
     *  contribution to phi at observation points within the mesh.
     *
     *  Mirrors ablastr::fields::computePhiIGF; the only difference is
     *  the `+ z_shift` in the z-coordinate passed to
     *  SumOfIntegratedPotential3D. Maintains its own cached
     *  OpenBCSolver because the Domain may differ from the direct one. */
    void compute_phi_IGF_shifted (
        amrex::MultiFab const&            rho,
        amrex::MultiFab&                  phi,
        std::array<amrex::Real, 3> const& cell_size,
        amrex::Real                       z_shift) const;

    /** Compute centroid and central-moment sigmas of the bunch in one
     *  pass. Used by the adaptive resizer. Returns false if the bunch
     *  is empty. */
    bool compute_bunch_stats (particles::TimeBunch& bunch,
                              amrex::Real& x_c,     amrex::Real& y_c,    amrex::Real& z_c,
                              amrex::Real& sigma_x, amrex::Real& sigma_y, amrex::Real& sigma_z) const;

    amrex::Geometry            m_geom;
    amrex::BoxArray            m_ba_nodal;
    amrex::DistributionMapping m_dm;

    amrex::MultiFab            m_rho;        // C/m^3,  node-centered
    amrex::MultiFab            m_phi;        // V,      node-centered, 1 ghost
    amrex::MultiFab            m_phi_image;  // V (image-charge contribution; allocated on demand)
    amrex::MultiFab            m_Ex;         // V/m,    node-centered, 1 ghost
    amrex::MultiFab            m_Ey;         // V/m,    node-centered, 1 ghost
    amrex::MultiFab            m_Ez;         // V/m,    node-centered, 1 ghost

    amrex::Real                m_last_beta_z    = 0.0;
    bool                       m_exact_range    = false;
    bool                       m_hybrid_z       = false;
    bool                       m_comoving       = false;
    bool                       m_adaptive       = false;
    amrex::Real                m_pad_factor     = 5.0;
    amrex::Real                m_min_pad_xy     = 1.0e-3;
    amrex::Real                m_min_pad_z      = 1.0e-3;
    bool                       m_image_plane_enabled = false;
    amrex::Real                m_image_plane_z       = 0.0;
    amrex::Real                m_image_cutoff        = 0.05;     // m
    int                        m_rho_smooth_passes = 0;
    int                        m_shape_order    = 1;     // 1 = CIC, 2 = TSC
    int                        m_n_recenters    = 0;
    int                        m_n_lut_rebuilds = 1;     // counts the constructor build

    // Per-component self-force shape table, size (N+1)^3 each, indexed
    // [ix*(N+1)^2 + iy*(N+1) + iz]. Built once in ctor; cell aspect
    // ratio is fixed for the lifetime of this SpaceCharge.
    std::array<std::vector<amrex::Real>, 3> m_self_force_lut;
};

} // namespace spacecharge
} // namespace lucretiatt
