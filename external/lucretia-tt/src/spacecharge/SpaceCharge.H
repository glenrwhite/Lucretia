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
#include <string>
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

    /** Resize hysteresis factor for adaptive mode. Resize triggers when
     *  the new pad is > hyst * current OR < (1/hyst) * current. Larger
     *  values mean fewer resizes (less per-step field-discontinuity noise
     *  in the gun region) but coarser tracking of bunch evolution.
     *  Default 2.0; 3.0-4.0 are reasonable for emittance-sensitive runs. */
    void set_resize_hyst (amrex::Real h) noexcept
    {
        m_resize_hyst = std::max(amrex::Real(1.1), h);
    }
    amrex::Real resize_hyst () const noexcept { return m_resize_hyst; }

    /** Centroid drift threshold for adaptive resize: triggers a resize
     *  when |new_centroid - old_centroid| > frac * current_half_extent
     *  along ANY axis. Default 0.5. Larger values defer resize events,
     *  letting the bunch drift further inside the existing mesh before
     *  forcing a re-solve. With a relativistic bunch traveling c*dt
     *  per step (~1.2 mm/step at dt=4 ps) this knob is the dominant
     *  control on resize FREQUENCY — at the default 0.5 with a 7-mm
     *  z half-extent, the threshold is hit every ~3 steps and produces
     *  thousands of field-reset events per run. Raising to 0.9-0.95 lets
     *  the bunch use most of the box before a resize fires.
     *  Range [0.1, 0.99]; default 0.5. */
    void set_cent_drift_threshold (amrex::Real frac) noexcept
    {
        m_cent_drift_threshold = std::min(std::max(frac, amrex::Real(0.1)), amrex::Real(0.99));
    }
    amrex::Real cent_drift_threshold () const noexcept { return m_cent_drift_threshold; }

    /** Integer-cell snap for centroid-only resizes. When on AND only the
     *  centroid drift trigger fires (not size_drift), the new box center
     *  is snapped to (old_center + N * dx) with N an integer per axis,
     *  so particles' fractional positions within their cells are
     *  preserved across the resize. With identical sub-cell deposit
     *  pattern, the IGF gives an identical per-cell phi field and the
     *  CIC gather at any fixed physical particle position returns the
     *  exact same E -- zero discretization noise from the resize. The
     *  Geometry RealBox is updated and m_phi/m_E setVal(0) still happens
     *  but is harmless because the IGF FFT overwrites phi from rho
     *  on the next solve.
     *  Default off (preserves prior behaviour). */
    void set_integer_cell_shift (bool on) noexcept { m_integer_cell_shift = on; }
    bool integer_cell_shift () const noexcept { return m_integer_cell_shift; }

    /** Diagnostic mode: when on, every solve() that triggers a mesh resize
     *  measures the per-particle field discontinuity. Before the resize,
     *  E_old is gathered at all alive particle positions. After the new
     *  deposit + solve, E_new is gathered at the SAME positions on the
     *  new mesh. Mean / std / max of (E_new - E_old) per axis are printed,
     *  along with the implied per-step momentum kick (q * dE * dt) using
     *  the most recent dt seen by the bunch. The diagnostic is O(N_alive)
     *  per resize event and is gated behind `space_charge.diag_resize_jump
     *  = 1` because of the extra gather pass.
     *  Default off. */
    void set_diag_resize_jump (bool on) noexcept { m_diag_resize_jump = on; }
    bool diag_resize_jump     () const noexcept { return m_diag_resize_jump; }

    /** Tell SpaceCharge what dt the calling integrator is using; used only
     *  by the resize-jump diagnostic to convert dE -> implied momentum
     *  kick. No effect on the physics. */
    void set_diag_dt_hint (amrex::Real dt) noexcept { m_diag_dt_hint = dt; }

    /** Dump mesh SC fields (rho, phi, Ex, Ey, Ez) to a binary file at a
     *  specific solve count. Set step > 0 to enable; default 0 = off.
     *  When the m_solve_count-th call to solve() runs, after the field
     *  computation completes, the mesh data + geometry are written to
     *  /tmp/sc_field_dump_step<N>.bin. Format:
     *    int32 nx, ny, nz
     *    double lo[3]      (RealBox lo, lab coords)
     *    double dx[3]      (cell sizes, lab coords)
     *    double gamma_b    (bunch-mean gamma used in solver)
     *    double rho [nx*ny*nz]   row-major (x fastest)
     *    double phi [nx*ny*nz]
     *    double Ex  [nx*ny*nz]
     *    double Ey  [nx*ny*nz]
     *    double Ez  [nx*ny*nz]
     *  Used for cross-code SC field comparison. */
    void set_dump_field_at_step (int step, std::string path = std::string()) noexcept
    {
        m_dump_field_at_step = step;
        m_dump_field_path    = path;
    }
    int dump_field_at_step () const noexcept { return m_dump_field_at_step; }

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

    /** Optional outlier kill in `exact_range` mode. When > 0, particles
     *  whose offset from the bunch centroid exceeds N*sigma in ANY axis
     *  are marked alive=0 (subsequent forces all skip them). This keeps
     *  the SC mesh tight around the bulk without leaving outlier particles
     *  half-gathered at the mesh boundary. Functionally a soft particle
     *  aperture computed each solve from the live bunch sigma. Once a
     *  particle is killed it stays killed (alive=0 is persistent).
     *  Recommended values in the 4-6 range; lower values clip more bulk
     *  and may impact downstream physics. Default 0 = no kill. */
    void set_outlier_kill_sigma (amrex::Real cap) noexcept
    {
        m_outlier_kill_sigma = std::max(amrex::Real(0.0), cap);
    }
    amrex::Real outlier_kill_sigma () const noexcept { return m_outlier_kill_sigma; }
    /** Absolute minimum on the per-axis kill cap. Effective cap is
     *  max(N*sigma, floor) — protects early-emission bunches with near-
     *  zero sigma in some axis from cascade kill. Defaults: 5 mm xy,
     *  10 mm z. */
    void set_outlier_kill_floor (amrex::Real floor_xy, amrex::Real floor_z) noexcept
    {
        m_outlier_kill_floor_xy = std::max(amrex::Real(0.0), floor_xy);
        m_outlier_kill_floor_z  = std::max(amrex::Real(0.0), floor_z);
    }
    void set_outlier_kill_verbose (bool on) noexcept { m_outlier_kill_verbose = on; }
    long n_outliers_killed () const noexcept { return m_n_outliers_killed; }

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

    /** Relative tolerance for the IGF Green-function cache (compute_phi_IGF_cached
     *  and compute_phi_IGF_shifted). 0 = exact match required (rebuild on every
     *  bit-changed cell_size or z_shift). >0 = skip setGreensFunction when each
     *  of cell_size[xyz] and z_shift has changed by less than this fraction
     *  relative to the last rebuilt value.
     *
     *  The Green function depends smoothly on cell_size and z_shift; a 1%
     *  tolerance introduces ~1% per-cell phi error (smaller than ABLASTR's
     *  intrinsic ~1-4% per-cell IGF noise) but skips the dominant per-step
     *  cost (the per-cell antiderivative evaluator). On lt's gamma-stretched
     *  z mesh this typically saves ~20-25% of total runtime. */
    void set_green_cache_tol (amrex::Real tol) noexcept
    {
        m_green_cache_tol = std::max(amrex::Real(0.0), tol);
    }
    amrex::Real green_cache_tol () const noexcept { return m_green_cache_tol; }

    /** Report Green-function cache hit/miss counts (direct + image solves)
     *  to amrex::Print(). Call once at end of run after all SC solves are
     *  done, e.g. alongside SimpleProfiler::print(). Only meaningful when
     *  green_cache_tol > 0 (otherwise misses == solve count). */
    void print_green_cache_stats () const;

    /** Restrict SC to particles with z >= z_min. When `on` is true, the
     *  charge deposit, the centroid/sigma, the bunch-range (for adaptive),
     *  and the bunch-mean beta_z all skip particles with z < z_min. Useful
     *  during cathode emission when behind-cathode-drifting particles
     *  shouldn't extend the SC mesh below the cathode plane (matches
     *  ImpactT's `flagpos=1` mode; see AccSimulator.f90:1469-1473 +
     *  Depositor.f90:273).
     *
     *  When off (default), all alive particles contribute regardless of z. */
    void set_z_filter_min (bool on, amrex::Real z_min = amrex::Real(0.0)) noexcept
    {
        m_z_filter_min_active = on;
        m_z_filter_min        = z_min;
    }
    bool        z_filter_min_active () const noexcept { return m_z_filter_min_active; }
    amrex::Real z_filter_min        () const noexcept { return m_z_filter_min; }

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

    /** Task #22 deposit-γ fix: when true, divide deposited rho by γ_bunch
     *  in deposit_charge() to match imp's chgdens_Depositor convention
     *  (Depositor.f90:80). lt's E-field magnitude was γ× imp's at high γ
     *  because lt deposited rho_lab while imp deposits rho_rest = rho_lab/γ.
     *  Pair with sc_transverse_gamma_boost=1 (the textbook external γ boost)
     *  to get the correct lab-frame E. */
    void set_deposit_rest_frame_rho (bool on) noexcept { m_deposit_rest_frame_rho = on; }
    bool deposit_rest_frame_rho () const noexcept { return m_deposit_rest_frame_rho; }

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
    // Multi-FAB layout: SC's m_rho/m_phi/m_Ex/m_Ey/m_Ez can have any number
    // of local FABs (when the caller's BoxArray has multiple boxes). The
    // gather-side accessor (gather_fabs()) returns a struct with parallel
    // vectors of (Ex, Ey, Ez) Array4s and their full grown boxes (valid + 1
    // ghost). The TrackingLoop iterates particles, finds the FAB whose grown
    // box contains the particle's cell, and gathers from that FAB's Array4.
    //
    // Backwards-compatibility helpers: Ex_array() etc. return the FIRST FAB
    // (only valid when m_ba_nodal has one box, the historical single-FAB
    // case). New code should use gather_fabs() and per-particle FAB lookup.

    struct GatherFABs {
        std::vector<amrex::Array4<const amrex::Real>> Ex;
        std::vector<amrex::Array4<const amrex::Real>> Ey;
        std::vector<amrex::Array4<const amrex::Real>> Ez;
        std::vector<amrex::Box> boxes;        // valid + 1 ghost (gather bounds)
        std::vector<amrex::Box> valid_boxes;  // interior only (FAB ownership)
        // Image E (only populated if image_active is true, task #61/#63).
        std::vector<amrex::Array4<const amrex::Real>> Ex_image;
        std::vector<amrex::Array4<const amrex::Real>> Ey_image;
        std::vector<amrex::Array4<const amrex::Real>> Ez_image;
        bool image_active = false;
    };

    /** Return all local FABs' Array4 + bounds for E_x, E_y, E_z. Used by
     *  the per-particle gather in TrackingLoop. With multiple FABs (and/or
     *  multiple ranks) the caller iterates particles and looks up the right
     *  FAB by checking which one's grown box contains the particle's cell. */
    GatherFABs gather_fabs () const
    {
        GatherFABs g;
        // Multi-rank: use the replica vectors populated by
        // replicate_e_fields_for_gather() so each rank sees ALL FABs from
        // every rank. Single-rank: fall back to local MFIter (no replicas).
        if (!m_Ex_replica.empty()) {
            for (std::size_t i = 0; i < m_Ex_replica.size(); ++i) {
                g.Ex.push_back(m_Ex_replica[i].const_array());
                g.Ey.push_back(m_Ey_replica[i].const_array());
                g.Ez.push_back(m_Ez_replica[i].const_array());
                g.boxes.push_back(m_replica_boxes[i]);
                g.valid_boxes.push_back(m_replica_valid_boxes[i]);
            }
        } else {
            for (amrex::MFIter mfi(m_Ex); mfi.isValid(); ++mfi) {
                g.Ex.push_back(m_Ex.const_array(mfi));
                g.Ey.push_back(m_Ey.const_array(mfi));
                g.Ez.push_back(m_Ez.const_array(mfi));
                g.boxes.push_back(m_Ex[mfi].box());        // valid + ghost
                g.valid_boxes.push_back(mfi.validbox());   // interior only
            }
        }
        g.image_active = m_image_active_this_solve;
        if (g.image_active) {
            if (!m_Ex_image_replica.empty()) {
                for (std::size_t i = 0; i < m_Ex_image_replica.size(); ++i) {
                    g.Ex_image.push_back(m_Ex_image_replica[i].const_array());
                    g.Ey_image.push_back(m_Ey_image_replica[i].const_array());
                    g.Ez_image.push_back(m_Ez_image_replica[i].const_array());
                }
            } else if (m_Ex_image.ok()) {
                for (amrex::MFIter mfi(m_Ex_image); mfi.isValid(); ++mfi) {
                    g.Ex_image.push_back(m_Ex_image.const_array(mfi));
                    g.Ey_image.push_back(m_Ey_image.const_array(mfi));
                    g.Ez_image.push_back(m_Ez_image.const_array(mfi));
                }
            }
        }
        return g;
    }

    /** Multi-rank field replication: after a solve, AllGather all FABs of
     *  m_Ex/m_Ey/m_Ez (and image versions) from every rank into per-rank
     *  m_Ex_replica vectors so gather_fabs() can serve any particle's
     *  stencil regardless of mesh-slab ownership. No-op for nprocs==1.
     *  Called from solve(). (Task #54 field-replication path.) */
    void replicate_e_fields_for_gather ();

    // Backwards-compat: single-FAB accessors (return the first local FAB).
    // Only valid when m_ba_nodal has exactly one box on this rank.
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
     *  CIC gather handles particles right at the box edge. Single-FAB only;
     *  use gather_fabs() for multi-FAB cases. */
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

    /** Cached drop-in replacement for ablastr::fields::computePhiIGF
     *  that skips OpenBCSolver::setGreensFunction when (domain, cell_size)
     *  are unchanged across calls. The Green function depends only on
     *  those, but the ablastr wrapper rebuilds it on every solve, which
     *  dominates ~30% of lt's SC time. Caching gives a ~30-40% SC speedup
     *  for runs where the mesh changes infrequently (static mesh,
     *  hysteretic adaptive). For sc_exact_range (resize every step),
     *  cell_size changes every call and caching is a no-op. */
    void compute_phi_IGF_cached (
        amrex::MultiFab const&            rho,
        amrex::MultiFab&                  phi,
        std::array<amrex::Real, 3> const& cell_size) const;

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
    amrex::MultiFab            m_phi;        // V,      node-centered, 1 ghost (DIRECT phi)
    amrex::MultiFab            m_phi_image;  // V (image-charge phi; allocated on demand, kept SEPARATE from m_phi for task #61/#63)
    amrex::MultiFab            m_Ex;         // V/m,    node-centered, 1 ghost (DIRECT E)
    amrex::MultiFab            m_Ey;         // V/m,    node-centered, 1 ghost
    amrex::MultiFab            m_Ez;         // V/m,    node-centered, 1 ghost
    // Image E kept separate. Applied with opposite B-field sign in
    // TrackingLoop (mirrors imp's gradEB_FieldQuant called twice with
    // opposite betC sign — Field.f90:445-449). Task #61/#63.
    amrex::MultiFab            m_Ex_image;
    amrex::MultiFab            m_Ey_image;
    amrex::MultiFab            m_Ez_image;
    bool                       m_image_active_this_solve = false;

    // Multi-rank field replicas: after each SC solve, all ranks AllGather
    // the E-field FABs into these per-rank vectors so that gather_fabs()
    // can serve any particle's stencil regardless of which rank originally
    // owned the FAB. Cost: ~1.6 MB total at 32^3, negligible. Doesn't
    // scale to huge meshes but adequate for the LCLS injector regime.
    // Empty when nprocs == 1 (gather_fabs falls back to local MFIter).
    // (Task #54 — field replication path.)
    std::vector<amrex::FArrayBox> m_Ex_replica;
    std::vector<amrex::FArrayBox> m_Ey_replica;
    std::vector<amrex::FArrayBox> m_Ez_replica;
    std::vector<amrex::Box>       m_replica_boxes;        // includes ghosts
    std::vector<amrex::Box>       m_replica_valid_boxes;  // interior only
    std::vector<amrex::FArrayBox> m_Ex_image_replica;
    std::vector<amrex::FArrayBox> m_Ey_image_replica;
    std::vector<amrex::FArrayBox> m_Ez_image_replica;

    amrex::Real                m_last_beta_z    = 0.0;
    bool                       m_exact_range    = false;
    amrex::Real                m_outlier_kill_sigma = 0.0;  // 0 = no kill; >0 = mark alive=0 if |x-x_c| > N*sigma in any axis
    amrex::Real                m_outlier_kill_floor_xy = 5.0e-3;  // m: min absolute cap (avoids cascade kill at near-zero sigma)
    amrex::Real                m_outlier_kill_floor_z  = 1.0e-2;  // m
    bool                       m_outlier_kill_verbose = false;
    long                       m_n_outliers_killed = 0;
    bool                       m_hybrid_z       = false;
    bool                       m_comoving       = false;
    bool                       m_adaptive       = false;
    amrex::Real                m_pad_factor     = 5.0;
    amrex::Real                m_min_pad_xy     = 1.0e-3;
    amrex::Real                m_min_pad_z      = 1.0e-3;
    amrex::Real                m_resize_hyst    = 2.0;     // adaptive resize hysteresis factor
    amrex::Real                m_cent_drift_threshold = 0.5; // adaptive: resize when centroid drifts > frac * half-extent
    bool                       m_integer_cell_shift = false; // adaptive: snap centroid-only resizes to integer cells
    bool                       m_diag_resize_jump = false; // diagnostic: per-particle dE at resize
    amrex::Real                m_diag_dt_hint     = 0.0;   // dt for dp = qE*dt readout (set by caller)
    int                        m_diag_jump_count  = 0;     // # resize-jump events reported so far
    int                        m_dump_field_at_step = 0;   // diagnostic: dump mesh SC fields at this solve count
    std::string                m_dump_field_path  = "";   // path; auto-generated if empty
    int                        m_solve_count      = 0;    // counts solve() calls
    bool                       m_image_plane_enabled = false;
    bool                       m_deposit_rest_frame_rho = false;   // task #22
    amrex::Real                m_image_plane_z       = 0.0;
    amrex::Real                m_image_cutoff        = 0.05;     // m
    int                        m_rho_smooth_passes = 0;
    amrex::Real                m_green_cache_tol   = 0.0;  // 0 = exact match; >0 = relative tolerance for IGF Green-fn cache

    // Cache hit/miss counters for diagnostic. mutable so that const cached
    // solvers can increment them. Reported via print_green_cache_stats().
    mutable long               m_igf_direct_hits   = 0;
    mutable long               m_igf_direct_misses = 0;
    mutable long               m_igf_image_hits    = 0;
    mutable long               m_igf_image_misses  = 0;
    int                        m_shape_order    = 1;     // 1 = CIC, 2 = TSC
    bool                       m_z_filter_min_active = false;
    amrex::Real                m_z_filter_min        = 0.0;
    int                        m_n_recenters    = 0;
    int                        m_n_lut_rebuilds = 1;     // counts the constructor build

    // Per-component self-force shape table, size (N+1)^3 each, indexed
    // [ix*(N+1)^2 + iy*(N+1) + iz]. Built once in ctor; cell aspect
    // ratio is fixed for the lifetime of this SpaceCharge.
    std::array<std::vector<amrex::Real>, 3> m_self_force_lut;
};

} // namespace spacecharge
} // namespace lucretiatt
