#pragma once

#include "particles/TimeBunch.H"

#include <AMReX_REAL.H>

#include <vector>


namespace lucretiatt {
namespace spacecharge {

/** Mesh-free 1D longitudinal space-charge model.
 *
 *  Bins the bunch into N z-slices and computes the on-axis longitudinal
 *  E_z at each slice centre via the standard "stack of charged disks"
 *  formula for a uniform-radius round beam (a = factor * sigma_xy):
 *
 *    E_z(z_i) = (1 / (2 pi eps0 a^2)) *
 *               Σ_j Q_j * sign(z_i - z_j) * (1 - g|z_i-z_j| / sqrt(g²(z_i-z_j)² + a²))
 *
 *  with g = gamma (Lorentz factor from mean beta_z) entering only
 *  through the z-distances — the formula is derived by Lorentz-boosting
 *  the rest-frame disk-stack to the lab and using that E_z is invariant
 *  under longitudinal boosts. This matches the standard 1D longitudinal
 *  SC kernel used in Astra, Parmela, etc.
 *
 *  WHY this exists alongside the 3D mesh SC:
 *  The 3D IGF mesh SC gets the transverse SC right but loses
 *  longitudinal resolution once the bunch compresses below one
 *  cell_z (typical for a photoinjector L0A run: bunch shrinks to
 *  ~0.1 mm but cell_z stays at ~1 mm). The mesh then computes E_z
 *  averaged over a too-large region, missing the strong z-gradient
 *  inside the bunch. The slice model is mesh-free in z, so it stays
 *  resolved no matter how short the bunch gets.
 *
 *  Use:
 *    - Construct once: SpaceChargeSlice slice_sc(n_slices = 256);
 *    - Each step: slice_sc.compute(bunch);
 *    - Kernel gathers E_z per particle via E_z_at(z_p).
 */
class SpaceChargeSlice
{
public:
    explicit SpaceChargeSlice (int n_slices = 256,
                               amrex::Real radius_factor = amrex::Real(2.0));

    /** Re-bin charge into z-slices and recompute E_z(z_slice).
     *
     *  - Computes bunch z extrema, mean beta_z, RMS sigma_xy, total bunch
     *    charge per slice via CIC-in-z deposition.
     *  - Sums the disk-stack Green function over slices to get E_z at
     *    each slice centre. */
    void compute (particles::TimeBunch& bunch);

    /** Lab-frame longitudinal E_z at particle position z, linearly
     *  interpolated between adjacent slice centres. Returns 0 outside
     *  the bunch z-extent (CIC clamp). */
    amrex::Real E_z_at (amrex::Real z) const noexcept;

    // ---- Diagnostics ----
    int  n_slices  () const noexcept { return m_n_slices; }
    amrex::Real bunch_radius () const noexcept { return m_a; }
    amrex::Real gamma_used   () const noexcept { return m_gamma; }
    amrex::Real total_Q      () const noexcept { return m_total_Q; }
    amrex::Real Ez_slice     (int i) const { return m_Ez_slice[i]; }
    amrex::Real z_slice      (int i) const
    { return m_z_lo + (amrex::Real(i) + amrex::Real(0.5)) * m_dz; }

private:
    int         m_n_slices;
    amrex::Real m_radius_factor;

    amrex::Real m_z_lo    = 0.0;
    amrex::Real m_z_hi    = 0.0;
    amrex::Real m_dz      = 0.0;
    amrex::Real m_a       = 0.0;     // characteristic transverse radius
    amrex::Real m_gamma   = 1.0;
    amrex::Real m_total_Q = 0.0;

    std::vector<amrex::Real> m_Q_slice;   // physical charge per slice (C)
    std::vector<amrex::Real> m_Ez_slice;  // E_z at slice centre (V/m)
};

} // namespace spacecharge
} // namespace lucretiatt
