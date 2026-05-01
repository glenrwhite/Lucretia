#pragma once

#include "particles/TimeBunch.H"

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>


namespace lucretiatt {
namespace elements {

/** Short-range wakefield element (Bane analytic formulas) for an iris-
 *  loaded TW or SW structure (S-band, X-band, etc.).
 *
 *  Per-step procedure (apply_wake, called by TrackingLoop):
 *    1. Bin live particles by z into n_slices over [z_start, z_end].
 *    2. Accumulate per-slice charge sum (Q_i) and dipole moments
 *       (Sx_i = sum q*x, Sy_i = sum q*y).
 *    3. ALLREDUCE the slice profiles across MPI ranks.
 *    4. Convolve causally (only slices AHEAD of slice i contribute):
 *         Ez_i = -sum_{j>=i} Q_j * w_long(z_j - z_i)
 *         Ex_i = -sum_{j>=i} Sx_j * w_trans(z_j - z_i)
 *         Ey_i = -sum_{j>=i} Sy_j * w_trans(z_j - z_i)
 *       The j == i (self-slice) term is taken at half-weight (the
 *       standard convention for in-slice wake).
 *    5. Apply per-particle kick: Δu = qm * (Ex,Ey,Ez) * dt for every
 *       particle whose z falls in [z_start, z_end].
 *
 *  Wake formulas (Bane 2003, "Short-range wakefields in linacs"):
 *    s_z0 = 0.41 * a^1.8 * g^1.6 / L^2.4       (m)
 *    w_long(s) = (Z0 c / pi a^2) * exp(-sqrt(s/s_z0))    (V/C/m)
 *    s_t0 = 0.169 * a^1.79 * g^0.38 / L^1.17   (m)
 *    w_trans(s) = (4 Z0 c s_t0 / pi a^4)
 *                 * (1 - (1 + sqrt(s/s_t0)) * exp(-sqrt(s/s_t0)))
 *  with a = iris radius, g = cell gap, L = cell period.
 *
 *  The wake is applied as an explicit impulse (Strang splitting after
 *  the Boris push). For typical short-range wakes this is adequate;
 *  for very strong wakes consider sub-stepping. Active gating: any
 *  particle whose z lies outside [z_start, z_end] is skipped.
 *
 *  longitudinal/transverse can be turned off independently via the
 *  m_long_on / m_trans_on flags (matches ImpactT type-(-6) v(5) v(6)
 *  flags). */
struct WakeField
    : public Named
    , public Position3D
    , public TimeWindow
{
    enum class WakeKind { ShortRangeBane };

    WakeField () = default;

    WakeField (std::string n,
               amrex::Real z_start,
               amrex::Real z_end,
               amrex::Real iris_a,
               amrex::Real gap_g,
               amrex::Real period_L,
               bool        longitudinal = true,
               bool        transverse   = true,
               int         n_slices     = 200)
        : Named(std::move(n))
        , Position3D(0.0, 0.0,
                     amrex::Real(0.5) * (z_start + z_end),
                     z_end - z_start)
        , TimeWindow()
        , m_z_start (z_start)
        , m_z_end   (z_end)
        , m_iris_a  (iris_a)
        , m_gap_g   (gap_g)
        , m_period_L(period_L)
        , m_long_on (longitudinal)
        , m_trans_on(transverse)
        , m_n_slices(n_slices)
    {}

    /** Per-step wake application. Called by TrackingLoop after the
     *  Boris push. */
    void apply_wake (particles::TimeBunch& bunch, amrex::Real dt) const;

    /** No external field gather. Wake is an instantaneous impulse on
     *  the bunch, applied separately. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
                     amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;
    }

    amrex::Real m_z_start  = 0.0;
    amrex::Real m_z_end    = 0.0;
    amrex::Real m_iris_a   = 0.0;
    amrex::Real m_gap_g    = 0.0;
    amrex::Real m_period_L = 0.0;
    bool        m_long_on  = true;
    bool        m_trans_on = true;
    int         m_n_slices = 200;
    WakeKind    m_kind     = WakeKind::ShortRangeBane;
};

} // namespace elements
} // namespace lucretiatt
