#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>


namespace lucretiatt {
namespace tracking {

/** Speed of light, m/s (CODATA 2019, exact). */
inline constexpr amrex::Real kSpeedOfLight = 299792458.0;

/** Relativistic Boris pusher, single particle.
 *
 *  Advances proper velocity (ux, uy, uz) = gamma * v by dt under fields
 *  (Ex, Ey, Ez) and (Bx, By, Bz), then drifts position (x, y, z) by
 *  v * dt where v = u / gamma at the new u. qm is per-particle q/m.
 *
 *  Synchronisation: u is centred at t + dt/2 on output, x at t + dt.
 *  For a leapfrog scheme, callers should desynchronise once at startup
 *  by half-stepping u backward (PushU(-dt/2)) — Phase 1.6 omits this
 *  because the cold-drift test is a pure free-flight check. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void boris_push_one (
    amrex::Real qm,
    amrex::Real Ex, amrex::Real Ey, amrex::Real Ez,
    amrex::Real Bx, amrex::Real By, amrex::Real Bz,
    amrex::Real dt,
    amrex::Real& x,  amrex::Real& y,  amrex::Real& z,
    amrex::Real& ux, amrex::Real& uy, amrex::Real& uz)
{
    constexpr amrex::Real c    = kSpeedOfLight;
    constexpr amrex::Real inv_c2 = amrex::Real(1.0) / (c * c);

    // 1) Half E-kick: u^- = u^n + (q/m) E (dt/2)
    const amrex::Real half_qm_dt = amrex::Real(0.5) * qm * dt;
    const amrex::Real ux_m = ux + half_qm_dt * Ex;
    const amrex::Real uy_m = uy + half_qm_dt * Ey;
    const amrex::Real uz_m = uz + half_qm_dt * Ez;

    // 2) gamma at u^-
    const amrex::Real u2_m  = ux_m*ux_m + uy_m*uy_m + uz_m*uz_m;
    const amrex::Real gamma_m = std::sqrt(amrex::Real(1.0) + u2_m * inv_c2);

    // 3) t = (q/m) B (dt / 2 gamma^-)
    const amrex::Real fac = half_qm_dt / gamma_m;
    const amrex::Real tx = fac * Bx;
    const amrex::Real ty = fac * By;
    const amrex::Real tz = fac * Bz;

    // 4) u' = u^- + u^- x t
    const amrex::Real upx = ux_m + (uy_m * tz - uz_m * ty);
    const amrex::Real upy = uy_m + (uz_m * tx - ux_m * tz);
    const amrex::Real upz = uz_m + (ux_m * ty - uy_m * tx);

    // 5) s = 2 t / (1 + |t|^2)
    const amrex::Real t2     = tx*tx + ty*ty + tz*tz;
    const amrex::Real s_norm = amrex::Real(2.0) / (amrex::Real(1.0) + t2);
    const amrex::Real sx = s_norm * tx;
    const amrex::Real sy = s_norm * ty;
    const amrex::Real sz = s_norm * tz;

    // 6) u^+ = u^- + u' x s
    const amrex::Real ux_p = ux_m + (upy * sz - upz * sy);
    const amrex::Real uy_p = uy_m + (upz * sx - upx * sz);
    const amrex::Real uz_p = uz_m + (upx * sy - upy * sx);

    // 7) Half E-kick: u^{n+1} = u^+ + (q/m) E (dt/2)
    ux = ux_p + half_qm_dt * Ex;
    uy = uy_p + half_qm_dt * Ey;
    uz = uz_p + half_qm_dt * Ez;

    // 8) Position drift with v = u / gamma at u^{n+1}
    const amrex::Real u2     = ux*ux + uy*uy + uz*uz;
    const amrex::Real inv_g  = amrex::Real(1.0) / std::sqrt(amrex::Real(1.0)
                                                           + u2 * inv_c2);
    x += ux * inv_g * dt;
    y += uy * inv_g * dt;
    z += uz * inv_g * dt;
}

} // namespace tracking
} // namespace lucretiatt
