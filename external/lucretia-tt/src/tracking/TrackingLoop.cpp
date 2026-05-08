#include "TrackingLoop.H"

#include "Boris.H"
#include "elements/CathodeSource.H"
#include "elements/WakeField.H"
#include "particles/TimeBunch.H"
#include "spacecharge/SpaceCharge.H"
#include "spacecharge/SpaceChargeSlice.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX_OpenMP.H>
#include <AMReX_ParIter.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <variant>
#include <vector>


namespace lucretiatt {
namespace tracking {

namespace {

// 1 / (4 pi eps0) -- Coulomb constant in SI (m / F)
constexpr amrex::Real kCoulombConstant = amrex::Real(8.987551787e9);

// TSC (triangle-shape-cloud) shape function weights. Particle is at
// fractional position p in [-0.5, 0.5) relative to the NEAREST node.
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

// Direct self-force computation: Coulomb sum from the deposited cloud
// of THIS particle at its own position. Replaces the LUT lookup, which
// is invalid when adaptive mesh changes cell sizes between the LUT
// build and the gather. Returns E_self[3] in unit (qw*kCoulomb) — the
// caller multiplies by qw_f = qw * kSelfForceFactor to get the field.
//
// shape_order: 1 = CIC (8-node cube), 2 = TSC (27-node cube)
// frac_x/y/z: CIC uses [0, 1) (fractional position from cell low-corner);
//              TSC uses [-0.5, +0.5) (fractional position from nearest node).
// dx, dy, dz: cell sizes (lab frame).
// eps2: softening for r²=0 (matches the LUT build).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void self_force_direct (int shape_order,
                        amrex::Real frac_x, amrex::Real frac_y, amrex::Real frac_z,
                        amrex::Real dx, amrex::Real dy, amrex::Real dz,
                        amrex::Real eps2,
                        amrex::Real* E_self) noexcept
{
    using amrex::Real;
    E_self[0] = Real(0.0); E_self[1] = Real(0.0); E_self[2] = Real(0.0);
    if (shape_order == 1) {
        // CIC: 8-node cube. wa = (a==0 ? 1-frac : frac). r_v = (frac-a)*dx.
        for (int a = 0; a < 2; ++a) {
            const Real wa   = (a == 0) ? (Real(1.0) - frac_x) : frac_x;
            const Real rx_v = (frac_x - Real(a)) * dx;
            for (int b = 0; b < 2; ++b) {
                const Real wb   = (b == 0) ? (Real(1.0) - frac_y) : frac_y;
                const Real ry_v = (frac_y - Real(b)) * dy;
                for (int cc = 0; cc < 2; ++cc) {
                    const Real wc   = (cc == 0) ? (Real(1.0) - frac_z) : frac_z;
                    const Real rz_v = (frac_z - Real(cc)) * dz;
                    const Real r2 = rx_v*rx_v + ry_v*ry_v + rz_v*rz_v + eps2;
                    const Real r3_inv = Real(1.0) / (r2 * std::sqrt(r2));
                    const Real factor = wa * wb * wc * r3_inv;
                    E_self[0] += factor * rx_v;
                    E_self[1] += factor * ry_v;
                    E_self[2] += factor * rz_v;
                }
            }
        }
    } else {
        // TSC: 27-node cube around nearest node. p in [-0.5, +0.5).
        // Weights: w0 = 0.5*(0.5-p)^2, w1 = 0.75-p^2, w2 = 0.5*(0.5+p)^2.
        // Distance from particle to nodes (i-1, i, i+1) along each axis:
        //   r = (p - (k-1))*dx for k=0,1,2 (relative offset -1, 0, +1)
        Real wx[3], wy[3], wz[3];
        tsc_weights(frac_x, wx[0], wx[1], wx[2]);
        tsc_weights(frac_y, wy[0], wy[1], wy[2]);
        tsc_weights(frac_z, wz[0], wz[1], wz[2]);
        for (int kk = 0; kk < 3; ++kk) {
            const Real wzz  = wz[kk];
            const Real rz_v = (frac_z - Real(kk - 1)) * dz;
            for (int jj = 0; jj < 3; ++jj) {
                const Real wyy  = wy[jj];
                const Real ry_v = (frac_y - Real(jj - 1)) * dy;
                for (int ii = 0; ii < 3; ++ii) {
                    const Real wxx  = wx[ii];
                    const Real rx_v = (frac_x - Real(ii - 1)) * dx;
                    const Real r2 = rx_v*rx_v + ry_v*ry_v + rz_v*rz_v + eps2;
                    const Real r3_inv = Real(1.0) / (r2 * std::sqrt(r2));
                    const Real factor = wxx * wyy * wzz * r3_inv;
                    E_self[0] += factor * rx_v;
                    E_self[1] += factor * ry_v;
                    E_self[2] += factor * rz_v;
                }
            }
        }
    }
}

// TSC gather: 27-node weighted sum of a nodal field at the particle position.
// Mirrors deposit symmetry so the kick is consistent with the deposit shape.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tsc_gather (amrex::Array4<const amrex::Real> const& field,
                        amrex::Real x, amrex::Real y, amrex::Real z,
                        amrex::GpuArray<amrex::Real, 3> const& dxi,
                        amrex::GpuArray<amrex::Real, 3> const& lo) noexcept
{
    using amrex::Real;
    const Real fx = (x - lo[0]) * dxi[0];
    const Real fy = (y - lo[1]) * dxi[1];
    const Real fz = (z - lo[2]) * dxi[2];
    const int  ix = int(amrex::Math::round(fx));
    const int  iy = int(amrex::Math::round(fy));
    const int  iz = int(amrex::Math::round(fz));
    const Real px = fx - Real(ix);
    const Real py = fy - Real(iy);
    const Real pz = fz - Real(iz);
    Real wx[3], wy[3], wz[3];
    tsc_weights(px, wx[0], wx[1], wx[2]);
    tsc_weights(py, wy[0], wy[1], wy[2]);
    tsc_weights(pz, wz[0], wz[1], wz[2]);
    Real result = Real(0.0);
    for (int kk = 0; kk < 3; ++kk) {
        const Real wzz = wz[kk];
        for (int jj = 0; jj < 3; ++jj) {
            const Real wyz = wzz * wy[jj];
            for (int ii = 0; ii < 3; ++ii) {
                result += field(ix - 1 + ii, iy - 1 + jj, iz - 1 + kk)
                        * wx[ii] * wyz;
            }
        }
    }
    return result;
}

} // anonymous


void TrackingLoop::step (
    particles::TimeBunch&                 bunch,
    const elements::Lattice&              lattice,
    amrex::Real                           t,
    amrex::Real                           dt,
    spacecharge::SpaceCharge*             sc,
    spacecharge::SpaceChargeSlice*        slice_sc)
{
    using namespace amrex;
    using namespace particles;

    // Dispatch to ImpactT-style drift-kick-drift integrator if enabled.
    // step_dkd increments m_step_count itself; we only need to bump it
    // when running the legacy path so n_steps_taken() stays consistent.
    if (m_use_dkd_integrator) {
        step_dkd(bunch, lattice, t, dt, sc, slice_sc);
        return;
    }
    ++m_step_count;

    // ---- 1. Emission ----
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::CathodeSource>) {
                e.emit_new_particles(bunch, t, dt);
            }
        }, el);
    }

    // ---- 1b. Centroid-z reference time (ImpactT compat) ----
    // When m_use_centroid_phase is set, compute the bunch z-centroid
    // and use t_centroid = z_centroid/c + offset as the wallclock
    // substitute for external-field gather. Mirrors ImpactT's
    // refptcl(5) = sgcenter(5) convention (AccSimulator.f90 line 1436).
    //
    // ImpactT computes sgcenter AFTER a half-step drift; we approximate
    // that by predicting the midstep centroid as
    //   z_mid = <z> + 0.5*dt*<v_z>
    // where <v_z> = <uz>/<gamma> is computed from the same alive
    // particles. Without this shift our t_centroid is the start-of-
    // step value, half a dt behind ImpactT's reference; the resulting
    // RF phase offset (~5° at non-rel cathode region) propagates into
    // a measurable chirp-slope error.
    Real t_centroid = t;   // default: identical to wallclock
    if (m_use_centroid_phase) {
        Real sum_z = 0.0, sum_vz = 0.0;
        long n_alive = 0;
        using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
        constexpr int lev = 0;
        constexpr Real inv_c2 = Real(1.0) / (kSpeedOfLight * kSpeedOfLight);
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            auto& soa = pti.GetStructOfArrays();
            const int np = pti.numParticles();
            const auto& zs     = soa.GetRealData(RealSoA::z);
            const auto& uxs    = soa.GetRealData(RealSoA::px);
            const auto& uys    = soa.GetRealData(RealSoA::py);
            const auto& uzs    = soa.GetRealData(RealSoA::pz);
            const auto& alives = soa.GetIntData(IntSoA::alive);
            for (int i = 0; i < np; ++i) {
                if (alives[i] == 0) { continue; }
                const Real ux = uxs[i], uy = uys[i], uz = uzs[i];
                const Real gamma = std::sqrt(Real(1.0) +
                    (ux*ux + uy*uy + uz*uz) * inv_c2);
                sum_z  += zs[i];
                sum_vz += uz / gamma;     // v_z = u_z / gamma
                ++n_alive;
            }
        }
        ParallelDescriptor::ReduceRealSum(sum_z);
        ParallelDescriptor::ReduceRealSum(sum_vz);
        ParallelDescriptor::ReduceLongSum(n_alive);
        if (n_alive > 0) {
            const Real z_centroid  = sum_z  / Real(n_alive);
            const Real vz_centroid = sum_vz / Real(n_alive);
            const Real z_mid = z_centroid + Real(0.5) * dt * vz_centroid;
            t_centroid = z_mid / kSpeedOfLight + m_centroid_t_offset;
        }
    }

    // ---- 2. Image-charge planes ----
    std::vector<Real> image_planes;
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::CathodeSource>) {
                if (e.m_image_charge_enabled) {
                    image_planes.push_back(e.m_z_cathode);
                }
            }
        }, el);
    }

    // ---- 3. Space-charge solve (once per step, all ranks/tiles) ----
    GpuArray<Real, 3>  dxi_sc{}, lo_sc{};
    Real const*        sf_lut[3] = {nullptr, nullptr, nullptr};
    int                sf_n      = 0;
    // Boost-back factor for SC E -> lab synchronous force.
    //
    // SpaceCharge::solve runs the IGF Poisson with cell sizes (dx, dy,
    // gamma * dz), giving phi as the Coulomb potential of the bunch
    // (full physical charge per cell, NOT divided by gamma) evaluated
    // at the stretched-z grid positions. Effectively:
    //   phi_solver(x_lab, y_lab, z_lab) = gamma * phi_rest_correct(x_lab, y_lab, gamma*z_lab)
    // The gradient computed in LAB coords (the kernel uses dx[2]) gives:
    //   E_x_solver = -d phi_solver / dx_lab  = gamma * E_x_rest    (dx_lab = dx_rest)
    //   E_z_solver = -d phi_solver / dz_lab  = gamma^2 * E_z_rest  (dz_rest = gamma*dz_lab,
    //                                                              so d/dz_lab = gamma * d/dz_rest)
    //
    // Lab-frame force on a synchronous test particle (moving at the
    // same beta as the bunch, with v x B canceling part of the boosted
    // E in the transverse direction):
    //   F_x_lab = q * E_x_rest / gamma   (transverse: 1/gamma^2 from boost cancellation)
    //   F_z_lab = q * E_z_rest           (longitudinal: invariant under z-boost)
    //
    // Required correction multipliers on the solver fields, BOTH the
    // same 1/gamma^2 = (1 - beta^2):
    //   E_x_solver * (1/gamma^2) = (gamma*E_x_rest) / gamma^2 = E_x_rest / gamma  -> F_x correct
    //   E_z_solver * (1/gamma^2) = (gamma^2*E_z_rest) / gamma^2 = E_z_rest          -> F_z correct
    //
    // At gamma = 1 (test_uniform_sphere) the factor is 1 → behaviour
    // unchanged. At gun exit (gamma ~ 12) the factor is ~1/144, so the
    // raw mesh contribution is correctly tamped down to physical scale.
    // sc_boost is now computed PER PARTICLE in the kernel (see below) —
    // this is the "infinite-bin" energy-binning limit, where each
    // particle's lab-frame SC force is corrected by ITS OWN 1/gamma^2
    // rather than the bunch-mean. Avoids the issue where a single
    // mean-gamma boost is wrong for both head and tail of a chirped
    // bunch (typical for photoinjector exit). Cost: one sqrt + 2 mul
    // per particle per step.
    // SC field arrays are looked up ONCE here (not per-pti) -- with
    // single-FAB SpaceCharge MultiFabs the same arrays serve all
    // particles, so caching outside the pti loop avoids repeated
    // MFIter constructions that AMReX rejects when nested with PIter.
    spacecharge::SpaceCharge::GatherFABs sc_fabs;
    int                sc_shape_order = 1;     // 1 = CIC, 2 = TSC
    if (sc) {
        sc->set_diag_dt_hint(dt);
        sc->solve(bunch);
        dxi_sc = sc->dxi();
        lo_sc  = sc->lo();
        sf_n      = sc->lut_n();
        sf_lut[0] = sc->lut_ptr(0);
        sf_lut[1] = sc->lut_ptr(1);
        sf_lut[2] = sc->lut_ptr(2);
        sc_fabs   = sc->gather_fabs();
        sc_shape_order = sc->shape_order();
    }

    // 1D longitudinal slice SC: solved (re-binned + integrated) once per
    // step for all ranks. Per-particle gather is just E_z_at(z_p) below.
    // The slice solver is mesh-free and stays resolved no matter how
    // short the bunch gets; it REPLACES the longitudinal component of
    // the 3D mesh SC contribution at gather time. The mesh SC keeps
    // doing transverse (Ex, Ey).
    if (slice_sc) {
        slice_sc->compute(bunch);
    }

    // ---- 4. Per-particle push ----
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;

    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();

        // SC field arrays were cached above (outside the pti loop)
        // since they're independent of pti when SC uses single-FAB layout.

        // CPU-only loop (Phase 4A). GPU portability deferred to Phase 7.
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int ip = 0; ip < np; ++ip)
        {
            // Skip dead particles (e.g. cathode re-cross kills below).
            // These remain in the bunch container for SC charge
            // conservation but get no field gather / Boris push.
            if (ptd.idata(IntSoA::alive)[ip] == 0) { continue; }

            Real x  = ptd.rdata(RealSoA::x)[ip];
            Real y  = ptd.rdata(RealSoA::y)[ip];
            Real z  = ptd.rdata(RealSoA::z)[ip];
            Real ux = ptd.rdata(RealSoA::px)[ip];
            Real uy = ptd.rdata(RealSoA::py)[ip];
            Real uz = ptd.rdata(RealSoA::pz)[ip];
            const Real q  = ptd.rdata(RealSoA::q)[ip];
            const Real qm = ptd.rdata(RealSoA::qm)[ip];

            // Per-particle effective dt and field-gather time. For freshly
            // emitted particles (t_birth in (t, t+dt]) the first-push
            // interval is [t_birth, t+dt]; gathering the lattice field at
            // the MIDPOINT of that interval gives each new particle its
            // own RF phase exposure on its first step. Old particles
            // (t_birth <= t) gather at t.
            //
            // dt_eff and t_field are MUTABLE because behind-cathode drift
            // may detect a cross-cathode this step and shrink the Boris
            // window to dt_post (= post-cross fraction), shifting t_field
            // accordingly. Mirrors ImpactT's fractional first-step kick
            // (AccSimulator.f90 lines 2154-2160).
            Real t_birth = ptd.rdata(RealSoA::t_birth)[ip];
            Real dt_eff  = std::min(dt, std::max(Real(0.0), t + dt - t_birth));
            // Phase-reference time for cos(omega*t_field + phi) in field gather.
            // Per-particle phase mode overrides centroid mode (z_i/c per particle).
            Real t_phase;
            if (m_use_particle_phase) {
                t_phase = z / kSpeedOfLight;
            } else if (m_use_centroid_phase) {
                t_phase = t_centroid;
            } else {
                t_phase = t;
            }
            Real t_field = (t_birth > t)
                ? Real(0.5) * (t_birth + t + dt) - (t - t_phase)
                : t_phase;

            // ImpactT-style behind-cathode drift. Particles with z < cathode_z
            // that have NEVER yet crossed (emerged == 0) advance at the
            // universal speed betazini*c (no field, Boris, or SC) -- mirror
            // of ImpactT's `driftemission_BeamBunch`.
            //
            // When a particle's drift carries z across cathode_z this step,
            // we apply a FRACTIONAL Boris kick over only the post-cross
            // remainder of the step. This mirrors ImpactT lines 2154-2160:
            // dt_post = (post-drift z) / (drift speed), particle's own
            // velocity is used for the post-cross position advance, and
            // Boris fires with the reduced dt_eff = dt_post.
            //
            // Particles already emerged (emerged == 1) that get pushed back
            // behind cathode by a decelerating Boris kick are KILLED and
            // PARKED at z=-1e9 -- the kill prevents the re-crossing ratchet
            // that otherwise produces ~15 outlier particles at extreme
            // gammas; the park keeps dead particles out of analysis-time
            // stats without breaking openPMD output.
            if (m_behind_cathode_drift_on && z < m_cathode_z) {
                if (ptd.idata(IntSoA::emerged)[ip] != 0) {
                    // Already emerged, now back behind cathode -- kill+park.
                    ptd.idata(IntSoA::alive)[ip] = 0;
                    ptd.rdata(RealSoA::x)[ip]  = Real(0.0);
                    ptd.rdata(RealSoA::y)[ip]  = Real(0.0);
                    ptd.rdata(RealSoA::z)[ip]  = Real(-1.0e9);
                    ptd.rdata(RealSoA::px)[ip] = Real(0.0);
                    ptd.rdata(RealSoA::py)[ip] = Real(0.0);
                    ptd.rdata(RealSoA::pz)[ip] = Real(0.0);
                    continue;
                }
                // Behind cathode, not yet emerged -- drift this step.
                const Real z_drift = z + dt_eff * m_behind_cathode_betazini * kSpeedOfLight;
                if (z_drift < m_cathode_z) {
                    // Still behind after drift -- drift only, skip Boris.
                    z = z_drift;
                    ptd.rdata(RealSoA::z)[ip] = z;
                    continue;
                }
                // Cross-cathode this step. Compute post-cross fraction;
                // pre-advance position using particle's OWN velocity (not
                // betazini); FALL THROUGH to apply Boris over the post-
                // cross dt_post. Mirrors ImpactT lines 2154-2160.
                const Real total_drift = z_drift - z;
                const Real post_drift  = z_drift - m_cathode_z;
                const Real post_frac   = (total_drift > Real(0.0))
                    ? (post_drift / total_drift) : Real(0.0);
                const Real dt_post = post_frac * dt_eff;
                constexpr Real inv_c2_local = Real(1.0) / (kSpeedOfLight * kSpeedOfLight);
                const Real recpgam = Real(1.0) /
                    std::sqrt(Real(1.0) + (ux*ux + uy*uy + uz*uz) * inv_c2_local);
                x += dt_post * ux * recpgam;
                y += dt_post * uy * recpgam;
                z  = m_cathode_z + dt_post * uz * recpgam;
                // Override dt_eff for the partial step; shift t_birth so
                // the existing field-gather machinery uses [t_cross, t+dt]
                // as the Boris interval.
                dt_eff  = dt_post;
                t_birth = t + dt - dt_post;
                ptd.rdata(RealSoA::t_birth)[ip] = t_birth;
                const Real midpoint = Real(0.5) * (t_birth + t + dt);
                if (m_use_particle_phase) {
                    t_field = z / kSpeedOfLight;
                } else if (m_use_centroid_phase) {
                    t_field = midpoint - (t - t_phase);
                } else {
                    t_field = midpoint;
                }
                ptd.idata(IntSoA::emerged)[ip] = 1;
                // Fall through to field gather + Boris below.
            } else if (m_behind_cathode_drift_on
                       && ptd.idata(IntSoA::emerged)[ip] == 0
                       && z >= m_cathode_z) {
                // Edge case: emitted directly at z >= cathode (e.g.
                // CathodeSource with z_init_spread = 0). Mark emerged
                // without t_birth tweaks; normal Boris with full dt.
                ptd.idata(IntSoA::emerged)[ip] = 1;
            }

            // External fields from elements. Optionally evaluate at
            // midstep position (x + 0.5*dt_eff*v) to mirror ImpactT's
            // half_drift -> gather -> half_drift centered scheme.
            // Without this, our scheme is gather at start-of-step
            // position, which is half a step behind ImpactT's gather
            // point. The 0.045 mm offset (at v=c, dt=0.3ps) times the
            // field gradient (~2 GV/m^2 in the gun) gives a ~0.1%
            // per-step field error that accumulates over many steps,
            // a likely source of the residual chirp-slope mismatch
            // vs ImpactT in the no-SC partcl-seed test.
            Real x_gather = x;
            Real y_gather = y;
            Real z_gather = z;
            if (m_use_midstep_field) {
                constexpr Real inv_c2_mid = Real(1.0) / (kSpeedOfLight * kSpeedOfLight);
                const Real recpgam_mid = Real(1.0) /
                    std::sqrt(Real(1.0) + (ux*ux + uy*uy + uz*uz) * inv_c2_mid);
                const Real half_dt = Real(0.5) * dt_eff;
                x_gather = x + half_dt * ux * recpgam_mid;
                y_gather = y + half_dt * uy * recpgam_mid;
                z_gather = z + half_dt * uz * recpgam_mid;
                // Also advance t_field by 0.5*dt to fully match ImpactT's
                // drift-kick-drift centering (half-drift advances t by dt/2
                // before kick, so field gather sees t_n+1/2). For wallclock
                // and centroid modes this is just an additive offset.
                if (!m_use_particle_phase) {
                    t_field += Real(0.5) * dt_eff;
                }
            }
            Real Ex = 0.0, Ey = 0.0, Ez = 0.0;
            Real Bx = 0.0, By = 0.0, Bz = 0.0;
            for (auto const& el : lattice) {
                Real Ex_e = 0.0, Ey_e = 0.0, Ez_e = 0.0;
                Real Bx_e = 0.0, By_e = 0.0, Bz_e = 0.0;
                std::visit([&] (auto const& e) {
                    if (e.active(t_field)) {
                        e.gather_E_B(x_gather, y_gather, z_gather, t_field,
                                     Ex_e, Ey_e, Ez_e,
                                     Bx_e, By_e, Bz_e);
                    }
                }, el);
                Ex += Ex_e; Ey += Ey_e; Ez += Ez_e;
                Bx += Bx_e; By += By_e; Bz += Bz_e;
            }

            // Self image-charge
            for (Real z_cath : image_planes) {
                const Real z_above = z - z_cath;
                if (z_above > Real(0.0)) {
                    Ez += -q * kCoulombConstant / (Real(4.0) * z_above * z_above);
                }
            }

            // Space-charge gather + per-particle self-force subtraction.
            // Gate on bounds: if the particle has drifted outside the SC
            // mesh (e.g. past the downstream end of a static box) the
            // nodal gather would read out-of-array cells. Skip the gather
            // for those particles -- they get no SC contribution but the
            // run continues.
            if (sc) {
                const Real fx_sc = (x - lo_sc[0]) * dxi_sc[0];
                const Real fy_sc = (y - lo_sc[1]) * dxi_sc[1];
                const Real fz_sc = (z - lo_sc[2]) * dxi_sc[2];

                // Footprint check: CIC needs {floor, floor+1}; TSC needs
                // {round-1, round, round+1}. Both fit in valid+1-ghost.
                int ix_lo, iy_lo, iz_lo, ix_hi, iy_hi, iz_hi;
                if (sc_shape_order == 1) {
                    ix_lo = int(std::floor(fx_sc));     ix_hi = ix_lo + 1;
                    iy_lo = int(std::floor(fy_sc));     iy_hi = iy_lo + 1;
                    iz_lo = int(std::floor(fz_sc));     iz_hi = iz_lo + 1;
                } else {
                    const int ix_n = int(std::round(fx_sc));
                    const int iy_n = int(std::round(fy_sc));
                    const int iz_n = int(std::round(fz_sc));
                    ix_lo = ix_n - 1;  ix_hi = ix_n + 1;
                    iy_lo = iy_n - 1;  iy_hi = iy_n + 1;
                    iz_lo = iz_n - 1;  iz_hi = iz_n + 1;
                }
                // Multi-FAB: find which FAB contains this particle's stencil.
                int fab_idx = -1;
                for (int f = 0, nf = int(sc_fabs.boxes.size()); f < nf; ++f) {
                    auto const& bx = sc_fabs.boxes[f];
                    if (ix_lo >= bx.smallEnd(0) && ix_hi <= bx.bigEnd(0) &&
                        iy_lo >= bx.smallEnd(1) && iy_hi <= bx.bigEnd(1) &&
                        iz_lo >= bx.smallEnd(2) && iz_hi <= bx.bigEnd(2))
                    { fab_idx = f; break; }
                }
                if (fab_idx < 0) {
                    goto sc_done;   // outside any FAB; keep external E,B
                }
                amrex::GpuArray<Real, 3> E_sc{};
                auto const& Ex_arr_f = sc_fabs.Ex[fab_idx];
                auto const& Ey_arr_f = sc_fabs.Ey[fab_idx];
                auto const& Ez_arr_f = sc_fabs.Ez[fab_idx];
                if (sc_shape_order == 1) {
                    auto E_cic = ablastr::particles::doGatherVectorFieldNodal(
                        ParticleReal(x), ParticleReal(y), ParticleReal(z),
                        Ex_arr_f, Ey_arr_f, Ez_arr_f,
                        dxi_sc, lo_sc);
                    E_sc[0] = E_cic[0]; E_sc[1] = E_cic[1]; E_sc[2] = E_cic[2];
                } else {
                    E_sc[0] = tsc_gather(Ex_arr_f, x, y, z, dxi_sc, lo_sc);
                    E_sc[1] = tsc_gather(Ey_arr_f, x, y, z, dxi_sc, lo_sc);
                    E_sc[2] = tsc_gather(Ez_arr_f, x, y, z, dxi_sc, lo_sc);
                }

                // Subtract Coulomb field of this particle's own deposited
                // shape cloud (8 nodes for CIC, 27 for TSC). Removes the
                // self-interaction the IGF + gather chain would otherwise
                // carry. Implementation: trilinear interp of a precomputed
                // shape table built once per SpaceCharge (see
                // build_self_force_lut). The table is built for the active
                // shape order with matching index conventions:
                //   CIC: indexed by (wx_frac, wy_frac, wz_frac) in [0, 1)
                //   TSC: indexed by (px, py, pz) in [-0.5, +0.5)
                const Real w  = ptd.rdata(RealSoA::w)[ip];
                const Real qw = q * w * kCoulombConstant;

                Real E_self[3];
                if (m_self_force_direct) {
                    // Direct mode: compute Coulomb sum from CURRENT cell sizes.
                    // Bypasses the LUT (which assumes static cell aspect/size and
                    // produces noise when adaptive mesh resizes between rebuilds).
                    Real frac_x, frac_y, frac_z;
                    if (sc_shape_order == 1) {
                        frac_x = fx_sc - std::floor(fx_sc);
                        frac_y = fy_sc - std::floor(fy_sc);
                        frac_z = fz_sc - std::floor(fz_sc);
                    } else {
                        frac_x = fx_sc - std::round(fx_sc);
                        frac_y = fy_sc - std::round(fy_sc);
                        frac_z = fz_sc - std::round(fz_sc);
                    }
                    const Real dx_cell = Real(1.0) / dxi_sc[0];
                    const Real dy_cell = Real(1.0) / dxi_sc[1];
                    const Real dz_cell = Real(1.0) / dxi_sc[2];
                    const Real eps2    = Real(1.0e-6) * (dx_cell * dx_cell);
                    self_force_direct(sc_shape_order, frac_x, frac_y, frac_z,
                                      dx_cell, dy_cell, dz_cell, eps2, E_self);
                } else {
                    // LUT mode: trilinear interpolation of pre-built table.
                    Real tx, ty, tz;
                    if (sc_shape_order == 1) {
                        const Real wx_frac = fx_sc - std::floor(fx_sc);
                        const Real wy_frac = fy_sc - std::floor(fy_sc);
                        const Real wz_frac = fz_sc - std::floor(fz_sc);
                        tx = wx_frac * Real(sf_n);
                        ty = wy_frac * Real(sf_n);
                        tz = wz_frac * Real(sf_n);
                    } else {
                        const Real px = fx_sc - std::round(fx_sc);
                        const Real py = fy_sc - std::round(fy_sc);
                        const Real pz = fz_sc - std::round(fz_sc);
                        tx = (px + Real(0.5)) * Real(sf_n);
                        ty = (py + Real(0.5)) * Real(sf_n);
                        tz = (pz + Real(0.5)) * Real(sf_n);
                    }
                    int ix_t = int(tx); if (ix_t >= sf_n) ix_t = sf_n - 1;
                    int iy_t = int(ty); if (iy_t >= sf_n) iy_t = sf_n - 1;
                    int iz_t = int(tz); if (iz_t >= sf_n) iz_t = sf_n - 1;
                    if (ix_t < 0) ix_t = 0;
                    if (iy_t < 0) iy_t = 0;
                    if (iz_t < 0) iz_t = 0;
                    const Real ax = tx - Real(ix_t);
                    const Real ay = ty - Real(iy_t);
                    const Real az = tz - Real(iz_t);
                    const int  s1 = sf_n + 1;
                    const int  s2 = s1 * s1;
                    const int  i000 = ix_t * s2 + iy_t * s1 + iz_t;

                    for (int c = 0; c < 3; ++c) {
                        const Real* T = sf_lut[c];
                        const Real v000 = T[i000];
                        const Real v001 = T[i000 + 1];
                        const Real v010 = T[i000 + s1];
                        const Real v011 = T[i000 + s1 + 1];
                        const Real v100 = T[i000 + s2];
                        const Real v101 = T[i000 + s2 + 1];
                        const Real v110 = T[i000 + s2 + s1];
                        const Real v111 = T[i000 + s2 + s1 + 1];
                        const Real v00  = v000 + (v001 - v000) * az;
                        const Real v01  = v010 + (v011 - v010) * az;
                        const Real v10  = v100 + (v101 - v100) * az;
                        const Real v11  = v110 + (v111 - v110) * az;
                        const Real v0   = v00  + (v01  - v00 ) * ay;
                        const Real v1   = v10  + (v11  - v10 ) * ay;
                        E_self[c] = v0 + (v1 - v0) * ax;
                    }
                }

                // Empirical fudge factor: Coulomb-of-point-charges over-
                // corrects vs the smeared IGF cloud. Calibrated on the
                // uniform-sphere test (1000 macros, R = 1 mm). TSC's
                // 27-node cloud has a smaller per-node Coulomb self-sum
                // than CIC's 8-node cloud (charge is spread further from
                // the particle), so the multiplicative compensation is
                // roughly 2x larger.
                const Real kSelfForceFactor =
                    m_self_force_disabled ? Real(0.0)
                                          : (sc_shape_order == 2 ? Real(1.27)
                                                                 : Real(0.62));
                const Real qw_f = qw * kSelfForceFactor;
                const Real Ex_corr = E_sc[0] - qw_f * E_self[0];
                const Real Ey_corr = E_sc[1] - qw_f * E_self[1];
                const Real Ez_corr = E_sc[2] - qw_f * E_self[2];

                if (m_sc_use_b_field) {
                    // ImpactT-style: explicit SC B field, Boris handles
                    // v×B (matches Field.f90:445-467). Equivalent to the
                    // boost shortcut for synchronous particles; adds the
                    // longitudinal v×B coupling for non-synchronous ones
                    // (chromatic SC effect we previously missed). Solver
                    // E_x,E_y are already E_x_lab,E_y_lab (transverse:
                    // solver gives γ*E_rest=E_lab). E_z_solver = γ²*E_z_lab
                    // because the gradient takes the rest-frame stretched
                    // z step against a lab dz cell -- divide by γ² to get
                    // E_z_lab.
                    const Real g_b      = sc->last_gamma();
                    const Real beta_z   = sc->last_beta_z();
                    const Real inv_g2   = Real(1.0) / (g_b * g_b);
                    constexpr Real inv_c = Real(1.0) / kSpeedOfLight;
                    const Real beta_inv_c = beta_z * inv_c;

                    Ex += Ex_corr;
                    Ey += Ey_corr;
                    if (!slice_sc) {
                        Ez += Ez_corr * inv_g2;
                    }
                    Bx += -beta_inv_c * Ey_corr;
                    By += +beta_inv_c * Ex_corr;
                    // Bz: SC contributes 0
                } else {
                    // Boost shortcut: equivalent to v×B for synchronous
                    // particles; misses non-synchronous coupling. Default
                    // for backward compatibility.
                    constexpr Real c     = kSpeedOfLight;
                    constexpr Real inv_c2 = Real(1.0) / (c * c);
                    Real sc_boost;
                    if (m_sc_boost_bunch_mean) {
                        const Real g_b = sc->last_gamma();
                        sc_boost = Real(1.0) / (g_b * g_b);
                    } else {
                        const Real u2_p = ux*ux + uy*uy + uz*uz;
                        sc_boost = Real(1.0) / (Real(1.0) + u2_p * inv_c2);
                    }
                    Ex += sc_boost * Ex_corr;
                    Ey += sc_boost * Ey_corr;
                    if (!slice_sc) {
                        // Mesh-only mode: longitudinal from the 3D IGF
                        // (with same boost factor as transverse).
                        Ez += sc_boost * Ez_corr;
                    }
                }
                // else: slice SC handles longitudinal — gathered below
                // (after the sc_done label, so it still runs for
                // particles that fell outside the mesh box).
            }
            sc_done:;
            if (slice_sc) {
                // 1D longitudinal slice SC: applies regardless of the
                // 3D mesh box bounds (slice grid spans the actual bunch
                // z-extent, not the mesh). Disk-stack formula is in the
                // lab frame (E_z invariant under longitudinal boost),
                // with gamma baked into the z-distance term, so no
                // additional boost factor here.
                Ez += slice_sc->E_z_at(z);
            }

            // dt_eff and t_birth are computed above (at the top of the
            // per-particle loop) so they are available to the field
            // gather. Just push with dt_eff here.
            boris_push_one(qm, Ex, Ey, Ez, Bx, By, Bz, dt_eff,
                           x, y, z, ux, uy, uz);

            ptd.rdata(RealSoA::x)[ip]  = x;
            ptd.rdata(RealSoA::y)[ip]  = y;
            ptd.rdata(RealSoA::z)[ip]  = z;
            ptd.rdata(RealSoA::px)[ip] = ux;
            ptd.rdata(RealSoA::py)[ip] = uy;
            ptd.rdata(RealSoA::pz)[ip] = uz;
        }
    }

    // ---- 5. Wake-field impulse (post-push, Strang splitting) ----
    // Applied to whatever particles ended up in each WakeField's
    // [z_start, z_end] range after the Boris push. Bane analytic
    // short-range wake; slice convolution; explicit per-step impulse.
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::WakeField>) {
                e.apply_wake(bunch, dt);
            }
        }, el);
    }

    // ---- 6. Aperture collimators ----
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::Collimator>) {
                e.apply(bunch);
            }
        }, el);
    }
}


// =====================================================================
// ImpactT-style drift-kick-drift integrator with first-order emission
// =====================================================================
//
// Mirrors AccSimulator.f90 main loop (lines 1407, 2110, 2141):
//   1. drifthalf_BeamBunch  (above-cathode particles, OLD velocity, 0.5*dt)
//   2. SC solve             (at midstep positions, after the half-drift)
//   3. kick2t_BeamBunch     (field gather at t+0.5*dt, full-dt Boris kick)
//   4. drifthalf_BeamBunch  (above-cathode, NEW velocity, 0.5*dt)
//   5. driftemission_BeamBunch (z<=0 + pz>=0 particles, full-dt at betazini)
//   6. first-order emission  (crossing particles: z = dtmp * uz/gamma,
//                             where dtmp = z_after_emission/betazini)
//
// Below-cathode particles with pz<0 are FROZEN (don't drift) -- ImpactT
// does this implicitly via the (pz>=0) check in driftemission. The fractional
// cross-cathode kick is REMOVED: emerging particles get their first Boris
// kick on the FOLLOWING step, not on the crossing step itself.
void TrackingLoop::step_dkd (
    particles::TimeBunch&                 bunch,
    const elements::Lattice&              lattice,
    amrex::Real                           t,
    amrex::Real                           dt,
    spacecharge::SpaceCharge*             sc,
    spacecharge::SpaceChargeSlice*        slice_sc)
{
    using namespace amrex;
    using namespace particles;
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;

    constexpr Real c        = kSpeedOfLight;
    constexpr Real inv_c2   = Real(1.0) / (c * c);
    constexpr int  lev      = 0;
    const     Real cathode_z = m_behind_cathode_drift_on
                                ? m_cathode_z : Real(-1e30);
    const     Real betazini_c = m_behind_cathode_drift_on
                                ? (m_behind_cathode_betazini * c) : Real(0.0);
    const     Real half_dt   = Real(0.5) * dt;

    // Per-particle SC-kick dump (cross-code comparison harness).
    ++m_step_count;
    bool dump_kicks_now = !m_dump_kicks_at_steps.empty()
        && std::find(m_dump_kicks_at_steps.begin(),
                     m_dump_kicks_at_steps.end(),
                     m_step_count) != m_dump_kicks_at_steps.end();
    // Time-triggered dump: fires at the FIRST step whose t >= each target.
    // Mark the trigger as fired so it can't re-fire on later steps.
    if (!m_dump_kicks_at_times.empty()) {
        for (std::size_t k = 0; k < m_dump_kicks_at_times.size(); ++k) {
            if (m_dump_kicks_times_fired[k]) continue;
            if (Real(t) >= Real(m_dump_kicks_at_times[k])) {
                m_dump_kicks_times_fired[k] = true;
                dump_kicks_now              = true;
            }
        }
    }
    int dump_total_np = 0;
    std::vector<double> kick_buf;        // 12 doubles per particle
    std::vector<std::int32_t> kick_keep; // 1 = include in dump, 0 = skip
    if (dump_kicks_now) {
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            dump_total_np += pti.numParticles();
        }
        kick_buf.assign(std::size_t(dump_total_np) * 12, 0.0);
        kick_keep.assign(std::size_t(dump_total_np), 0);
    }

    // ---- 1. Emission ----
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::CathodeSource>) {
                e.emit_new_particles(bunch, t, dt);
            }
        }, el);
    }

    // ---- 2. First half-drift: above-cathode particles only, OLD velocity ----
    // Mirrors drifthalf_BeamBunch (BeamBunch.f90 line 103).
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int ip = 0; ip < np; ++ip) {
            if (ptd.idata(IntSoA::alive)[ip] == 0) continue;
            const Real z = ptd.rdata(RealSoA::z)[ip];
            if (z <= cathode_z) continue;
            const Real ux = ptd.rdata(RealSoA::px)[ip];
            const Real uy = ptd.rdata(RealSoA::py)[ip];
            const Real uz = ptd.rdata(RealSoA::pz)[ip];
            const Real recpgam = Real(1.0) / std::sqrt(
                Real(1.0) + (ux*ux + uy*uy + uz*uz) * inv_c2);
            ptd.rdata(RealSoA::x)[ip] += half_dt * ux * recpgam;
            ptd.rdata(RealSoA::y)[ip] += half_dt * uy * recpgam;
            ptd.rdata(RealSoA::z)[ip] += half_dt * uz * recpgam;
        }
    }

    // ---- 3. Phase reference time (centroid mode only -- otherwise unused) ----
    Real t_centroid = t;
    if (m_use_centroid_phase) {
        Real sum_z = 0.0, sum_vz = 0.0;
        long n_alive = 0;
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            auto& soa = pti.GetStructOfArrays();
            const int np = pti.numParticles();
            const auto& zs     = soa.GetRealData(RealSoA::z);
            const auto& uxs    = soa.GetRealData(RealSoA::px);
            const auto& uys    = soa.GetRealData(RealSoA::py);
            const auto& uzs    = soa.GetRealData(RealSoA::pz);
            const auto& alives = soa.GetIntData(IntSoA::alive);
            for (int i = 0; i < np; ++i) {
                if (alives[i] == 0) continue;
                const Real ux = uxs[i], uy = uys[i], uz = uzs[i];
                const Real gamma = std::sqrt(Real(1.0) +
                    (ux*ux + uy*uy + uz*uz) * inv_c2);
                sum_z  += zs[i];
                sum_vz += uz / gamma;
                ++n_alive;
            }
        }
        ParallelDescriptor::ReduceRealSum(sum_z);
        ParallelDescriptor::ReduceRealSum(sum_vz);
        ParallelDescriptor::ReduceLongSum(n_alive);
        if (n_alive > 0) {
            const Real z_centroid = sum_z / Real(n_alive);
            t_centroid = z_centroid / c + m_centroid_t_offset;
        }
    }

    // ---- 4. Image-charge planes (collect cathode z's from CathodeSource) ----
    std::vector<Real> image_planes;
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::CathodeSource>) {
                if (e.m_image_charge_enabled) {
                    image_planes.push_back(e.m_z_cathode);
                }
            }
        }, el);
    }

    // ---- 5. Space-charge solve (at midstep positions, post half-drift) ----
    GpuArray<Real, 3>  dxi_sc{}, lo_sc{};
    Real const*        sf_lut[3] = {nullptr, nullptr, nullptr};
    int                sf_n      = 0;
    spacecharge::SpaceCharge::GatherFABs sc_fabs;
    int                sc_shape_order = 1;     // 1 = CIC, 2 = TSC
    if (sc) {
        sc->set_diag_dt_hint(dt);
        sc->solve(bunch);
        dxi_sc    = sc->dxi();
        lo_sc     = sc->lo();
        sf_n      = sc->lut_n();
        sf_lut[0] = sc->lut_ptr(0);
        sf_lut[1] = sc->lut_ptr(1);
        sf_lut[2] = sc->lut_ptr(2);
        sc_fabs   = sc->gather_fabs();
        sc_shape_order = sc->shape_order();
    }
    if (slice_sc) {
        slice_sc->compute(bunch);
    }

    // ---- 6. Field gather + Boris kick (above-cathode only, full dt) ----
    // Field is gathered at midstep position (current x after half-drift)
    // and midstep time (t + 0.5*dt for wallclock; t_centroid for centroid mode).
    // Below-cathode particles get NO field, NO kick this step.
    const Real t_field_default = m_use_centroid_phase ? t_centroid
                               : (t + half_dt);
    int dump_pti_offset = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
        const int dump_my_offset = dump_pti_offset;
        dump_pti_offset += np;
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int ip = 0; ip < np; ++ip) {
            if (ptd.idata(IntSoA::alive)[ip] == 0) continue;
            const Real x = ptd.rdata(RealSoA::x)[ip];
            const Real y = ptd.rdata(RealSoA::y)[ip];
            const Real z = ptd.rdata(RealSoA::z)[ip];
            if (z <= cathode_z) continue;

            Real ux = ptd.rdata(RealSoA::px)[ip];
            Real uy = ptd.rdata(RealSoA::py)[ip];
            Real uz = ptd.rdata(RealSoA::pz)[ip];
            const Real q  = ptd.rdata(RealSoA::q)[ip];
            const Real qm = ptd.rdata(RealSoA::qm)[ip];

            const Real t_field = m_use_particle_phase
                ? (z / c) : t_field_default;

            Real Ex = 0, Ey = 0, Ez = 0;
            Real Bx = 0, By = 0, Bz = 0;
            for (auto const& el : lattice) {
                Real Ex_e = 0, Ey_e = 0, Ez_e = 0;
                Real Bx_e = 0, By_e = 0, Bz_e = 0;
                std::visit([&] (auto const& e) {
                    if (e.active(t_field)) {
                        e.gather_E_B(x, y, z, t_field,
                                     Ex_e, Ey_e, Ez_e,
                                     Bx_e, By_e, Bz_e);
                    }
                }, el);
                Ex += Ex_e; Ey += Ey_e; Ez += Ez_e;
                Bx += Bx_e; By += By_e; Bz += Bz_e;
            }

            // Image-charge planes (one Coulomb mirror per cathode plane).
            for (Real z_cath : image_planes) {
                const Real z_above = z - z_cath;
                if (z_above > Real(0.0)) {
                    Ez += -q * kCoulombConstant
                        / (Real(4.0) * z_above * z_above);
                }
            }

            // Snapshot E,B before SC contribution -- the dump records
            // the per-particle delta (SC-only) for cross-code comparison.
            const Real Ex_pre_sc = Ex, Ey_pre_sc = Ey, Ez_pre_sc = Ez;
            const Real Bx_pre_sc = Bx, By_pre_sc = By, Bz_pre_sc = Bz;

            // Space-charge gather + per-particle self-force subtraction
            // + per-particle 1/gamma^2 boost (same pattern as kick-drift).
            if (sc) {
                const Real fx_sc = (x - lo_sc[0]) * dxi_sc[0];
                const Real fy_sc = (y - lo_sc[1]) * dxi_sc[1];
                const Real fz_sc = (z - lo_sc[2]) * dxi_sc[2];

                int ix_lo, iy_lo, iz_lo, ix_hi, iy_hi, iz_hi;
                if (sc_shape_order == 1) {
                    ix_lo = int(std::floor(fx_sc));     ix_hi = ix_lo + 1;
                    iy_lo = int(std::floor(fy_sc));     iy_hi = iy_lo + 1;
                    iz_lo = int(std::floor(fz_sc));     iz_hi = iz_lo + 1;
                } else {
                    const int ix_n = int(std::round(fx_sc));
                    const int iy_n = int(std::round(fy_sc));
                    const int iz_n = int(std::round(fz_sc));
                    ix_lo = ix_n - 1;  ix_hi = ix_n + 1;
                    iy_lo = iy_n - 1;  iy_hi = iy_n + 1;
                    iz_lo = iz_n - 1;  iz_hi = iz_n + 1;
                }
                // Multi-FAB gather: find which FAB's grown box contains the
                // particle's stencil footprint. Single-FAB case: one iteration,
                // straight match. Multi-FAB: walk the FABs (typically <30).
                int fab_idx = -1;
                for (int f = 0, nf = int(sc_fabs.boxes.size()); f < nf; ++f) {
                    auto const& bx = sc_fabs.boxes[f];
                    if (ix_lo >= bx.smallEnd(0) && ix_hi <= bx.bigEnd(0) &&
                        iy_lo >= bx.smallEnd(1) && iy_hi <= bx.bigEnd(1) &&
                        iz_lo >= bx.smallEnd(2) && iz_hi <= bx.bigEnd(2))
                    {
                        fab_idx = f;
                        break;
                    }
                }
                if (fab_idx >= 0)
                {
                    amrex::GpuArray<Real, 3> E_sc{};
                    auto const& Ex_arr_f = sc_fabs.Ex[fab_idx];
                    auto const& Ey_arr_f = sc_fabs.Ey[fab_idx];
                    auto const& Ez_arr_f = sc_fabs.Ez[fab_idx];
                    if (sc_shape_order == 1) {
                        auto E_cic = ablastr::particles::doGatherVectorFieldNodal(
                            ParticleReal(x), ParticleReal(y), ParticleReal(z),
                            Ex_arr_f, Ey_arr_f, Ez_arr_f,
                            dxi_sc, lo_sc);
                        E_sc[0] = E_cic[0]; E_sc[1] = E_cic[1]; E_sc[2] = E_cic[2];
                    } else {
                        E_sc[0] = tsc_gather(Ex_arr_f, x, y, z, dxi_sc, lo_sc);
                        E_sc[1] = tsc_gather(Ey_arr_f, x, y, z, dxi_sc, lo_sc);
                        E_sc[2] = tsc_gather(Ez_arr_f, x, y, z, dxi_sc, lo_sc);
                    }

                    const Real w  = ptd.rdata(RealSoA::w)[ip];
                    const Real qw = q * w * kCoulombConstant;

                    Real E_self[3];
                    if (m_self_force_direct) {
                        Real frac_x, frac_y, frac_z;
                        if (sc_shape_order == 1) {
                            frac_x = fx_sc - std::floor(fx_sc);
                            frac_y = fy_sc - std::floor(fy_sc);
                            frac_z = fz_sc - std::floor(fz_sc);
                        } else {
                            frac_x = fx_sc - std::round(fx_sc);
                            frac_y = fy_sc - std::round(fy_sc);
                            frac_z = fz_sc - std::round(fz_sc);
                        }
                        const Real dx_cell = Real(1.0) / dxi_sc[0];
                        const Real dy_cell = Real(1.0) / dxi_sc[1];
                        const Real dz_cell = Real(1.0) / dxi_sc[2];
                        const Real eps2    = Real(1.0e-6) * (dx_cell * dx_cell);
                        self_force_direct(sc_shape_order, frac_x, frac_y, frac_z,
                                          dx_cell, dy_cell, dz_cell, eps2, E_self);
                    } else {
                        Real tx, ty, tz;
                        if (sc_shape_order == 1) {
                            const Real wx_frac = fx_sc - std::floor(fx_sc);
                            const Real wy_frac = fy_sc - std::floor(fy_sc);
                            const Real wz_frac = fz_sc - std::floor(fz_sc);
                            tx = wx_frac * Real(sf_n);
                            ty = wy_frac * Real(sf_n);
                            tz = wz_frac * Real(sf_n);
                        } else {
                            const Real px = fx_sc - std::round(fx_sc);
                            const Real py = fy_sc - std::round(fy_sc);
                            const Real pz = fz_sc - std::round(fz_sc);
                            tx = (px + Real(0.5)) * Real(sf_n);
                            ty = (py + Real(0.5)) * Real(sf_n);
                            tz = (pz + Real(0.5)) * Real(sf_n);
                        }
                        int ix_t = int(tx); if (ix_t >= sf_n) ix_t = sf_n - 1;
                        int iy_t = int(ty); if (iy_t >= sf_n) iy_t = sf_n - 1;
                        int iz_t = int(tz); if (iz_t >= sf_n) iz_t = sf_n - 1;
                        if (ix_t < 0) ix_t = 0;
                        if (iy_t < 0) iy_t = 0;
                        if (iz_t < 0) iz_t = 0;
                        const Real ax = tx - Real(ix_t);
                        const Real ay = ty - Real(iy_t);
                        const Real az = tz - Real(iz_t);
                        const int  s1 = sf_n + 1;
                        const int  s2 = s1 * s1;
                        const int  i000 = ix_t * s2 + iy_t * s1 + iz_t;

                        for (int cc = 0; cc < 3; ++cc) {
                            const Real* T = sf_lut[cc];
                            const Real v000 = T[i000];
                            const Real v001 = T[i000 + 1];
                            const Real v010 = T[i000 + s1];
                            const Real v011 = T[i000 + s1 + 1];
                            const Real v100 = T[i000 + s2];
                            const Real v101 = T[i000 + s2 + 1];
                            const Real v110 = T[i000 + s2 + s1];
                            const Real v111 = T[i000 + s2 + s1 + 1];
                            const Real v00  = v000 + (v001 - v000) * az;
                            const Real v01  = v010 + (v011 - v010) * az;
                            const Real v10  = v100 + (v101 - v100) * az;
                            const Real v11  = v110 + (v111 - v110) * az;
                            const Real v0   = v00  + (v01  - v00 ) * ay;
                            const Real v1   = v10  + (v11  - v10 ) * ay;
                            E_self[cc] = v0 + (v1 - v0) * ax;
                        }
                    }
                    const Real kSelfForceFactor =
                        m_self_force_disabled ? Real(0.0)
                                              : (sc_shape_order == 2 ? Real(1.27)
                                                                     : Real(0.62));
                    const Real qw_f = qw * kSelfForceFactor;
                    const Real Ex_corr = E_sc[0] - qw_f * E_self[0];
                    const Real Ey_corr = E_sc[1] - qw_f * E_self[1];
                    const Real Ez_corr = E_sc[2] - qw_f * E_self[2];

                    if (m_sc_use_b_field) {
                        const Real g_b      = sc->last_gamma();
                        const Real beta_z   = sc->last_beta_z();
                        const Real inv_g2   = Real(1.0) / (g_b * g_b);
                        constexpr Real inv_c = Real(1.0) / kSpeedOfLight;
                        const Real beta_inv_c = beta_z * inv_c;

                        Ex += Ex_corr;
                        Ey += Ey_corr;
                        if (!slice_sc) {
                            Ez += Ez_corr * inv_g2;
                        }
                        Bx += -beta_inv_c * Ey_corr;
                        By += +beta_inv_c * Ex_corr;
                    } else {
                        Real sc_boost;
                        if (m_sc_boost_bunch_mean) {
                            const Real g_b = sc->last_gamma();
                            sc_boost = Real(1.0) / (g_b * g_b);
                        } else {
                            const Real u2_p = ux*ux + uy*uy + uz*uz;
                            sc_boost = Real(1.0) / (Real(1.0) + u2_p * inv_c2);
                        }
                        Ex += sc_boost * Ex_corr;
                        Ey += sc_boost * Ey_corr;
                        if (!slice_sc) {
                            Ez += sc_boost * Ez_corr;
                        }
                    }
                }
            }
            if (slice_sc) {
                Ez += slice_sc->E_z_at(z);
            }

            // Capture per-particle SC contribution + state for the dump.
            if (dump_kicks_now) {
                const std::size_t base = std::size_t(dump_my_offset + ip) * 12;
                kick_buf[base+0]  = double(x);
                kick_buf[base+1]  = double(y);
                kick_buf[base+2]  = double(z);
                kick_buf[base+3]  = double(ux);
                kick_buf[base+4]  = double(uy);
                kick_buf[base+5]  = double(uz);
                kick_buf[base+6]  = double(Ex - Ex_pre_sc);
                kick_buf[base+7]  = double(Ey - Ey_pre_sc);
                kick_buf[base+8]  = double(Ez - Ez_pre_sc);
                kick_buf[base+9]  = double(Bx - Bx_pre_sc);
                kick_buf[base+10] = double(By - By_pre_sc);
                kick_buf[base+11] = double(Bz - Bz_pre_sc);
                kick_keep[std::size_t(dump_my_offset + ip)] = 1;
            }

            // Velocity-only Boris kick (full dt). Position drift is handled
            // by separate half-drift passes around this kick.
            boris_kick_only(qm, Ex, Ey, Ez, Bx, By, Bz, dt,
                            ux, uy, uz);

            ptd.rdata(RealSoA::px)[ip] = ux;
            ptd.rdata(RealSoA::py)[ip] = uy;
            ptd.rdata(RealSoA::pz)[ip] = uz;
        }
    }

    // ---- 6b. Write per-particle SC-kick dump if scheduled at this step ----
    if (dump_kicks_now) {
        std::size_t n_alive = 0;
        for (auto k : kick_keep) if (k) ++n_alive;

        std::string path = m_dump_kicks_path_prefix
            + std::to_string(m_step_count) + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f) {
            const std::int32_t hdr[4] = {
                std::int32_t(m_step_count),
                std::int32_t(n_alive),
                std::int32_t(12),       // n_doubles_per_particle
                std::int32_t(0)         // reserved
            };
            std::fwrite(hdr, sizeof(std::int32_t), 4, f);
            const double sc_gamma  = sc ? double(sc->last_gamma())  : 1.0;
            const double sc_beta_z = sc ? double(sc->last_beta_z()) : 0.0;
            const double meta[4] = { double(t), double(dt), sc_gamma, sc_beta_z };
            std::fwrite(meta, sizeof(double), 4, f);
            for (std::size_t i = 0; i < kick_keep.size(); ++i) {
                if (!kick_keep[i]) continue;
                std::fwrite(kick_buf.data() + i * 12, sizeof(double), 12, f);
            }
            std::fclose(f);
            amrex::Print() << "[tracking.dump_kicks] wrote " << path
                           << " (step=" << m_step_count
                           << ", n_alive=" << n_alive << ")\n";
        } else {
            amrex::Print() << "[tracking.dump_kicks] WARNING: failed to open "
                           << path << " for writing\n";
        }
    }

    // ---- 7. Second half-drift + driftemission + first-order emission ----
    // Mirrors AccSimulator.f90 lines 2141-2167.
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int ip = 0; ip < np; ++ip) {
            if (ptd.idata(IntSoA::alive)[ip] == 0) continue;
            const Real ux = ptd.rdata(RealSoA::px)[ip];
            const Real uy = ptd.rdata(RealSoA::py)[ip];
            const Real uz = ptd.rdata(RealSoA::pz)[ip];
            const Real recpgam = Real(1.0) / std::sqrt(
                Real(1.0) + (ux*ux + uy*uy + uz*uz) * inv_c2);
            const Real z_pre = ptd.rdata(RealSoA::z)[ip];

            if (z_pre > cathode_z) {
                // Above cathode: half-drift at NEW velocity.
                ptd.rdata(RealSoA::x)[ip] += half_dt * ux * recpgam;
                ptd.rdata(RealSoA::y)[ip] += half_dt * uy * recpgam;
                ptd.rdata(RealSoA::z)[ip] += half_dt * uz * recpgam;
            } else if (uz >= Real(0.0)) {
                // Below cathode + forward velocity: full-step drift at
                // universal betazini (driftemission_BeamBunch).
                const Real z_after = z_pre + dt * betazini_c;
                ptd.rdata(RealSoA::z)[ip] = z_after;

                // First-order emission: did we cross cathode this step?
                // ImpactT condition: tmpzz<=0 (was below) AND z>=0 (now above).
                if (z_after >= cathode_z) {
                    const Real dtmp = (z_after - cathode_z) / betazini_c;
                    // x, y get ADDITIONAL own-velocity drift over dtmp
                    ptd.rdata(RealSoA::x)[ip] += dtmp * ux * recpgam;
                    ptd.rdata(RealSoA::y)[ip] += dtmp * uy * recpgam;
                    // z is OVERRIDDEN to own-velocity result over dtmp
                    ptd.rdata(RealSoA::z)[ip] = cathode_z + dtmp * uz * recpgam;
                }
            }
            // else: z_pre <= cathode AND uz < 0 -- frozen (no motion).
        }
    }

    // ---- 8. Wake-field impulse (post-push, Strang splitting) ----
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::WakeField>) {
                e.apply_wake(bunch, dt);
            }
        }, el);
    }

    // ---- 9. Aperture collimators (kill particles outside x/y bounds) ----
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, elements::Collimator>) {
                e.apply(bunch);
            }
        }, el);
    }
}

} // namespace tracking
} // namespace lucretiatt
