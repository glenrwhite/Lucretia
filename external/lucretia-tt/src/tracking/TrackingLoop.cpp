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

#include <cmath>
#include <variant>
#include <vector>


namespace lucretiatt {
namespace tracking {

namespace {

// 1 / (4 pi eps0) -- Coulomb constant in SI (m / F)
constexpr amrex::Real kCoulombConstant = amrex::Real(8.987551787e9);

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
    Array4<const Real> Ex_sc_arr, Ey_sc_arr, Ez_sc_arr;
    Box                sc_box;
    if (sc) {
        sc->solve(bunch);
        dxi_sc = sc->dxi();
        lo_sc  = sc->lo();
        sf_n      = sc->lut_n();
        sf_lut[0] = sc->lut_ptr(0);
        sf_lut[1] = sc->lut_ptr(1);
        sf_lut[2] = sc->lut_ptr(2);
        Ex_sc_arr = sc->Ex_array();
        Ey_sc_arr = sc->Ey_array();
        Ez_sc_arr = sc->Ez_array();
        sc_box    = sc->mesh_box();
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
            const Real t_phase = m_use_centroid_phase ? t_centroid : t;
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
                t_field = m_use_centroid_phase
                    ? midpoint - (t - t_phase)
                    : midpoint;
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
                const int  ix_sc = int(std::floor(fx_sc));
                const int  iy_sc = int(std::floor(fy_sc));
                const int  iz_sc = int(std::floor(fz_sc));
                if (ix_sc < sc_box.smallEnd(0) || ix_sc + 1 > sc_box.bigEnd(0) ||
                    iy_sc < sc_box.smallEnd(1) || iy_sc + 1 > sc_box.bigEnd(1) ||
                    iz_sc < sc_box.smallEnd(2) || iz_sc + 1 > sc_box.bigEnd(2))
                {
                    goto sc_done;   // exit the SC block; keep external E,B
                }
                auto E_sc = ablastr::particles::doGatherVectorFieldNodal(
                    ParticleReal(x), ParticleReal(y), ParticleReal(z),
                    Ex_sc_arr, Ey_sc_arr, Ez_sc_arr,
                    dxi_sc, lo_sc);

                // Subtract Coulomb field of this particle's own deposited
                // CIC cloud (8 weighted point charges at the surrounding
                // nodes). Removes the self-interaction the IGF + gather
                // chain would otherwise carry. Implementation: trilinear
                // interp of a precomputed shape table (built once per
                // SpaceCharge in build_self_force_lut, see SpaceCharge.cpp)
                // indexed on the particle's fractional cell position. The
                // table holds E per unit qw=1 -- the per-particle qw
                // factor is applied after the lookup. ~20 cycles +
                // 8 cached reads per particle vs ~150 cycles for the
                // 8 sqrt+div sum that this replaces.
                const Real w  = ptd.rdata(RealSoA::w)[ip];
                const Real qw = q * w * kCoulombConstant;

                const Real fx = (x - lo_sc[0]) * dxi_sc[0];
                const Real fy = (y - lo_sc[1]) * dxi_sc[1];
                const Real fz = (z - lo_sc[2]) * dxi_sc[2];
                const Real wx_frac = fx - std::floor(fx);
                const Real wy_frac = fy - std::floor(fy);
                const Real wz_frac = fz - std::floor(fz);

                const Real tx = wx_frac * Real(sf_n);
                const Real ty = wy_frac * Real(sf_n);
                const Real tz = wz_frac * Real(sf_n);
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

                Real E_self[3];
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

                // Empirical fudge factor: Coulomb-of-point-charges over-
                // corrects vs the smeared IGF cloud. Calibrated on the
                // uniform-sphere test (1000 macros, R = 1 mm).
                constexpr Real kSelfForceFactor = Real(0.62);
                const Real qw_f = qw * kSelfForceFactor;
                // sc_boost = 1/gamma = sqrt(1 - beta^2) — see comment at
                // SC solve. Both the gathered IGF field and the self-force
                // subtraction get the same boost-back factor (they are
                // both computed in the bunch rest frame, with the same
                // stretched-z geometry).
                // Per-particle 1/gamma^2 boost (infinite-bin limit of
                // ImpactT-style energy binning): use THIS particle's
                // u^2 = ux^2 + uy^2 + uz^2 to compute its own gamma,
                // not the bunch-mean. Critical for chirped bunches
                // where head and tail differ by a factor of two or
                // more in gamma — applying mean-gamma boost gives
                // wrong forces to the off-mean particles and inflates
                // slice emittance with mesh-noise spread.
                constexpr Real c     = kSpeedOfLight;
                constexpr Real inv_c2 = Real(1.0) / (c * c);
                const Real u2_p     = ux*ux + uy*uy + uz*uz;
                const Real sc_boost = Real(1.0) / (Real(1.0) + u2_p * inv_c2);   // 1/gamma^2
                Ex += sc_boost * (E_sc[0] - qw_f * E_self[0]);
                Ey += sc_boost * (E_sc[1] - qw_f * E_self[1]);
                if (!slice_sc) {
                    // Mesh-only mode: longitudinal from the 3D IGF
                    // (with same boost factor as transverse).
                    Ez += sc_boost * (E_sc[2] - qw_f * E_self[2]);
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
}

} // namespace tracking
} // namespace lucretiatt
