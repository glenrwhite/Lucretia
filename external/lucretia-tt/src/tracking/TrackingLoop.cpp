#include "TrackingLoop.H"

#include "Boris.H"
#include "elements/CathodeSource.H"
#include "elements/WakeField.H"
#include "particles/TimeBunch.H"
#include "spacecharge/SpaceCharge.H"

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
    spacecharge::SpaceCharge*             sc)
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
    if (sc) {
        sc->solve(bunch);
        dxi_sc = sc->dxi();
        lo_sc  = sc->lo();
        sf_n      = sc->lut_n();
        sf_lut[0] = sc->lut_ptr(0);
        sf_lut[1] = sc->lut_ptr(1);
        sf_lut[2] = sc->lut_ptr(2);
    }

    // ---- 4. Per-particle push ----
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;

    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();

        // Per-tile SC field arrays (one Array4 per FAB owned by this rank).
        Array4<const Real> Ex_sc_arr, Ey_sc_arr, Ez_sc_arr;
        Box                sc_box;
        if (sc) {
            Ex_sc_arr = sc->Ex_array(pti);
            Ey_sc_arr = sc->Ey_array(pti);
            Ez_sc_arr = sc->Ez_array(pti);
            sc_box    = sc->mesh_box(pti);
        }

        // CPU-only loop (Phase 4A). GPU portability deferred to Phase 7.
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int ip = 0; ip < np; ++ip)
        {
            Real x  = ptd.rdata(RealSoA::x)[ip];
            Real y  = ptd.rdata(RealSoA::y)[ip];
            Real z  = ptd.rdata(RealSoA::z)[ip];
            Real ux = ptd.rdata(RealSoA::px)[ip];
            Real uy = ptd.rdata(RealSoA::py)[ip];
            Real uz = ptd.rdata(RealSoA::pz)[ip];
            const Real q  = ptd.rdata(RealSoA::q)[ip];
            const Real qm = ptd.rdata(RealSoA::qm)[ip];

            // External fields from elements
            Real Ex = 0.0, Ey = 0.0, Ez = 0.0;
            Real Bx = 0.0, By = 0.0, Bz = 0.0;
            for (auto const& el : lattice) {
                Real Ex_e = 0.0, Ey_e = 0.0, Ez_e = 0.0;
                Real Bx_e = 0.0, By_e = 0.0, Bz_e = 0.0;
                std::visit([&] (auto const& e) {
                    if (e.active(t)) {
                        e.gather_E_B(x, y, z, t,
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
                Ex += E_sc[0] - qw_f * E_self[0];
                Ey += E_sc[1] - qw_f * E_self[1];
                Ez += E_sc[2] - qw_f * E_self[2];
            }
            sc_done:;

            boris_push_one(qm, Ex, Ey, Ez, Bx, By, Bz, dt,
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
