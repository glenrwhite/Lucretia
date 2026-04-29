#include "TrackingLoop.H"

#include "Boris.H"
#include "elements/CathodeSource.H"
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

    // ---- 3. Space-charge solve (once per step) ----
    Array4<const Real> Ex_sc_arr, Ey_sc_arr, Ez_sc_arr;
    GpuArray<Real, 3>  dxi_sc{}, lo_sc{};
    Box                sc_box;
    if (sc) {
        sc->solve(bunch);
        Ex_sc_arr = sc->ExArr();
        Ey_sc_arr = sc->EyArr();
        Ez_sc_arr = sc->EzArr();
        dxi_sc    = sc->dxi();
        lo_sc     = sc->lo();
        sc_box    = sc->mesh_box();
    }

    // ---- 4. Per-particle push ----
    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;

    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();

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
                // chain would otherwise carry. Soft-core to avoid the
                // singularity when wx,wy,wz lands exactly on a node.
                const Real w  = ptd.rdata(RealSoA::w)[ip];
                const Real qw = q * w * kCoulombConstant;

                const Real fx = (x - lo_sc[0]) * dxi_sc[0];
                const Real fy = (y - lo_sc[1]) * dxi_sc[1];
                const Real fz = (z - lo_sc[2]) * dxi_sc[2];
                const Real wx_frac = fx - std::floor(fx);
                const Real wy_frac = fy - std::floor(fy);
                const Real wz_frac = fz - std::floor(fz);

                const Real dx = Real(1.0) / dxi_sc[0];
                const Real dy = Real(1.0) / dxi_sc[1];
                const Real dz = Real(1.0) / dxi_sc[2];
                const Real eps2 = Real(1.0e-6) * (dx * dx);   // ~10^-3 cell

                Real Ex_self = 0.0, Ey_self = 0.0, Ez_self = 0.0;
                for (int a = 0; a < 2; ++a) {
                    const Real wa = (a == 0) ? (Real(1.0) - wx_frac) : wx_frac;
                    const Real rx_v = (wx_frac - Real(a)) * dx;
                    for (int b = 0; b < 2; ++b) {
                        const Real wb = (b == 0) ? (Real(1.0) - wy_frac) : wy_frac;
                        const Real ry_v = (wy_frac - Real(b)) * dy;
                        for (int c = 0; c < 2; ++c) {
                            const Real wc = (c == 0) ? (Real(1.0) - wz_frac) : wz_frac;
                            const Real rz_v = (wz_frac - Real(c)) * dz;
                            const Real r2 = rx_v*rx_v + ry_v*ry_v + rz_v*rz_v + eps2;
                            const Real r3_inv = Real(1.0) / (r2 * std::sqrt(r2));
                            const Real factor = qw * wa * wb * wc * r3_inv;
                            Ex_self += factor * rx_v;
                            Ey_self += factor * ry_v;
                            Ez_self += factor * rz_v;
                        }
                    }
                }

                // Empirical fudge factor: Coulomb-of-point-charges over-
                // corrects vs the smeared IGF cloud. Calibrated on the
                // uniform-sphere test (1000 macros, R = 1 mm). Replace
                // with proper IGF-self lookup table in a future polish.
                constexpr Real kSelfForceFactor = Real(0.62);
                Ex += E_sc[0] - kSelfForceFactor * Ex_self;
                Ey += E_sc[1] - kSelfForceFactor * Ey_self;
                Ez += E_sc[2] - kSelfForceFactor * Ez_self;
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
}

} // namespace tracking
} // namespace lucretiatt
