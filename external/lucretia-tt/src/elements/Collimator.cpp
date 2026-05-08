#include "Collimator.H"

#include "particles/TimeBunch.H"

#include <AMReX_OpenMP.H>
#include <AMReX_ParIter.H>


namespace lucretiatt {
namespace elements {

void Collimator::apply (particles::TimeBunch& bunch) const
{
    using namespace amrex;
    using namespace particles;

    if (m_xmax <= m_xmin && !m_kill_backward) return;
    if (m_ymax <= m_ymin && !m_kill_backward) return;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;

    // ---- fire_once mode: fire when bunch centroid crosses m_z_target ----
    // Mirrors imp's `if(distance.le.tcol .and. (distance+dzz).ge.tcol)` +
    // `lostXY_BeamBunch` (which checks ALL alive particles' xy regardless
    // of each particle's own z).
    if (m_fire_once) {
        if (m_fired) return;
        // Compute alive-particle z-centroid
        Real sum_z = Real(0.0);
        long n_alive = 0;
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            auto& soa = pti.GetStructOfArrays();
            const int np = pti.numParticles();
            const auto& zs     = soa.GetRealData(RealSoA::z);
            const auto& alives = soa.GetIntData(IntSoA::alive);
            for (int i = 0; i < np; ++i) {
                if (alives[i] == 0) continue;
                if (zs[i] < Real(-1.0e6)) continue;   // skip dead-parked
                sum_z += zs[i];
                ++n_alive;
            }
        }
        if (n_alive == 0) return;
        const Real z_centroid = sum_z / Real(n_alive);
        if (z_centroid < m_z_target) return;
        // Fire: kill all alive particles outside aperture (xy only;
        // ignore each particle's individual z).
        long n_killed = 0;
        for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
            auto ptd = pti.GetParticleTile().getParticleTileData();
            const int np = pti.numParticles();
            for (int p = 0; p < np; ++p) {
                if (ptd.idata(IntSoA::alive)[p] == 0) continue;
                const Real z = ptd.rdata(RealSoA::z)[p];
                if (z < Real(-1.0e6)) continue;
                const Real x = ptd.rdata(RealSoA::x)[p];
                const Real y = ptd.rdata(RealSoA::y)[p];
                const bool out_xy = (x < m_xmin) || (x > m_xmax)
                                 || (y < m_ymin) || (y > m_ymax);
                if (out_xy) {
                    ptd.idata(IntSoA::alive)[p] = 0;
                    ptd.rdata(RealSoA::x )[p] = Real(0.0);
                    ptd.rdata(RealSoA::y )[p] = Real(0.0);
                    ptd.rdata(RealSoA::z )[p] = Real(-1.0e9);
                    ptd.rdata(RealSoA::px)[p] = Real(0.0);
                    ptd.rdata(RealSoA::py)[p] = Real(0.0);
                    ptd.rdata(RealSoA::pz)[p] = Real(0.0);
                    ++n_killed;
                }
            }
        }
        m_fired = true;
        amrex::Print() << "[Collimator " << name() << "] fired at z_centroid="
                       << z_centroid << " m  (target=" << m_z_target
                       << "); killed " << n_killed << " particles outside ["
                       << m_xmin << "," << m_xmax << "] x ["
                       << m_ymin << "," << m_ymax << "]\n";
        return;
    }

    // ---- z-range mode (legacy / stop_bkw): per-particle z check ----
    if (m_z_end <= m_z_start) return;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int p = 0; p < np; ++p) {
            if (ptd.idata(IntSoA::alive)[p] == 0) continue;
            const Real z = ptd.rdata(RealSoA::z)[p];
            if (z < m_z_start || z > m_z_end) continue;
            const Real x  = ptd.rdata(RealSoA::x )[p];
            const Real y  = ptd.rdata(RealSoA::y )[p];
            const Real uz = ptd.rdata(RealSoA::pz)[p];
            const bool out_xy = (x < m_xmin) || (x > m_xmax)
                             || (y < m_ymin) || (y > m_ymax);
            const bool back   = m_kill_backward && (uz < Real(0.0));
            if (out_xy || back) {
                ptd.idata(IntSoA::alive)[p] = 0;
                ptd.rdata(RealSoA::x )[p] = Real(0.0);
                ptd.rdata(RealSoA::y )[p] = Real(0.0);
                ptd.rdata(RealSoA::z )[p] = Real(-1.0e9);
                ptd.rdata(RealSoA::px)[p] = Real(0.0);
                ptd.rdata(RealSoA::py)[p] = Real(0.0);
                ptd.rdata(RealSoA::pz)[p] = Real(0.0);
            }
        }
    }
}

} // namespace elements
} // namespace lucretiatt
