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

    if (m_z_end <= m_z_start) return;
    if (m_xmax <= m_xmin)     return;
    if (m_ymax <= m_ymin)     return;

    using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
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
