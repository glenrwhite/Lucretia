#include "BeamIO.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>


#ifdef LUCRETIATT_USE_OPENPMD

#include "particles/TimeBunch.H"

#include <AMReX_ParIter.H>

#include <openPMD/openPMD.hpp>

#include <cstdint>
#include <vector>


namespace lucretiatt {
namespace io {

namespace detail {

class OpenPMDImpl
{
public:
    explicit OpenPMDImpl (const std::string& path_pattern);
    void write (particles::TimeBunch& bunch, int step, amrex::Real t);

private:
    std::unique_ptr<openPMD::Series> m_series;
};


OpenPMDImpl::OpenPMDImpl (const std::string& path_pattern)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        m_series = std::make_unique<openPMD::Series>(
            path_pattern, openPMD::Access::CREATE);
        m_series->setAuthor("lucretia-tt");
        m_series->setSoftware("lucretia-tt");
    }
}


void OpenPMDImpl::write (particles::TimeBunch& bunch, int step, amrex::Real t)
{
    using namespace particles;

    // Phase 1.7: single-rank only.
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    if (!m_series) { return; }

    // ---- 1. Pull SoA into flat host vectors ----
    std::vector<amrex::ParticleReal> x_v, y_v, z_v;
    std::vector<amrex::ParticleReal> px_v, py_v, pz_v;
    std::vector<amrex::ParticleReal> q_v, w_v;
    std::vector<uint64_t>            id_v;

    using PIter = amrex::ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
    constexpr int lev = 0;
    for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
        const int np = pti.numParticles();
        if (np == 0) { continue; }

        auto& soa = pti.GetStructOfArrays();
        const auto& xs   = soa.GetRealData(RealSoA::x);
        const auto& ys   = soa.GetRealData(RealSoA::y);
        const auto& zs   = soa.GetRealData(RealSoA::z);
        const auto& uxs  = soa.GetRealData(RealSoA::px);
        const auto& uys  = soa.GetRealData(RealSoA::py);
        const auto& uzs  = soa.GetRealData(RealSoA::pz);
        const auto& qs   = soa.GetRealData(RealSoA::q);
        const auto& qms  = soa.GetRealData(RealSoA::qm);
        const auto& ws   = soa.GetRealData(RealSoA::w);
        const auto& idcpu = soa.GetIdCPUData();

        for (int i = 0; i < np; ++i) {
            x_v.push_back(xs[i]);
            y_v.push_back(ys[i]);
            z_v.push_back(zs[i]);
            // p = m * u where m = q/qm (per-particle).
            const amrex::ParticleReal m = qs[i] / qms[i];
            px_v.push_back(m * uxs[i]);
            py_v.push_back(m * uys[i]);
            pz_v.push_back(m * uzs[i]);
            q_v.push_back(qs[i]);
            w_v.push_back(ws[i]);
            id_v.push_back(idcpu[i]);
        }
    }

    const uint64_t n = x_v.size();
    if (n == 0) {
        // openPMD requires non-empty datasets in CREATE mode; skip the
        // dump entirely when the bunch is empty (e.g. before any
        // CathodeSource emission has fired).
        return;
    }
    const std::vector<amrex::ParticleReal> zero_v(n, 0.0);

    // ---- 2. Open iteration ----
    auto iter = m_series->iterations[step];
    iter.setTime(static_cast<double>(t));

    auto species = iter.particles["beam"];

    // ---- 3. Define + store records ----
    auto store_real = [&] (const std::string& record,
                           const std::string& comp,
                           const std::vector<amrex::ParticleReal>& data)
    {
        auto rc = species[record][comp];
        rc.resetDataset(openPMD::Dataset(
            openPMD::determineDatatype<amrex::ParticleReal>(),
            openPMD::Extent{n}));
        rc.storeChunkRaw(data.data(), openPMD::Offset{0}, openPMD::Extent{n});
    };

    store_real("position",       "x", x_v);
    store_real("position",       "y", y_v);
    store_real("position",       "z", z_v);
    store_real("positionOffset", "x", zero_v);
    store_real("positionOffset", "y", zero_v);
    store_real("positionOffset", "z", zero_v);
    store_real("momentum",       "x", px_v);
    store_real("momentum",       "y", py_v);
    store_real("momentum",       "z", pz_v);

    {
        auto rc = species["weighting"][openPMD::RecordComponent::SCALAR];
        rc.resetDataset(openPMD::Dataset(
            openPMD::determineDatatype<amrex::ParticleReal>(),
            openPMD::Extent{n}));
        rc.storeChunkRaw(w_v.data(), openPMD::Offset{0}, openPMD::Extent{n});
    }
    {
        auto rc = species["charge"][openPMD::RecordComponent::SCALAR];
        rc.resetDataset(openPMD::Dataset(
            openPMD::determineDatatype<amrex::ParticleReal>(),
            openPMD::Extent{n}));
        rc.storeChunkRaw(q_v.data(), openPMD::Offset{0}, openPMD::Extent{n});
    }
    {
        auto rc = species["id"][openPMD::RecordComponent::SCALAR];
        rc.resetDataset(openPMD::Dataset(
            openPMD::determineDatatype<uint64_t>(),
            openPMD::Extent{n}));
        rc.storeChunkRaw(id_v.data(), openPMD::Offset{0}, openPMD::Extent{n});
    }

    // ---- 4. Unit dimensions (openPMD spec) ----
    species["position"].setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
    species["positionOffset"].setUnitDimension({{openPMD::UnitDimension::L, 1.0}});
    species["momentum"].setUnitDimension({
        {openPMD::UnitDimension::M,  1.0},
        {openPMD::UnitDimension::L,  1.0},
        {openPMD::UnitDimension::T, -1.0}});
    species["charge"].setUnitDimension({
        {openPMD::UnitDimension::T, 1.0},
        {openPMD::UnitDimension::I, 1.0}});

    // ---- 5. macroWeighted / weightingPower attributes ----
    species["position"].setAttribute("macroWeighted",       std::uint32_t(0));
    species["position"].setAttribute("weightingPower",      0.0);
    species["positionOffset"].setAttribute("macroWeighted", std::uint32_t(0));
    species["positionOffset"].setAttribute("weightingPower", 0.0);
    species["momentum"].setAttribute("macroWeighted",       std::uint32_t(0));
    species["momentum"].setAttribute("weightingPower",      1.0);
    species["charge"].setAttribute("macroWeighted",         std::uint32_t(0));
    species["charge"].setAttribute("weightingPower",        1.0);
    species["weighting"].setAttribute("macroWeighted",      std::uint32_t(1));
    species["weighting"].setAttribute("weightingPower",     1.0);
    species["id"].setAttribute("macroWeighted",             std::uint32_t(0));
    species["id"].setAttribute("weightingPower",            0.0);

    // Synchronous flush (vectors above stay alive until this returns).
    m_series->flush();
}

} // namespace detail


OpenPMDWriter::OpenPMDWriter (const std::string& path_pattern)
    : m_impl(std::make_unique<detail::OpenPMDImpl>(path_pattern))
{}

OpenPMDWriter::~OpenPMDWriter ()                                 = default;
OpenPMDWriter::OpenPMDWriter (OpenPMDWriter&&)            noexcept = default;
OpenPMDWriter& OpenPMDWriter::operator= (OpenPMDWriter&&) noexcept = default;

void OpenPMDWriter::write (particles::TimeBunch& bunch, int step, amrex::Real t)
{ m_impl->write(bunch, step, t); }

} // namespace io
} // namespace lucretiatt

#else  // LUCRETIATT_USE_OPENPMD not defined

namespace lucretiatt {
namespace io {

void OpenPMDWriter::write (particles::TimeBunch& /*bunch*/,
                           int /*step*/, amrex::Real /*t*/)
{
    static bool warned = false;
    if (!warned && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[OpenPMDWriter] openPMD-api not compiled in "
                       << "(LUCRETIATT_USE_OPENPMD undefined); skipping.\n";
        warned = true;
    }
}

} // namespace io
} // namespace lucretiatt

#endif
