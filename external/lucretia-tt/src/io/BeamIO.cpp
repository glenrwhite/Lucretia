#include "BeamIO.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>


#ifdef LUCRETIATT_USE_OPENPMD

#include "particles/TimeBunch.H"

#include <AMReX_ParIter.H>
#include <AMReX_Vector.H>

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

    // ---- 1. Pull this rank's SoA into flat host vectors ----
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

    // ---- 1b. MPI gather to rank 0 ----
    // openPMD-api was built with serial HDF5, so only rank 0 writes. To
    // capture the full bunch we need to gather every rank's local
    // particles to rank 0 first. Single-rank: Gatherv is a no-op copy.
    const int io_root  = amrex::ParallelDescriptor::IOProcessorNumber();
    const int my_rank  = amrex::ParallelDescriptor::MyProc();
    const int n_ranks  = amrex::ParallelDescriptor::NProcs();
    const int local_n  = static_cast<int>(x_v.size());

    amrex::Vector<int> counts(n_ranks, 0);
    amrex::ParallelDescriptor::Gather(&local_n, 1, counts.data(), 1, io_root);

    amrex::Vector<int> displs;
    long total_n = 0;
    if (my_rank == io_root) {
        displs.assign(n_ranks, 0);
        total_n = counts[0];
        for (int i = 1; i < n_ranks; ++i) {
            displs[i] = displs[i-1] + counts[i-1];
            total_n += counts[i];
        }
    }

    auto gather_real = [&] (std::vector<amrex::ParticleReal>& v) {
        std::vector<amrex::ParticleReal> recv;
        if (my_rank == io_root) { recv.assign(static_cast<size_t>(total_n), 0); }
        amrex::ParallelDescriptor::Gatherv(
            v.data(), local_n,
            recv.data(), counts, displs, io_root);
        if (my_rank == io_root) { v = std::move(recv); }
    };
    auto gather_u64 = [&] (std::vector<uint64_t>& v) {
        std::vector<uint64_t> recv;
        if (my_rank == io_root) { recv.assign(static_cast<size_t>(total_n), 0); }
        amrex::ParallelDescriptor::Gatherv(
            v.data(), local_n,
            recv.data(), counts, displs, io_root);
        if (my_rank == io_root) { v = std::move(recv); }
    };
    gather_real(x_v);  gather_real(y_v);  gather_real(z_v);
    gather_real(px_v); gather_real(py_v); gather_real(pz_v);
    gather_real(q_v);  gather_real(w_v);
    gather_u64(id_v);

    if (my_rank != io_root) { return; }
    if (!m_series) { return; }

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
