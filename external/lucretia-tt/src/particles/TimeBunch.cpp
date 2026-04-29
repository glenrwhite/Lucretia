#include "TimeBunch.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>


namespace lucretiatt {
namespace particles {

namespace {
    constexpr amrex::ParticleReal kElectronCharge = -1.602176634e-19;   // C
    constexpr amrex::ParticleReal kElectronMass   =  9.1093837015e-31;  // kg
}


TimeBunch::TimeBunch (const amrex::Geometry& geom,
                      const amrex::DistributionMapping& dm,
                      const amrex::BoxArray& ba)
    : TimeBunchBase(geom, dm, ba)
{
    SetSoACompileTimeNames(
        {"x", "y", "z", "px", "py", "pz", "q", "qm", "w", "t_birth"},
        {"alive"});
}


void TimeBunch::AddParticles (
    int                 n,
    amrex::ParticleReal x,  amrex::ParticleReal y,  amrex::ParticleReal z,
    amrex::ParticleReal ux, amrex::ParticleReal uy, amrex::ParticleReal uz,
    amrex::ParticleReal weight)
{
    using namespace amrex;

    if (ParallelDescriptor::IOProcessor() && n > 0)
    {
        auto& ptile = DefineAndReturnParticleTile(0, 0, 0);
        const int old_size = ptile.numParticles();
        ptile.resize(old_size + n);

        auto& soa = ptile.GetStructOfArrays();
        uint64_t* const idcpu = soa.GetIdCPUData().data() + old_size;

        const Long pid_base = ParticleType::NextID();
        ParticleType::NextID(pid_base + n);
        const int cpuid = ParallelDescriptor::MyProc();

        auto ptd = ptile.getParticleTileData();

        for (int i = 0; i < n; ++i)
        {
            const int idx = old_size + i;
            idcpu[i] = SetParticleIDandCPU(pid_base + i, cpuid);

            ptd.rdata(RealSoA::x)[idx]       = x;
            ptd.rdata(RealSoA::y)[idx]       = y;
            ptd.rdata(RealSoA::z)[idx]       = z;
            ptd.rdata(RealSoA::px)[idx]      = ux;
            ptd.rdata(RealSoA::py)[idx]      = uy;
            ptd.rdata(RealSoA::pz)[idx]      = uz;
            ptd.rdata(RealSoA::q)[idx]       = kElectronCharge;
            ptd.rdata(RealSoA::qm)[idx]      = kElectronCharge / kElectronMass;
            ptd.rdata(RealSoA::w)[idx]       = weight;
            ptd.rdata(RealSoA::t_birth)[idx] = ParticleReal(0.0);
            ptd.idata(IntSoA::alive)[idx]    = 1;
        }
    }
    Redistribute();
}


void TimeBunch::AddParticlesFromArrays (
    int                        n,
    const amrex::ParticleReal* xs,  const amrex::ParticleReal* ys,  const amrex::ParticleReal* zs,
    const amrex::ParticleReal* uxs, const amrex::ParticleReal* uys, const amrex::ParticleReal* uzs,
    amrex::ParticleReal        weight,
    amrex::ParticleReal        t_birth)
{
    using namespace amrex;

    if (ParallelDescriptor::IOProcessor() && n > 0)
    {
        auto& ptile = DefineAndReturnParticleTile(0, 0, 0);
        const int old_size = ptile.numParticles();
        ptile.resize(old_size + n);

        auto& soa = ptile.GetStructOfArrays();
        uint64_t* const idcpu = soa.GetIdCPUData().data() + old_size;

        const Long pid_base = ParticleType::NextID();
        ParticleType::NextID(pid_base + n);
        const int cpuid = ParallelDescriptor::MyProc();

        auto ptd = ptile.getParticleTileData();

        for (int i = 0; i < n; ++i)
        {
            const int idx = old_size + i;
            idcpu[i] = SetParticleIDandCPU(pid_base + i, cpuid);

            ptd.rdata(RealSoA::x)[idx]       = xs[i];
            ptd.rdata(RealSoA::y)[idx]       = ys[i];
            ptd.rdata(RealSoA::z)[idx]       = zs[i];
            ptd.rdata(RealSoA::px)[idx]      = uxs[i];
            ptd.rdata(RealSoA::py)[idx]      = uys[i];
            ptd.rdata(RealSoA::pz)[idx]      = uzs[i];
            ptd.rdata(RealSoA::q)[idx]       = kElectronCharge;
            ptd.rdata(RealSoA::qm)[idx]      = kElectronCharge / kElectronMass;
            ptd.rdata(RealSoA::w)[idx]       = weight;
            ptd.rdata(RealSoA::t_birth)[idx] = t_birth;
            ptd.idata(IntSoA::alive)[idx]    = 1;
        }
    }
    Redistribute();
}


void TimeBunch::PrintSummary (const std::string& tag)
{
    const long my_n    = NumberOfParticlesAtLevel(0, /*only_valid*/ true,
                                                  /*only_local*/ true);
    const long total_n = TotalNumberOfParticles(/*only_valid*/ true,
                                                /*only_local*/ false);

    amrex::Print() << "[TimeBunch:" << tag << "] total particles: "
                   << total_n << "\n";
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        amrex::AllPrint() << "[TimeBunch:" << tag << "]   rank "
                          << amrex::ParallelDescriptor::MyProc()
                          << " holds " << my_n << "\n";
    }
}

} // namespace particles
} // namespace lucretiatt
