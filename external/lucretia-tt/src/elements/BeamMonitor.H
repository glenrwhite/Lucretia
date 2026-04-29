#pragma once

#include "mixin/Diagnostic.H"
#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>


namespace lucretiatt {
namespace elements {

/** Time-step-triggered beam dump.
 *
 *  Writes the entire bunch state (positions, momentum, charge, weight, id)
 *  to openPMD HDF5 every `m_dump_every` steps. Phase 9 will add z-plane
 *  crossing trigger for the freeze-plane handoff. */
struct BeamMonitor
    : public Named
    , public Position3D
    , public TimeWindow
    , public Diagnostic
{
    BeamMonitor () = default;
    BeamMonitor (std::string n, int dump_every)
        : Named(std::move(n))
        , Position3D()
        , TimeWindow()
        , m_dump_every(dump_every)
    {}

    [[nodiscard]] bool should_dump (int step) const noexcept
    { return m_dump_every > 0 && (step % m_dump_every) == 0; }

    /** No external field contribution. Kept on the element for uniform
     *  std::visit dispatch with field-source elements. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/,
                     amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;
    }

    int m_dump_every = 0;
};

} // namespace elements
} // namespace lucretiatt
