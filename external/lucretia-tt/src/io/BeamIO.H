#pragma once

#include "particles/TimeBunch.H"

#include <AMReX_REAL.H>

#include <memory>
#include <string>


namespace lucretiatt {
namespace io {

#ifdef LUCRETIATT_USE_OPENPMD

namespace detail { class OpenPMDImpl; }

/** Writer for time-stepped openPMD particle dumps.
 *
 *  Single instance per output series; call write() at each diagnostic
 *  step. fileBased iteration encoding: each step gets its own file
 *  (e.g. diags/openpmd_0000010.h5). Phase 1.7 is single-rank only;
 *  multi-rank parallel HDF5 lands in Phase 6. */
class OpenPMDWriter
{
public:
    explicit OpenPMDWriter (const std::string& path_pattern);
    ~OpenPMDWriter();

    OpenPMDWriter (const OpenPMDWriter&)            = delete;
    OpenPMDWriter& operator= (const OpenPMDWriter&) = delete;
    OpenPMDWriter (OpenPMDWriter&&)            noexcept;
    OpenPMDWriter& operator= (OpenPMDWriter&&) noexcept;

    /** Write the bunch state at iteration `step`, simulation time `t` (s). */
    void write (particles::TimeBunch& bunch, int step, amrex::Real t);

private:
    std::unique_ptr<detail::OpenPMDImpl> m_impl;
};

#else  // LUCRETIATT_USE_OPENPMD not defined

class OpenPMDWriter
{
public:
    explicit OpenPMDWriter (const std::string& /*path_pattern*/) {}
    void write (particles::TimeBunch& /*bunch*/, int /*step*/, amrex::Real /*t*/);
};

#endif

} // namespace io
} // namespace lucretiatt
