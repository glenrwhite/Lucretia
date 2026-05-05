#pragma once

#include <AMReX_REAL.H>

#include <limits>


namespace lucretiatt {
namespace elements {

/** Mixin: time window during which the element is active.
 *
 *  Default window is [0, +inf): always-on. RF cavities and the cathode
 *  source override these to model finite pulses. */
struct TimeWindow
{
    TimeWindow () = default;
    TimeWindow (amrex::Real t_start, amrex::Real t_stop)
        : m_t_start(t_start), m_t_stop(t_stop) {}

    [[nodiscard]] bool active (amrex::Real t) const noexcept
    { return t >= m_t_start && t <= m_t_stop; }

    amrex::Real m_t_start = -std::numeric_limits<amrex::Real>::infinity();
    amrex::Real m_t_stop  =  std::numeric_limits<amrex::Real>::infinity();
};

} // namespace elements
} // namespace lucretiatt
