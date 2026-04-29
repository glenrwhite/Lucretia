#pragma once

#include <AMReX_REAL.H>


namespace lucretiatt {
namespace elements {

/** Mixin: lab-frame position + length of a time-based element.
 *
 *  (m_x, m_y, m_z) is the upstream (low-z) face centre; the element extends
 *  from m_z to m_z + m_length. Orientation is axis-aligned for now; later
 *  phases can add a rotation matrix mixin without disturbing this one. */
struct Position3D
{
    Position3D () = default;
    Position3D (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real length)
        : m_x(x), m_y(y), m_z(z), m_length(length) {}

    /** Inclusive z-range test. Used to short-circuit field gather. */
    [[nodiscard]] bool contains_z (amrex::Real z) const noexcept
    { return z >= m_z && z <= m_z + m_length; }

    amrex::Real m_x      = 0.0;
    amrex::Real m_y      = 0.0;
    amrex::Real m_z      = 0.0;
    amrex::Real m_length = 0.0;
};

} // namespace elements
} // namespace lucretiatt
