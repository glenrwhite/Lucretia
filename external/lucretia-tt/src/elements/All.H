#pragma once

#include "BeamMonitor.H"
#include "CathodeSource.H"
#include "Drift3D.H"
#include "FieldMap3D.H"
#include "ImpactTField.H"
#include "QuadAnalytic.H"
#include "Solenoid3D.H"
#include "SolenoidAnalytic.H"
#include "SWCavity.H"
#include "TWCavity.H"
#include "UniformB.H"
#include "WakeField.H"

#include <variant>
#include <vector>


namespace lucretiatt {
namespace elements {

/** Compile-time enumeration of all element types known to lucretia-tt.
 *
 *  Each lattice is a std::vector<KnownElements>. Elements are dispatched
 *  via std::visit, which the compiler inlines into the tracking kernel
 *  (no virtuals, no vtables). */
using KnownElements = std::variant<
    Drift3D,
    BeamMonitor,
    UniformB,
    SWCavity,
    TWCavity,
    QuadAnalytic,
    SolenoidAnalytic,
    Solenoid3D,
    FieldMap3D,
    CathodeSource,
    ImpactTField,
    WakeField
>;

/** A lattice is an ordered sequence of elements. */
using Lattice = std::vector<KnownElements>;

} // namespace elements
} // namespace lucretiatt
