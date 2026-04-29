#pragma once


namespace lucretiatt {
namespace elements {

/** Marker mixin: element produces output but contributes no field.
 *
 *  Used to short-circuit field gather and to drive run-loop diagnostic
 *  dispatch (e.g. BeamMonitor invokes the writer instead of the pusher). */
struct Diagnostic {};

} // namespace elements
} // namespace lucretiatt
