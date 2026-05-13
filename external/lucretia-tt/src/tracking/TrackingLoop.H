#pragma once

#include "elements/All.H"
#include "particles/TimeBunch.H"

#include <AMReX_REAL.H>

#include <string>
#include <vector>


namespace lucretiatt {

namespace spacecharge {
    class SpaceCharge;
    class SpaceChargeSlice;
}

namespace tracking {

/** Time-stepping orchestrator.
 *
 *  Phase 4A: gathers E,B from each lattice element via std::visit, adds
 *  optional space-charge E from a SpaceCharge solver, and calls Boris
 *  per particle. Image-charge contribution from CathodeSource elements
 *  is also added inline. */
class TrackingLoop
{
public:
    /** Advance `bunch` by one timestep dt at simulation time t.
     *
     *  When `sc` is non-null, a Poisson solve is run once at the start
     *  of the step and per-particle E_sc is gathered and added to the
     *  external-field E before the Boris call.
     *
     *  When `slice_sc` is non-null, the 1D longitudinal slice SC is
     *  computed and its E_z REPLACES the longitudinal component of the
     *  3D mesh SC contribution (mesh handles transverse only, slice
     *  handles longitudinal). The slice model is mesh-free in z and
     *  stays resolved even when the bunch compresses below one mesh
     *  cell — which is the normal case once a photoinjector bunch
     *  crosses the gun. */
    void step (
        particles::TimeBunch&                 bunch,
        const elements::Lattice&              lattice,
        amrex::Real                           t,
        amrex::Real                           dt,
        spacecharge::SpaceCharge*             sc       = nullptr,
        spacecharge::SpaceChargeSlice*        slice_sc = nullptr);

    /** Enable ImpactT-style behind-cathode drift. Particles with
     *  z < cathode_z advance at the universal speed `betazini * c`
     *  (no field gather, no Boris push, no SC contribution). Once a
     *  particle's z >= cathode_z, normal tracking takes over.
     *
     *  This synchronizes the cross-cathode arrival of particles emitted
     *  with thermal pz spread: instead of dispersing over ~30 ps (and
     *  ~30° of RF phase) the way per-particle drift would, all particles
     *  arrive at z=0 within ~ z_init_spread / (betazini*c) of each other
     *  -- typically ~1-2 ps for the LCLS injector setup -- which matches
     *  ImpactT's `driftemission_BeamBunch` behaviour.
     *
     *  Default off (cathode_z = -inf, betazini = 0). */
    void set_behind_cathode_drift (amrex::Real cathode_z,
                                   amrex::Real betazini) noexcept
    {
        m_cathode_z                = cathode_z;
        m_behind_cathode_betazini  = betazini;
        m_behind_cathode_drift_on  = true;
    }
    bool behind_cathode_drift_enabled () const noexcept { return m_behind_cathode_drift_on; }

    /** Use ImpactT-style centroid-z reference time for external-field
     *  gather. ImpactT line 1436 of AccSimulator.f90 sets
     *  refptcl(5) = sgcenter(5) (bunch z-centroid in normalized units),
     *  then field gather uses tg = refptcl(5). The effective field
     *  formula becomes `cos(omega * z_centroid/c + phase)` -- a SPATIAL
     *  phase tied to the bunch's progress through the lattice, NOT
     *  wallclock time. For relativistic bunches z_centroid/c == t (modulo
     *  offset) so both conventions agree. During the cathode region
     *  (beta<1), they diverge by tens of degrees of RF phase across the
     *  emission window, which appears to be the source of the chirp-
     *  sign reversal seen in the no-SC sanity check vs ImpactT.
     *
     *  When enabled, TrackingLoop computes a per-step centroid z and
     *  uses t_eff = z_centroid/c + t_offset for the field gather time
     *  (replacing wallclock t). t_offset is set so t_eff matches
     *  wallclock t at the moment of bunch emission start (typically
     *  Tini), preserving the RF phase reference. */
    void set_centroid_field_phase (bool on, amrex::Real t_offset = amrex::Real(0.0)) noexcept
    {
        m_use_centroid_phase = on;
        m_centroid_t_offset  = t_offset;
    }
    bool centroid_field_phase_enabled () const noexcept { return m_use_centroid_phase; }

    /** Use midstep particle position for external-field gather (ImpactT
     *  compat). ImpactT's main loop does half_drift -> gather -> Boris ->
     *  half_drift, so the field is evaluated at x_n+1/2 = x_n + 0.5*dt*v_n.
     *  Lucretia-tt's default scheme gathers at x_n (start of step), then
     *  Boris + full drift. The half-step position offset (~0.045 mm at
     *  v=c, dt=0.3ps) times the field gradient can introduce a sub-percent
     *  per-step error that accumulates over many steps -- a likely source
     *  of the 3x chirp-slope residual vs ImpactT in the no-SC partcl-seed
     *  test. Default off; enable via tracking.use_midstep_field = 1. */
    void set_midstep_field (bool on) noexcept { m_use_midstep_field = on; }
    bool midstep_field_enabled () const noexcept { return m_use_midstep_field; }

    /** Use PER-PARTICLE phase: each particle sees cos(omega*z_i/c + phi)
     *  where z_i is the particle's own z. This is in contrast to the
     *  centroid-phase mode (cos(omega*z_centroid/c + phi), same for all
     *  particles) and the wallclock mode (cos(omega*t + phi)).
     *
     *  Per-particle phase adds a phase-derivative term to the chirp source:
     *  with centroid phase, only the spatial gradient dE_z/dz contributes
     *  to chirp (dgamma/dz ~ q/m * dE_z/dz * cos). With per-particle phase,
     *  there is also -E_z * (omega/c) * sin (the phase derivative across
     *  the bunch). For LCLS gun phase 305 deg these terms can partially
     *  cancel, reducing chirp by a factor of 2-3. Test mode for the no-SC
     *  partcl-seed comparison vs ImpactT.
     *
     *  Default off. Mutually exclusive with centroid-phase mode. */
    void set_particle_phase (bool on) noexcept { m_use_particle_phase = on; }
    bool particle_phase_enabled () const noexcept { return m_use_particle_phase; }

    /** Use BUNCH-MEAN gamma instead of per-particle gamma for the SC
     *  Lorentz boost-back factor (1/gamma^2). Default true (matches the
     *  IGF solver's stretched-z mesh convention: the solver stretches by
     *  bunch-mean gamma, so the boost-back must use the same gamma).
     *
     *  Setting false reverts to per-particle gamma — formally the
     *  "infinite-bin energy-binning limit" but with only a SINGLE solve
     *  it's an inconsistent approximation that empirically over-amplifies
     *  the head/tail energy spread of chirped bunches in L0A acceleration
     *  (sigma_gamma 4× too big at LCLS L0A exit). */
    void set_sc_boost_bunch_mean (bool on) noexcept { m_sc_boost_bunch_mean = on; }
    bool sc_boost_bunch_mean_enabled () const noexcept { return m_sc_boost_bunch_mean; }

    /** Compute the SC self-force subtraction term DIRECTLY from cell
     *  sizes at gather time, instead of looking up a pre-built table.
     *
     *  The LUT (built once at SpaceCharge construction, rebuilt on
     *  cell aspect ratio change >5%) returns the Coulomb field of the
     *  particle's own deposited cloud at its position, normalised by
     *  qw*kCoulomb. With adaptive mesh the cell sizes change every step,
     *  causing repeated LUT rebuilds and resize-discontinuity noise that
     *  inflates per-particle SC kicks (eps_nx grows).
     *
     *  Direct mode computes the same Coulomb sum (8 nodes for CIC,
     *  27 for TSC) at gather time using the CURRENT cell sizes. ~7x
     *  more per-particle work than the LUT lookup but produces no
     *  rebuild noise. Result is bit-equivalent to a freshly-built LUT.
     *
     *  Default off (LUT used). */
    void set_self_force_direct (bool on) noexcept { m_self_force_direct = on; }
    bool self_force_direct_enabled () const noexcept { return m_self_force_direct; }

    /** Apply SC magnetic field explicitly via the Boris pusher (matches
     *  ImpactT's algorithm) instead of the 1/gamma^2 boost shortcut.
     *
     *  Mathematically equivalent for SYNCHRONOUS particles (transverse
     *  v=0; v_z=beta*c): both give F_x_lab = q*E_x_lab/gamma^2. For
     *  NON-SYNCHRONOUS particles, only the explicit-B path captures the
     *  v×B coupling -- in particular the longitudinal kick term
     *      F_z_extra = q*(beta/c)*(v_x*E_x_lab + v_y*E_y_lab)
     *  that ImpactT applies but the boost shortcut omits. This is a
     *  chromatic-emittance-relevant coupling for the L0A regime.
     *
     *  When on, gathered fields go to the Boris pusher as:
     *    Ex_in = Ex_solver         (= E_x_lab; transverse: solver gives γ*E_x_rest=E_x_lab)
     *    Ey_in = Ey_solver         (= E_y_lab)
     *    Ez_in = Ez_solver / γ²    (= E_z_lab; longitudinal: solver gives γ²*E_z_lab)
     *    Bx_sc = -(β/c) * Ey_solver
     *    By_sc = +(β/c) * Ex_solver
     *    Bz_sc = 0
     *  Boris's v×B then reproduces the 1/γ² synchronous reduction
     *  automatically and adds the non-synchronous coupling.
     *
     *  Default: off (keeps the existing boost shortcut for backward
     *  comparison). */
    void set_sc_use_b_field (bool on) noexcept { m_sc_use_b_field = on; }
    bool sc_use_b_field_enabled () const noexcept { return m_sc_use_b_field; }

    /** Per-particle SC-kick dump. When the integrator's internal step
     *  counter equals one of `steps`, write all alive particles' state +
     *  SC field/momentum-kick contributions to
     *    {path_prefix}{N}.bin
     *  for cross-code comparison with imp's lt_cmp_dump_particles
     *  (mesh fields are dumped separately by SpaceCharge::set_dump_field_at_step).
     *
     *  Per-particle record (96 bytes = 12 doubles): x, y, z, ux, uy, uz,
     *  dE_sc_x, dE_sc_y, dE_sc_z, dB_sc_x, dB_sc_y, dB_sc_z. The dE/dB are
     *  the increments that SC actually contributed to the total fields fed
     *  into the Boris kick (i.e., the IGF gather - self-force, with
     *  whatever boost / B-field convention is active, plus any slice SC).
     *
     *  Header (48 bytes): int32 step, int32 n_alive, int32 n_per_part=12,
     *  int32 reserved=0; double t, double dt, double sc_gamma, double sc_beta_z.
     *
     *  Default off (empty steps list). Use only on the DKD integrator path
     *  (the legacy step() is not instrumented). */
    void set_dump_kicks_at_steps (std::vector<int> const& steps,
                                  std::string const& path_prefix
                                      = std::string("/tmp/lt_part_kicks_step")) noexcept
    {
        m_dump_kicks_at_steps    = steps;
        m_dump_kicks_path_prefix = path_prefix;
    }
    /** Time-based variant of set_dump_kicks_at_steps. Triggers a dump on
     *  the FIRST step where wallclock simulation time t reaches >= one of
     *  the listed times (each time fires at most once). Useful for cross-
     *  code comparison when the two integrators use different dt schedules
     *  -- step indices won't align, but physical times will. The dump file
     *  uses the actual triggered step index in its filename. */
    void set_dump_kicks_at_times (std::vector<double> const& times,
                                  std::string const& path_prefix
                                      = std::string("/tmp/lt_part_kicks_step")) noexcept
    {
        m_dump_kicks_at_times    = times;
        m_dump_kicks_path_prefix = path_prefix;
        m_dump_kicks_times_fired.assign(times.size(), false);
    }
    int n_steps_taken () const noexcept { return m_step_count; }

    /** Use ImpactT-style drift-kick-drift integrator with first-order
     *  emission model. Replaces the default kick-drift scheme. The DKD
     *  step is:
     *    1. Half-drift (above-cathode particles only, OLD velocity)
     *    2. Field gather at midstep position + midstep time (t + 0.5*dt)
     *    3. Boris kick (full dt, velocity-only — no position drift)
     *    4. Half-drift (above-cathode particles only, NEW velocity)
     *    5. driftemission: below-cathode + pz>=0 particles drift FULL dt
     *       at universal betazini
     *    6. First-order emission: particles that crossed cathode this step
     *       get z reset to (z/betazini) * pz/gamma (own-velocity drift)
     *
     *  Differences from kick-drift mode:
     *   - Field is gathered at midstep position + time (not start-of-step)
     *   - No fractional Boris kick on cathode-crossing step (Boris waits
     *     until next step when particle is fully above cathode)
     *   - Below-cathode + pz<0 particles are FROZEN (don't drift)
     *
     *  Together with use_centroid_phase=false (i.e., wallclock cos), this
     *  matches ImpactT's main integration loop almost exactly.
     *
     *  Default off; enable via tracking.use_dkd_integrator = 1. */
    void set_dkd_integrator (bool on) noexcept { m_use_dkd_integrator = on; }
    bool dkd_integrator_enabled () const noexcept { return m_use_dkd_integrator; }

private:
    void step_dkd (
        particles::TimeBunch&                 bunch,
        const elements::Lattice&              lattice,
        amrex::Real                           t,
        amrex::Real                           dt,
        spacecharge::SpaceCharge*             sc,
        spacecharge::SpaceChargeSlice*        slice_sc);

    bool        m_behind_cathode_drift_on = false;
    amrex::Real m_cathode_z               = 0.0;
    amrex::Real m_behind_cathode_betazini = 0.0;
    bool        m_use_centroid_phase      = false;
    amrex::Real m_centroid_t_offset       = 0.0;
    bool        m_use_midstep_field       = false;
    bool        m_use_particle_phase      = false;
    bool        m_use_dkd_integrator      = false;
    bool        m_sc_boost_bunch_mean     = true;   // default: use sc->last_gamma() for 1/gamma^2 boost
    bool        m_self_force_disabled     = false;  // diagnostic: skip the self-force LUT subtraction
    bool        m_self_force_direct       = false;  // off: LUT lookup; on: direct Coulomb sum each step (no rebuild noise w/ adaptive mesh)
    bool        m_sc_use_b_field          = false;  // off: 1/gamma^2 boost shortcut; on: explicit B + Boris (matches ImpactT)
    int         m_step_count              = 0;      // incremented by step() / step_dkd() before any work
    std::vector<int>    m_dump_kicks_at_steps;      // step indices at which to dump per-particle SC kicks
    std::vector<double> m_dump_kicks_at_times;      // physical times (s) at which to dump (fires on first step with t>=target)
    std::vector<bool>   m_dump_kicks_times_fired;   // mirror of m_dump_kicks_at_times: true once fired
    std::string m_dump_kicks_path_prefix  = "/tmp/lt_part_kicks_step";

    // Per-particle trajectory trace: write selected particles' phase-space
    // state every m_trace_interval steps to a binary file for cross-code
    // comparison. Each record: int32 step, double t, int32 n_traced, then
    // n_traced × 7 doubles (global_idx, x, y, z, ux, uy, uz).
    std::vector<int> m_trace_ids;          // global particle indices to trace
    int              m_trace_interval = 0; // write every N steps (0 = off)
    std::string      m_trace_path     = "/tmp/lt_particle_trace.bin";
    FILE*            m_trace_file     = nullptr;
public:
    void set_self_force_disabled (bool on) noexcept { m_self_force_disabled = on; }
    bool self_force_disabled () const noexcept { return m_self_force_disabled; }

    /** Enable per-particle trajectory trace. ids = global particle indices
     *  to track, interval = steps between writes, path = output file. */
    void set_particle_trace (const std::vector<int>& ids, int interval,
                             const std::string& path = "/tmp/lt_particle_trace.bin") noexcept
    {
        m_trace_ids      = ids;
        m_trace_interval = interval;
        m_trace_path     = path;
    }
};

} // namespace tracking
} // namespace lucretiatt
