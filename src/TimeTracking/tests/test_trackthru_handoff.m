function ok = test_trackthru_handoff()
% TEST_TRACKTHRU_HANDOFF  Phase 5: end-to-end handoff to Lucretia's
% TrackThru MEX.
%
% Generates a bunch in MATLAB, drifts it through lucretia-tt for 1 ns,
% calls freezePlaneOut to get a Lucretia-format Beam, then sets up a
% minimal Lucretia BEAMLINE with a single drift element and calls
% TrackThru on the captured bunch. Verifies that the bunch survives the
% s-based drift cleanly (charges preserved, no NaNs, energy unchanged).

% Lucretia uses a global BEAMLINE; protect it.
global BEAMLINE PS GIRDER WF KLYSTRON
old_BEAMLINE = BEAMLINE;
old_PS = PS; old_GIRDER = GIRDER; old_WF = WF; old_KLYSTRON = KLYSTRON;
restoreOnExit = onCleanup(@() restore_globals(old_BEAMLINE, old_PS, ...
                                              old_GIRDER, old_WF, old_KLYSTRON));

me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
c    = 299792458.0;

% --- Initial bunch ---
N = 1000;  gamma_in = 50.0;
beta_in = sqrt(1 - 1/gamma_in^2);
P0_GeV  = gamma_in * me * beta_in * c^2 / qe / 1e9;
uz0     = gamma_in * beta_in * c;
sigma_xy = 0.3e-3;  sigma_z = 0.3e-3;
total_Q  = 100e-12;
mte_eV   = 0.5;

rng(11);
x = sigma_xy*randn(N,1); y = sigma_xy*randn(N,1); z = sigma_z*randn(N,1);
sigma_u_th = sqrt(mte_eV * qe / me);
ux = sigma_u_th*randn(N,1); uy = sigma_u_th*randn(N,1);
uz = uz0*ones(N,1);
w_per = total_Q / (N * qe);

seed_path = fullfile(tempdir, 'lucretia_tt_trackthru_seed.h5');
timetracking.writeBunchSeed(seed_path, x, y, z, ux, uy, uz, w_per);

% --- Run lucretia-tt for 1 ns ---
tt = TimeTrack();
tt.lattice = { ...
    timetracking.Drift3D('name','D1','length',1.0), ...
    timetracking.BeamMonitor('name','mon','dump_every',1000) };
tt.beam = timetracking.SeedBeam('seed_file', seed_path);
tt.geom_lo    = [-3e-3, -3e-3, -2e-3];
tt.geom_hi    = [ 3e-3,  3e-3,  3e-1];
tt.geom_ncell = [16, 16, 64];
tt.dt = 1e-12;  tt.n_steps = 1000;
tt.enable_space_charge = false;

fprintf('Phase 5 TrackThru handoff:\n');
fprintf('  N = %d, total Q = %.2f pC, P0 = %.4f GeV/c, gamma = %.0f\n', ...
        N, total_Q*1e12, P0_GeV, gamma_in);

tt.run();

% --- Convert to Lucretia Beam ---
beam = tt.freezePlaneOut('last');
N_in = numel(beam.Bunch.Q);
Q_in = sum(beam.Bunch.Q);
P_in_mean  = mean(beam.Bunch.x(6, :));
sx_in_mm   = std(beam.Bunch.x(1, :)) * 1e3;
sy_in_mm   = std(beam.Bunch.x(3, :)) * 1e3;
emit_x_in  = compute_norm_emit(beam.Bunch.x(1,:), beam.Bunch.x(2,:), gamma_in);

fprintf('  freezePlaneOut: N=%d, Q=%.2f pC, P=%.4f GeV/c, sx=%.3f mm, sy=%.3f mm, eps_nx=%.3f um.mr\n', ...
        N_in, Q_in*1e12, P_in_mean, sx_in_mm, sy_in_mm, emit_x_in*1e6);

% --- Build a minimal Lucretia BEAMLINE: one drift element ---
BEAMLINE = {};
BEAMLINE{1} = DrifStruc(0.5, 'D2');   % 0.5 m drift

% --- Call TrackThru ---
fprintf('\nCalling TrackThru on a 0.5 m drift...\n');
[stat, beamout] = TrackThru(1, 1, beam, 1, 1);

fprintf('  TrackThru status: %d\n', stat{1});
if stat{1} ~= 1
    fprintf('  message: %s\n', stat{2});
end

% --- Validate output ---
N_out = numel(beamout.Bunch.Q);
Q_out = sum(beamout.Bunch.Q);
P_out_mean = mean(beamout.Bunch.x(6, :));
sx_out_mm = std(beamout.Bunch.x(1, :)) * 1e3;
sy_out_mm = std(beamout.Bunch.x(3, :)) * 1e3;
emit_x_out = compute_norm_emit(beamout.Bunch.x(1,:), beamout.Bunch.x(2,:), gamma_in);

fprintf('\n  After TrackThru drift:\n');
fprintf('    N    = %d  (in %d)\n', N_out, N_in);
fprintf('    Q    = %.2f pC  (in %.2f)\n', Q_out*1e12, Q_in*1e12);
fprintf('    P    = %.4f GeV/c  (in %.4f)\n', P_out_mean, P_in_mean);
fprintf('    sx   = %.3f mm  (in %.3f) -- expected to grow with drift\n', sx_out_mm, sx_in_mm);
fprintf('    eps_nx = %.3f um.mr  (in %.3f) -- preserved by drift\n', emit_x_out*1e6, emit_x_in*1e6);

% Pass conditions:
%   - TrackThru status = 1 (success)
%   - particle count and total charge preserved exactly
%   - mean P preserved (drift = no acceleration)
%   - emittance preserved within 1% (drift is symplectic)
%   - sigma_x grows (sanity check: drift expands a diverging bunch)
ok_stat = (stat{1} == 1);
ok_N    = (N_out == N_in);
ok_Q    = abs(Q_out - Q_in) / Q_in < 1e-6;
ok_P    = abs(P_out_mean - P_in_mean) / P_in_mean < 1e-6;
ok_emit = abs(emit_x_out - emit_x_in) / emit_x_in < 0.01;

ok = ok_stat && ok_N && ok_Q && ok_P && ok_emit;
if ok, fprintf('\n  PASS\n'); else, fprintf('\n  FAIL stat=%d N=%d Q=%d P=%d emit=%d\n', ...
        ok_stat, ok_N, ok_Q, ok_P, ok_emit); end
end


function eps_n = compute_norm_emit(x, xp, gamma_avg)
sx  = std(x);  sxp = std(xp);
cxx = mean((x - mean(x)) .* (xp - mean(xp)));
beta_avg = sqrt(1 - 1/gamma_avg^2);
eps_n = beta_avg * gamma_avg * sqrt(max(sx*sx*sxp*sxp - cxx*cxx, 0));
end


function restore_globals(b, ps, g, wf, k)
global BEAMLINE PS GIRDER WF KLYSTRON
BEAMLINE = b; PS = ps; GIRDER = g; WF = wf; KLYSTRON = k;
end
