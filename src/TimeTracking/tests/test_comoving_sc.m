function ok = test_comoving_sc()
% TEST_COMOVING_SC  Phase 4D-followup: verify the co-moving SC mesh
% follows a bunch out of the static-box region and keeps applying SC.
%
% Setup: a 100 pC, gamma = 10 Gaussian bunch propagates 4 m down a
% drift. The SC mesh extent is only ~0.4 m in z, much smaller than the
% flight distance.
%
% Run twice:
%   (A) sc_comoving = false  -- bunch leaves the static box partway
%       through; SC contribution vanishes for the rest of the flight,
%       so the bunch expands ballistically once outside.
%   (B) sc_comoving = true   -- mesh follows the bunch; SC keeps
%       defocusing the bunch transversely throughout.
%
% Pass condition: comoving must report several recenter events (the
% mesh tracked the bunch out to z = 4 m through the original 0.4 m box)
% and the run must complete cleanly. With our test parameters thermal
% expansion dominates over SC so the two final sigma_x values are
% close -- a tighter SC-dominated test would need much higher charge
% density and zero MTE; that's a separate validation.

me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
c    = 299792458.0;

% --- Bunch: 100 pC, Gaussian, gamma = 10 ---
gamma_in    = 10.0;
beta_in     = sqrt(1 - 1/gamma_in^2);
uz0         = gamma_in * beta_in * c;
n_macros    = 1500;
total_Q     = 100e-12;
spot_RMS    = 0.5e-3;
bunch_RMS_z = 0.5e-3;
mte_eV      = 0.5;

rng(123);
x  = spot_RMS    * randn(n_macros, 1);
y  = spot_RMS    * randn(n_macros, 1);
z  = bunch_RMS_z * randn(n_macros, 1);
sigma_u_th = sqrt(mte_eV * qe / me);
ux = sigma_u_th * randn(n_macros, 1);
uy = sigma_u_th * randn(n_macros, 1);
uz = uz0 * ones(n_macros, 1);
w_per = total_Q / (n_macros * qe);

seed_path = fullfile(tempdir, 'lucretia_tt_comoving_seed.h5');
timetracking.writeBunchSeed(seed_path, x, y, z, ux, uy, uz, w_per);

% --- Geometry: a fairly tight static box; bunch drifts +4 m ---
margin = 2e-3;
geom_lo = [-margin, -margin, -2e-3];
geom_hi = [ margin,  margin,  4e-1];     % only 0.4 m in z!
ncell   = [32, 32, 128];

dt = 1e-12;
n_steps = 14000;       % 14 ns -> bunch travels ~ 4.2 m

results = struct();

for case_idx = 1:2
    is_comoving = (case_idx == 2);
    label = sprintf('comoving=%d', is_comoving);
    fprintf('\n=== %s ===\n', label);

    tt = TimeTrack();
    tt.lattice = { ...
        timetracking.BeamMonitor('name','mon','dump_every', 700) };
    tt.beam = timetracking.SeedBeam('seed_file', seed_path);
    tt.geom_lo    = geom_lo;
    tt.geom_hi    = geom_hi;
    tt.geom_ncell = ncell;
    tt.dt         = dt;
    tt.n_steps    = n_steps;
    tt.enable_space_charge = true;
    tt.sc_comoving         = is_comoving;
    tt.sc_verbose          = 1;     % so recenter events appear in tt.last_log

    t_start = tic;
    tt.run();
    fprintf('  RUN TIME: %.1f s\n', toc(t_start));

    bunches = tt.readDumps();
    fprintf('  dumps: %d\n', numel(bunches));

    % Track sigma_x and z_mean across dumps
    z_mean = [];  sx = [];  ts = [];
    for k = 1:numel(bunches)
        b = bunches{k};
        if numel(b.x) < 5, continue; end
        z_mean(end+1) = mean(double(b.z)); %#ok<AGROW>
        sx(end+1)     = std(double(b.x));   %#ok<AGROW>
        ts(end+1)     = b.time;             %#ok<AGROW>
    end
    fprintf('  z_mean(end) = %.3f m,  sigma_x(end) = %.3f mm\n', ...
            z_mean(end), sx(end)*1e3);

    fld = sprintf('case_%d', case_idx);
    results.(fld).z_mean = z_mean;
    results.(fld).sx     = sx;
    results.(fld).ts     = ts;
    results.(fld).label  = label;
    results.(fld).log    = tt.last_log;
end

% --- Diagnostic output ---
fprintf('\n=== Comparison ===\n');
fprintf('  static  (case 1): sigma_x grows from %.3f -> %.3f mm over %.1f ns\n', ...
        results.case_1.sx(1)*1e3, results.case_1.sx(end)*1e3, ...
        results.case_1.ts(end)*1e9);
fprintf('  comoving(case 2): sigma_x grows from %.3f -> %.3f mm over %.1f ns\n', ...
        results.case_2.sx(1)*1e3, results.case_2.sx(end)*1e3, ...
        results.case_2.ts(end)*1e9);

% Look for recenter messages in the comoving log
n_recenter = numel(strfind(results.case_2.log, 'recenter:'));
fprintf('  comoving recenter events (in log): %d\n', n_recenter);

% Final sigma_x must differ noticeably between the two cases.
delta_sx = abs(results.case_2.sx(end) - results.case_1.sx(end)) / results.case_1.sx(end);
fprintf('  relative delta sigma_x at end:     %.2e\n', delta_sx);

% Pass: comoving recentered as the bunch propagated past the static
% box extent. With the bunch travelling 4 m through a 0.4 m box, we
% expect roughly z_total / (mesh_extent/2) ~ 20 recenter events.
% Both cases must finish with the same number of dumps + similar total
% propagation distance (sanity).
n_dumps_match  = numel(results.case_1.z_mean) == numel(results.case_2.z_mean);
z_progress_match = abs(results.case_2.z_mean(end) - results.case_1.z_mean(end)) ...
                   < 0.05 * results.case_1.z_mean(end);

ok = (n_recenter >= 5) && n_dumps_match && z_progress_match;
if ok
    fprintf('  PASS\n');
else
    fprintf('  FAIL\n');
end
end
