function trace_compare(varargin)
% TRACE_COMPARE  Per-step bunch-mean comparison vs ImpactT fort.18.
% Identifies the first timestep where gamma trajectories diverge.
%
% Run lucretia-tt with tracking.trace_file enabled to produce CSV;
% then compare to ImpactT fort.18 (also per-diagnostic-step).

p = inputParser;
p.addParameter('ltt_trace',    '/tmp/partcl_seed_run_trace/trace.csv', @(x) ischar(x) || isstring(x));
p.addParameter('impactt_dir',  '/tmp/impactT_noSC',                   @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

% --- Load lucretia-tt trace ---
fprintf('Loading lucretia-tt trace from %s\n', opts.ltt_trace);
T = readtable(opts.ltt_trace);
ltt.t       = T.t;
ltt.z       = T.z_mean;
ltt.sz      = T.sigma_z;
ltt.gamma   = T.gamma_mean;
ltt.sg      = T.sigma_gamma;
ltt.sx      = T.sigma_x;
ltt.vz      = T.vz_mean;
ltt.t_eff   = T.t_centroid_eff;
fprintf('  rows: %d  t span: [%.4g, %.4g] s\n', height(T), ltt.t(1), ltt.t(end));

% --- Load ImpactT fort.18 ---
imp = timetracking.readImpactTFort18(char(opts.impactt_dir));
% Trim to overlap window
tmax = min(ltt.t(end), imp.t(end));
imp_keep = imp.t <= tmax;
imp.t = imp.t(imp_keep);
imp.z = imp.z_mean(imp_keep);
imp.gamma = imp.gamma(imp_keep);
imp.sx = imp.sigma_x(imp_keep);
imp.sz = imp.sigma_z(imp_keep);

% --- Side by side at canonical times ---
fprintf('\n%-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
        't [ns]', 'ltt z[mm]', 'imp z[mm]', 'ltt gam', 'imp gam', 'd_gam', 'd_gam %');
ts = [-0.018, -0.010, 0.000, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.000, 2.000, 3.000];
for tt = ts
    tns = tt;
    tsec = tt * 1e-9;
    [~, kl] = min(abs(ltt.t - tsec));
    [~, ki] = min(abs(imp.t - tsec));
    if abs(ltt.t(kl) - tsec) > 5e-11 || abs(imp.t(ki) - tsec) > 5e-11
        continue;
    end
    dg = ltt.gamma(kl) - imp.gamma(ki);
    if imp.gamma(ki) > 1.001
        dg_pct = dg / imp.gamma(ki) * 100;
    else
        dg_pct = 0;
    end
    fprintf('%-10.3f %-10.4f %-10.4f %-10.5f %-10.5f %-10.5f %-10.4f\n', ...
            tns, ltt.z(kl)*1e3, imp.z(ki)*1e3, ltt.gamma(kl), imp.gamma(ki), dg, dg_pct);
end

% --- Plot gamma vs t and difference ---
figure('Position', [50, 50, 1200, 700]);

subplot(2,2,1);
plot(ltt.t*1e9, ltt.gamma, 'b-', 'DisplayName', 'lucretia-tt');
hold on;
plot(imp.t*1e9, imp.gamma, 'r--', 'DisplayName', 'ImpactT');
xlabel('t [ns]');  ylabel('mean gamma');  title('Bunch-mean gamma vs time');
legend('Location', 'best'); grid on;

subplot(2,2,2);
% Interpolate ImpactT onto ltt's t grid
imp_g_at_ltt = interp1(imp.t, imp.gamma, ltt.t, 'linear', 'extrap');
plot(ltt.t*1e9, ltt.gamma - imp_g_at_ltt, 'k-');
xlabel('t [ns]');  ylabel('ltt gamma - imp gamma');
title('Gamma deficit vs time'); grid on;

subplot(2,2,3);
plot(ltt.t*1e9, ltt.z*1e3, 'b-', 'DisplayName', 'lucretia-tt');
hold on;
plot(imp.t*1e9, imp.z*1e3, 'r--', 'DisplayName', 'ImpactT');
xlabel('t [ns]');  ylabel('mean z [mm]');  title('Bunch-mean z vs time');
legend('Location', 'best'); grid on;

subplot(2,2,4);
imp_z_at_ltt = interp1(imp.t, imp.z, ltt.t, 'linear', 'extrap');
plot(ltt.t*1e9, (ltt.z - imp_z_at_ltt)*1e6, 'k-');
xlabel('t [ns]');  ylabel('ltt z - imp z [um]');
title('Position deficit vs time'); grid on;

print('-dpng', '-r150', 'trace_compare.png');
fprintf('\nSaved -> trace_compare.png\n');

% --- Find FIRST step where gamma deficit > some threshold ---
defs = ltt.gamma - imp_g_at_ltt;
abs_defs = abs(defs);
first_idx = find(abs_defs > 0.05, 1);
if ~isempty(first_idx)
    fprintf('\nFIRST gamma deficit > 0.05 at step %d, t=%.4g s, z=%.4f mm:\n', ...
            first_idx, ltt.t(first_idx), ltt.z(first_idx)*1e3);
    fprintf('  ltt gamma = %.5f, imp gamma = %.5f, deficit = %g\n', ...
            ltt.gamma(first_idx), imp_g_at_ltt(first_idx), defs(first_idx));
end
% Find when deficit first reaches 1%
pct_defs = defs ./ max(imp_g_at_ltt, 1) * 100;
first_pct = find(abs(pct_defs) > 1.0, 1);
if ~isempty(first_pct)
    fprintf('FIRST |gamma deficit %%| > 1%% at step %d, t=%.4g s, z=%.4f mm:\n', ...
            first_pct, ltt.t(first_pct), ltt.z(first_pct)*1e3);
end
end
