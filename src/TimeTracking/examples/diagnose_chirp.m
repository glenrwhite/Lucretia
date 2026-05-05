function diagnose_chirp(varargin)
% DIAGNOSE_CHIRP  Analyze the (z, gamma) chirp at gun exit in lucretia-tt
% and compare to ImpactT's fort.101 dump (z=0.954m).
%
% Usage:
%   diagnose_chirp                            % default work_dir + ImpactT
%   diagnose_chirp('work_dir', '/tmp/foo')

p = inputParser;
p.addParameter('work_dir', '/tmp/cathode_imptT_run_cathode_imptT', @(x) ischar(x) || isstring(x));
p.addParameter('impactt_dir', '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

c     = 299792458.0;
me_kg = 9.1093837015e-31;
me_GeV_c2 = me_kg * c^2 / 1.602176634e-19 / 1e9;

% --- Load lucretia-tt dumps ---
tt = TimeTrack();
tt.work_dir   = char(opts.work_dir);
tt.output_dir = 'diags';
bunches = tt.readDumps();
fprintf('lucretia-tt: %d dumps loaded from %s\n', numel(bunches), opts.work_dir);

% Use the LAST dump (deepest z reached this run)
b = bunches{end};
% Filter parked dead particles (z=-1e9 sentinel from TrackingLoop kill)
zr_all = double(b.z(:));
keep_alive = zr_all > -1.0;
n_dead = sum(~keep_alive);
if n_dead > 0
    fprintf('lucretia-tt last dump: %d killed/parked particles excluded\n', n_dead);
end
xx = double(b.x(keep_alive));  yy = double(b.y(keep_alive));  zz = double(b.z(keep_alive));
px = double(b.px(keep_alive)); py = double(b.py(keep_alive)); pz = double(b.pz(keep_alive));
N = numel(xx);
gam_ltt_raw = sqrt(1 + (px.^2 + py.^2 + pz.^2) / (me_kg*c)^2);
fprintf('lucretia-tt last dump: N(alive)=%d  step=%d\n', N, b.step);
fprintf('  RAW <z>=%.3f m  sigma_z=%.3f mm  <gamma>=%.3f  sigma_gamma=%.4f\n', ...
        mean(zz), std(zz)*1e3, mean(gam_ltt_raw), std(gam_ltt_raw));
fprintf('  RAW gamma quantiles: 1%%=%.2f  50%%=%.2f  99%%=%.2f  max=%.2f\n', ...
        quantile(gam_ltt_raw, [0.01, 0.5, 0.99]), max(gam_ltt_raw));
fprintf('  N(gamma>50): %d  N(gamma>500): %d  N(gamma>2000): %d\n', ...
        sum(gam_ltt_raw>50), sum(gam_ltt_raw>500), sum(gam_ltt_raw>2000));

% Clip runaways for the chirp comparison: keep only particles within
% +/- 5 std of the median gamma.
g_med  = median(gam_ltt_raw);
g_iqr  = iqr(gam_ltt_raw);
g_keep = gam_ltt_raw < g_med + 5*g_iqr & gam_ltt_raw > 0.5;
fprintf('  Clipped: keeping %d / %d (gamma < %.2f)\n', ...
        sum(g_keep), numel(g_keep), g_med + 5*g_iqr);
zz = zz(g_keep);  xx = xx(g_keep);  yy = yy(g_keep);
px = px(g_keep);  py = py(g_keep);  pz = pz(g_keep);
gam_ltt = gam_ltt_raw(g_keep);
fprintf('  CLIPPED <z>=%.3f m  sigma_z=%.3f mm  <gamma>=%.3f  sigma_gamma=%.4f\n', ...
        mean(zz), std(zz)*1e3, mean(gam_ltt), std(gam_ltt));

% Chirp slope: linear fit z vs gamma
% Centered for numerical stability
z_c = zz - mean(zz);
g_c = gam_ltt - mean(gam_ltt);
alpha_ltt = (z_c' * g_c) / (z_c' * z_c);     % d gamma / d z [1/m]
fprintf('  chirp d(gamma)/d(z) = %.3f /m\n', alpha_ltt);
% In units of relative-energy/z (% per mm):
fprintf('  -> %.4g %% per mm\n', alpha_ltt / mean(gam_ltt) * 100 * 1e-3);

% --- Load ImpactT fort.101 ---
fort101 = fullfile(char(opts.impactt_dir), 'fort.101');
A = readmatrix(fort101, 'FileType', 'text');
% 9 cols: x px/(mc) y py/(mc) z pz/(mc) q/m q id
xi = A(:,1);  pxi_d = A(:,2);  yi = A(:,3);  pyi_d = A(:,4);
zi = A(:,5);  pzi_d = A(:,6);
% gamma = sqrt(1 + (px_dim^2 + py_dim^2 + pz_dim^2))
gam_imp = sqrt(1 + pxi_d.^2 + pyi_d.^2 + pzi_d.^2);
fprintf('\nImpactT fort.101: N=%d\n', numel(zi));
fprintf('  <z>=%.3f m  sigma_z=%.3f mm  <gamma>=%.3f  sigma_gamma=%.4f\n', ...
        mean(zi), std(zi)*1e3, mean(gam_imp), std(gam_imp));

z_c_imp = zi - mean(zi);
g_c_imp = gam_imp - mean(gam_imp);
alpha_imp = (z_c_imp' * g_c_imp) / (z_c_imp' * z_c_imp);
fprintf('  chirp d(gamma)/d(z) = %.3f /m\n', alpha_imp);
fprintf('  -> %.4g %% per mm\n', alpha_imp / mean(gam_imp) * 100 * 1e-3);

% --- Compare chirps in same units (relative-energy slope per relative-z) ---
% Compute correlation coefficient and z-vs-gamma scatter spread
r_ltt = corr(zz, gam_ltt);
r_imp = corr(zi, gam_imp);
fprintf('\nChirp correlations:\n');
fprintf('  lucretia-tt: corr(z, gamma) = %+.4f\n', r_ltt);
fprintf('  ImpactT    : corr(z, gamma) = %+.4f\n', r_imp);
fprintf('  (sign: + means head/leading particles have higher gamma)\n');

% --- Plot ---
figure('Position', [100, 100, 1100, 480]);

subplot(1,2,1);
scatter((zz - mean(zz))*1e3, gam_ltt - mean(gam_ltt), 4, '.');
hold on;
zfit = linspace(min(zz)-mean(zz), max(zz)-mean(zz), 100);
plot(zfit*1e3, alpha_ltt*zfit, 'r-', 'LineWidth', 1.5);
xlabel('z - <z> [mm]');  ylabel('gamma - <gamma>');
title(sprintf('lucretia-tt: <z>=%.3f m  sigma_z=%.2f mm  alpha=%.2f /m', ...
              mean(zz), std(zz)*1e3, alpha_ltt));
grid on;

subplot(1,2,2);
scatter((zi - mean(zi))*1e3, gam_imp - mean(gam_imp), 4, '.');
hold on;
zfit_imp = linspace(min(zi)-mean(zi), max(zi)-mean(zi), 100);
plot(zfit_imp*1e3, alpha_imp*zfit_imp, 'r-', 'LineWidth', 1.5);
xlabel('z - <z> [mm]');  ylabel('gamma - <gamma>');
title(sprintf('ImpactT (z=%.3f m): sigma_z=%.2f mm  alpha=%.2f /m', ...
              mean(zi), std(zi)*1e3, alpha_imp));
grid on;

sgtitle('z-gamma chirp comparison (last lucretia-tt dump vs ImpactT fort.101)');

print('-dpng', '-r150', 'chirp_compare.png');
fprintf('\nSaved -> chirp_compare.png\n');
end
