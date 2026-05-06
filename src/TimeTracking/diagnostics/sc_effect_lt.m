function sc_effect_lt()
% Compute per-particle SC effect in lt by matching with-SC vs no-SC
% trajectories. Quantifies what our SC actually does to each particle
% over the full bench run, isolated from lattice-physics differences.

cd('/Users/glenwhite/code/Lucretia');
addpath('src/TimeTracking');
addpath('/tmp/claude');

bunches_sc = readBeamOpenPMD('/tmp/bench_full_smooth0/diags');
bunches_nosc = readBeamOpenPMD('/tmp/bench_full_nosc/diags');
b_sc = bunches_sc{end};
b_nosc = bunches_nosc{end};
fprintf('lt with-SC final: t=%g, N=%d\n', b_sc.time, numel(b_sc.x));
fprintf('lt no-SC  final: t=%g, N=%d\n', b_nosc.time, numel(b_nosc.x));

% Decode IDs
N_max = 50000;
sc_id   = double(uint64(b_sc.id))   / 2^24;   sc_id   = sc_id   - min(sc_id)   + 1;
nosc_id = double(uint64(b_nosc.id)) / 2^24;   nosc_id = nosc_id - min(nosc_id) + 1;

% Filter dead
sc_alive   = double(b_sc.z)   > -1e6;
nosc_alive = double(b_nosc.z) > -1e6;

sc_idx_by_id   = zeros(N_max, 1);
nosc_idx_by_id = zeros(N_max, 1);
for k = 1:numel(sc_id),  if sc_id(k)>=1 && sc_id(k)<=N_max && sc_alive(k),  sc_idx_by_id(sc_id(k))=k; end; end
for k = 1:numel(nosc_id), if nosc_id(k)>=1 && nosc_id(k)<=N_max && nosc_alive(k), nosc_idx_by_id(nosc_id(k))=k; end; end
matched = (sc_idx_by_id > 0) & (nosc_idx_by_id > 0);
fprintf('Matched %d / %d particles (lt with-SC alive=%d, no-SC alive=%d)\n', ...
        sum(matched), N_max, sum(sc_alive), sum(nosc_alive));

me = 9.1093837015e-31; c = 299792458;
sc_x  = double(b_sc.x(sc_idx_by_id(matched)));
sc_y  = double(b_sc.y(sc_idx_by_id(matched)));
sc_z  = double(b_sc.z(sc_idx_by_id(matched)));
sc_pxn = double(b_sc.px(sc_idx_by_id(matched))) / (me*c);
sc_pyn = double(b_sc.py(sc_idx_by_id(matched))) / (me*c);
sc_pzn = double(b_sc.pz(sc_idx_by_id(matched))) / (me*c);

nosc_x  = double(b_nosc.x(nosc_idx_by_id(matched)));
nosc_y  = double(b_nosc.y(nosc_idx_by_id(matched)));
nosc_z  = double(b_nosc.z(nosc_idx_by_id(matched)));
nosc_pxn = double(b_nosc.px(nosc_idx_by_id(matched))) / (me*c);
nosc_pyn = double(b_nosc.py(nosc_idx_by_id(matched))) / (me*c);
nosc_pzn = double(b_nosc.pz(nosc_idx_by_id(matched))) / (me*c);

dx_sc  = sc_x  - nosc_x;
dy_sc  = sc_y  - nosc_y;
dz_sc  = sc_z  - nosc_z;
dpx_sc = sc_pxn - nosc_pxn;
dpy_sc = sc_pyn - nosc_pyn;
dpz_sc = sc_pzn - nosc_pzn;

g_sc = sqrt(1 + sc_pxn.^2 + sc_pyn.^2 + sc_pzn.^2);
g_nosc = sqrt(1 + nosc_pxn.^2 + nosc_pyn.^2 + nosc_pzn.^2);
dg_sc = g_sc - g_nosc;

fprintf('\nlt SC effect per particle (with-SC - no-SC) at L0A end:\n');
fprintf('  Delta_x  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dx_sc)*1e6, std(dx_sc)*1e6, max(abs(dx_sc))*1e6);
fprintf('  Delta_y  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dy_sc)*1e6, std(dy_sc)*1e6, max(abs(dy_sc))*1e6);
fprintf('  Delta_z  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dz_sc)*1e6, std(dz_sc)*1e6, max(abs(dz_sc))*1e6);
fprintf('  Delta_px : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpx_sc), std(dpx_sc), max(abs(dpx_sc)));
fprintf('  Delta_py : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpy_sc), std(dpy_sc), max(abs(dpy_sc)));
fprintf('  Delta_pz : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpz_sc), std(dpz_sc), max(abs(dpz_sc)));
fprintf('  Delta_g  : mean=%+.4f     std=%.4f\n', mean(dg_sc), std(dg_sc));

% Bin by initial radius (use no-SC final r since SC perturbs it)
r_init = sqrt(nosc_x.^2 + nosc_y.^2);
% The SC kick accumulates radially; component along (x_no_sc, y_no_sc):
% dr_radial = (dx*x_no_sc + dy*y_no_sc) / r_no_sc → positive = SC pushed OUT
dr_rad = (dx_sc .* nosc_x + dy_sc .* nosc_y) ./ max(r_init, 1e-30);
dpr_rad = (dpx_sc .* nosc_x + dpy_sc .* nosc_y) ./ max(r_init, 1e-30);

edges = linspace(0, 2e-3, 11);
[~, ~, bin] = histcounts(r_init, edges);
fprintf('\nlt SC radial effect, binned by no-SC r (no-SC -> with-SC):\n');
fprintf('  r [mm]    N      <dr_rad> [um]  std(dr_rad) [um]   <dpr_rad>      std(dpr_rad)\n');
for k = 1:numel(edges)-1
    msk = (bin == k);
    if sum(msk) > 5
        fprintf('  %4.2f-%4.2f   %5d   %+8.0f       %8.0f       %+9.4f      %.4f\n', ...
            edges(k)*1e3, edges(k+1)*1e3, sum(msk), ...
            mean(dr_rad(msk))*1e6, std(dr_rad(msk))*1e6, ...
            mean(dpr_rad(msk)), std(dpr_rad(msk)));
    end
end

save('/Users/glenwhite/code/Lucretia/sc_effect_lt_data.mat', ...
     'sc_x', 'sc_y', 'sc_z', 'sc_pxn', 'sc_pyn', 'sc_pzn', ...
     'nosc_x', 'nosc_y', 'nosc_z', 'nosc_pxn', 'nosc_pyn', 'nosc_pzn', ...
     'dx_sc', 'dy_sc', 'dz_sc', 'dpx_sc', 'dpy_sc', 'dpz_sc', 'dg_sc');
fprintf('\nSaved -> sc_effect_lt_data.mat\n');
end
