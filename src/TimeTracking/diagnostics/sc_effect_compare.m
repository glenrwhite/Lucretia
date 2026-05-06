function sc_effect_compare()
% Compare per-particle SC effect (with-SC - no-SC) between lt and imp.
% Final answer: is our SC magnitude similar to imp's, or systematically off?

cd('/Users/glenwhite/code/Lucretia');
addpath('src/TimeTracking');
addpath('/tmp/claude');

% --- lt SC effect (already computed in sc_effect_lt) ---
sc_effect_lt();
ld = load('sc_effect_lt_data.mat');
fprintf('\n=========================================\n');

% --- imp SC effect: read fort.50 from with-SC run + no-SC run ---
read_imp = @(d) read_imp_fort50(fullfile(d, 'fort.50'));
[imp_sc,   imp_sc_id]   = read_imp('/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT');
[imp_nosc, imp_nosc_id] = read_imp(fullfile(getenv('TMPDIR'), 'imp_nosc'));
fprintf('imp with-SC: N=%d (collimator-killed: %d)\n', size(imp_sc, 1), 50000 - size(imp_sc, 1));
fprintf('imp no-SC  : N=%d (collimator-killed: %d)\n', size(imp_nosc, 1), 50000 - size(imp_nosc, 1));

N_max = 50000;
sc_idx_by_id   = zeros(N_max, 1);
nosc_idx_by_id = zeros(N_max, 1);
for k = 1:size(imp_sc, 1)
    if imp_sc_id(k) >= 1 && imp_sc_id(k) <= N_max, sc_idx_by_id(imp_sc_id(k)) = k; end
end
for k = 1:size(imp_nosc, 1)
    if imp_nosc_id(k) >= 1 && imp_nosc_id(k) <= N_max, nosc_idx_by_id(imp_nosc_id(k)) = k; end
end
matched = (sc_idx_by_id > 0) & (nosc_idx_by_id > 0);
fprintf('Matched in imp: %d\n', sum(matched));

imp_sc_x   = imp_sc(sc_idx_by_id(matched),   1);
imp_sc_y   = imp_sc(sc_idx_by_id(matched),   3);
imp_sc_z   = imp_sc(sc_idx_by_id(matched),   5);
imp_sc_pxn = imp_sc(sc_idx_by_id(matched),   2);
imp_sc_pyn = imp_sc(sc_idx_by_id(matched),   4);
imp_sc_pzn = imp_sc(sc_idx_by_id(matched),   6);
imp_nosc_x   = imp_nosc(nosc_idx_by_id(matched),   1);
imp_nosc_y   = imp_nosc(nosc_idx_by_id(matched),   3);
imp_nosc_z   = imp_nosc(nosc_idx_by_id(matched),   5);
imp_nosc_pxn = imp_nosc(nosc_idx_by_id(matched),   2);
imp_nosc_pyn = imp_nosc(nosc_idx_by_id(matched),   4);
imp_nosc_pzn = imp_nosc(nosc_idx_by_id(matched),   6);

dx_imp  = imp_sc_x  - imp_nosc_x;
dy_imp  = imp_sc_y  - imp_nosc_y;
dpx_imp = imp_sc_pxn - imp_nosc_pxn;
dpy_imp = imp_sc_pyn - imp_nosc_pyn;
dpz_imp = imp_sc_pzn - imp_nosc_pzn;
dg_imp  = sqrt(1 + imp_sc_pxn.^2 + imp_sc_pyn.^2 + imp_sc_pzn.^2) ...
        - sqrt(1 + imp_nosc_pxn.^2 + imp_nosc_pyn.^2 + imp_nosc_pzn.^2);

fprintf('\nimp SC effect per particle (with-SC - no-SC) at L0A end:\n');
fprintf('  Delta_x  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dx_imp)*1e6, std(dx_imp)*1e6, max(abs(dx_imp))*1e6);
fprintf('  Delta_y  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dy_imp)*1e6, std(dy_imp)*1e6, max(abs(dy_imp))*1e6);
fprintf('  Delta_px : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpx_imp), std(dpx_imp), max(abs(dpx_imp)));
fprintf('  Delta_py : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpy_imp), std(dpy_imp), max(abs(dpy_imp)));
fprintf('  Delta_pz : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpz_imp), std(dpz_imp), max(abs(dpz_imp)));
fprintf('  Delta_g  : mean=%+.4f     std=%.4f\n', mean(dg_imp), std(dg_imp));

% --- Side-by-side comparison ---
fprintf('\n========== lt vs imp SC effect at L0A end ==========\n');
fprintf('Quantity      |    lt              |    imp            |  ratio (lt/imp)\n');
fprintf('--------------|--------------------|--------------------|---------------\n');
fprintf('std(Dx) [um]  |  %8.0f          |  %8.0f          |  %.2f\n', std(ld.dx_sc)*1e6, std(dx_imp)*1e6, std(ld.dx_sc)/std(dx_imp));
fprintf('std(Dy) [um]  |  %8.0f          |  %8.0f          |  %.2f\n', std(ld.dy_sc)*1e6, std(dy_imp)*1e6, std(ld.dy_sc)/std(dy_imp));
fprintf('std(Dpx)      |  %.4f             |  %.4f             |  %.2f\n', std(ld.dpx_sc), std(dpx_imp), std(ld.dpx_sc)/std(dpx_imp));
fprintf('std(Dpy)      |  %.4f             |  %.4f             |  %.2f\n', std(ld.dpy_sc), std(dpy_imp), std(ld.dpy_sc)/std(dpy_imp));
fprintf('std(Dpz)      |  %.4f             |  %.4f             |  %.2f\n', std(ld.dpz_sc), std(dpz_imp), std(ld.dpz_sc)/std(dpz_imp));
fprintf('std(Dg)       |  %.4f             |  %.4f             |  %.2f\n', std(ld.dg_sc), std(dg_imp), std(ld.dg_sc)/std(dg_imp));

% Bin by initial radius
r_init_imp = sqrt(imp_nosc_x.^2 + imp_nosc_y.^2);
dr_rad_imp = (dx_imp .* imp_nosc_x + dy_imp .* imp_nosc_y) ./ max(r_init_imp, 1e-30);
dpr_rad_imp = (dpx_imp .* imp_nosc_x + dpy_imp .* imp_nosc_y) ./ max(r_init_imp, 1e-30);

% Plot or print binned imp SC radial effect
edges = linspace(0, 2e-3, 11);
[~, ~, bin] = histcounts(r_init_imp, edges);
fprintf('\nimp SC radial effect, binned by no-SC r:\n');
fprintf('  r [mm]    N      <dr_rad> [um]  std(dr_rad) [um]   <dpr_rad>      std(dpr_rad)\n');
for k = 1:numel(edges)-1
    msk = (bin == k);
    if sum(msk) > 5
        fprintf('  %4.2f-%4.2f   %5d   %+8.0f       %8.0f       %+9.4f      %.4f\n', ...
            edges(k)*1e3, edges(k+1)*1e3, sum(msk), ...
            mean(dr_rad_imp(msk))*1e6, std(dr_rad_imp(msk))*1e6, ...
            mean(dpr_rad_imp(msk)), std(dpr_rad_imp(msk)));
    end
end

save('/Users/glenwhite/code/Lucretia/sc_effect_compare_data.mat', ...
     'dx_imp', 'dy_imp', 'dpx_imp', 'dpy_imp', 'dpz_imp', 'dg_imp', ...
     'imp_sc_x', 'imp_sc_y', 'imp_sc_z', 'imp_nosc_x', 'imp_nosc_y', 'imp_nosc_z');
fprintf('\nSaved -> sc_effect_compare_data.mat\n');
end


function [P, IDs] = read_imp_fort50(path)
fid = fopen(path, 'r');
D = textscan(fid, '%f %f %f %f %f %f %f %f %f');
fclose(fid);
% cols: x, px, y, py, z, pz, q/m, q_macro, id  (per IMPACT-T Distribution.f90)
P   = [D{1} D{2} D{3} D{4} D{5} D{6}];
IDs = D{9};
end
