function sc_kick_compare()
% SC_KICK_COMPARE  Particle-by-particle comparison of lt vs imp at L0A end.
%
% Loads the final particle states from both codes (WITH SC), matches by
% particle ID (both codes preserve ordering when seeded from partcl.data),
% computes per-particle differences, and dumps stats + histograms.

cd('/Users/glenwhite/code/Lucretia');
addpath('src/TimeTracking');
% readBeamOpenPMD lives in TimeTrack's private/, can't addpath; copy/inline
copyfile('src/TimeTracking/@TimeTrack/private/readBeamOpenPMD.m', '/tmp/claude/');
addpath('/tmp/claude');

% --- Read lt final particle state ---
% Use the smooth0 baseline run dir which is the WITH-SC reference.
lt_dir = '/tmp/bench_full_smooth0/diags';
fprintf('Reading lt particles from %s\n', lt_dir);

% Find last dump
files = dir(fullfile(lt_dir, '*.h5'));
if isempty(files)
    files = dir(fullfile(lt_dir, '**', '*.h5'));
end
fprintf('Found %d HDF5 dumps\n', numel(files));
if isempty(files), error('no lt dumps'); end
% Get the last one (highest step number)
[~, ix] = sort({files.name});
files = files(ix);
last_dump = fullfile(files(end).folder, files(end).name);
fprintf('Last lt dump: %s\n', last_dump);

% Use existing reader
bunches = readBeamOpenPMD(lt_dir);
fprintf('readBeamOpenPMD returned %d bunches\n', numel(bunches));
b = bunches{end};
fprintf('lt last dump: t=%g, N=%d, fields:\n', b.time, numel(b.x));
disp(fieldnames(b));

% Quick check on ID field
% Decode lt IDs: stride is 2^24 (CPU bits), AMReX packs (cpu, id).
% Computed lt_id = raw/2^24 has range with offset; subtract min to get 0..N-1
% then add 1 to get sequential 1..N matching imp's convention.
raw = uint64(b.id);
lt_id_raw = double(raw) / double(2^24);
lt_id = lt_id_raw - min(lt_id_raw) + 1;
fprintf('lt decoded IDs range: %d to %d\n', min(lt_id), max(lt_id));

% Filter out lt dead particles (parked at z=-1e9 by collimator/cathode kill)
lt_alive = double(b.z) > -1e6;
fprintf('lt alive particles: %d / %d\n', sum(lt_alive), numel(b.z));

% Read imp fort.50 (final lab-frame particles).
% Per IMPACT-T source (Distribution.f90 lines 220-222 in active repo),
% Pts1 has 9 cols: (x, px, y, py, z, pz, q/m, q_macro, id) where
% px/py/pz are normalised momenta beta*gamma. The earlier interpretation
% of col 5 as beta_z and col 6 as gamma was wrong -- they're z and pz.
imp_dir = '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT';
fid = fopen(fullfile(imp_dir, 'fort.50'), 'r');
D = textscan(fid, '%f %f %f %f %f %f %f %f %f');
fclose(fid);
imp_x   = D{1};  imp_pxn = D{2};
imp_y   = D{3};  imp_pyn = D{4};
imp_z   = D{5};  imp_pzn = D{6};
imp_qm  = D{7};  imp_qmacro = D{8};  imp_id = D{9};
imp_g   = sqrt(1 + imp_pxn.^2 + imp_pyn.^2 + imp_pzn.^2);
fprintf('imp fort.50: N=%d, IDs range %d to %d, unique=%d\n', ...
        numel(imp_x), min(imp_id), max(imp_id), numel(unique(imp_id)));

% Match by ID: build a 50000-index vector of lt particles and imp particles.
% Skip lt dead particles -- only use those that survived to L0A end.
N_max = 50000;
lt_idx_by_id  = zeros(N_max, 1);
imp_idx_by_id = zeros(N_max, 1);
for k = 1:numel(lt_id)
    if lt_id(k) >= 1 && lt_id(k) <= N_max && lt_alive(k)
        lt_idx_by_id(lt_id(k)) = k;
    end
end
for k = 1:numel(imp_id)
    if imp_id(k) >= 1 && imp_id(k) <= N_max
        imp_idx_by_id(imp_id(k)) = k;
    end
end
matched = (lt_idx_by_id > 0) & (imp_idx_by_id > 0);
fprintf('Matched %d / %d particles (%d in lt, %d in imp)\n', ...
        sum(matched), N_max, sum(lt_idx_by_id > 0), sum(imp_idx_by_id > 0));

% For matched particles, compare positions and momenta
me = 9.1093837015e-31; c = 299792458;
lt_x_m  = b.x(lt_idx_by_id(matched));
lt_y_m  = b.y(lt_idx_by_id(matched));
lt_z_m  = b.z(lt_idx_by_id(matched));
lt_px_m = b.px(lt_idx_by_id(matched));   % kg.m/s
lt_py_m = b.py(lt_idx_by_id(matched));
lt_pz_m = b.pz(lt_idx_by_id(matched));
% Convert lt to imp's normalised units: px_norm = px/(m*c) = beta*gamma
lt_pxn = double(lt_px_m) / (me * c);
lt_pyn = double(lt_py_m) / (me * c);
lt_pzn = double(lt_pz_m) / (me * c);
lt_g   = sqrt(1 + lt_pxn.^2 + lt_pyn.^2 + lt_pzn.^2);

imp_x_m  = imp_x  (imp_idx_by_id(matched));
imp_y_m  = imp_y  (imp_idx_by_id(matched));
imp_z_m  = imp_z  (imp_idx_by_id(matched));
imp_pxn_m = imp_pxn(imp_idx_by_id(matched));
imp_pyn_m = imp_pyn(imp_idx_by_id(matched));
imp_pzn_m = imp_pzn(imp_idx_by_id(matched));
imp_g_m  = imp_g  (imp_idx_by_id(matched));

fprintf('\nMatched particle stats:\n');
fprintf('  lt  sigma_x = %.0f um, sigma_y = %.0f um\n', std(lt_x_m)*1e6, std(lt_y_m)*1e6);
fprintf('  imp sigma_x = %.0f um, sigma_y = %.0f um\n', std(imp_x_m)*1e6, std(imp_y_m)*1e6);
fprintf('  lt  mean gamma = %.2f\n', mean(lt_g));
fprintf('  imp mean gamma = %.2f\n', mean(imp_g_m));

% Per-particle differences
dx  = double(lt_x_m)  - imp_x_m;
dy  = double(lt_y_m)  - imp_y_m;
dz  = double(lt_z_m)  - imp_z_m;
dpx = lt_pxn  - imp_pxn_m;
dpy = lt_pyn  - imp_pyn_m;
dpz = lt_pzn  - imp_pzn_m;
dg  = lt_g    - imp_g_m;
fprintf('\nPer-particle (lt - imp) at L0A end (matched %d particles):\n', sum(matched));
fprintf('  Delta_x  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dx)*1e6, std(dx)*1e6, max(abs(dx))*1e6);
fprintf('  Delta_y  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dy)*1e6, std(dy)*1e6, max(abs(dy))*1e6);
fprintf('  Delta_z  : mean=%+.0f um  std=%.0f um  max|.|=%.0f um\n', mean(dz)*1e6, std(dz)*1e6, max(abs(dz))*1e6);
fprintf('  Delta_px : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpx), std(dpx), max(abs(dpx)));
fprintf('  Delta_py : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpy), std(dpy), max(abs(dpy)));
fprintf('  Delta_pz : mean=%+.4f     std=%.4f     max|.|=%.4f\n', mean(dpz), std(dpz), max(abs(dpz)));
fprintf('  Delta_g  : mean=%+.4f     std=%.4f\n', mean(dg), std(dg));

% Categorise: how does dx vary with initial radius? Use imp_x as proxy
% (initial x correlated with final x for matched particles).
r_imp = sqrt(imp_x_m.^2 + imp_y_m.^2);
dr   = sqrt(dx.^2 + dy.^2);
edges = linspace(0, 3e-3, 11);
[~, ~, bin] = histcounts(r_imp, edges);
fprintf('\nDelta-position binned by imp r (lt - imp):\n');
fprintf('  r_bin [mm]    N     mean|dr| [um]   std|dr| [um]\n');
for k = 1:numel(edges)-1
    msk = (bin == k);
    if sum(msk) > 5
        fprintf('  %4.1f-%4.1f    %5d   %8.0f       %8.0f\n', ...
            edges(k)*1e3, edges(k+1)*1e3, sum(msk), ...
            mean(dr(msk))*1e6, std(dr(msk))*1e6);
    end
end

% Save data for further analysis
save('/Users/glenwhite/code/Lucretia/sc_compare_data.mat', ...
     'lt_x_m', 'lt_y_m', 'lt_z_m', 'lt_pxn', 'lt_pyn', 'lt_pzn', 'lt_g', ...
     'imp_x_m', 'imp_y_m', 'imp_z_m', 'imp_pxn_m', 'imp_pyn_m', 'imp_pzn_m', 'imp_g_m', ...
     'dx', 'dy', 'dz', 'dpx', 'dpy', 'dpz', 'dg');
fprintf('\nSaved -> sc_compare_data.mat\n');
end
