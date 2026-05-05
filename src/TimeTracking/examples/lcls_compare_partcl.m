function res = lcls_compare_partcl(varargin)
% Optional name/value pairs:
%   'slice_sc'      (true)
%   'mesh_sc'       (true)
%   'sc_adaptive'   (false)
%   'tag'           ('')
%   'n_steps'       (1500)
%   'impactt_dir'   ('common/ImpactT')
p = inputParser;
p.addParameter('slice_sc',     true,  @islogical);
p.addParameter('mesh_sc',      true,  @islogical);
p.addParameter('sc_adaptive',  false, @islogical);
p.addParameter('tag',          '',    @(x) ischar(x) || isstring(x));
p.addParameter('n_steps',      1500,  @isnumeric);
p.addParameter('impactt_dir',  '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;
% LCLS_COMPARE_PARTCL  Cross-code calibration: lucretia-tt seeded directly
% from ImpactT's partcl.data, vs ImpactT's own gun-region trajectory.
%
% Eliminates CathodeSource emission as a degree of freedom: both codes
% start from byte-identical particles. Any remaining sigma_x / sigma_z
% divergence in the gun region must come from tracking + SC, not from
% emission timing or cold-start statistics.
%
% Per memory project_lucretia_tt_mesh_silent_zero.md (2026-05-04 fourth
% session): the recommended next test was "seed lucretia-tt directly
% from partcl.data, run the gun region, see if sigma_x ratio drops from
% 2.2x to ~1.0x". This driver implements that test.

impactt_dir = char(opts.impactt_dir);

fprintf('=== LCLS gun: partcl.data seed comparison ===\n\n');

% --- Parse ImpactT.in to inherit lattice + header config ---
[lattice, ~, ~, ~, info] = timetracking.readImpactTLattice( ...
    impactt_dir, 'n_macros', 50000);

% Strip CathodeSource and trim to GUN + SOL1 + monitor (gun-region focus)
keep = false(size(lattice));
for k = 1:numel(lattice)
    keep(k) = ismember(lattice{k}.name, {'GUN', 'SOL1'});
end
lattice = lattice(keep);

% Read ImpactT.in header for charge + Tini
fid = fopen(fullfile(impactt_dir, 'ImpactT.in'), 'r');
hd = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
hd = hd{1};
% Lines starting with ! are comments; strip and find the Bcurr line (h9)
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');     % Bcurr Bkenergy Bmass Bcharge Bfreq Tini
Bcurr = h9(1);  Bfreq = h9(5);  Tini = h9(6);
q_total = abs(Bcurr) / Bfreq;          % C
fprintf('  charge      = %.3g C   (= Bcurr/Bfreq)\n', q_total);
fprintf('  t_init      = %.4g s   (Tini header)\n', Tini);

% Append a per-step BeamMonitor for diagnostics
mon = timetracking.BeamMonitor('name', 'mon', 'dump_every', 50);
lattice{end+1} = mon;

% --- Build TimeTrack ---
tt = TimeTrack();
tt.lattice = lattice;

tt.beam = struct( ...
    'partcl_file',    fullfile(impactt_dir, 'partcl.data'), ...
    'partcl_q_total', q_total, ...
    't_init',         Tini);

% Geometry sized for the gun region (z<0.65m, transverse +/- 3mm)
tt.geom_lo    = [-3e-3, -3e-3, -0.05];
tt.geom_hi    = [ 3e-3,  3e-3,  0.70];
tt.geom_ncell = [48, 48, 256];

% Tracking. Start at Tini so RF phase exposure matches ImpactT's wall-
% clock time. First-pass run length ~750 ps to clear the gun region
% (z=0 to ~200 mm). Memory's gun-region table goes to z=200mm; that's
% the slice of the lattice where the 2.2x sigma_x divergence is set.
% A longer run can follow once we know whether the issue persists.
tt.t_start = Tini;
tt.dt      = 0.5e-12;
tt.n_steps = opts.n_steps;

% SC config
tt.enable_space_charge = opts.mesh_sc;
tt.sc_image_plane      = opts.mesh_sc;
tt.sc_image_plane_z    = 0.0;
tt.sc_image_cutoff     = 0.05;
tt.sc_adaptive         = opts.sc_adaptive;
tt.sc_pad_factor       = 5.0;
tt.sc_min_pad_xy       = 1e-3;
tt.sc_min_pad_z        = 1e-3;
tt.enable_slice_sc     = opts.slice_sc;
tt.slice_sc_n          = 256;
tt.slice_sc_radius_factor = 2.0;

% ImpactT-style behind-cathode drift. partcl.data particles are at
% z slightly negative (~-7 um) -- same as ImpactT's loaded state.
% Drift them at universal betazini until z=0 just like ImpactT's
% driftemission_BeamBunch does. Plus the kill-on-recross from
% TrackingLoop guards against ratcheting.
% betazini = sqrt(1 - 1/(1 + Bkenergy/Bmass)^2), Bkenergy=h9(2)=1eV,
% Bmass=h9(3)=511005eV.
Bkenergy = h9(2);  Bmass = h9(3);
tt.behind_cathode_z        = 0.0;
tt.behind_cathode_betazini = sqrt(1.0 - 1.0/(1.0 + Bkenergy/Bmass)^2);

% Keep dumps in a known location so re-analysis doesn't require a re-run.
if isempty(opts.tag)
    tt.work_dir = '/tmp/partcl_seed_run';
else
    tt.work_dir = ['/tmp/partcl_seed_run_' char(opts.tag)];
end

fprintf('\nRun config:\n');
fprintf('  geom: lo=[%g %g %g] hi=[%g %g %g] m, ncell=[%d %d %d]\n', ...
        tt.geom_lo, tt.geom_hi, tt.geom_ncell);
fprintf('  dt = %.1f ps, n_steps = %d, total %.2f ns\n', ...
        tt.dt*1e12, tt.n_steps, tt.dt*tt.n_steps*1e9);
fprintf('  t_start = %.4g s\n', tt.t_start);
fprintf('  SC: mesh+image_charge=%d, slice_sc=%d, slice_n=%d\n', ...
        tt.sc_image_plane, tt.enable_slice_sc, tt.slice_sc_n);

% --- Run ---
t0 = tic;
tt.run();
fprintf('  RUN TIME: %.1f s\n', toc(t0));

bunches = tt.readDumps();
fprintf('  dumps:    %d\n', numel(bunches));

% --- Per-dump diagnostics: gather sigmas + gamma vs <z> ---
n = numel(bunches);
ltt = struct('z', zeros(1,n), 'sx', zeros(1,n), 'sy', zeros(1,n), ...
             'sz', zeros(1,n), 'gamma', zeros(1,n), 'N', zeros(1,n));
me_kg = 9.1093837015e-31;
c     = 299792458.0;
for k = 1:n
    b = bunches{k};
    % Filter out parked dead particles (TrackingLoop kill-on-recross
    % parks them at z=-1e9). Anything below z=-1m is the dead pool.
    keep = double(b.z) > -1.0;
    xx_k = double(b.x(keep));  yy_k = double(b.y(keep));  zz_k = double(b.z(keep));
    px_k = double(b.px(keep)); py_k = double(b.py(keep)); pz_k = double(b.pz(keep));
    p2  = px_k.^2 + py_k.^2 + pz_k.^2;
    gam = sqrt(1 + p2 / (me_kg*c)^2);
    ltt.z(k)     = mean(zz_k);
    ltt.sx(k)    = std(xx_k);
    ltt.sy(k)    = std(yy_k);
    ltt.sz(k)    = std(zz_k);
    ltt.gamma(k) = mean(gam);
    ltt.N(k)     = numel(zz_k);
end

% --- ImpactT reference ---
fprintf('\nLoading ImpactT fort.18 / fort.24-26 ...\n');
imp = timetracking.readImpactTFort18(impactt_dir);

% --- Side-by-side at canonical z planes ---
% Limit to z planes our run actually reached.
ltt_z_max = max(ltt.z);
zs = [0.005, 0.020, 0.060, 0.100, 0.150, 0.200, 0.300, 0.500, 0.640];
zs = zs(zs <= ltt_z_max + 1e-3);
fprintf('\n%-8s %-12s %-12s %-7s %-12s %-12s %-12s %-7s\n', ...
        'z [m]', 'ltt_sx[mm]', 'imp_sx[mm]', 'ratio', ...
        'ltt_sz[mm]', 'imp_sz[mm]', 'ltt_gam', 'imp_gam');
out = struct('z', zs);
out.ltt_sx = nan(size(zs));
out.imp_sx = nan(size(zs));
out.ltt_sz = nan(size(zs));
out.imp_sz = nan(size(zs));
out.ltt_gamma = nan(size(zs));
out.imp_gamma = nan(size(zs));
for j = 1:numel(zs)
    z_target = zs(j);
    [~, kl] = min(abs(ltt.z      - z_target));
    [~, ki] = min(abs(imp.z_mean - z_target));
    out.ltt_sx(j)    = ltt.sx(kl) * 1e3;
    out.imp_sx(j)    = imp.sigma_x(ki) * 1e3;
    out.ltt_sz(j)    = ltt.sz(kl) * 1e3;
    out.imp_sz(j)    = imp.sigma_z(ki) * 1e3;
    out.ltt_gamma(j) = ltt.gamma(kl);
    out.imp_gamma(j) = imp.gamma(ki);
    fprintf('%-8.3f %-12.3f %-12.3f %-7.2f %-12.3f %-12.3f %-12.3f %-7.3f\n', ...
            z_target, out.ltt_sx(j), out.imp_sx(j), out.ltt_sx(j)/out.imp_sx(j), ...
            out.ltt_sz(j), out.imp_sz(j), out.ltt_gamma(j), out.imp_gamma(j));
end

res = struct('ltt', ltt, 'imp', imp, 'out', out, 'opts', opts);
save_name = 'partcl_seed_compare.mat';
if ~isempty(opts.tag)
    save_name = ['partcl_seed_compare_' char(opts.tag) '.mat'];
end
save(save_name, '-struct', 'res');
fprintf('\nSaved -> %s\n', save_name);
end
