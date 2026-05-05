function res = lcls_compare_cathode_imptT(varargin)
% LCLS_COMPARE_CATHODE_IMPTT  Validate the ImpactT-style augmentation to
% CathodeSource: thermal half-Maxwell uz + universal-betazini drift behind
% the cathode. Same gun-region comparison vs ImpactT as the partcl-seed
% driver, but using CathodeSource (not partcl.data) so we test the actual
% emission code path that lucretia-tt's standard runs use.
%
% Optional name/value pairs:
%   'tag'        ('cathode_imptT')
%   'n_steps'    (1500)
%   'slice_sc'   (true)
%   'mesh_sc'    (true)

p = inputParser;
p.addParameter('tag',           'cathode_imptT', @(x) ischar(x) || isstring(x));
p.addParameter('n_steps',       2500,            @isnumeric);   % matches ImpactT dt=0.3ps for ~750 ps
p.addParameter('slice_sc',      true,            @islogical);
p.addParameter('mesh_sc',       true,            @islogical);
p.addParameter('z_init_spread', 1.6e-5,          @isnumeric);
p.addParameter('long_thermal',  true,            @islogical);
% Sigma_x diagnostic knobs
p.addParameter('sc_adaptive',   false,           @islogical);
p.addParameter('sc_pad_factor', 5.0,             @isnumeric);
p.addParameter('mte_eV',        [],              @(x) isempty(x) || isnumeric(x));
p.addParameter('geom_ncell',    [],              @(x) isempty(x) || (isnumeric(x) && numel(x)==3));
% Custom ImpactT reference directory (for no-SC sanity checks, etc.).
% Must contain ImpactT.in, partcl.data, rfdata*, and fort.18+24-26 from
% a completed ImpactT run.
p.addParameter('impactt_dir',   '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT', ...
                                                 @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

impactt_dir = char(opts.impactt_dir);

fprintf('=== LCLS gun: CathodeSource + ImpactT-style emission ===\n\n');

% Read ImpactT.in to inherit lattice + tracking + geom suggestions
[lattice, ~, geom_imp, tracking_imp, info] = timetracking.readImpactTLattice( ...
    impactt_dir, 'n_macros', 50000);

% Read Tini and Bcurr from ImpactT.in header (same logic as the partcl driver)
fid = fopen(fullfile(impactt_dir, 'ImpactT.in'), 'r');
hd  = textscan(fid, '%s', 'Delimiter', '\n');  fclose(fid);
hd  = hd{1};
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');
Bcurr = h9(1);  Bfreq = h9(5);  Tini = h9(6);
% ImpactT's behind-cathode drift speed: betazini =
% sqrt(1 - 1/(1 + Bkenergy/Bmass)^2). With Bkenergy=h9(2)=1 eV,
% Bmass=h9(3)=511005 eV, betazini ~ 1.98e-3.
Bkenergy = h9(2);  Bmass = h9(3);
betazini = sqrt(1.0 - 1.0/(1.0 + Bkenergy/Bmass)^2);
fprintf('  Tini      = %.4g s   (matched in tracking.t_start)\n', Tini);
fprintf('  betazini  = %.4g     (Bkenergy=%g eV, Bmass=%g eV)\n', betazini, Bkenergy, Bmass);

% Find the CathodeSource that readImpactTLattice prepended and augment it
% with the ImpactT-style options.
cat_idx = find(cellfun(@(e) strcmp(e.type, 'cathode_source'), lattice), 1);
assert(~isempty(cat_idx), 'readImpactTLattice did not insert a CathodeSource');
lattice{cat_idx}.longitudinal_thermal = double(opts.long_thermal);
lattice{cat_idx}.z_init_spread        = opts.z_init_spread;
if ~isempty(opts.mte_eV)
    lattice{cat_idx}.mte = opts.mte_eV;
end
% When ImpactT.in has Flagdist=16, ImpactT IGNORES the header sigx/sigy
% and uses partcl.data positions directly. readImpactTLattice currently
% takes the (header) sigx as spot_size, which gave 0.6 mm here vs the
% partcl.data actual 0.286 mm -- a 2x mismatch in initial bunch size.
% Compute the partcl.data-actual sigma_x and override.
partcl_path = fullfile(impactt_dir, 'partcl.data');
if exist(partcl_path, 'file')
    fid = fopen(partcl_path, 'r');
    fscanf(fid, '%d', 1);  % header: N
    A = fscanf(fid, '%g', [6, Inf])';
    fclose(fid);
    partcl_sigx = std(A(:,1));
    partcl_sigy = std(A(:,3));
    spot_actual = max(partcl_sigx, partcl_sigy);
    fprintf('  partcl.data actual: sigma_x=%g mm  sigma_y=%g mm  (header had %g mm)\n', ...
            partcl_sigx*1e3, partcl_sigy*1e3, lattice{cat_idx}.spot_size*1e3);
    lattice{cat_idx}.spot_size = spot_actual;
end
fprintf('  CathodeSource augmented: longitudinal_thermal=%d, z_init_spread=%g m, mte=%g eV, spot=%g mm\n', ...
        lattice{cat_idx}.longitudinal_thermal, lattice{cat_idx}.z_init_spread, ...
        lattice{cat_idx}.mte, lattice{cat_idx}.spot_size*1e3);

% Trim to GUN + SOL1 + monitor for fast first-pass test
keep = false(size(lattice));
for k = 1:numel(lattice)
    keep(k) = ismember(lattice{k}.name, {'cat', 'GUN', 'SOL1'}) ...
              || strcmp(lattice{k}.type, 'cathode_source');
end
lattice = lattice(keep);
% Append per-step BeamMonitor
lattice{end+1} = timetracking.BeamMonitor('name', 'mon', 'dump_every', 50);

% --- Build TimeTrack ---
tt = TimeTrack();
tt.lattice = lattice;
tt.beam    = struct('n_particles', 0);    % cathode emits

tt.geom_lo    = [-3e-3, -3e-3, -0.05];
tt.geom_hi    = [ 3e-3,  3e-3,  0.70];
% Match ImpactT.in's grid by default (line 9 of LCLS deck = [48 48 48]).
% Override via opts.geom_ncell when sweeping.
if isempty(opts.geom_ncell)
    tt.geom_ncell = geom_imp.ncell(:).';
else
    tt.geom_ncell = opts.geom_ncell;
end

tt.t_start = Tini;
% Match ImpactT.in's dt schedule: dt0=0.3ps for the cathode/gun region,
% switching to 4ps after z=0.25m (encoded in the type-(-4) lattice
% element). readImpactTLattice extracts the schedule into tracking_imp;
% we use it verbatim. Override dt only if opts.n_steps is supplied
% (then we honor that count and the override dt the caller wants).
if isfield(tracking_imp, 'dt') && ~isempty(tracking_imp.dt)
    tt.dt = tracking_imp.dt;
else
    tt.dt = 0.5e-12;
end
if isfield(tracking_imp, 'dt_change') && ~isempty(tracking_imp.dt_change)
    tt.dt_change_t = tracking_imp.dt_change;
end
if isfield(tracking_imp, 'dt_after') && ~isempty(tracking_imp.dt_after)
    tt.dt_after = tracking_imp.dt_after;
end
tt.n_steps = opts.n_steps;
fprintf('  tracking: dt=%.2g ps, n_steps=%d', tt.dt*1e12, tt.n_steps);
if ~isempty(tt.dt_change_t)
    fprintf(', dt_change_t=%.3g ns -> dt_after=%.2g ps', ...
            tt.dt_change_t*1e9, tt.dt_after*1e12);
end
fprintf('\n');

% ImpactT-style behind-cathode drift. Disable when z_init_spread <= 0
% (particles emit AT or just past cathode; no behind-cathode region).
if opts.z_init_spread > 0
    tt.behind_cathode_z        = lattice{cat_idx}.z_cathode;
    tt.behind_cathode_betazini = betazini;
end

tt.enable_space_charge = opts.mesh_sc;
tt.sc_image_plane      = opts.mesh_sc;
tt.sc_image_plane_z    = lattice{cat_idx}.z_cathode;
tt.sc_image_cutoff     = 0.05;
tt.sc_adaptive         = opts.sc_adaptive;
tt.sc_pad_factor       = opts.sc_pad_factor;
tt.sc_min_pad_xy       = 1e-3;
tt.sc_min_pad_z        = 1e-3;
tt.enable_slice_sc     = opts.slice_sc;
tt.slice_sc_n          = 256;
tt.slice_sc_radius_factor = 2.0;

tt.work_dir = ['/tmp/cathode_imptT_run_' char(opts.tag)];

fprintf('\nRun config:\n');
fprintf('  geom: lo=[%g %g %g] hi=[%g %g %g] m, ncell=[%d %d %d]\n', ...
        tt.geom_lo, tt.geom_hi, tt.geom_ncell);
fprintf('  dt = %.1f ps, n_steps = %d, total %.2f ns\n', ...
        tt.dt*1e12, tt.n_steps, tt.dt*tt.n_steps*1e9);
fprintf('  behind-cathode drift: z<%g m at betazini=%.4g\n', ...
        tt.behind_cathode_z, tt.behind_cathode_betazini);
fprintf('  SC: mesh=%d slice=%d\n', tt.enable_space_charge, tt.enable_slice_sc);

% --- Run ---
t0 = tic;
tt.run();
fprintf('  RUN TIME: %.1f s\n', toc(t0));

bunches = tt.readDumps();
fprintf('  dumps: %d\n', numel(bunches));

% --- Per-dump diagnostics ---
n = numel(bunches);
ltt = struct('z', zeros(1,n), 'sx', zeros(1,n), 'sy', zeros(1,n), ...
             'sz', zeros(1,n), 'gamma', zeros(1,n), 'N', zeros(1,n));
me_kg = 9.1093837015e-31;
c     = 299792458.0;
for k = 1:n
    b   = bunches{k};
    % Filter out parked dead particles (TrackingLoop kills cathode-
    % re-crossers and parks them at z = -1e9). Anything below z=-1m is
    % the dead pool, not the bunch.
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

fprintf('\nLoading ImpactT fort.18 / fort.24-26 ...\n');
imp = timetracking.readImpactTFort18(impactt_dir);

ltt_z_max = max(ltt.z);
zs = [0.005, 0.020, 0.060, 0.100, 0.150, 0.200, 0.300, 0.500, 0.640];
zs = zs(zs <= ltt_z_max + 1e-3);
fprintf('\n%-8s %-12s %-12s %-7s %-12s %-12s %-12s %-7s\n', ...
        'z [m]', 'ltt_sx[mm]', 'imp_sx[mm]', 'ratio', ...
        'ltt_sz[mm]', 'imp_sz[mm]', 'ltt_gam', 'imp_gam');
out = struct('z', zs);
out.ltt_sx = nan(size(zs));  out.imp_sx = nan(size(zs));
out.ltt_sz = nan(size(zs));  out.imp_sz = nan(size(zs));
out.ltt_gamma = nan(size(zs));  out.imp_gamma = nan(size(zs));
for j = 1:numel(zs)
    [~, kl] = min(abs(ltt.z      - zs(j)));
    [~, ki] = min(abs(imp.z_mean - zs(j)));
    out.ltt_sx(j)    = ltt.sx(kl) * 1e3;
    out.imp_sx(j)    = imp.sigma_x(ki) * 1e3;
    out.ltt_sz(j)    = ltt.sz(kl) * 1e3;
    out.imp_sz(j)    = imp.sigma_z(ki) * 1e3;
    out.ltt_gamma(j) = ltt.gamma(kl);
    out.imp_gamma(j) = imp.gamma(ki);
    fprintf('%-8.3f %-12.3f %-12.3f %-7.2f %-12.3f %-12.3f %-12.3f %-7.3f\n', ...
            zs(j), out.ltt_sx(j), out.imp_sx(j), out.ltt_sx(j)/out.imp_sx(j), ...
            out.ltt_sz(j), out.imp_sz(j), out.ltt_gamma(j), out.imp_gamma(j));
end

res = struct('ltt', ltt, 'imp', imp, 'out', out, 'opts', opts);
save(['cathode_imptT_compare_' char(opts.tag) '.mat'], '-struct', 'res');
fprintf('\nSaved -> cathode_imptT_compare_%s.mat\n', char(opts.tag));
end
