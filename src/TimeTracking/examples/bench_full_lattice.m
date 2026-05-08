function bench_full_lattice(varargin)
% BENCH_FULL_LATTICE  Comprehensive cross-code benchmark vs ImpactT for the
% LCLS injector through L0A (~5 m), with space charge enabled.
%
% Usage: bench_full_lattice('impactt_dir', '/path/to/ImpactT', 'n_macros', 50000)
%
% Compares (at end of L0A):
%   - sigma_x, sigma_y, sigma_z        (transverse + longitudinal beam sizes)
%   - bunch length (FWHM-equivalent)
%   - sigma_gamma (energy spread, MeV)
%   - eps_nx, eps_ny                   (normalised transverse emittance)
%   - eps_nz                           (longitudinal emittance, m·rad)
%   - slice eps_nx, eps_ny             (per z-slice, peak-current slice)
%
% Uses TimeTrack defaults (DKD + wallclock cos), adaptive 3D SC + slice SC,
% ImpactT partcl.data as initial distribution (apples-to-apples seed).
%
% Defaults (updated 2026-05-08 after bunch-size-divergence investigation):
%   sc_static_xrad = []   -- adaptive mesh (was 0.015 mirroring imp's deck Xrad,
%                            but imp's Xrad is a particle-loss aperture, NOT the
%                            SC mesh extent; static cell_xy=625um massively
%                            under-resolved the early-emission bunch sigma~100-300um).
%   sc_use_b_field = true        -- explicit SC B via Boris (matches ImpactT v×B).
%   self_force_direct = true     -- avoids LUT-rebuild noise w/ adaptive mesh.
%   sc_integer_cell_shift = true -- snap centroid-only resizes to integer cells.
%   sc_cent_drift_threshold = 0.9 -- defer centroid resizes (less per-step noise).
%   sc_rho_smooth_passes = 2     -- binomial filter on rho before IGF solve.
%                                   Best of 5 noise-knob experiments at end-of-L0A
%                                   eps_nx (-13% vs 0 passes; sc_exact_range,
%                                   sc_shape_order=2/TSC, pad_factor=2 all gave
%                                   smaller or no improvement).
%   disable_self_force = true    -- imp does NO self-force subtraction; lt's
%                                   kSelfForceFactor=0.62 is empirically neutral
%                                   at high gamma and IMPROVES per-particle noise
%                                   (-28%) at low gamma. No measurable downside.

p = inputParser;
p.addParameter('impactt_dir', '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT', @(x) ischar(x) || isstring(x));
p.addParameter('n_macros',    50000, @isnumeric);
p.addParameter('n_steps',     9000,  @isnumeric);   % ~16.6 ns to match ImpactT end
p.addParameter('dt_initial',  0.3e-12, @isnumeric);
p.addParameter('dt_after',    4e-12,  @isnumeric);
p.addParameter('dt_change_z', [],     @(x) isempty(x) || isnumeric(x));   % m: z at which dt switches (overrides dt_change_t; matches ImpactT type-(-4))
p.addParameter('dt_change_t', [],     @(x) isempty(x) || isnumeric(x));   % s: wall-clock fallback if dt_change_z is empty
p.addParameter('n_slice',     30,    @isnumeric);
p.addParameter('tag',         'bench_full', @(x) ischar(x) || isstring(x));
p.addParameter('sc_mode',     'full', @(x) ischar(x) || isstring(x));   % 'full' (mesh+slice), 'mesh', 'slice', 'off'
p.addParameter('sc_static_xrad', [], @(x) isempty(x) || isnumeric(x));   % m: STATIC mesh ±xrad. [] (default) = ADAPTIVE (matches imp's per-step bunch-tracking; static at 15mm under-resolves the small early-emission bunch).
p.addParameter('sc_pad_factor',  5.0,  @isnumeric);                       % adaptive: half-extent = pad_factor * sigma
p.addParameter('sc_resize_hyst', 2.0,  @isnumeric);                       % adaptive: resize hysteresis factor (larger = fewer resizes, less noise)
p.addParameter('sc_cent_drift_threshold', 0.9, @isnumeric);              % adaptive: centroid drift trigger (frac of half-extent). 0.9 reduces resize freq for relativistic bunches.
p.addParameter('sc_integer_cell_shift', true, @islogical);               % adaptive: snap centroid-only resizes to integer cells (preserves deposit pattern -> zero per-particle field jump for that resize event)
p.addParameter('slice_gamma_off', [],  @(x) isempty(x) || isnumeric(x));  % disable slice SC when bunch mean gamma >= this (lets 3D mesh handle longitudinal at high gamma)
p.addParameter('disable_self_force', true, @islogical);                   % skip the empirical self-force subtraction. Default ON: imp does NO self-force subtraction; lt's kSelfForceFactor=0.62 is empirically neutral at high gamma and IMPROVES per-particle noise (-28%) at low gamma. No measurable downside in tracking results.
p.addParameter('sc_exact_range', false, @islogical);                      % ImpactT-style exact bunch range adaptive mesh (no padding, every step)
p.addParameter('sc_rho_smooth_passes', 2, @isnumeric);                   % # of binomial-smoother passes on rho before IGF solve. Default 2: best-found end-of-L0A eps_nx (~13% reduction vs 0 passes); cheap (one stencil pass per rho).
p.addParameter('slice_radius_factor',  0, @isnumeric);                   % override slice SC bunch radius: a = factor * sigma_xy (default 2.0; pass 0 to use default)
p.addParameter('sc_hybrid_z_adaptive', false, @islogical);               % SC mesh hybrid mode: static xy + adaptive z (overrides sc_static_xrad behavior in z)
p.addParameter('sc_shape_order', 1, @isnumeric);                         % particle shape: 1 = CIC (default), 2 = TSC (smoother per-particle field gradient at ~3x deposit/gather cost)
p.addParameter('sc_use_b_field', true, @islogical);                      % apply SC B field via Boris (matches ImpactT) -- captures non-synchronous v×B coupling. Default ON (matches ImpactT v×B convention).
p.addParameter('slice_profile',  0,   @isnumeric);                       % slice SC transverse profile: 0 = uniform disk (default), 1 = Gaussian disk
p.addParameter('self_force_direct', true, @islogical);                   % compute SC self-force directly each step (no LUT; ~7x cost; avoids LUT-rebuild noise w/ adaptive mesh). Default ON because adaptive mesh is now default.
p.addParameter('sc_diag_resize_jump', 0, @isnumeric);                    % if >0, lucretia-tt prints per-particle dE statistics on each mesh resize (diagnostic of field discontinuity)
p.addParameter('sc_dump_field_at_step', 0, @isnumeric);                   % >0 -> dump SC mesh field (rho/phi/Ex/Ey/Ez) at this solve count
p.addParameter('sc_dump_field_path', '', @(x) ischar(x) || isstring(x));  % path for sc mesh dump; empty -> /tmp/sc_field_dump_step<N>.bin
p.addParameter('dump_kicks_at_steps', [], @isnumeric);                   % vector of step indices: dump per-particle SC kicks for cross-code comparison
p.addParameter('dump_kicks_at_times', [], @isnumeric);                   % vector of physical times (s): dump per-particle SC kicks at first step crossing each time
p.addParameter('dump_kicks_path_prefix', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;
impactt_dir = char(opts.impactt_dir);

fprintf('=== Full-lattice benchmark (LCLS injector + L0A, with SC) ===\n');
fprintf('  ImpactT dir: %s\n', impactt_dir);
fprintf('  n_macros: %d  n_steps: %d\n', opts.n_macros, opts.n_steps);

[lattice, ~, geom_imp, tracking_imp, info] = timetracking.readImpactTLattice( ...
    impactt_dir, 'n_macros', opts.n_macros);

% Read header for partcl-seed setup
fid = fopen(fullfile(impactt_dir, 'ImpactT.in'), 'r');
hd = textscan(fid, '%s', 'Delimiter', '\n');  fclose(fid);  hd = hd{1};
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');
Bcurr = h9(1); Bfreq = h9(5); Tini = h9(6);
Bkenergy = h9(2); Bmass = h9(3);
q_total = abs(Bcurr) / Bfreq;

% Filter out cathode_source -- we are using partcl.data as the seed,
% otherwise the cathode would emit a second batch of particles in parallel.
keep = true(size(lattice));
for k = 1:numel(lattice)
    if strcmp(lattice{k}.type, 'cathode_source'), keep(k) = false; end
end
lattice = lattice(keep);

% Replace BeamMonitor with cadence that gives ~25 dumps over the run
mon_idx = find(cellfun(@(e) strcmp(e.type, 'beam_monitor'), lattice), 1);
dump_every = max(100, floor(opts.n_steps / 25));
if ~isempty(mon_idx)
    lattice{mon_idx} = timetracking.BeamMonitor('name', 'mon', 'dump_every', dump_every);
else
    lattice{end+1} = timetracking.BeamMonitor('name', 'mon', 'dump_every', dump_every);
end

fprintf('  lattice elements: %d  (BeamMonitor every %d steps)\n', numel(lattice), dump_every);
for k = 1:numel(lattice)
    el = lattice{k};
    fprintf('    [%2d] %-22s type=%-18s\n', k, el.name, el.type);
end

% --- Build TimeTrack (uses new DKD + wallclock defaults) ---
tt = TimeTrack();
tt.lattice = lattice;
tt.beam    = struct('partcl_file', fullfile(impactt_dir, 'partcl.data'), ...
                    'partcl_q_total', q_total, 't_init', Tini);

% Geometry (initial; SC adaptive will resize per step)
tt.geom_lo    = [-3e-3, -3e-3, -0.05];
tt.geom_hi    = [ 3e-3,  3e-3,  0.70];
tt.geom_ncell = geom_imp.ncell(:).';

% Tracking schedule
tt.t_start     = Tini;
tt.dt          = opts.dt_initial;
% Prefer z-trigger (matches ImpactT exactly); fall back to t-trigger or
% to whatever readImpactTLattice extracted from the type-(-4) element.
if ~isempty(opts.dt_change_z)
    tt.dt_change_z = opts.dt_change_z;
elseif ~isempty(opts.dt_change_t)
    tt.dt_change_t = opts.dt_change_t;
elseif isfield(tracking_imp, 'dt_change_z') && ~isempty(tracking_imp.dt_change_z)
    tt.dt_change_z = tracking_imp.dt_change_z;
elseif isfield(tracking_imp, 'dt_change') && ~isempty(tracking_imp.dt_change)
    tt.dt_change_t = tracking_imp.dt_change;
end
tt.dt_after    = opts.dt_after;
tt.n_steps     = opts.n_steps;

% Behind-cathode drift (universal-betazini, ImpactT compat)
tt.behind_cathode_z        = 0.0;
tt.behind_cathode_betazini = sqrt(1.0 - 1.0/(1.0 + Bkenergy/Bmass)^2);

% Space charge: configurable. 'full' = 3D adaptive + 1D slice + cathode image.
sc_mode = lower(char(opts.sc_mode));
tt.enable_space_charge = ismember(sc_mode, {'full', 'mesh'});
tt.sc_adaptive         = ismember(sc_mode, {'full', 'mesh'});
tt.enable_slice_sc     = ismember(sc_mode, {'full', 'slice'});
tt.sc_image_plane      = ismember(sc_mode, {'full', 'mesh'});
tt.sc_image_plane_z    = 0.0;
tt.sc_image_cutoff     = 0.05;
% Optional: STATIC SC mesh ±xrad transverse (ImpactT convention) instead
% of per-step adaptive sizing. Reduces LUT-noise by giving more particles
% per cell at the cost of mesh resolution.
if ~isempty(opts.sc_static_xrad)
    xrad = opts.sc_static_xrad;
    z_half = 0.005;  % co-moving z half-span (5 mm = ~4 sigma_z at L0A exit)
    tt.geom_lo = [-xrad, -xrad, -z_half];
    tt.geom_hi = [ xrad,  xrad,  z_half];
    tt.sc_adaptive = false;
    tt.sc_comoving = true;   % co-move in z; mesh size stays fixed at 2*z_half
    fprintf('  STATIC mesh: xrad=%.3f m  (cell_xy=%.4f mm, cell_z=%.4f mm)\n', ...
        xrad, 2*xrad/tt.geom_ncell(1)*1e3, 2*z_half/tt.geom_ncell(3)*1e3);
end
tt.sc_pad_factor = opts.sc_pad_factor;
if opts.sc_resize_hyst ~= 2.0
    tt.sc_resize_hyst = opts.sc_resize_hyst;
end
if opts.sc_cent_drift_threshold ~= 0.5
    tt.sc_cent_drift_threshold = opts.sc_cent_drift_threshold;
end
if opts.sc_integer_cell_shift
    tt.sc_integer_cell_shift = true;
    fprintf('  integer-cell snap on centroid-only resizes (zero discretization noise)\n');
end
if ~isempty(opts.slice_gamma_off)
    tt.slice_sc_gamma_off = opts.slice_gamma_off;
end
if opts.disable_self_force
    tt.disable_self_force = true;
end
if opts.sc_exact_range
    tt.sc_exact_range = true;
    tt.sc_adaptive = false;          % superseded
    tt.sc_comoving = false;
end
if opts.sc_rho_smooth_passes > 0
    tt.sc_rho_smooth_passes = round(opts.sc_rho_smooth_passes);
end
if isfield(opts, 'slice_radius_factor') && opts.slice_radius_factor > 0
    tt.slice_sc_radius_factor = opts.slice_radius_factor;
end
if opts.sc_hybrid_z_adaptive
    tt.sc_hybrid_z_adaptive = true;
    tt.sc_adaptive          = false;     % superseded
    tt.sc_comoving          = false;
    tt.sc_exact_range       = false;
    fprintf('  HYBRID mesh: static xy=[%.0f mm] + adaptive z (pad_factor=%.1f * sigma_z)\n', ...
            (tt.geom_hi(1) - tt.geom_lo(1))*1e3, opts.sc_pad_factor);
end
if opts.sc_shape_order ~= 1
    tt.sc_shape_order = round(opts.sc_shape_order);
    name = 'CIC'; if tt.sc_shape_order == 2, name = 'TSC'; end
    fprintf('  shape order = %d (%s)\n', tt.sc_shape_order, name);
end
if opts.sc_use_b_field
    tt.sc_use_b_field = true;
    fprintf('  SC B field via Boris (matches ImpactT v×B convention)\n');
end
if opts.slice_profile ~= 0
    tt.slice_sc_profile = round(opts.slice_profile);
    name = 'uniform-disk'; if tt.slice_sc_profile == 1, name = 'Gaussian-disk'; end
    fprintf('  slice SC profile = %d (%s)\n', tt.slice_sc_profile, name);
end
if opts.self_force_direct
    tt.self_force_direct = true;
    fprintf('  self-force computed directly each step (no LUT)\n');
end
if opts.sc_diag_resize_jump > 0
    tt.sc_diag_resize_jump = round(opts.sc_diag_resize_jump);
    tt.verbose_run = true;   % diagnostic prints go to stdout; stream them
    fprintf('  diagnostic: per-particle dE on each resize (will stream stdout)\n');
end
if ~isempty(opts.dump_kicks_at_steps)
    tt.dump_kicks_at_steps = round(opts.dump_kicks_at_steps);
    fprintf('  dump_kicks_at_steps = [%s]\n', ...
        strtrim(sprintf('%d ', tt.dump_kicks_at_steps)));
end
if ~isempty(opts.dump_kicks_at_times)
    tt.dump_kicks_at_times = double(opts.dump_kicks_at_times);
    fprintf('  dump_kicks_at_times = [%s] s\n', ...
        strtrim(sprintf('%.6g ', tt.dump_kicks_at_times)));
end
if opts.sc_dump_field_at_step > 0
    tt.sc_dump_field_at_step = round(opts.sc_dump_field_at_step);
    if ~isempty(opts.sc_dump_field_path)
        tt.sc_dump_field_path = char(opts.sc_dump_field_path);
    end
    fprintf('  sc_dump_field_at_step = %d -> %s\n', tt.sc_dump_field_at_step, ...
        char(string(tt.sc_dump_field_path)));
end
if ~isempty(opts.dump_kicks_at_steps) || ~isempty(opts.dump_kicks_at_times)
    if ~isempty(opts.dump_kicks_path_prefix)
        tt.dump_kicks_path_prefix = char(opts.dump_kicks_path_prefix);
    end
    fprintf('    -> %s<step>.bin\n', char(string(tt.dump_kicks_path_prefix)));
end
fprintf('  sc_mode = %s (mesh=%d, slice=%d, image=%d, adaptive=%d)\n', sc_mode, ...
    tt.enable_space_charge, tt.enable_slice_sc, tt.sc_image_plane, tt.sc_adaptive);

tt.work_dir = ['/tmp/bench_full_' char(opts.tag)];
if exist(tt.work_dir, 'dir'), rmdir(tt.work_dir, 's'); end

fprintf('\nDKD=%d wallclock=%d (centroid=%d, particle=%d)\n', ...
    coalesce(tt.use_dkd_integrator), coalesce(~ifempty(tt.use_centroid_phase, false)), ...
    coalesce(tt.use_centroid_phase), coalesce(tt.use_particle_phase));

t0 = tic;
tt.run();
fprintf('Run time: %.1f s\n', toc(t0));

% --- Compute lucretia-tt stats ---
bunches = tt.readDumps();
fprintf('Dumps: %d\n', numel(bunches));

me = 9.1093837015e-31; c = 299792458;
last = bunches{end};
[bm_lt, sl_lt] = compute_stats(last, me, c, opts.n_slice);

% --- Read ImpactT final particles + bunch evolution ---
[bm_imp, sl_imp] = read_imp_final(impactt_dir, me, c, opts.n_slice);
ref = timetracking.readImpactTFort18(impactt_dir);

% --- Print final-state comparison ---
print_final_comparison(bm_lt, bm_imp, sl_lt, sl_imp, last, me, c);

% --- Print evolution comparison ---
print_evolution(bunches, ref, me, c);

end


% ====================================================================
function [bm, sl] = compute_stats(b, me, c, n_slice)
% Compute bunch-mean and slice stats from a particle dump struct.
% FILTERS dead particles (collimator-killed, parked at z=-1e9).
xx = double(b.x); yy = double(b.y); zz = double(b.z);
px = double(b.px); py = double(b.py); pz = double(b.pz);
alive = (zz > -1e6);   % dead particles parked at z=-1e9
xx = xx(alive); yy = yy(alive); zz = zz(alive);
px = px(alive); py = py(alive); pz = pz(alive);
N  = numel(xx);
p2 = px.*px + py.*py + pz.*pz;
g  = sqrt(1 + p2/(me*c)^2);

bm.N        = N;
bm.t        = b.time;
bm.z_mean   = mean(zz);
bm.gamma    = mean(g);
bm.sigma_x  = std(xx);
bm.sigma_y  = std(yy);
bm.sigma_z  = std(zz);
bm.sigma_gamma = std(g);
bm.sigma_E_MeV = std(g) * 0.5109989461;        % MeV
bm.bunch_len_fwhm = sigma_to_fwhm(zz);          % m (FWHM of z dist)
bm.eps_nx   = norm_emittance(xx, px./max(pz,1e-30), g);
bm.eps_ny   = norm_emittance(yy, py./max(pz,1e-30), g);
% Longitudinal: eps_nz = sqrt(<dz^2><dgamma^2> - <dz*dgamma>^2) in m
dz = zz - mean(zz);
dg = g  - mean(g);
bm.eps_nz   = sqrt(max(mean(dz.^2)*mean(dg.^2) - mean(dz.*dg)^2, 0));

% Slice emittances (peak-current slice)
edges = linspace(min(zz), max(zz), n_slice+1);
[~, ~, bin] = histcounts(zz, edges);
sl.eps_nx_slice = nan(n_slice, 1);
sl.eps_ny_slice = nan(n_slice, 1);
sl.current      = nan(n_slice, 1);
sl.z_centers    = 0.5*(edges(1:end-1) + edges(2:end));
for k = 1:n_slice
    msk = (bin == k);
    if sum(msk) < 5, continue; end
    xk = xx(msk); pxk = px(msk); pzk = pz(msk);
    yk = yy(msk); pyk = py(msk);
    gk = g(msk);
    sl.eps_nx_slice(k) = norm_emittance(xk, pxk./max(pzk,1e-30), gk);
    sl.eps_ny_slice(k) = norm_emittance(yk, pyk./max(pzk,1e-30), gk);
    dz = edges(k+1) - edges(k);
    Nk = sum(msk);
    sl.current(k) = Nk / N * dz;   % normalised slice density
end
[~, peak_k] = max(sl.current);
bm.eps_nx_peak_slice = sl.eps_nx_slice(peak_k);
bm.eps_ny_peak_slice = sl.eps_ny_slice(peak_k);
bm.peak_slice_z      = sl.z_centers(peak_k);
end


function [bm, sl] = read_imp_final(impactt_dir, me, c, n_slice)
% Read fort.50 (final particles in lab frame) and compute matching stats.
fid = fopen(fullfile(impactt_dir, 'fort.50'), 'r');
D = textscan(fid, '%f %f %f %f %f %f %f %f %f');
fclose(fid);
% fort.50 columns per IMPACT-T source (Distribution.f90 in active repo,
% Pts1(9, Nptlocal) layout): (x, px, y, py, z, pz, q/m, q_macro, id)
% with px/py/pz in normalised units = beta*gamma. Earlier interpretation of
% col 5 as beta_z and col 6 as gamma was wrong; gamma = sqrt(1+|p|^2)
% computed from cols 2,4,6 instead.
xx = D{1}; px_n = D{2}; yy = D{3}; py_n = D{4}; zz = D{5}; pz_n = D{6};
gam = sqrt(1 + px_n.^2 + py_n.^2 + pz_n.^2);

% z-position from fort.50 not directly stored — fort.50 is at FIXED TIME, all
% particles drifted to common t. The "z" we need is from fort.40 (initial)
% propagated forward. As ImpactT outputs gun-exit projection in fort.50, we
% need a different file for z. Use fort.26 evolution + fort.50 transverse.
%
% For SLICE EMITTANCE in z, ImpactT provides fort.60+ but the format varies.
% Best approach: use fort.50 transverse stats and fort.26 longitudinal.

% So we report what we CAN from fort.50:
bm.N        = numel(xx);
bm.gamma    = mean(gam);
bm.sigma_x  = std(xx);
bm.sigma_y  = std(yy);
bm.sigma_gamma = std(gam);
bm.sigma_E_MeV = std(gam) * 0.5109989461;
bm.eps_nx   = norm_emittance(xx, px_n./max(pz_n,1e-30), gam);
bm.eps_ny   = norm_emittance(yy, py_n./max(pz_n,1e-30), gam);

% Pull longitudinal stats from fort.26 (last row)
ref = timetracking.readImpactTFort18(impactt_dir);
bm.t        = ref.t(end);
bm.z_mean   = ref.z_mean(end);
bm.sigma_z  = ref.sigma_z(end);
bm.bunch_len_fwhm = bm.sigma_z * 2.3548;   % approx FWHM = 2*sqrt(2 ln 2)*sigma
bm.eps_nz   = ref.eps_nz(end);

% Slice transverse emittance from fort.50 — but no z per particle!
% Approximate: assume fort.50's particle ORDERING preserves emission time
% (often true for ImpactT). Bin by ID to get pseudo-slices.
% This is an approximation; for true z-binned slice, would need fort.27/28
% slice diagnostics.
sl.note = 'ImpactT slice emittance not directly available without full z; approximated via fort.50 N-bin if needed.';
sl.eps_nx_slice = [];
sl.eps_ny_slice = [];
sl.current = [];
sl.z_centers = [];
bm.eps_nx_peak_slice = NaN;
bm.eps_ny_peak_slice = NaN;
bm.peak_slice_z      = NaN;
end


function eps_n = norm_emittance(x, xp, g)
sx  = std(x);
sxp = std(xp);
cxxp = mean((x - mean(x)) .* (xp - mean(xp)));
eps_geom = sqrt(max(sx^2 * sxp^2 - cxxp^2, 0));
bg = sqrt(g.^2 - 1);
eps_n = mean(bg) * eps_geom;
end


function fwhm = sigma_to_fwhm(z)
[counts, edges] = histcounts(z, 100);
ctrs = 0.5*(edges(1:end-1) + edges(2:end));
mx = max(counts);
above = counts > 0.5*mx;
if sum(above) < 2, fwhm = 0; return; end
fwhm = ctrs(find(above, 1, 'last')) - ctrs(find(above, 1, 'first'));
end


function v = coalesce(x)
if isempty(x), v = false; else, v = logical(x); end
end


function v = ifempty(x, dflt)
if isempty(x), v = dflt; else, v = x; end
end


function print_final_comparison(bm_lt, bm_imp, sl_lt, sl_imp, last, me, c)
fprintf('\n=== FINAL STATE COMPARISON (end of L0A) ===\n');
fprintf('%-22s %-14s %-14s %-10s\n', 'Quantity', 'lucretia-tt', 'ImpactT', 'ratio');
fprintf('%s\n', repmat('-', 1, 70));
P = @(name, lt, imp, fmt) fprintf('%-22s %-14s %-14s %-10.3f\n', name, ...
        sprintf(fmt, lt), sprintf(fmt, imp), lt/imp);

fprintf('%-22s %-14d %-14d   --\n', 'N (alive)', bm_lt.N, bm_imp.N);
P('t [ns]',          bm_lt.t*1e9,        bm_imp.t*1e9,         '%.3f');
P('z_mean [m]',      bm_lt.z_mean,       bm_imp.z_mean,        '%.4f');
P('gamma_mean',      bm_lt.gamma,        bm_imp.gamma,         '%.3f');
P('sigma_x [um]',    bm_lt.sigma_x*1e6,  bm_imp.sigma_x*1e6,   '%.2f');
P('sigma_y [um]',    bm_lt.sigma_y*1e6,  bm_imp.sigma_y*1e6,   '%.2f');
P('sigma_z [um]',    bm_lt.sigma_z*1e6,  bm_imp.sigma_z*1e6,   '%.2f');
P('FWHM_z [um]',     bm_lt.bunch_len_fwhm*1e6, bm_imp.bunch_len_fwhm*1e6, '%.2f');
P('sigma_gamma',     bm_lt.sigma_gamma,  bm_imp.sigma_gamma,   '%.4f');
P('sigma_E [keV]',   bm_lt.sigma_E_MeV*1e3, bm_imp.sigma_E_MeV*1e3, '%.2f');
P('eps_nx [um.rad]', bm_lt.eps_nx*1e6,   bm_imp.eps_nx*1e6,    '%.4f');
P('eps_ny [um.rad]', bm_lt.eps_ny*1e6,   bm_imp.eps_ny*1e6,    '%.4f');
P('eps_nz [um.rad]', bm_lt.eps_nz*1e6,   bm_imp.eps_nz*1e6,    '%.4f');

fprintf('\n  Slice (peak-current) eps_nx [um.rad]: ltt=%.4f  imp=%s\n', ...
    bm_lt.eps_nx_peak_slice*1e6, ...
    iif(isnan(bm_imp.eps_nx_peak_slice), '--N/A--', sprintf('%.4f', bm_imp.eps_nx_peak_slice*1e6)));
fprintf('  Slice (peak-current) eps_ny [um.rad]: ltt=%.4f  imp=%s\n', ...
    bm_lt.eps_ny_peak_slice*1e6, ...
    iif(isnan(bm_imp.eps_ny_peak_slice), '--N/A--', sprintf('%.4f', bm_imp.eps_ny_peak_slice*1e6)));

% Save .mat for later inspection
save('bench_full_final.mat', 'bm_lt', 'bm_imp', 'sl_lt', 'sl_imp');
fprintf('\nSaved -> bench_full_final.mat\n');
end


function print_evolution(bunches, ref, me, c)
fprintf('\n=== EVOLUTION (lucretia-tt dumps interpolated to ImpactT) ===\n');
fprintf('%-9s %-8s | %-7s %-7s | %-8s %-8s | %-8s %-8s | %-9s %-9s\n', ...
    't[ns]', 'z[mm]', 'lt-gam', 'imp-gam', 'lt-sx[mm]', 'imp-sx[mm]', ...
    'lt-sz[mm]', 'imp-sz[mm]', 'lt-eps[um]', 'imp-eps[um]');
fprintf('%s\n', repmat('-', 1, 100));
for k = 1:numel(bunches)
    b = bunches{k};
    if numel(b.x) < 5, continue; end
    xx=double(b.x); yy=double(b.y); zz=double(b.z);
    px=double(b.px); py=double(b.py); pz=double(b.pz);
    alive = (zz > -1e6);   % filter dead (collimator-killed) particles
    xx=xx(alive); zz=zz(alive); px=px(alive); py=py(alive); pz=pz(alive);
    if numel(xx) < 5, continue; end
    gam = sqrt(1 + (px.^2+py.^2+pz.^2)/(me*c)^2);
    xp = px./max(pz,1e-30);
    sx = std(xx);
    sz = std(zz);
    eps = norm_emittance(xx, xp, gam);
    tk = b.time;
    if tk < ref.t(1) || tk > ref.t(end), continue; end
    g_imp = interp1(ref.t, ref.gamma,   tk, 'linear');
    sx_imp = interp1(ref.t, ref.sigma_x, tk, 'linear');
    sz_imp = interp1(ref.t, ref.sigma_z, tk, 'linear');
    eps_imp = interp1(ref.t, ref.eps_nx, tk, 'linear');
    fprintf('%-9.2f %-8.1f | %-7.2f %-7.2f | %-9.3f %-9.3f | %-9.4f %-9.4f | %-10.3f %-10.3f\n', ...
        tk*1e9, mean(zz)*1e3, mean(gam), g_imp, ...
        sx*1e3, sx_imp*1e3, sz*1e3, sz_imp*1e3, ...
        eps*1e6, eps_imp*1e6);
end
end


function v = iif(cond, a, b)
if cond, v = a; else, v = b; end
end
