function bench_gun_only(varargin)
% BENCH_GUN_ONLY  Fast gun-region-only cross-code benchmark vs ImpactT.
%
% Strips L0A and beyond from the LCLS injector lattice. Keeps GUN + SOL1
% + corrector quads + collimator (truncated at z<0.7m). Runs ~1 ns of
% tracking through gun + solenoid drift, ending around z=0.30 m.
%
% Use this for fast iteration on SC field structure / kick comparisons.
% Full-lattice run takes ~5 min; gun-only takes ~30 sec.
%
% Usage: bench_gun_only('sc_mode', 'mesh', 'sc_static_xrad', [], ...)
%
% All bench_full_lattice options pass through.

p = inputParser;
p.KeepUnmatched = true;
p.addParameter('impactt_dir', '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT', @(x) ischar(x) || isstring(x));
p.addParameter('n_macros',    50000, @isnumeric);
p.addParameter('lattice_end_z', 0.30, @isnumeric);   % m: drop lattice elements past this z
p.addParameter('n_steps_override', [], @(x) isempty(x) || isnumeric(x));
p.addParameter('tag',         'bench_gun', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;
extra = p.Unmatched;

% --- Estimate n_steps to reach the target z ---
% imp's dt_change_z is 0.25m. Until then dt=0.3ps. Beyond, dt=4ps.
% Time to reach z=0.25m: bunch accelerates through gun region; centroid
% takes ~500ps (1700 steps at dt=0.3ps). Beyond 0.25m, c*dt_after = 1.2mm
% per step. For LCLS injector the relevant gun-only span is z<0.30m so
% only a few steps beyond dt_change.
if isempty(opts.n_steps_override)
    n_steps = 2200;   % covers z up to ~0.30m
else
    n_steps = opts.n_steps_override;
end
fprintf('=== Gun-only bench (LCLS injector, lattice_end_z = %.2f m) ===\n', opts.lattice_end_z);
fprintf('  n_steps: %d  n_macros: %d\n', n_steps, opts.n_macros);

% --- Read full lattice, strip past lattice_end_z ---
[lattice, ~, geom_imp, tracking_imp, ~] = timetracking.readImpactTLattice( ...
    opts.impactt_dir, 'n_macros', opts.n_macros);
% Filter by element type + name: we want everything in the gun region
% only. For LCLS injector that's GUN + SOL1 + the corrector quads.
% Drop wakefield_L0A and any L0A_* elements and the post-L0A stop_bkw.
% Keep the gun-region collimator if any. Always keep cathode_source +
% beam_monitor (the latter is replaced below).
drop_names = {'wakefield_L0A', 'L0A_entrance', 'L0A_body_1', 'L0A_body_2', ...
              'L0A_exit', 'stop_bkw'};
keep = true(size(lattice));
for k = 1:numel(lattice)
    if ismember(lattice{k}.name, drop_names)
        keep(k) = false;
    end
end
lattice = lattice(keep);

% Pass to the standard bench harness via a wrapper. The cleanest way:
% Build TimeTrack here mirroring bench_full_lattice but with our truncated
% lattice. Reuse common code.
fid = fopen(fullfile(opts.impactt_dir, 'ImpactT.in'), 'r');
hd = textscan(fid, '%s', 'Delimiter', '\n');  fclose(fid);  hd = hd{1};
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');
Bcurr = h9(1); Bfreq = h9(5); Tini = h9(6);
Bkenergy = h9(2); Bmass = h9(3);
q_total = abs(Bcurr) / Bfreq;

keep = true(size(lattice));
for k = 1:numel(lattice)
    if strcmp(lattice{k}.type, 'cathode_source'), keep(k) = false; end
end
lattice = lattice(keep);

mon_idx = find(cellfun(@(e) strcmp(e.type, 'beam_monitor'), lattice), 1);
dump_every = max(50, floor(n_steps / 25));
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

tt = TimeTrack();
tt.lattice = lattice;
tt.beam = struct('partcl_file', fullfile(opts.impactt_dir, 'partcl.data'), ...
                 'partcl_q_total', q_total, 't_init', Tini);
tt.geom_lo    = [-3e-3, -3e-3, -0.05];
tt.geom_hi    = [ 3e-3,  3e-3,  0.70];
tt.geom_ncell = geom_imp.ncell(:).';
tt.t_start    = Tini;
tt.dt         = 0.3e-12;
if isfield(tracking_imp, 'dt_change_z') && ~isempty(tracking_imp.dt_change_z)
    tt.dt_change_z = tracking_imp.dt_change_z;
elseif isfield(tracking_imp, 'dt_change')  && ~isempty(tracking_imp.dt_change)
    tt.dt_change_t = tracking_imp.dt_change;
end
tt.dt_after = 4e-12;
tt.n_steps  = n_steps;
tt.behind_cathode_z        = 0.0;
tt.behind_cathode_betazini = sqrt(1.0 - 1.0/(1.0 + Bkenergy/Bmass)^2);

% --- SC defaults: imp-match config ---
tt.enable_space_charge = true;
tt.sc_adaptive         = true;
tt.sc_pad_factor       = 5.0;
tt.sc_image_plane      = true;
tt.sc_image_plane_z    = 0.0;
tt.sc_image_cutoff     = 0.05;
tt.sc_use_b_field      = true;
tt.self_force_direct   = true;
tt.sc_integer_cell_shift = true;

% Apply any extra options from caller
extra_fields = fieldnames(extra);
for k = 1:numel(extra_fields)
    fn = extra_fields{k};
    val = extra.(fn);
    % Map bench_full_lattice-style names to TimeTrack properties
    switch fn
        case 'sc_mode'
            sc_mode = lower(char(val));
            tt.enable_space_charge = ismember(sc_mode, {'full', 'mesh'});
            tt.sc_adaptive         = ismember(sc_mode, {'full', 'mesh'});
            tt.enable_slice_sc     = ismember(sc_mode, {'full', 'slice'});
            tt.sc_image_plane      = ismember(sc_mode, {'full', 'mesh'});
        case 'sc_static_xrad'
            if ~isempty(val)
                xrad = val;
                tt.geom_lo = [-xrad, -xrad, -0.005];
                tt.geom_hi = [ xrad,  xrad,  0.005];
                tt.sc_adaptive = false;
                tt.sc_comoving = true;
            end
        otherwise
            if isprop(tt, fn), tt.(fn) = val;
            else, fprintf('  (skipping unknown extra opt: %s)\n', fn);
            end
    end
end

tt.work_dir = ['/tmp/bench_gun_' char(opts.tag)];
if exist(tt.work_dir, 'dir'), rmdir(tt.work_dir, 's'); end

t0 = tic; tt.run(); fprintf('Run time: %.1f s\n', toc(t0));

% --- Compute final stats and compare to imp at the same z ---
bunches = tt.readDumps();
fprintf('Dumps: %d\n', numel(bunches));
if isempty(bunches)
    fprintf('No dumps -- aborting analysis.\n');
    return;
end

me = 9.1093837015e-31; c = 299792458;
last = bunches{end};
xx = double(last.x); yy = double(last.y); zz = double(last.z);
px = double(last.px); py = double(last.py); pz = double(last.pz);
% Filter to alive AND emerged. Particles still drifting behind cathode
% (z<0) are "alive" but haven't been accelerated yet -- they inflate
% sigma_z and eps if included. Use z > 0 cutoff to match imp's flagpos=1
% diagnostic convention. Also filter dead particles (z<-1e6).
alive = (zz > 0); xx=xx(alive); yy=yy(alive); zz=zz(alive); px=px(alive); py=py(alive); pz=pz(alive);
N  = numel(xx);
g  = sqrt(1 + (px.^2 + py.^2 + pz.^2)/(me*c)^2);
sx = std(xx); sy = std(yy); sz = std(zz);
sg = std(g);
xp = px./max(pz, 1e-30);
yp = py./max(pz, 1e-30);
eps_nx = sqrt(max(std(xx)^2*std(xp)^2 - mean((xx-mean(xx)).*(xp-mean(xp)))^2, 0)) * mean(sqrt(g.^2-1));
eps_ny = sqrt(max(std(yy)^2*std(yp)^2 - mean((yy-mean(yy)).*(yp-mean(yp)))^2, 0)) * mean(sqrt(g.^2-1));

fprintf('\n=== END-OF-GUN STATE (lt) ===\n');
fprintf('  t = %.3f ns  z_mean = %.4f m\n', last.time*1e9, mean(zz));
fprintf('  N alive = %d\n', N);
fprintf('  gamma = %.3f  sigma_gamma = %.3f\n', mean(g), sg);
fprintf('  sigma_x = %.0f um  sigma_y = %.0f um  sigma_z = %.0f um\n', sx*1e6, sy*1e6, sz*1e6);
fprintf('  eps_nx = %.3f um.rad  eps_ny = %.3f um.rad\n', eps_nx*1e6, eps_ny*1e6);

% --- Read imp evolution and find imp's state at the same z_mean ---
ref = timetracking.readImpactTFort18(opts.impactt_dir);
target_z = mean(zz);
[~, ki] = min(abs(ref.z_mean - target_z));
fprintf('\n=== imp at matching z = %.4f m (closest sample) ===\n', ref.z_mean(ki));
fprintf('  t = %.3f ns  z = %.4f m\n', ref.t(ki)*1e9, ref.z_mean(ki));
fprintf('  gamma = %.3f\n', ref.gamma(ki));
fprintf('  sigma_x = %.0f um  sigma_y = %.0f um  sigma_z = %.0f um\n', ...
        ref.sigma_x(ki)*1e6, ref.sigma_y(ki)*1e6, ref.sigma_z(ki)*1e6);
fprintf('  eps_nx = %.3f um.rad  eps_ny = %.3f um.rad\n', ...
        ref.eps_nx(ki)*1e6, ref.eps_ny(ki)*1e6);

fprintf('\n=== RATIOS lt/imp ===\n');
P = @(name, lt, imp) fprintf('  %-12s  lt=%.4g  imp=%.4g  ratio=%.3f\n', ...
                              name, lt, imp, lt/imp);
P('gamma',        mean(g),          ref.gamma(ki));
P('sigma_x [um]', sx*1e6,           ref.sigma_x(ki)*1e6);
P('sigma_z [um]', sz*1e6,           ref.sigma_z(ki)*1e6);
P('eps_nx [um]',  eps_nx*1e6,       ref.eps_nx(ki)*1e6);

save('bench_gun_final.mat', 'last', 'ref', 'ki', 'sx', 'sz', 'eps_nx', 'eps_ny', 'sg', 'N');
fprintf('\nSaved -> bench_gun_final.mat\n');
end
