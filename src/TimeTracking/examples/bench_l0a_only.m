function bench_l0a_only(varargin)
% BENCH_L0A_ONLY  Lattice-swap test: inject imp's bunch at z=0.20m into
% lt and run only through the L0A region. Compares to imp's L0A end.
%
% This isolates whether the lt-vs-imp residual eps_nx 4.5x is from:
%   - GUN physics (different gun-imprinted phase-space orientation)
%   - L0A physics (different wakefield, RF, SC during L0A acceleration)
%
% If lt evolves imp's bunch to imp's end-state, the gun is the culprit.
% If lt evolves imp's bunch to its OWN end-state (4.5x eps_nx), L0A is.

p = inputParser;
p.addParameter('impactt_dir', '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT', @(x) ischar(x) || isstring(x));
p.addParameter('seed_file',   '/tmp/claude/imp_z200_seed.data', @(x) ischar(x) || isstring(x));
p.addParameter('t_start',     0.68369593e-9, @isnumeric);   % imp's t at z=0.199m (per fort.18)
p.addParameter('z_start',     0.199, @isnumeric);             % bunch centroid z at start
p.addParameter('n_steps',     1300, @isnumeric);              % roughly to L0A end
p.addParameter('tag',         'l0a_only', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

% --- Read lattice, strip everything BEFORE L0A_entrance ---
[lattice, ~, geom_imp, tracking_imp, ~] = timetracking.readImpactTLattice( ...
    opts.impactt_dir, 'n_macros', 48442);
fid = fopen(fullfile(opts.impactt_dir, 'ImpactT.in'), 'r');
hd = textscan(fid, '%s', 'Delimiter', '\n'); fclose(fid); hd = hd{1};
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');
Bcurr = h9(1); Bfreq = h9(5); Tini = h9(6);
q_total = abs(Bcurr) / Bfreq;

% Drop pre-L0A elements (gun, SOL1, etc.) and cathode_source
keep_names = {'wakefield_L0A','L0A_entrance','L0A_body_1','L0A_body_2',...
              'L0A_exit','stop_bkw','mon','e8','global_xrad_aperture'};
keep = false(size(lattice));
for k = 1:numel(lattice)
    if ismember(lattice{k}.name, keep_names), keep(k) = true; end
    % Always drop cathode (we have a seed file; no emission)
    if isfield(lattice{k}, 'type') && strcmp(lattice{k}.type, 'cathode_source')
        keep(k) = false;
    end
end
lattice = lattice(keep);

mon_idx = find(cellfun(@(e) strcmp(e.type, 'beam_monitor'), lattice), 1);
dump_every = max(20, floor(opts.n_steps / 30));
if ~isempty(mon_idx)
    lattice{mon_idx} = timetracking.BeamMonitor('name', 'mon', 'dump_every', dump_every);
else
    lattice{end+1} = timetracking.BeamMonitor('name', 'mon', 'dump_every', dump_every);
end
fprintf('  L0A-only lattice elements: %d  (BeamMonitor every %d steps)\n', ...
        numel(lattice), dump_every);
for k = 1:numel(lattice)
    el = lattice{k};
    fprintf('    [%2d] %-22s type=%-18s\n', k, el.name, el.type);
end

% --- Build TimeTrack with imp's z=0.20 bunch as seed ---
tt = TimeTrack();
tt.lattice = lattice;
tt.beam = struct('partcl_file', opts.seed_file, ...
                 'partcl_q_total', q_total, 't_init', opts.t_start);

tt.geom_lo = [-3e-3, -3e-3, opts.z_start - 5e-3];
tt.geom_hi = [ 3e-3,  3e-3, opts.z_start + 5e-3];
tt.geom_ncell = geom_imp.ncell(:).';

tt.t_start  = opts.t_start;
tt.dt       = 4e-12;             % L0A region uses dt=4ps
tt.n_steps  = opts.n_steps;
tt.behind_cathode_z        = -1;  % disable behind-cathode mode (all particles emerged)
tt.behind_cathode_betazini = 0;

tt.enable_space_charge = true;
tt.sc_adaptive   = true;
tt.sc_pad_factor = 5.0;
tt.sc_image_plane = false;        % bunch is past gun, no image needed
tt.sc_use_b_field = true;
tt.self_force_direct = true;
tt.sc_integer_cell_shift = true;

tt.work_dir = ['/tmp/bench_l0a_' char(opts.tag)];
if exist(tt.work_dir, 'dir'), rmdir(tt.work_dir, 's'); end

t0 = tic; tt.run(); fprintf('Run time: %.1f s\n', toc(t0));

bunches = tt.readDumps();
if isempty(bunches)
    fprintf('No dumps. Aborting.\n'); return;
end
me = 9.1093837015e-31; c = 299792458;
last = bunches{end};
xx = double(last.x); yy = double(last.y); zz = double(last.z);
px = double(last.px); py = double(last.py); pz = double(last.pz);
alive = (zz > 0); xx=xx(alive); yy=yy(alive); zz=zz(alive); px=px(alive); py=py(alive); pz=pz(alive);
g  = sqrt(1 + (px.^2 + py.^2 + pz.^2)/(me*c)^2);
sx = std(xx); sz = std(zz); sg = std(g);
xp = px./max(pz, 1e-30);
cxxp = mean((xx-mean(xx)) .* (xp-mean(xp)));
eps_nx = sqrt(max(std(xx)^2*std(xp)^2 - cxxp^2, 0)) * mean(sqrt(g.^2-1));

fprintf('\n=== L0A-only end-state (lt evolved IMP bunch) ===\n');
fprintf('  t=%.3fns  z_mean=%.4fm  N=%d\n', last.time*1e9, mean(zz), numel(xx));
fprintf('  gamma=%.3f  sigma_gamma=%.3f\n', mean(g), sg);
fprintf('  sigma_x=%.0f um  sigma_z=%.0f um\n', sx*1e6, sz*1e6);
fprintf('  eps_nx=%.3f um.rad\n', eps_nx*1e6);

% Read imp's L0A end (fort.50)
fprintf('\n=== imp final state (its own L0A end) ===\n');
imp_d = readmatrix(fullfile(opts.impactt_dir, 'fort.50'), 'FileType','text');
imp_xs = imp_d(:,1); imp_pxs = imp_d(:,2);
imp_pzs = imp_d(:,6);
imp_gam = sqrt(1 + imp_pxs.^2 + imp_d(:,4).^2 + imp_pzs.^2);
imp_xp  = imp_pxs ./ max(imp_pzs, 1e-30);
imp_sx  = std(imp_xs);
imp_cxxp = mean((imp_xs-mean(imp_xs)) .* (imp_xp-mean(imp_xp)));
imp_eps  = sqrt(max(std(imp_xs)^2*std(imp_xp)^2 - imp_cxxp^2, 0)) * mean(sqrt(imp_gam.^2-1));
fprintf('  N=%d  gamma=%.3f  sigma_x=%.0f um  eps_nx=%.3f\n', ...
        numel(imp_xs), mean(imp_gam), imp_sx*1e6, imp_eps*1e6);

fprintf('\n=== RATIOS lt-evolved-imp-bunch / imp-original ===\n');
fprintf('  sigma_x=%.3f  eps_nx=%.3f\n', sx/imp_sx, eps_nx/imp_eps);

save('bench_l0a_only_final.mat', 'last','sx','sz','eps_nx','sg', ...
     'imp_xs','imp_pxs','imp_eps','imp_sx');
end
