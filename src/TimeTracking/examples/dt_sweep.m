function dt_sweep(varargin)
% DT_SWEEP  Test if smaller dt closes the sigma_gamma residual.
%
% Usage:
%   dt_sweep('dt_ps', 0.1, 'tag', 'dt01')

p = inputParser;
p.addParameter('dt_ps',     0.3,                @isnumeric);
p.addParameter('tag',       'dt_sweep',         @(x) ischar(x) || isstring(x));
p.addParameter('impactt_dir', '/tmp/impactT_noSC', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;

impactt_dir = char(opts.impactt_dir);
[lattice, ~, geom_imp, tracking_imp, ~] = timetracking.readImpactTLattice( ...
    impactt_dir, 'n_macros', 50000);
keep = false(size(lattice));
for k = 1:numel(lattice)
    keep(k) = ismember(lattice{k}.name, {'GUN', 'SOL1'});
end
lattice = lattice(keep);

fid = fopen(fullfile(impactt_dir, 'ImpactT.in'), 'r');
hd = textscan(fid, '%s', 'Delimiter', '\n'); fclose(fid); hd = hd{1};
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');
Bcurr = h9(1); Bfreq = h9(5); Tini = h9(6);
Bkenergy = h9(2); Bmass = h9(3);
q_total = abs(Bcurr) / Bfreq;

mon = timetracking.BeamMonitor('name', 'mon', 'dump_every', 200);
lattice{end+1} = mon;

tt = TimeTrack();
tt.lattice = lattice;
tt.beam = struct('partcl_file', fullfile(impactt_dir, 'partcl.data'), ...
                 'partcl_q_total', q_total, 't_init', Tini);
tt.geom_lo    = [-3e-3, -3e-3, -0.05];
tt.geom_hi    = [ 3e-3,  3e-3,  0.70];
tt.geom_ncell = geom_imp.ncell(:).';
tt.t_start    = Tini;
tt.dt         = opts.dt_ps * 1e-12;
% Scale n_steps to reach z=1.6m at the new dt
factor = 0.3 / opts.dt_ps;
tt.n_steps    = round(6500 * factor);
tt.dt_change_t = tracking_imp.dt_change;
% Keep the post-switch dt at the same SCALE relative to dt
tt.dt_after   = tracking_imp.dt_after * (opts.dt_ps / 0.3);
tt.behind_cathode_z        = 0.0;
tt.behind_cathode_betazini = sqrt(1.0 - 1.0/(1.0 + Bkenergy/Bmass)^2);
tt.use_centroid_phase      = true;
tt.centroid_t_offset       = 0.0;
tt.work_dir = ['/tmp/partcl_seed_run_' char(opts.tag)];

fprintf('dt_sweep: dt=%.2g ps, n_steps=%d, dt_after=%.2g ps\n', ...
        tt.dt*1e12, tt.n_steps, tt.dt_after*1e12);
if exist(tt.work_dir, 'dir'), rmdir(tt.work_dir, 's'); end

t0 = tic;
tt.run();
fprintf('  RUN TIME: %.1f s\n', toc(t0));
diagnose_chirp('work_dir', tt.work_dir, 'impactt_dir', impactt_dir);
end
