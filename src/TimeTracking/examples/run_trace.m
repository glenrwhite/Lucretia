function run_trace(varargin)
% RUN_TRACE  Wrapper to run partcl-seed test with per-step trace enabled.

p = inputParser;
p.addParameter('tag',        'trace',         @(x) ischar(x) || isstring(x));
p.addParameter('n_steps',    6500,            @isnumeric);
p.addParameter('trace_every', 1,              @isnumeric);
p.addParameter('dt_override', [],             @(x) isempty(x) || isnumeric(x));   % override tracking.dt (s)
p.addParameter('use_particle_phase', false, @islogical);   % per-particle z/c phase (overrides centroid)
p.addParameter('use_wallclock_phase', false, @islogical);  % wallclock t (overrides centroid; ImpactT default)
p.addParameter('use_midstep_field', false, @islogical);    % gather field at x + 0.5*dt*v
p.addParameter('use_dkd', true, @islogical);               % ImpactT drift-kick-drift integrator (default ON)
p.addParameter('enable_sc', false, @islogical);            % enable mesh + slice space-charge
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
hd = textscan(fid, '%s', 'Delimiter', '\n');  fclose(fid);  hd = hd{1};
data_lines = hd(~startsWith(strtrim(hd), '!') & ~cellfun('isempty', strtrim(hd)));
h9 = sscanf(data_lines{9}, '%f');
Bcurr = h9(1); Bfreq = h9(5); Tini = h9(6);
Bkenergy = h9(2); Bmass = h9(3);
q_total = abs(Bcurr) / Bfreq;

mon = timetracking.BeamMonitor('name', 'mon', 'dump_every', 1000);
lattice{end+1} = mon;

tt = TimeTrack();
tt.lattice = lattice;
tt.beam = struct('partcl_file', fullfile(impactt_dir, 'partcl.data'), ...
                 'partcl_q_total', q_total, 't_init', Tini);
tt.geom_lo    = [-3e-3, -3e-3, -0.05];
tt.geom_hi    = [ 3e-3,  3e-3,  0.70];
tt.geom_ncell = geom_imp.ncell(:).';
tt.t_start    = Tini;
if ~isempty(opts.dt_override)
    tt.dt = opts.dt_override;
    % Hold the small dt all the way through gun region; switch only for far drift
    tt.dt_change_t = tracking_imp.dt_change;
    tt.dt_after    = tracking_imp.dt_after;
else
    tt.dt          = tracking_imp.dt;
    tt.dt_change_t = tracking_imp.dt_change;
    tt.dt_after    = tracking_imp.dt_after;
end
tt.n_steps    = opts.n_steps;
tt.behind_cathode_z        = 0.0;
tt.behind_cathode_betazini = sqrt(1.0 - 1.0/(1.0 + Bkenergy/Bmass)^2);
% Phase mode: TimeTrack defaults to wallclock + DKD (matches ImpactT). Opt-in
% to centroid or per-particle phase for diagnostic comparisons.
if opts.use_particle_phase
    tt.use_centroid_phase  = false;
    tt.use_particle_phase  = true;
elseif opts.use_wallclock_phase
    % already default
end
if opts.use_midstep_field
    tt.use_midstep_field   = true;
end
tt.use_dkd_integrator = opts.use_dkd;
if opts.enable_sc
    tt.enable_space_charge   = true;
    tt.sc_adaptive           = true;
    tt.enable_slice_sc       = true;
    tt.sc_image_plane        = true;
end
tt.work_dir = ['/tmp/partcl_seed_run_' char(opts.tag)];
tt.trace_file  = fullfile(tt.work_dir, 'trace.csv');
tt.trace_every = opts.trace_every;

if exist(tt.work_dir, 'dir'), rmdir(tt.work_dir, 's'); end
fprintf('run_trace: dt=%.2g ps n_steps=%d trace_every=%d\n', ...
        tt.dt*1e12, tt.n_steps, opts.trace_every);
tic; tt.run(); fprintf('  RUN TIME: %.1f s\n', toc);
trace_compare('ltt_trace', tt.trace_file, 'impactt_dir', impactt_dir);
end
