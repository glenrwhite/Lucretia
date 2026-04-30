function results = bench_scaling(varargin)
% BENCH_SCALING  Strong-scaling sweep for lucretia-tt on the current host.
%
% Runs the same problem (drift + space-charge) at multiple
% (nranks, omp_threads_per_rank) combinations and reports wall time +
% strong-scaling efficiency relative to the smallest config. Lets you
% answer "am I making full use of this machine?".
%
% Usage:
%   results = bench_scaling()                                 % defaults
%   results = bench_scaling('macros',  5e7, 'n_steps', 200)
%   results = bench_scaling('configs', {{1,24},{2,12},{4,6}}, ...
%                           'numa_bind', true)
%
% Each config is {nranks, omp_threads_per_rank}. Total threads should
% match the number of physical cores you want to exercise. Use
% `lscpu | grep -E '^Socket|^Core'` to see the layout.
%
% Defaults target a 96-physical-core box (e.g. c4-standard-192) with
% 1 nC / 10^7 macros / 100 steps in a 64^3 SC mesh. Run takes ~1-3
% minutes per config. Total wall time for the default sweep: ~10-15 min.
%
% NUMA binding: pass numa_bind=true to add `--bind-to socket --map-by
% socket` to mpirun. Pins each rank to one socket so OMP threads stay
% NUMA-local. Strongly recommended on multi-socket boxes; harmless
% on single-socket.
%
% Strong-scaling efficiency = (speedup * baseline_threads) /
% (current_threads). 100% is perfect linear scaling. Below ~70% on
% multi-rank configs usually means NUMA crosstalk, FFT communication
% overhead, or MPI binding mismatch.

p = inputParser;
addParameter(p, 'configs', {{1,24}, {1,48}, {1,96}, {2,48}, {4,24}, {8,12}});
addParameter(p, 'macros',  1e7);
addParameter(p, 'n_steps', 100);
addParameter(p, 'numa_bind', false);
parse(p, varargin{:});
opts = p.Results;

fprintf('\nbench_scaling: %.0e macros, %d steps, 64^3 SC mesh\n', ...
        opts.macros, opts.n_steps);
fprintf('NUMA binding via --bind-to socket: %s\n', mat2str(opts.numa_bind));
[~, lscpu] = system('lscpu | grep -E "^(Socket|Core|Model name)" 2>/dev/null');
if ~isempty(lscpu), fprintf('host:\n%s', lscpu); end
fprintf('\n');

seed = build_seed_(opts.macros);

results = struct('nranks', {}, 'omp', {}, 'wall_s', {}, 'pps', {});

old_omp = getenv('OMP_NUM_THREADS');
restore = onCleanup(@() setenv('OMP_NUM_THREADS', old_omp));

for i = 1:numel(opts.configs)
    cfg = opts.configs{i};
    nranks = cfg{1};
    omp    = cfg{2};
    setenv('OMP_NUM_THREADS', num2str(omp));

    tt = TimeTrack();
    if nranks > 1
        tt.mpi_nranks = nranks;
        if opts.numa_bind
            tt.mpirun_extra_args = '--bind-to socket --map-by socket';
        end
    end
    tt.lattice = { ...
        timetracking.Drift3D('name','D','length',1.0), ...
        timetracking.BeamMonitor('name','m','dump_every',opts.n_steps) };
    tt.beam = timetracking.SeedBeam('seed_file', seed);
    tt.geom_lo    = [-3e-3, -3e-3, -2e-3];
    tt.geom_hi    = [ 3e-3,  3e-3,  3e-1];
    tt.geom_ncell = [64, 64, 64];
    tt.dt = 1e-12;
    tt.n_steps = opts.n_steps;
    tt.enable_space_charge = true;

    fprintf('  [%d/%d] nranks=%d omp=%d (total=%d) ... ', ...
            i, numel(opts.configs), nranks, omp, nranks*omp);
    t0 = tic;
    try
        tt.run();
        wall = toc(t0);
        pps  = opts.macros * opts.n_steps / wall;
        fprintf('wall=%6.1f s, %.2e part-step/s\n', wall, pps);
    catch ME
        wall = toc(t0);
        pps  = NaN;
        fprintf('FAILED (%.1f s): %s\n', wall, ME.message);
    end
    results(end+1).nranks = nranks; %#ok<AGROW>
    results(end).omp     = omp;
    results(end).wall_s  = wall;
    results(end).pps     = pps;
end

% Strong-scaling summary
fprintf('\nStrong-scaling summary (relative to first config):\n');
fprintf('   nranks   omp   total   wall(s)   speedup   efficiency\n');
base = results(1).wall_s;
base_threads = results(1).nranks * results(1).omp;
for i = 1:numel(results)
    r = results(i);
    total_threads = r.nranks * r.omp;
    speedup = base / r.wall_s;
    eff = speedup * base_threads / total_threads;
    fprintf('   %5d  %4d  %5d  %8.1f  %7.2fx  %7.0f%%\n', ...
            r.nranks, r.omp, total_threads, r.wall_s, speedup, eff*100);
end

fprintf('\nInterpretation:\n');
fprintf('  - Efficiency >80%%: scaling well, you''re using the resources.\n');
fprintf('  - Efficiency 50-80%%: real speedup but losing some to overhead\n');
fprintf('    (FFT communication, NUMA crosstalk, MPI launch, etc.).\n');
fprintf('  - Efficiency <50%%: bottleneck somewhere; try numa_bind=true\n');
fprintf('    or fewer ranks. The IGF FFT solve floors scaling at small\n');
fprintf('    macro counts -- bump --macros 5e7 or higher to expose\n');
fprintf('    the push-bound regime.\n');
end


function path = build_seed_(N)
me  = 9.1093837015e-31;
qe  = 1.602176634e-19;
c   = 299792458.0;
gamma_in = 50.0;
beta = sqrt(1 - 1/gamma_in^2);
uz0  = gamma_in * beta * c;

path = fullfile(tempdir, sprintf('bench_scaling_seed_%.0e.h5', N));
if exist(path, 'file'), return; end

rng(42);
x  = 0.3e-3*randn(N,1);
y  = 0.3e-3*randn(N,1);
z  = 0.3e-3*randn(N,1);
sigma_u_th = sqrt(0.5 * qe / me);
ux = sigma_u_th*randn(N,1);
uy = sigma_u_th*randn(N,1);
uz = uz0*ones(N,1);
w_per = 1e-9 / (N * qe);    % 1 nC total
timetracking.writeBunchSeed(path, x, y, z, ux, uy, uz, w_per);
end
