function [status, log] = invokeBinary(obj, in_file)
% Run the lucretia-tt binary on the given input file inside work_dir,
% capturing combined stdout/stderr. Uses mpirun when obj.mpi_nranks > 1.

nranks = 1;
if isprop(obj, 'mpi_nranks') && ~isempty(obj.mpi_nranks)
    nranks = max(1, round(double(obj.mpi_nranks)));
end

% On Linux, strip MATLAB's bundled libs from LD_LIBRARY_PATH so the
% lucretia-tt binary (linked against the host's libstdc++) can find
% the right symbols. No-op on Mac and when LD_LIBRARY_PATH has no
% MATLAB entries.
envP = timetracking.linux_matlab_env_prefix();

if nranks <= 1
    cmd = sprintf('cd "%s" && %s"%s" "%s" 2>&1', ...
                  obj.work_dir, envP, obj.binary, in_file);
else
    mpirun = '';
    if isprop(obj, 'mpirun_bin') && ~isempty(obj.mpirun_bin)
        mpirun = obj.mpirun_bin;
    end
    if isempty(mpirun)
        for cand = {'/opt/homebrew/bin/mpirun', '/usr/local/bin/mpirun', 'mpirun'}
            [s, ~] = system(sprintf('command -v %s', cand{1}));
            if s == 0, mpirun = strtrim(cand{1}); break; end
        end
    end
    if isempty(mpirun)
        error('TimeTrack:invokeBinary:noMpirun', ...
              ['mpi_nranks=%d but no mpirun found. Install Open MPI ' ...
               '(brew install open-mpi) or set obj.mpirun_bin.'], nranks);
    end
    extra = '';
    if isprop(obj, 'mpirun_extra_args') && ~isempty(obj.mpirun_extra_args)
        extra = [obj.mpirun_extra_args ' '];
    end
    % Pin OMP threads per rank to total_physical_cores / nranks so that
    % nranks * OMP_NUM_THREADS does not oversubscribe physical cores.
    % Otherwise default OMP=10 + nranks=4 = 40 threads on a 10-core box ->
    % scheduler thrashing kills MPI scaling. Override via mpirun_extra_args
    % for fine-tuning (e.g. '-x OMP_NUM_THREADS=4 ...' to set explicitly).
    omp_per_rank = '';
    [s, c] = system('sysctl -n hw.physicalcpu 2>/dev/null');
    if s == 0
        ncores = str2double(strtrim(c));
        if isfinite(ncores) && ncores > 0
            per = max(1, floor(ncores / nranks));
            omp_per_rank = sprintf('-x OMP_NUM_THREADS=%d ', per);
        end
    end
    cmd = sprintf('cd "%s" && %s"%s" %s%s-np %d "%s" "%s" 2>&1', ...
                  obj.work_dir, envP, mpirun, omp_per_rank, extra, ...
                  nranks, obj.binary, in_file);
end

% '-echo' streams stdout to the MATLAB Command Window AS THE BINARY
% RUNS while still capturing the full transcript into `log`. Useful for
% long Injector / production runs (BeamMonitor lines, AMReX milestones).
% For batch / regression tests we keep the run quiet (default).
verbose = isprop(obj, 'verbose_run') && ~isempty(obj.verbose_run) && obj.verbose_run;
if verbose
    [status, log] = system(cmd, '-echo');
else
    [status, log] = system(cmd);
end
end
