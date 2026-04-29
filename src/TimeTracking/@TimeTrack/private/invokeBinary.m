function [status, log] = invokeBinary(obj, in_file)
% Run the lucretia-tt binary on the given input file inside work_dir,
% capturing combined stdout/stderr. Uses mpirun when obj.mpi_nranks > 1.

nranks = 1;
if isprop(obj, 'mpi_nranks') && ~isempty(obj.mpi_nranks)
    nranks = max(1, round(double(obj.mpi_nranks)));
end

if nranks <= 1
    cmd = sprintf('cd "%s" && "%s" "%s" 2>&1', ...
                  obj.work_dir, obj.binary, in_file);
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
    cmd = sprintf('cd "%s" && "%s" -np %d "%s" "%s" 2>&1', ...
                  obj.work_dir, mpirun, nranks, obj.binary, in_file);
end
[status, log] = system(cmd);
end
