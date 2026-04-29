function bunches = readBeamOpenPMD(diag_dir)
% READBEAMOPENPMD  Read all openpmd_*.h5 files in diag_dir into a cell
% array of bunch structs (sorted by simulation step).
%
% Each struct has: step, time, x, y, z, px, py, pz, q, w, id.
% Positions in m, momenta in kg.m/s, charge in C, time in s.

if ~exist(diag_dir, 'dir')
    error('readBeamOpenPMD:noDir', ...
          'Diag directory does not exist: %s', diag_dir);
end

files = dir(fullfile(diag_dir, 'openpmd_*.h5'));
n     = numel(files);
bunches = cell(1, n);

for i = 1:n
    fpath = fullfile(diag_dir, files(i).name);

    % Find the iteration group under /data/<step>/
    inf  = h5info(fpath, '/data');
    if isempty(inf.Groups)
        warning('readBeamOpenPMD:noStep', ...
                'No /data/<step> group in %s; skipping.', files(i).name);
        continue;
    end
    step_path = inf.Groups(1).Name;            % e.g. '/data/100'
    step_str  = regexprep(step_path, '^/data/', '');
    step_id   = str2double(step_str);

    % openPMD time/dt/timeUnitSI live as attributes on /data/<step>
    try
        t        = h5readatt(fpath, step_path, 'time');
        tUnitSI  = h5readatt(fpath, step_path, 'timeUnitSI');
        time_s   = double(t) * double(tUnitSI);
    catch
        time_s = NaN;
    end

    base = [step_path '/particles/beam'];
    bunches{i} = struct( ...
        'step', step_id, ...
        'time', time_s, ...
        'x',    h5read(fpath, [base '/position/x']), ...
        'y',    h5read(fpath, [base '/position/y']), ...
        'z',    h5read(fpath, [base '/position/z']), ...
        'px',   h5read(fpath, [base '/momentum/x']), ...
        'py',   h5read(fpath, [base '/momentum/y']), ...
        'pz',   h5read(fpath, [base '/momentum/z']), ...
        'q',    h5read(fpath, [base '/charge']), ...
        'w',    h5read(fpath, [base '/weighting']), ...
        'id',   h5read(fpath, [base '/id']));
end

% Drop empties (skipped files), sort by step.
mask    = ~cellfun(@isempty, bunches);
bunches = bunches(mask);
[~, ord] = sort(cellfun(@(b) b.step, bunches));
bunches  = bunches(ord);
end
