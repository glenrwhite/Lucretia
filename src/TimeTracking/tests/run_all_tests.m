function results = run_all_tests(varargin)
% RUN_ALL_TESTS  Run every test_*.m in this directory and report pass/fail.
%
% Each test is a function handle taking no args and returning a single
% boolean ok. Tests run in alphabetical order. The script captures the
% return value, the wall-clock time, and any error thrown.
%
% Usage:
%   results = run_all_tests()           % run everything
%   results = run_all_tests('exclude', {'test_trackthru_handoff'})
%
% Returns a struct array with fields .name, .ok, .elapsed_s, .err.

p = inputParser;
addParameter(p, 'exclude', {});
parse(p, varargin{:});
exclude = p.Results.exclude;

thisDir = fileparts(mfilename('fullpath'));
files = dir(fullfile(thisDir, 'test_*.m'));
names = {files.name};
[~, base, ~] = cellfun(@fileparts, names, 'UniformOutput', false);
base = setdiff(base, exclude, 'stable');
base = sort(base);

n = numel(base);
fprintf('\n=== run_all_tests: %d tests ===\n', n);

results = repmat(struct('name','', 'ok',false, 'elapsed_s',0, 'err',''), n, 1);

for k = 1:n
    name = base{k};
    fprintf('\n--- [%d/%d] %s ---\n', k, n, name);
    fn = str2func(name);
    t0 = tic;
    try
        ok = fn();
        elapsed = toc(t0);
        results(k).name      = name;
        results(k).ok        = logical(ok);
        results(k).elapsed_s = elapsed;
        results(k).err       = '';
        fprintf('  -> %s in %.1f s\n', ternary(ok,'PASS','FAIL'), elapsed);
    catch ME
        elapsed = toc(t0);
        results(k).name      = name;
        results(k).ok        = false;
        results(k).elapsed_s = elapsed;
        results(k).err       = ME.message;
        fprintf('  -> ERROR in %.1f s: %s\n', elapsed, ME.message);
    end
end

n_pass = sum([results.ok]);
fprintf('\n=== %d / %d PASSED ===\n', n_pass, n);
if n_pass < n
    fprintf('Failures:\n');
    for k = 1:n
        if ~results(k).ok
            fprintf('  - %s  (%s)\n', results(k).name, results(k).err);
        end
    end
end
end


function v = ternary(cond, a, b)
if cond, v = a; else, v = b; end
end
