function show_kill_timing()
% Show kill timing for several runs we have
cd('/Users/glenwhite/code/Lucretia');
addpath('src/TimeTracking');
addpath('/tmp/claude');

runs = {
    'baseline (static)',  '/tmp/bench_full_smooth0/diags';
    'full adaptive',      '/tmp/bench_full_fulladapt/diags';
    'imp-match',          '/tmp/bench_full_imp_match2/diags';
};

for r = 1:size(runs, 1)
    fprintf('\n=== %s ===\n', runs{r,1});
    bunches = readBeamOpenPMD(runs{r,2});
    fprintf('%-9s %-8s %-7s %-7s\n', 't[ns]', 'z[mm]', 'N_alive', 'killed');
    prev_alive = 50000;
    for k = 1:numel(bunches)
        b = bunches{k};
        z = double(b.z);
        n_alive = sum(z > -1e6);
        n_killed = prev_alive - n_alive;
        z_mean = mean(z(z > -1e6));
        if n_killed > 0 || k == numel(bunches) || k == 1
            fprintf('%-9.3f %-8.0f %-7d %-7d\n', b.time*1e9, z_mean*1e3, n_alive, n_killed);
        end
        prev_alive = n_alive;
    end
end
end
