function dt_test()
% DT_TEST  Diagnostic: does a finer dt let us reach ImpactT's gamma plateau?
%
% Same LCLS setup as lcls_compare but truncated to a 1.2 ns run (one full
% gun traversal at v ~ c) and run twice -- once with dt = 0.5 ps (the
% Phase 4D baseline that plateaued at gamma ~ 8.8) and once with dt =
% 0.05 ps (10x finer; should resolve the cathode-region dynamics that
% ImpactT runs at dt = 0.01 ps). If gamma climbs noticeably with the
% finer dt, dt is the bottleneck and the variable-dt scheme below will
% help. If gamma stays at ~8.8, the bottleneck is elsewhere
% (image-charge model, emission timing, etc.).

impactt_dir = '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT';
[lattice0, beam, ~, ~, ~] = timetracking.readImpactTLattice( ...
    impactt_dir, 'n_macros', 500);   % fewer macros for speed

% Trim to gun + solenoid only (skip L0A linac for this gun-region test)
keep = false(size(lattice0));
for k = 1:numel(lattice0)
    keep(k) = ismember(lattice0{k}.name, {'cat','GUN','SOL1','mon'});
end
lattice = lattice0(keep);
mon_idx = find(cellfun(@(e) strcmp(e.type, 'beam_monitor'), lattice), 1);
lattice{mon_idx} = timetracking.BeamMonitor('name','mon','dump_every', 200);

me = 9.1093837015e-31;  qe = 1.602176634e-19;  c = 299792458.0;

cases = {struct('label','baseline 0.5 ps', 'dt', 0.5e-12, 'n_steps', 2400), ...
         struct('label','fine 0.05 ps',     'dt', 0.05e-12, 'n_steps', 24000)};

for k = 1:numel(cases)
    cfg = cases{k};
    fprintf('\n=== %s (dt=%g ps, n_steps=%d, total %.2f ns) ===\n', ...
            cfg.label, cfg.dt*1e12, cfg.n_steps, cfg.dt*cfg.n_steps*1e9);

    tt = TimeTrack();
    tt.lattice = lattice;
    tt.beam    = beam;
    tt.geom_lo    = [-3e-3, -3e-3, -0.16];
    tt.geom_hi    = [ 3e-3,  3e-3,  0.55];
    tt.geom_ncell = [32, 32, 256];
    tt.dt         = cfg.dt;
    tt.n_steps    = cfg.n_steps;
    tt.enable_space_charge = true;

    t_start = tic;
    tt.run();
    fprintf('  RUN TIME: %.1f s\n', toc(t_start));

    bunches = tt.readDumps();
    fprintf('  dumps: %d\n', numel(bunches));

    % Final & peak gamma
    g_max = 0;  g_final = NaN;  z_final = NaN;
    for j = 1:numel(bunches)
        b = bunches{j};
        if numel(b.x) < 5, continue; end
        px = double(b.px); py = double(b.py); pz = double(b.pz);
        p2 = px.*px + py.*py + pz.*pz;
        g  = mean(sqrt(1 + p2/(me*c)^2));
        g_max = max(g_max, g);
        g_final = g;
        z_final = mean(double(b.z));
    end
    fprintf('  peak  gamma = %.2f (KE = %.2f MeV)\n', g_max, (g_max-1)*0.511);
    fprintf('  final gamma = %.2f (KE = %.2f MeV)  at z = %.3f m\n', ...
            g_final, (g_final-1)*0.511, z_final);
end

fprintf('\nReference (ImpactT fort.18 gun-exit plateau): gamma ~ 11.68\n');
end
