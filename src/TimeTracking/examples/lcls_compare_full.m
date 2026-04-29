function lcls_compare_full()
% LCLS_COMPARE_FULL  Phase 4D-followup: full LCLS injector run vs ImpactT,
% using the co-moving SC mesh so the bunch can be tracked all the way
% through the L0A linac (~5 m) without the static box being absurdly big.
%
% Lattice: cathode + GUN + SOL1 + L0A_entrance + L0A_body_{1,2} + L0A_exit.
% Auto-translated from ImpactT.in via timetracking.readImpactTLattice
% (which now passes phase through unchanged, uses the narrow Nemission *
% dt_initial emission window, and sets t_start = Tini).
%
% Configuration:
%   - Geometry initially sized for the gun + solenoid region.
%   - SC mesh co-moves with the bunch (sc_comoving = true).
%   - Variable dt: fine (0.2 ps) for the gun-and-solenoid region where
%     RF and image-charge dynamics matter, coarse (1 ps) afterwards
%     during the L0A drift / linac transit.

impactt_dir = '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT';

fprintf('=== LCLS injector compare (FULL lattice, co-moving SC) ===\n');
[lattice, beam, ~, tracking_sug, info] = timetracking.readImpactTLattice( ...
    impactt_dir, 'n_macros', 2000);

fprintf('  cathode at z = %.4f m\n', info.cathode_z);
fprintf('  total charge = %.1f pC, t_emission = %.2f ps\n', ...
        info.total_charge*1e12, info.t_emission*1e12);
fprintf('  lattice elements: %d\n', numel(lattice));
for k = 1:numel(lattice)
    el = lattice{k};
    z_str = '';
    if isfield(el, 'z'),         z_str = sprintf('z=%.4f m', el.z); end
    if isfield(el, 'z_cathode'), z_str = sprintf('z_cathode=%.4f m', el.z_cathode); end
    fprintf('    [%2d] %-20s type=%-18s %s\n', k, el.name, el.type, z_str);
end

% Replace the BeamMonitor with one whose dump_every fits our step count.
mon_idx = find(cellfun(@(e) strcmp(e.type, 'beam_monitor'), lattice), 1);
if ~isempty(mon_idx)
    lattice{mon_idx} = timetracking.BeamMonitor('name', 'mon', 'dump_every', 200);
end

% --- Build TimeTrack ---
tt = TimeTrack();
tt.lattice = lattice;
tt.beam    = beam;

% Geometry: initially sized to fit gun + solenoid (z in roughly [-0.2, 0.6]).
% Co-moving SC will shift this in z as the bunch propagates downstream.
tt.geom_lo    = [-3e-3, -3e-3, -0.16];
tt.geom_hi    = [ 3e-3,  3e-3,  0.55];
tt.geom_ncell = [32, 32, 256];

% Tracking: start at Tini, fine dt during cathode + gun, coarse after.
% Total time ~ 17 ns to traverse L0A (~5 m at c).
tt.t_start     = tracking_sug.t_start;   % = Tini = -13.06 ps for LCLS
tt.dt          = 0.5e-12;       % fine in cathode + gun region
tt.dt_change_t = 1.5e-9;        % switch at t = 1.5 ns (post-gun, post-solenoid)
tt.dt_after    = 1.0e-12;       % coarser through L0A
tt.n_steps     = 18000;         % ~17 ns total wall-clock at the schedule above
tt.enable_space_charge = true;
tt.sc_comoving         = true;

fprintf('\nRun config:\n');
fprintf('  geom: lo=[%g %g %g] hi=[%g %g %g] m, ncell=[%d %d %d] (initial; co-moving)\n', ...
        tt.geom_lo, tt.geom_hi, tt.geom_ncell);
fprintf('  dt = %.2f ps until t = %.1f ns, then %.2f ps\n', ...
        tt.dt*1e12, tt.dt_change_t*1e9, tt.dt_after*1e12);
fprintf('  n_steps = %d, t_start = %.3f ps\n', tt.n_steps, tt.t_start*1e12);

t_start = tic;
tt.run();
fprintf('  RUN TIME: %.1f s\n', toc(t_start));

bunches = tt.readDumps();
fprintf('  dumps: %d\n', numel(bunches));

% Count recenter events from the run log.
n_recenter = numel(strfind(tt.last_log, 'recenter:'));
fprintf('  SC recenter events: %d\n', n_recenter);

% --- ImpactT reference ---
fprintf('\nLoading ImpactT reference...\n');
ref = timetracking.readImpactTFort18(impactt_dir);

% --- Lucretia-tt summary per dump ---
me = 9.1093837015e-31;  c = 299792458.0;
n_d  = numel(bunches);
ours = struct('t', zeros(n_d,1), 'z', zeros(n_d,1), 'gamma', zeros(n_d,1), ...
              'sx', zeros(n_d,1), 'eps_nx', zeros(n_d,1), 'n', zeros(n_d,1));
for k = 1:n_d
    b = bunches{k};
    n = numel(b.x);
    if n < 5, continue; end
    ours.t(k)   = b.time;
    px = double(b.px);  py = double(b.py);  pz = double(b.pz);
    xx = double(b.x);
    p2 = px.*px + py.*py + pz.*pz;
    g  = sqrt(1 + p2 / (me*c)^2);
    bg = sqrt(g.^2 - 1);
    xp = px ./ max(pz, 1e-30);
    sx = std(xx); spx = std(xp);
    cxxp = mean((xx - mean(xx)) .* (xp - mean(xp)));
    ours.z(k)      = mean(double(b.z));
    ours.gamma(k)  = mean(g);
    ours.sx(k)     = sx;
    ours.eps_nx(k) = mean(bg) * sqrt(max(sx^2 * spx^2 - cxxp^2, 0));
    ours.n(k)      = n;
end
mask = ours.n >= 5;
ours = filter_struct(ours, mask);

% --- Side-by-side print ---
fprintf('\n=== Side-by-side (interpolated to lucretia-tt dump times) ===\n');
fprintf('%-9s %-8s %-5s | %-8s %-8s | %-8s %-8s | %-12s %-12s\n', ...
        't[ns]','z[mm]','N','LT-gam','IT-gam','LT-sx[mm]','IT-sx[mm]', ...
        'LT-eps[um]','IT-eps[um]');
fprintf('%s\n', repmat('-', 1, 105));

for k = 1:numel(ours.t)
    tk = ours.t(k);
    if tk < ref.t(1) || tk > ref.t(end), continue; end
    g_ref   = interp1(ref.t, ref.gamma,   tk, 'linear');
    sx_ref  = interp1(ref.t, ref.sigma_x, tk, 'linear');
    eps_ref = interp1(ref.t, ref.eps_nx,  tk, 'linear');
    fprintf('%-9.2f %-8.1f %-5d | %-8.2f %-8.2f | %-9.4f %-9.4f | %-12.3f %-12.3f\n', ...
        tk*1e9, ours.z(k)*1e3, ours.n(k), ...
        ours.gamma(k), g_ref, ...
        ours.sx(k)*1e3, sx_ref*1e3, ...
        ours.eps_nx(k)*1e6, eps_ref*1e6);
end

% --- Final summary ---
[~, klast] = max(ours.t);
fprintf('\n=== Final state ===\n');
fprintf('  Lucretia-TT: t=%.2f ns z=%.2f m gamma=%.2f sx=%.2f mm eps_nx=%.3g um.mr  N=%d\n', ...
        ours.t(klast)*1e9, ours.z(klast), ours.gamma(klast), ...
        ours.sx(klast)*1e3, ours.eps_nx(klast)*1e6, ours.n(klast));
fprintf('  ImpactT:     t=%.2f ns z=%.2f m gamma=%.2f sx=%.2f mm eps_nx=%.3g um.mr\n', ...
        ref.t(end)*1e9, ref.z_mean(end), ref.gamma(end), ...
        ref.sigma_x(end)*1e3, ref.eps_nx(end)*1e6);
end


function s = filter_struct(s, mask)
fnames = fieldnames(s);
for i = 1:numel(fnames)
    s.(fnames{i}) = s.(fnames{i})(mask);
end
end
