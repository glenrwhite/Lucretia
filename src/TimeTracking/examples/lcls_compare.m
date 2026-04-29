function lcls_compare()
% LCLS_COMPARE  Phase 4D auto-translated LCLS injector run with overlay
% against ImpactT's fort.18/24/25/26 diagnostics.
%
% Workflow:
%   1. Parse ImpactT.in via timetracking.readImpactTLattice -> lattice cell.
%   2. Override the parser's tracking suggestion with sensible workstation
%      defaults (the ImpactT input asks for 5e6 steps at 0.01 ps; we run
%      ~5000 steps at 0.5 ps -- still ~25 ns of flight, ~7 m at c).
%   3. Run with SC enabled.
%   4. Read fort.18 via timetracking.readImpactTFort18.
%   5. Interpolate ImpactT reference onto our dump times and print
%      side-by-side: gamma, sigma_x, eps_nx.

impactt_dir = '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT';

fprintf('=== LCLS injector compare ===\n\n');
fprintf('Parsing %s/ImpactT.in...\n', impactt_dir);
[lattice, beam, geom, ~, info] = timetracking.readImpactTLattice( ...
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

% --- Trim lattice to just gun + solenoid for a focused 2.5 ns run.
% Skipping L0A linac sections keeps the SC mesh resolution usable
% (otherwise the bunch is sub-cell in the z direction across a 6 m box).
keep = false(size(lattice));
for k = 1:numel(lattice)
    n = lattice{k}.name;
    keep(k) = ismember(n, {'cat','GUN','SOL1','mon'});
end
lattice = lattice(keep);

% Replace the BeamMonitor with one whose dump_every makes sense for our run.
mon_idx = find(cellfun(@(e) strcmp(e.type, 'beam_monitor'), lattice), 1);
if ~isempty(mon_idx)
    lattice{mon_idx} = timetracking.BeamMonitor('name', 'mon', 'dump_every', 100);
end

% --- Build TimeTrack ---
tt = TimeTrack();
tt.lattice = lattice;
tt.beam    = beam;
tt.geom_lo    = [-3e-3, -3e-3, -0.16];
tt.geom_hi    = [ 3e-3,  3e-3,  0.55];
tt.geom_ncell = [32, 32, 256];

% Pragmatic dt + n_steps for a ~2.5 ns run.
tt.dt      = 0.5e-12;
tt.n_steps = 5000;
tt.enable_space_charge = true;

fprintf('\nRun config:\n');
fprintf('  geom: lo=[%g %g %g] hi=[%g %g %g] m, ncell=[%d %d %d]\n', ...
        tt.geom_lo, tt.geom_hi, tt.geom_ncell);
fprintf('  dt = %.1f ps, n_steps = %d, total %.2f ns\n', ...
        tt.dt*1e12, tt.n_steps, tt.dt*tt.n_steps*1e9);

% --- Run ---
t_start = tic;
tt.run();
fprintf('  RUN TIME: %.1f s\n', toc(t_start));

bunches = tt.readDumps();
fprintf('  dumps: %d\n', numel(bunches));

% --- Read ImpactT reference ---
fprintf('\nLoading ImpactT reference...\n');
ref = timetracking.readImpactTFort18(impactt_dir);

% --- Compute lucretia-tt summary at each dump ---
me = 9.1093837015e-31;  qe = 1.602176634e-19;  c = 299792458.0;
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
    sx = std(xx);   spx = std(xp);
    cxxp = mean((xx - mean(xx)) .* (xp - mean(xp)));
    ours.z(k)      = mean(double(b.z));
    ours.gamma(k)  = mean(g);
    ours.sx(k)     = sx;
    ours.eps_nx(k) = mean(bg) * sqrt(max(sx^2 * spx^2 - cxxp^2, 0));
    ours.n(k)      = n;
end

% Filter empty entries (where N was too low)
mask = ours.n >= 5;
ours = filter_struct(ours, mask);

% --- Overlay print ---
fprintf('\n=== Side-by-side (interpolated to lucretia-tt dump times) ===\n');
fprintf('%-9s %-8s | %-8s %-8s %-12s %-13s | %-8s %-8s %-12s %-13s\n', ...
        't[ns]','z[mm]','LT-gam','IT-gam','LT-sx[mm]','IT-sx[mm]', ...
        'LT-eps[um.mr]','LT-relgrowth','IT-eps[um.mr]','rel-err');
fprintf('%s\n', repmat('-',1,130));

for k = 1:numel(ours.t)
    tk = ours.t(k);
    if tk < ref.t(1) || tk > ref.t(end), continue; end
    g_ref   = interp1(ref.t, ref.gamma,  tk, 'linear');
    sx_ref  = interp1(ref.t, ref.sigma_x, tk, 'linear');
    eps_ref = interp1(ref.t, ref.eps_nx, tk, 'linear');
    rel_err_eps = abs(ours.eps_nx(k) - eps_ref) / max(eps_ref, 1e-15);
    fprintf('%-9.2f %-8.2f | %-8.2f %-8.2f %-12.4f %-13.4f | %-13.3f %-12.4f %-12.3f %.2e\n', ...
        tk*1e9, ours.z(k)*1e3, ...
        ours.gamma(k), g_ref, ...
        ours.sx(k)*1e3, sx_ref*1e3, ...
        ours.eps_nx(k)*1e6, ...
        ours.eps_nx(k) / ours.eps_nx(1), ...
        eps_ref*1e6, rel_err_eps);
end

% --- Final summary ---
fprintf('\n=== Final state ===\n');
[~, klast] = max(ours.t);
fprintf('  Lucretia-TT: t=%.2f ns z=%.2f m gamma=%.2f sx=%.2f mm eps_nx=%.3g um.mr\n', ...
        ours.t(klast)*1e9, ours.z(klast), ours.gamma(klast), ...
        ours.sx(klast)*1e3, ours.eps_nx(klast)*1e6);
fprintf('  ImpactT (run ended): t=%.2f ns z=%.2f m gamma=%.2f sx=%.2f mm eps_nx=%.3g um.mr\n', ...
        ref.t(end)*1e9, ref.z_mean(end), ref.gamma(end), ...
        ref.sigma_x(end)*1e3, ref.eps_nx(end)*1e6);
end


function s = filter_struct(s, mask)
fnames = fieldnames(s);
for i = 1:numel(fnames)
    s.(fnames{i}) = s.(fnames{i})(mask);
end
end
