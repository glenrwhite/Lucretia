function ok = test_cyclotron()
% TEST_CYCLOTRON  End-to-end Phase 1.10 test of lucretia-tt via @TimeTrack.
%
% A single electron at the origin with proper velocity (ux, 0, 0) =
% (1e7, 0, 0) m/s is launched into a uniform B field B = (0, 0, Bz)
% with Bz = 1e-3 T. The resulting motion is a circle in the xy-plane.
%
% Analytic:
%   gamma  = sqrt(1 + (ux/c)^2)
%   r_L    = m_e * ux / (|qe| * Bz)        (= gamma*m*v/(|qe|*Bz))
%   T_c    = 2*pi*gamma*m_e / (|qe|*Bz)
%
% Orbit centre (electron with v_x>0, B_z>0): (0, +r_L, 0).
%
% Pass conditions:
%   |mean orbit radius - r_L| / r_L  <  1e-3
%   max  |orbit radius - r_L| / r_L  <  1e-2
%   bunch returns to within 1% of (0,0) at every integer multiple of T_c
%
% Returns true on pass, false on fail. Prints summary either way.

% --- Constants & analytic ---
me  = 9.1093837015e-31;
qe  = 1.602176634e-19;
c   = 299792458.0;

ux0 = 1e7;
Bz  = 1e-3;
gamma_p = sqrt(1 + (ux0/c)^2);
rL  = me * ux0 / (qe * Bz);
Tc  = 2*pi * gamma_p * me / (qe * Bz);

fprintf('Cyclotron benchmark:\n');
fprintf('  ux_0    = %.3g m/s\n', ux0);
fprintf('  Bz      = %.3g T\n',   Bz);
fprintf('  gamma   = %.6f\n',     gamma_p);
fprintf('  r_L     = %.4f mm\n',  rL*1e3);
fprintf('  T_c     = %.3f ns\n',  Tc*1e9);

% --- TimeTrack setup ---
tt = TimeTrack();
tt.lattice = { ...
    timetracking.UniformB('name', 'uB1', 'Bz', Bz), ...
    timetracking.BeamMonitor('name', 'mon1', 'dump_every', 10) };
tt.beam    = timetracking.SeedBeam('n_particles', 1, 'ux', ux0);

% Geometry: 4 r_L margin each way; coarse grid OK (no space charge in P1).
margin     = 4 * rL;
tt.geom_lo = [-margin, -margin, -1e-4];
tt.geom_hi = [ margin,  margin,  1e-4];
tt.geom_ncell = [32, 32, 16];

% Time stepping: T_c/1000 step, 2 cyclotron periods (-> ~2000 steps).
tt.dt      = Tc / 1000;
tt.n_steps = 2000;

fprintf('  dt      = %.4g s  (T_c / %d)\n', tt.dt, round(Tc/tt.dt));
fprintf('  n_steps = %d  (%.2f periods)\n', tt.n_steps, tt.n_steps*tt.dt/Tc);

tt.run();
bunches = tt.readDumps();
fprintf('  dumps   = %d\n', numel(bunches));

% --- Verify ---
xs = cellfun(@(b) double(b.x(1)), bunches);
ys = cellfun(@(b) double(b.y(1)), bunches);
ts = cellfun(@(b) b.time,         bunches);

r       = sqrt(xs.^2 + (ys - rL).^2);
mean_r  = mean(r);
max_dev = max(abs(r - rL)) / rL;
mean_dev = abs(mean_r - rL) / rL;

fprintf('  mean orbit radius:    %.6f mm  (analytic %.6f mm)\n', ...
        mean_r*1e3, rL*1e3);
fprintf('  |<r> - r_L| / r_L:    %.2e\n', mean_dev);
fprintf('  max |r - r_L| / r_L:  %.2e\n', max_dev);

% Closest dump to t = T_c (1 period) and t = 2*T_c (2 periods)
[~, k1] = min(abs(ts - Tc));
[~, k2] = min(abs(ts - 2*Tc));
d1 = sqrt(xs(k1)^2 + ys(k1)^2);
d2 = sqrt(xs(k2)^2 + ys(k2)^2);
fprintf('  return to origin @ T_c:    |xy| = %.4f mm  (%.2f%% of r_L)\n', ...
        d1*1e3, 100*d1/rL);
fprintf('  return to origin @ 2*T_c:  |xy| = %.4f mm  (%.2f%% of r_L)\n', ...
        d2*1e3, 100*d2/rL);

ok = (mean_dev < 1e-3) && (max_dev < 1e-2) && (d2/rL < 1e-2);

if ok
    fprintf('\n  PASS\n');
else
    fprintf('\n  FAIL\n');
end
end
