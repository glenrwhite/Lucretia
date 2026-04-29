function ok = test_image_charge()
% TEST_IMAGE_CHARGE  Phase 3 validation: single-particle image-force trajectory.
%
% A single electron is placed at z = z0 above a conducting cathode plane
% at z = 0. With image-charge enabled, it experiences the self-mirror force:
%
%   F_z = -e^2 / (16*pi*eps0*z^2)        (toward the cathode)
%
% leading to non-relativistic 1D motion governed by
%
%   m_e * dv/dt = -e^2 / (16*pi*eps0*z^2)
%   dz/dt       = v
%
% We integrate this in MATLAB via ode45 and compare to the lucretia-tt
% trajectory. Pass: max relative error in z(t) < 5%.

me = 9.1093837015e-31;
qe = 1.602176634e-19;
eps0 = 8.854187817e-12;

z0 = 1.0e-6;          % start 1 um above cathode
v0 = 0;               % at rest
T  = 30.0e-12;        % integrate to 30 ps

% --- Analytic ODE solve ---
K = qe^2 / (16 * pi * eps0 * me);   % m^3 / s^2
% dy/dt = [v; -K/z^2], y = [z; v]
ode_rhs    = @(t, y) [y(2); -K / y(1)^2];
ode_events = @(t, y) ode_event_z_floor(t, y);
opts = odeset('AbsTol', 1e-15, 'RelTol', 1e-10, 'Events', ode_events);
[t_ode, y_ode] = ode45(ode_rhs, [0, T], [z0; v0], opts);
z_ode = y_ode(:, 1);
v_ode = y_ode(:, 2);

t_end_ode = t_ode(end);
z_end_ode = z_ode(end);
v_end_ode = v_ode(end);

fprintf('Image-charge analytic (ode45):\n');
fprintf('  z0 = %.3g m, v0 = %.3g m/s\n', z0, v0);
fprintf('  Stopped at t = %.3f ps, z = %.4f nm, v = %+.3e m/s\n', ...
        t_end_ode*1e12, z_end_ode*1e9, v_end_ode);

% --- TimeTrack run ---
% CathodeSource with no emission, image_charge enabled. Particle seeded
% manually via SeedBeam.
margin = 5e-6;
tt = TimeTrack();
tt.lattice = { ...
    timetracking.CathodeSource('name', 'cat', 'z_cathode', 0.0, ...
                               'image_charge', true, ...
                               'n_macroparticles_total', 0), ...
    timetracking.BeamMonitor('name', 'mon', 'dump_every', 1) };
tt.beam = timetracking.SeedBeam('n_particles', 1, ...
                                 'x', 0, 'y', 0, 'z', z0, ...
                                 'ux', 0, 'uy', 0, 'uz', 0);
tt.geom_lo    = [-margin, -margin, -1e-7];
tt.geom_hi    = [ margin,  margin,  3*z0];
tt.geom_ncell = [16, 16, 32];

% Choose dt small enough to resolve the late-time fast acceleration.
% Empirically dt ~ 1/200 of the analytic t_end is enough.
tt.dt      = t_end_ode / 1000;
tt.n_steps = ceil(t_end_ode / tt.dt);

fprintf('  dt = %.3g s, n_steps = %d\n', tt.dt, tt.n_steps);

tt.run();
bunches = tt.readDumps();
fprintf('  dumps = %d\n', numel(bunches));

t_meas = cellfun(@(b) b.time, bunches);
z_meas = cellfun(@(b) double(b.z(1)), bunches);
uz_meas = cellfun(@(b) double(b.pz(1)) / me, bunches);  % proper-velocity is u; for non-rel, u ~ v

% Interpolate analytic onto measured times for direct comparison.
% (ode45 sample points and our dump times don't align.)
mask_ok   = t_meas <= t_end_ode;
t_cmp     = t_meas(mask_ok);
z_cmp     = z_meas(mask_ok);
z_pred    = interp1(t_ode, z_ode, t_cmp, 'linear');

% Skip the first dump (step 0, exactly z0 in both)
err = abs(z_cmp(2:end) - z_pred(2:end)) ./ z_pred(2:end);
max_err  = max(err);
mean_err = mean(err);

fprintf('  trajectory: %d sample points compared\n', sum(mask_ok)-1);
fprintf('  z(t_end_ode) measured = %.4f nm,  predicted = %.4f nm\n', ...
        z_cmp(end)*1e9, z_pred(end)*1e9);
fprintf('  max  relative error in z(t):  %.2e\n', max_err);
fprintf('  mean relative error in z(t):  %.2e\n', mean_err);

ok = max_err < 0.05;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end


function [value, isterminal, direction] = ode_event_z_floor(~, y)
% Stop the ode45 integration when the particle reaches z = 1 nm.
value      = y(1) - 1.0e-9;
isterminal = 1;
direction  = -1;
end
