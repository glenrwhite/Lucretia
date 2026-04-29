function ok = test_solenoid()
% TEST_SOLENOID  Phase 2B validation of SolenoidAnalytic via Larmor rotation.
%
% A particle entering offset at (x0, 0, 0) with proper velocity (0, 0, uz)
% through a solenoid acquires a half-Larmor-angle rotation of the position
% vector in the (x, y) plane:
%
%   theta_L     = |q| * integral(B_z dz) / p
%   integral    = peak_Bz * sigma * sqrt(pi)             (full Gaussian)
%   x_out       = x0 * cos(theta_L / 2)
%   y_out       = -sign(q) * x0 * sin(theta_L / 2)
%
% Sign convention: for B_z > 0 and electron (q < 0), the position rotates
% toward +y. We test the magnitude of the rotation angle.
%
% Pass: |theta_meas - theta_L/2| / |theta_L/2| < 5%.

me = 9.1093837015e-31;
qe = 1.602176634e-19;
c  = 299792458.0;

% --- Beam: gamma = 10 electron ---
gamma_in = 10.0;
beta_in  = sqrt(1 - 1/gamma_in^2);
uz_in    = gamma_in * beta_in * c;          % proper velocity m/s
p_in     = gamma_in * me * beta_in * c;     % kg.m/s

% --- Solenoid (Gaussian B_z) ---
peak_Bz  = 1.0e-4;     % 0.1 mT (small enough that theta_L < 1 rad)
sigma    = 0.03;       % 30 mm
length_s = 0.20;       % 200 mm = ~6.7 sigma => integral well captured
z_sol    = 0.05;       % start at z = 50 mm
z_peak   = length_s/2; % peak in middle

integral_Bz = peak_Bz * sigma * sqrt(pi);
theta_L     = qe * integral_Bz / p_in;        % full Larmor angle (positive)
half_theta  = theta_L / 2;

x0 = 1.0e-3;          % 1 mm offset

fprintf('Solenoid Larmor-rotation test:\n');
fprintf('  gamma_in     = %.2f\n',  gamma_in);
fprintf('  peak_Bz      = %.2g T\n', peak_Bz);
fprintf('  integral B_z = %.4e T.m\n', integral_Bz);
fprintf('  theta_L      = %.4f rad   (= %.3f deg)\n', theta_L, theta_L*180/pi);
fprintf('  half angle   = %.4f rad   (= %.3f deg)\n', half_theta, half_theta*180/pi);

% --- Geometry / time stepping ---
margin   = 5.0e-3;
geom_lo  = [-margin, -margin, -1.0e-3];
geom_hi  = [ margin,  margin,  z_sol + length_s + 5.0e-3];

v       = uz_in / gamma_in;
% Need to resolve the length scale of B_z variation (sigma)
dt_run  = (sigma / v) / 100;
total_t = (geom_hi(3) - 0) / v + 5*dt_run;
n_steps = ceil(total_t / dt_run);

fprintf('  dt           = %.3g s,  n_steps = %d\n', dt_run, n_steps);

tt = TimeTrack();
tt.lattice = { ...
    timetracking.SolenoidAnalytic('name','SOL1', 'z', z_sol, ...
        'length', length_s, 'peak_Bz', peak_Bz, 'sigma', sigma, ...
        'z_peak', z_peak), ...
    timetracking.BeamMonitor('name','mon1','dump_every',n_steps) };
tt.beam       = timetracking.SeedBeam('n_particles', 1, ...
                                       'x', x0, 'uz', uz_in);
tt.geom_lo    = geom_lo;
tt.geom_hi    = geom_hi;
tt.geom_ncell = [16, 16, 64];
tt.dt         = dt_run;
tt.n_steps    = n_steps;
tt.run();

bunches = tt.readDumps();
last = bunches{end};
x_out = double(last.x(1));
y_out = double(last.y(1));

theta_meas = atan2(y_out, x_out);              % signed angle
r_out      = sqrt(x_out^2 + y_out^2);
% Compare magnitudes (sign of rotation depends on sign(q) which we leave
% to the underlying physics).
abs_err   = abs(abs(theta_meas) - half_theta);
rel_err   = abs_err / half_theta;
r_err     = abs(r_out - x0) / x0;

fprintf('  measured (x,y) = (%.3f, %+.3f) mm\n', x_out*1e3, y_out*1e3);
fprintf('  measured |theta| = %.4f rad   (predicted %.4f rad)\n', ...
        abs(theta_meas), half_theta);
fprintf('  rel err (angle)  = %.2e\n', rel_err);
fprintf('  rel err (radius) = %.2e   (transverse position should be conserved)\n', r_err);

ok = rel_err < 0.05 && r_err < 0.05;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
