function ok = test_solenoid_3d()
% TEST_SOLENOID_3D  Phase 2C validation of file-based Solenoid3D loading.
%
% Generates a 2D RZ B_z(r,z) HDF5 map with the same Gaussian profile as
% test_solenoid (SolenoidAnalytic) and verifies the Larmor rotation
% matches the analytic prediction. Should reproduce the SolenoidAnalytic
% result within bilinear-interpolation accuracy.

me = 9.1093837015e-31;
qe = 1.602176634e-19;
c  = 299792458.0;

% --- Beam ---
gamma_in = 10.0;
beta_in  = sqrt(1 - 1/gamma_in^2);
uz_in    = gamma_in * beta_in * c;
p_in     = gamma_in * me * beta_in * c;

% --- Solenoid ---
peak_Bz  = 1.0e-4;
sigma    = 0.03;
length_s = 0.20;
z_sol    = 0.05;
z_peak_inside = length_s/2;       % peak relative to element start

x0 = 1.0e-3;

integral_Bz = peak_Bz * sigma * sqrt(pi);
theta_L     = qe * integral_Bz / p_in;
half_theta  = theta_L / 2;

% --- Build the 2D RZ B_z map (radius-independent, since paraxial test) ---
Nr = 32; Nz = 256;
dr = (2 * x0) / (Nr - 1);             % small radial range; particle stays at ~1mm
dz = length_s / (Nz - 1);
r_grid = (0:Nr-1) * dr;
z_grid = (0:Nz-1) * dz;
% B_z is independent of r (axisymmetric, on-axis profile assumed valid up
% to the small radii we test at).
[Z, ~]  = meshgrid(z_grid, r_grid);   % size [Nr, Nz]
Bz_map  = peak_Bz * exp(-((Z - z_peak_inside)/sigma).^2);

map_path = fullfile(tempdir, 'lucretia_tt_sol2d.h5');
timetracking.writeFieldMap2D(map_path, Bz_map, [dr, dz], [0, 0]);

fprintf('Solenoid3D (file-based) Larmor test:\n');
fprintf('  rL grid: Nr=%d Nz=%d, dr=%.3g m, dz=%.3g m\n', Nr, Nz, dr, dz);
fprintf('  theta_L = %.4f rad   half = %.4f rad\n', theta_L, half_theta);

% --- TimeTrack run ---
margin = 5.0e-3;
v       = uz_in / gamma_in;
dt_run  = (sigma / v) / 100;
total_t = (z_sol + length_s + 5e-3) / v + 5*dt_run;
n_steps = ceil(total_t / dt_run);

tt = TimeTrack();
tt.lattice = { ...
    timetracking.Solenoid3D('name','SOL3D','z',z_sol, ...
                            'length',length_s,'path',map_path), ...
    timetracking.BeamMonitor('name','mon','dump_every',n_steps) };
tt.beam = timetracking.SeedBeam('n_particles',1,'x',x0,'uz',uz_in);
tt.geom_lo    = [-margin, -margin, -1e-3];
tt.geom_hi    = [ margin,  margin,  z_sol + length_s + 5e-3];
tt.geom_ncell = [16, 16, 64];
tt.dt         = dt_run;
tt.n_steps    = n_steps;
tt.run();

bunches = tt.readDumps();
last = bunches{end};
x_out = double(last.x(1));
y_out = double(last.y(1));

theta_meas = atan2(y_out, x_out);
r_out      = sqrt(x_out^2 + y_out^2);
abs_err    = abs(abs(theta_meas) - half_theta);
rel_err    = abs_err / half_theta;
r_err      = abs(r_out - x0) / x0;

fprintf('  measured (x,y) = (%.4f, %+.4f) mm\n', x_out*1e3, y_out*1e3);
fprintf('  measured |theta| = %.6f rad   (predicted %.6f)\n', ...
        abs(theta_meas), half_theta);
fprintf('  rel err (angle)  = %.2e\n', rel_err);
fprintf('  rel err (radius) = %.2e\n', r_err);

ok = rel_err < 0.05 && r_err < 0.05;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
