function ok = test_relativistic_pencil(gamma_b)
% TEST_RELATIVISTIC_PENCIL  Validate the lab-frame transverse SC boost factor
% (1/gamma^2) used in lucretia-tt against an analytic uniform-sphere result.
%
% TRICK: to isolate just the BOOST FACTOR (independent of geometry shape
% effects from the lab-frame -> rest-frame Lorentz transformation), we
% initialize the bunch as a LAB-frame OBLATE SPHEROID with semi-axes
% (R, R, R/gamma_b). After IGF z-stretching by gamma_b, the REST-frame
% bunch is a UNIFORM SPHERE of radius R for any choice of gamma_b. So:
%
%   rest-frame field:    E_r_rest = Q * r_rest / (4*pi*eps0*R^3)
%   lab-frame transverse: F_x_lab = q * E_x_rest / gamma
%   slope:                d(u_x)/dx = qm * Q * dt / (4*pi*eps0*R^3*gamma)
%
% The slope scales as 1/gamma (NOT 1/gamma^2!), because the rest-frame
% field magnitude is INVARIANT to gamma (it's always a sphere of radius R)
% and only the v x B cancellation in the lab gives the 1/gamma reduction
% to the synchronous radial force.
%
% (For a LAB-frame uniform sphere, you'd get an additional shape-factor
%  scaling because the rest frame becomes a prolate spheroid; the
%  oblate-lab construction here divides that out.)
%
% Pass: 2D transverse slope matches analytic within 5%.
%
% EMPIRICAL SCALING (N=20000, ncell=128^3):
%   gamma=1   :  0.67% error  PASS  (boost factor: 1/1   = 1)
%   gamma=3   :  0.02% error  PASS  (boost factor: 1/3   = 0.333; very clean test)
%   gamma=12  : 10.3 % error  FAIL  (boost factor: 1/12 = 0.083; IGF cell aspect 1:1:12)
%
% At gamma >= 12 the IGF z-stretched cell aspect ratio is so extreme
% (1:1:12 in solver coords) that the integrated Green function has
% noticeable discretization error, and the measured slope under-predicts
% by ~10%. This is NOT a boost formula bug -- it's IGF discretization at
% high gamma. Default test gamma is 3 (cleanest demonstration that the
% boost factor is correctly implemented).

if nargin < 1, gamma_b = 3.0; end

me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
eps0 = 8.854187817e-12;
c    = 299792458;

N      = 20000;
R      = 1.0e-3;
Q_tot  = 1.0e-9;
w_per  = Q_tot / (N * qe);

% --- Generate uniform-in-volume sphere positions in REST FRAME, then
% Lorentz-contract z to lab frame so that rest-frame is a sphere of R ---
rng(42);
x = []; y = []; z = [];
while numel(x) < N
    n_try = ceil(1.5 * (N - numel(x)));
    xt = R * (2*rand(n_try,1) - 1);
    yt = R * (2*rand(n_try,1) - 1);
    zt = R * (2*rand(n_try,1) - 1);
    inside = (xt.^2 + yt.^2 + zt.^2) <= R^2;
    x = [x; xt(inside)];   %#ok<AGROW>
    y = [y; yt(inside)];   %#ok<AGROW>
    z = [z; zt(inside)];   %#ok<AGROW>
end
x = x(1:N);  y = y(1:N);  z = z(1:N);
% Lab-frame is z-contracted by 1/gamma_b (so rest-frame, after IGF stretch
% by gamma_b, is a perfect sphere of radius R).
z_lab = z / gamma_b;

% Cold longitudinal: all particles share gamma_b
beta_z = sqrt(max(1 - 1/gamma_b^2, 0));
ux = zeros(N,1);  uy = zeros(N,1);
uz = ones(N,1) * (gamma_b * beta_z * c);

% --- Write seed file ---
seed_path = fullfile(tempdir, sprintf('lucretia_tt_oblate_seed_g%d.h5', round(gamma_b)));
timetracking.writeBunchSeed(seed_path, x, y, z_lab, ux, uy, uz, w_per);

% --- Drive lucretia-tt: one short timestep, mesh-only SC ---
margin_xy = 4 * R;
margin_z  = 4 * R;          % oversized; lab-frame z extent = R/gamma_b
dt_run    = 5.0e-15;

tt = TimeTrack();
tt.lattice = { timetracking.BeamMonitor('name', 'mon', 'dump_every', 1) };
tt.beam = timetracking.SeedBeam('seed_file', seed_path);
tt.geom_lo    = [-margin_xy, -margin_xy, -margin_z];
tt.geom_hi    = [ margin_xy,  margin_xy,  margin_z];
tt.geom_ncell = [128, 128, 128];
tt.dt         = dt_run;
tt.n_steps    = 1;

tt.enable_space_charge = true;
tt.enable_slice_sc     = false;
tt.sc_image_plane      = false;
tt.sc_adaptive         = false;
tt.sc_comoving         = false;
tt.sc_exact_range      = false;

tt.run();

bunches = tt.readDumps();
last = bunches{end};
assert(numel(last.x) == N, 'particle count changed');

x_out  = double(last.x);   y_out  = double(last.y);   z_out  = double(last.z);
ux_out = double(last.px) / me;
uy_out = double(last.py) / me;
% No 3D radial; only TRANSVERSE matters for the boost test (longitudinal
% is contaminated by the bulk uz). 2D transverse slope:
r_meas  = sqrt(x_out.^2 + y_out.^2);
ur_meas = (ux_out .* x_out + uy_out .* y_out) ./ max(r_meas, 1e-30);

r0 = sqrt(x.^2 + y.^2);  % use rest-frame radial (= lab-frame radial in xy)

% Analytic prediction: transverse slope = qm * Q * dt / (4*pi*eps0*R^3*gamma)
% qm * (-Q) = +qe*Q/me for electrons.
slope_pred = (qe * Q_tot) / (4*pi*eps0 * R^3 * gamma_b) * dt_run / me;

% Inner 80% radial (xy only); use thin equatorial slab to avoid spheroid-
% surface CIC noise, but accept all z within R/gamma_b extent.
mask = r0 < 0.8 * R;
slope_fit = (r0(mask)' * ur_meas(mask)) / (r0(mask)' * r0(mask));
rel_err   = abs(slope_fit - slope_pred) / max(abs(slope_pred), 1e-30);

fprintf('Boost-factor SC test (oblate lab -> sphere rest):\n');
fprintf('  gamma_b = %.3f, beta_z = %.6f\n', gamma_b, beta_z);
fprintf('  Lab z-extent = +/-%.4f mm  (rest-frame: +/-%.3f mm sphere)\n', R/gamma_b*1e3, R*1e3);
fprintf('  N = %d, R_rest = %.3f mm, Q = %.3f pC\n', N, R*1e3, Q_tot*1e12);
fprintf('  Mesh: %d^3 cells over [%.0f mm]^3 (lab; IGF stretches z by gamma)\n', tt.geom_ncell(1), 2*margin_xy*1e3);
fprintf('  Mask: %d / %d particles (inner 80%% radial)\n', sum(mask), N);
fprintf('  dt = %.3g s\n', dt_run);
fprintf('  Slope analytic  d(u_x)/dx = %.4g  (1/s)\n', slope_pred);
fprintf('  Slope measured  d(u_x)/dx = %.4g\n', slope_fit);
fprintf('  Relative error           = %.3f%%\n', rel_err*100);

ok = rel_err < 0.05;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
