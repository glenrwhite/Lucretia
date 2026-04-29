function ok = test_quad()
% TEST_QUAD  Phase 2A validation of QuadAnalytic in the thin-lens limit.
%
% A relativistic electron (gamma=10, ~5 MeV) enters offset at (x0, 0, 0)
% with proper velocity (0, 0, uz). Inside a quad of length L and gradient
% G the field is B = (G*y, G*x, 0). For the offset particle B_y = G*x0,
% so the longitudinal v_z crossed with B_y gives a transverse Lorentz
% force in -x for positive charge / +x for electron.
%
% Thin-lens momentum kick (assume x ~ x0 throughout transit):
%   dp_x = -q * v_z * B_y * (L/v_z) = -q * L * G * x0
% In our SoA we store proper velocity (u = p/m), so:
%   d(ux) = dp_x / m = -(q/m) * L * G * x0
% For an electron q = -|e|, so d(ux) = +(|e|/m) * L * G * x0  (defocusing
% in x for G > 0 and electron).
%
% Pass condition: |d(ux)_meas - d(ux)_pred| / |d(ux)_pred| < 10% (the
% thin-lens approximation has finite-thickness corrections of similar
% size for our chosen geometry).

me = 9.1093837015e-31;
qe = 1.602176634e-19;
c  = 299792458.0;
qm_e = -qe / me;          % electron q/m

% --- Beam: gamma = 10 (~5 MeV) ---
gamma_in = 10.0;
beta_in  = sqrt(1 - 1/gamma_in^2);
uz_in    = gamma_in * beta_in * c;

% --- Quad ---
x0       = 1.0e-3;        % 1 mm offset
length_q = 5.0e-2;        % 5 cm
gradient = 1.0;           % T/m
z_q      = 0.02;          % quad starts at z = 20 mm

% Analytic kick:  d(ux) = -(q/m) * L * G * x0
dux_pred = -qm_e * length_q * gradient * x0;

% --- Geometry / time stepping ---
margin = 5.0e-3;
geom_lo = [-margin, -margin, -1.0e-3];
geom_hi = [ margin,  margin,  z_q + length_q + 1.0e-2];

v       = uz_in / gamma_in;
dt_run  = (length_q / v) / 200;   % 200 steps inside quad
total_t = (geom_hi(3) - 0) / v + 5*dt_run;
n_steps = ceil(total_t / dt_run);

fprintf('Quad thin-lens test:\n');
fprintf('  gamma_in = %.2f, uz = %.3g m/s\n', gamma_in, uz_in);
fprintf('  x0 = %.3f mm, L = %.0f mm, G = %.2f T/m\n', ...
        x0*1e3, length_q*1e3, gradient);
fprintf('  d(ux) predicted = %+.4g m/s\n', dux_pred);
fprintf('  dt = %.3g s,  n_steps = %d\n', dt_run, n_steps);

tt = TimeTrack();
tt.lattice = { ...
    timetracking.QuadAnalytic('name','Q1','z',z_q, ...
                              'length',length_q,'gradient',gradient), ...
    timetracking.BeamMonitor('name','mon1','dump_every',n_steps) };
tt.beam       = timetracking.SeedBeam('n_particles',1,'x',x0,'uz',uz_in);
tt.geom_lo    = geom_lo;
tt.geom_hi    = geom_hi;
tt.geom_ncell = [16, 16, 64];
tt.dt         = dt_run;
tt.n_steps    = n_steps;
tt.run();

bunches = tt.readDumps();
last = bunches{end};
px = double(last.px(1));
ux_out = px / me;
dux_meas = ux_out - 0.0;          % initial ux = 0
fprintf('  d(ux) measured  = %+.4g m/s\n', dux_meas);

rel_err = abs(dux_meas - dux_pred) / abs(dux_pred);
fprintf('  relative error  = %.2e\n', rel_err);

ok = rel_err < 0.10;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
