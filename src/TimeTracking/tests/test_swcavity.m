function ok = test_swcavity()
% TEST_SWCAVITY  Phase 2A validation of the analytic SWCavity element.
%
% A relativistic electron (gamma=2, KE = 0.511 MeV) drifts through a thin
% N=1 SWCavity. The cavity is short (5 mm) and the RF frequency low
% (100 MHz, lambda = 3 m), so transit-time factor ~ 1 and the energy
% gain per pass should be:
%
%   dW(phi) = -|q| * peak_E * (2*L/pi) * cos(omega * t_pass + phi)
%
% where t_pass is the time the particle is at the cavity centre and phi
% is the input phase parameter. Phase 0 puts E_z at peak at t = 0.
%
% Pass condition: relative error in dW vs analytic < 5% on the high-|dW|
% phases, ignoring sub-keV phases where rounding dominates.

me = 9.1093837015e-31;
qe = 1.602176634e-19;
c  = 299792458.0;

% --- Beam: gamma = 2 electron ---
gamma_in = 2.0;
beta_in  = sqrt(1 - 1/gamma_in^2);
uz_in    = gamma_in * beta_in * c;        % proper velocity, m/s (along +z = beam axis)
KE_in_eV = (gamma_in - 1) * me * c^2 / qe;

% --- Cavity (thin: L << lambda_RF) ---
f_RF       = 1.0e8;      % 100 MHz
length_cav = 5.0e-3;     % 5 mm
peak_E     = 5.0e6;      % 5 MV/m
n_cells    = 1;
z_cav      = 50.0e-3;    % cavity at z = 50 mm
z_cav_ctr  = z_cav + length_cav / 2;
v          = uz_in / gamma_in;
omega      = 2 * pi * f_RF;
t_pass     = z_cav_ctr / v;

% --- Geometry / time stepping ---
margin   = 5.0e-3;
geom_lo  = [-margin, -margin, -1.0e-3];
geom_hi  = [ margin,  margin,  z_cav + length_cav + 5.0e-3];
dt_run   = (length_cav / v) / 200;        % ~200 steps inside cavity
total_t  = (geom_hi(3) - 0) / v + 5 * dt_run;
n_steps  = ceil(total_t / dt_run);

% --- Analytic dW (thin-cavity, transit-time factor = 1) ---
dW_unit_eV = peak_E * (2 * length_cav / pi);   % positive scalar, eV

fprintf('SWCavity phase scan:\n');
fprintf('  KE_in       = %.3f keV\n',   KE_in_eV * 1e-3);
fprintf('  dW_unit     = %.3f keV\n',   dW_unit_eV * 1e-3);
fprintf('  omega t_pass= %.4f rad\n',   omega * t_pass);
fprintf('  dt          = %.3g s   (RF/dt = %d)\n', dt_run, round(2*pi/(omega*dt_run)));
fprintf('  n_steps     = %d\n',         n_steps);

phases  = linspace(-pi, pi, 9);
phases  = phases(1:end-1);                % drop the duplicate +pi == -pi
dW_meas = zeros(size(phases));
dW_pred = -dW_unit_eV * cos(omega * t_pass + phases);

for k = 1:numel(phases)
    phi = phases(k);
    tt = TimeTrack();
    tt.lattice = { ...
        timetracking.SWCavity('name', 'SW1', 'z', z_cav, ...
                              'length', length_cav, 'n_cells', n_cells, ...
                              'f_RF', f_RF, 'peak_field', peak_E, ...
                              'phase', phi), ...
        timetracking.BeamMonitor('name', 'mon1', 'dump_every', n_steps) };
    tt.beam       = timetracking.SeedBeam('n_particles', 1, 'uz', uz_in);
    tt.geom_lo    = geom_lo;
    tt.geom_hi    = geom_hi;
    tt.geom_ncell = [16, 16, 64];
    tt.dt         = dt_run;
    tt.n_steps    = n_steps;
    tt.run();

    bunches = tt.readDumps();
    last = bunches{end};
    px = double(last.px(1));   py = double(last.py(1));   pz = double(last.pz(1));
    p_over_mc2 = (px*px + py*py + pz*pz) / (me * c)^2;
    gamma_out = sqrt(1 + p_over_mc2);
    KE_out_eV = (gamma_out - 1) * me * c^2 / qe;
    dW_meas(k) = KE_out_eV - KE_in_eV;
    fprintf('  phi = %+6.3f: dW_meas = %+7.3f keV   pred = %+7.3f keV\n', ...
            phi, dW_meas(k)*1e-3, dW_pred(k)*1e-3);
end

% Compare on phases with |dW_pred| > 1 keV (noise threshold)
mask    = abs(dW_pred) > 1e3;
rel_err = abs(dW_meas(mask) - dW_pred(mask)) ./ abs(dW_pred(mask));
max_re  = max(rel_err);

fprintf('  max relative error on |dW| > 1 keV phases: %.2e\n', max_re);

ok = max_re < 5e-2;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
