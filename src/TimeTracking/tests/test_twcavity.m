function ok = test_twcavity()
% TEST_TWCAVITY  Phase 2B validation of the analytic TWCavity element.
%
% A near-synchronous (gamma=100, ~50 MeV) electron traverses a
% v_phase = c traveling-wave cavity. For exact synchronism the phase the
% particle sees stays constant at its entry value, so the energy gain is:
%
%   dW(phi) = -|q| * peak_field * L * cos(phi + omega * z_cav / c)
%
% (the "drift phase" omega*z_cav/c is the phase the wave at z_cav has
% relative to the wave at z=0, t=0). Our particle is at gamma=100, so
% (1 - beta) ~ 5e-5: the residual phase slip during transit is sub-mrad
% and we can compare directly against the synchronous formula.

me = 9.1093837015e-31;
qe = 1.602176634e-19;
c  = 299792458.0;

% --- Beam: gamma = 100 (~50 MeV) ---
gamma_in = 100.0;
beta_in  = sqrt(1 - 1/gamma_in^2);
uz_in    = gamma_in * beta_in * c;
KE_in_eV = (gamma_in - 1) * me * c^2 / qe;

% --- Cavity ---
f_RF       = 2.856e9;     % S-band
length_cav = 5.0e-2;      % 5 cm
peak_E     = 5.0e6;       % 5 MV/m
z_cav      = 0.10;        % cavity at z = 100 mm
v_phase    = c;

omega = 2*pi*f_RF;
v     = uz_in / gamma_in;
% Drift phase from origin to cavity entrance
phi_drift = omega * z_cav / c;

% --- Analytic dW (synchronous limit, transit-time factor = 1) ---
dW_unit_eV = peak_E * length_cav;             % positive scalar, eV

% --- Geometry / time stepping ---
margin   = 1.0e-2;
geom_lo  = [-margin, -margin, -1.0e-3];
geom_hi  = [ margin,  margin,  z_cav + length_cav + 5.0e-2];
% Need ~100 RF samples per period and ~200 steps inside cavity
dt_run   = min((length_cav / v) / 200, 1/(f_RF * 50));
total_t  = (geom_hi(3) - 0) / v + 5*dt_run;
n_steps  = ceil(total_t / dt_run);

fprintf('TWCavity phase scan:\n');
fprintf('  KE_in       = %.3f MeV\n', KE_in_eV * 1e-6);
fprintf('  dW_unit     = %.3f keV\n', dW_unit_eV * 1e-3);
fprintf('  phi_drift   = %.4f rad\n', phi_drift);
fprintf('  dt          = %.3g s   (RF/dt = %d)\n', dt_run, round(2*pi/(omega*dt_run)));
fprintf('  n_steps     = %d\n',     n_steps);

phases  = linspace(-pi, pi, 9);
phases  = phases(1:end-1);
dW_meas = zeros(size(phases));
dW_pred = -dW_unit_eV * cos(phases + phi_drift);

for k = 1:numel(phases)
    phi = phases(k);
    tt = TimeTrack();
    tt.lattice = { ...
        timetracking.TWCavity('name', 'TW1', 'z', z_cav, ...
                              'length', length_cav, 'f_RF', f_RF, ...
                              'peak_field', peak_E, 'phase', phi, ...
                              'v_phase', v_phase), ...
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
    p2 = px*px + py*py + pz*pz;
    gamma_out = sqrt(1 + p2 / (me * c)^2);
    KE_out_eV = (gamma_out - 1) * me * c^2 / qe;
    dW_meas(k) = KE_out_eV - KE_in_eV;
    fprintf('  phi = %+6.3f: dW_meas = %+8.3f keV   pred = %+8.3f keV\n', ...
            phi, dW_meas(k)*1e-3, dW_pred(k)*1e-3);
end

mask    = abs(dW_pred) > 5e3;
rel_err = abs(dW_meas(mask) - dW_pred(mask)) ./ abs(dW_pred(mask));
fprintf('  max relative error on |dW| > 5 keV phases: %.2e\n', max(rel_err));

ok = max(rel_err) < 0.02;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
