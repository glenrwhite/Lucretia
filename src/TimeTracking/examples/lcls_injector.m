function lcls_injector()
% LCLS_INJECTOR  Phase 4C end-to-end run using real LCLS gun + solenoid
% field maps (rfdata201, rfdata102) loaded from the ImpactT lattice at
% /Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT.
%
% Lattice (ImpactT.in lines 25-27):
%   GUN:  L=0.15, scale_E=4.75282e+07 V/m, f_RF=2.856 GHz, phase=304.668 deg
%         file = rfdata201, z_lab edge = 0.0
%   SOL1: L=0.49308, scale_B=0.219982 T, f_RF=0 (DC), file = rfdata102,
%         z_lab edge = 0.0422
%
% Bunch parameters (ImpactT.in lines 13-19):
%   sigma_x = sigma_y = 0.6 mm RMS, sigma_z = 3.66 um (~12 fs bunch length),
%   I = 2.91 A, total Q ~ I * t_emission = 2.91 * 35.5e-12 ~ 0.10 nC.
%
% This example uses our cathode emission element with parameters chosen
% to approximate the ImpactT input. Compare exit energy / RMS spot vs
% fort.18 / fort.24 from the ImpactT reference run.

impact_dir = '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT';

% --- Beam parameters from ImpactT.in ---
total_Q       = 100e-12;        % 100 pC (I*t_em from ImpactT)
sigma_x       = 0.6e-3;         % 0.6 mm transverse RMS
pulse_FWHM    = 35.5e-12;       % 35.5 ps emission window (use as flat-top length)
mte_eV        = 0.5;            % typical photocathode MTE
n_macros      = 2000;
% Use flat-top to mirror ImpactT's flagdist=16 (uniform-in-t emission)
pulse_t0      = 0.0;            % start at t=0

% --- Lattice ---
gun_E0        = 4.75282e+07;    % V/m
gun_freq      = 2.856e9;        % Hz
gun_phase_deg = 304.668;        % degrees (ImpactT convention)
gun_z_edge    = 0.0;            % field map's z=0 maps to lab z = 0
gun_path      = fullfile(impact_dir, 'rfdata201');

sol_z_edge    = 0.0422;         % from ImpactT.in
sol_B_scale   = 0.219982;       % T
sol_path      = fullfile(impact_dir, 'rfdata102');

% Cathode sits at the START of the gun field map (lab z = m_z + zmin).
% For our element convention, m_z is the lab position of the field map's
% intrinsic z = 0; the cathode plate is at z_field = zmin = -0.15.
cathode_z  = gun_z_edge - 0.15;    % = -0.15 m

% --- Geometry: covers gun (z in [-0.15, +0.15]) + solenoid (z in [-0.49, +0.54])
geom_lo    = [-2e-3, -2e-3, -0.16];
geom_hi    = [ 2e-3,  2e-3,  0.55];
geom_ncell = [32, 32, 256];     % coarser than production but faster

dt         = 0.5e-12;
n_steps    = 5000;              % 2.5 ns total -> bunch travels ~75 cm

dump_every = 100;

% --- Build TimeTrack ---
tt = TimeTrack();
tt.lattice = { ...
    timetracking.CathodeSource('name','cat',  'z_cathode', cathode_z, ...
        'image_charge', true, ...
        'n_macroparticles_total', n_macros, ...
        'total_charge', total_Q, ...
        'pulse_shape', 'flat_top', ...
        'pulse_duration', pulse_FWHM, 'pulse_t0', pulse_t0, ...
        'transverse_profile', 'gaussian', 'spot_size', sigma_x, ...
        'mte', mte_eV), ...
    timetracking.ImpactTField('name','gun', 'z', gun_z_edge, ...
        'path', gun_path, ...
        'scale_E', gun_E0, 'f_RF', gun_freq, 'phase_deg', gun_phase_deg), ...
    timetracking.ImpactTField('name','sol', 'z', sol_z_edge, ...
        'path', sol_path, ...
        'scale_B', sol_B_scale, 'f_RF', 0, 'phase_deg', 0), ...
    timetracking.BeamMonitor('name','mon', 'dump_every', dump_every) };
tt.beam = timetracking.SeedBeam('n_particles', 0);
tt.geom_lo    = geom_lo;
tt.geom_hi    = geom_hi;
tt.geom_ncell = geom_ncell;
tt.dt         = dt;
tt.n_steps    = n_steps;
tt.enable_space_charge = true;

fprintf('LCLS injector run (Phase 4C):\n');
fprintf('  cathode: Q = %.0f pC, FWHM %.1f ps (flat-top), spot %.2f mm RMS, MTE %.1f eV\n', ...
        total_Q*1e12, pulse_FWHM*1e12, sigma_x*1e3, mte_eV);
fprintf('  gun:     %s, scale_E = %.3e V/m, phase = %.3f deg\n', ...
        gun_path, gun_E0, gun_phase_deg);
fprintf('  sol:     %s, scale_B = %.4f T at z=%.4f m\n', ...
        sol_path, sol_B_scale, sol_z_edge);
fprintf('  dt = %.1f ps, n_steps = %d, total %.2f ns\n', ...
        dt*1e12, n_steps, dt*n_steps*1e9);

t_start = tic;
tt.run();
elapsed = toc(t_start);
fprintf('  RUN TIME: %.1f s\n', elapsed);

bunches = tt.readDumps();
fprintf('  dumps: %d\n', numel(bunches));

% --- Post-process emittance + energy ---
me = 9.1093837015e-31;  qe = 1.602176634e-19;  c = 299792458.0;

fprintf('\n  step   t[ps]   z_mean[mm]   gamma   KE[MeV]   sx[um]   sy[um]   eps_nx[um.mrad]\n');
fprintf('  ---------------------------------------------------------------------------------\n');
last = struct();
for k = 1:numel(bunches)
    b = bunches{k};
    n = numel(b.x);
    if n < 5, continue; end
    px = double(b.px);  py = double(b.py);  pz = double(b.pz);
    xx = double(b.x);   yy = double(b.y);   zz = double(b.z);
    p2 = px.*px + py.*py + pz.*pz;
    g  = sqrt(1 + p2 / (me*c)^2);
    bg = sqrt(g.^2 - 1);
    xp = px ./ max(pz, 1e-30);
    sx = std(xx); sy = std(yy);
    spx = std(xp);
    cxxp = mean((xx - mean(xx)) .* (xp - mean(xp)));
    eps_nx = mean(bg) * sqrt(max(sx^2 * spx^2 - cxxp^2, 0));
    fprintf('  %5d  %6.1f  %9.3f   %5.2f   %6.3f   %6.1f   %6.1f    %8.3f\n', ...
            b.step, b.time*1e12, mean(zz)*1e3, mean(g), (mean(g)-1)*0.511, ...
            sx*1e6, sy*1e6, eps_nx*1e6);
    last = struct('time',b.time,'z_mean',mean(zz),'gamma',mean(g), ...
                  'sx',sx,'sy',sy,'eps_nx',eps_nx,'n',n);
end

fprintf('\n=== Final state ===\n');
if ~isempty(fieldnames(last))
    fprintf('  t = %.2f ns, z = %.1f mm, gamma = %.2f (KE = %.3f MeV), N = %d\n', ...
            last.time*1e9, last.z_mean*1e3, last.gamma, ...
            (last.gamma-1)*0.511, last.n);
    fprintf('  RMS spot = (%.3f, %.3f) mm,  eps_nx = %.3f um.mrad\n', ...
            last.sx*1e3, last.sy*1e3, last.eps_nx*1e6);
end

fprintf('\nImpactT reference (commissioning numbers):\n');
fprintf('  At gun exit (z=0.15m): ~5.5 MeV, ~0.6 mm RMS spot\n');
fprintf('  Post-solenoid:         <= 1 mm RMS, eps_n ~ 0.5-1 um.mrad\n');
fprintf('  Fine-grained comparison: see fort.18 / fort.24-26 in %s\n', impact_dir);
end
