function carlsten_lcls()
% CARLSTEN_LCLS  Phase 4B end-to-end SC-tracking demo (Carlsten-flavoured).
%
% A scaled-down photoinjector-style benchmark using the lucretia-tt
% machinery built up through Phase 4A:
%   - Pre-seeded relativistic bunch (gamma = 10, ~5 MeV) with a Gaussian
%     transverse profile and short Gaussian bunch length, total charge
%     100 pC.
%   - Optional solenoid downstream (Gaussian B_z profile, ~0.18 T peak).
%   - Space-charge solver on throughout (with self-force subtraction
%     calibrated in Phase 4B).
%
% Why pre-seed instead of emitting from a real cathode? The pure-sine
% SWCavity profile we ship can't break cold electrons out of the cathode
% region in a single half-cycle, so cathode-to-relativistic dynamics need
% a more sophisticated gun field model (Phase 5+ TODO). Skipping cathode
% dynamics here lets us focus on the SC + element-gather + emittance
% machinery, which is the Phase 4B story.
%
% Reports normalised RMS emittance evolution. Compare qualitatively to
% LCLS injector commissioning numbers (Akre et al. 2008 PRSTAB):
%   Final eps_n ~ 0.4-0.6 um.mrad at 100-250 pC after compensation.

me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
c    = 299792458.0;

% --- Bunch ---
gamma_in    = 10.0;
beta_in     = sqrt(1 - 1/gamma_in^2);
uz0         = gamma_in * beta_in * c;     % proper velocity m/s
n_macros    = 2000;
total_Q     = 100e-12;                    % 100 pC
spot_RMS    = 0.3e-3;                     % 0.3 mm Gaussian RMS
bunch_RMS_z = 0.3e-3;                     % 0.3 mm RMS in z (~ 1 ps at gamma=10)
mte_eV      = 0.5;                        % thermal momentum spread

% Generate the seeded bunch (Gaussian transverse, Gaussian z, cold uz)
rng(42);
x = spot_RMS * randn(n_macros, 1);
y = spot_RMS * randn(n_macros, 1);
z = bunch_RMS_z * randn(n_macros, 1);   % centred on 0
% Add MTE thermal momentum (ux, uy)
sigma_u_thermal = sqrt(mte_eV * qe / me);
ux = sigma_u_thermal * randn(n_macros, 1);
uy = sigma_u_thermal * randn(n_macros, 1);
uz = uz0 * ones(n_macros, 1);
w_per = total_Q / (n_macros * qe);

seed_path = fullfile(tempdir, 'lucretia_tt_carlsten_seed.h5');
timetracking.writeBunchSeed(seed_path, x, y, z, ux, uy, uz, w_per);

% --- Lattice ---
% Bunch starts centred at z = 0. Push it +z. Solenoid at z ~ 0.05 m.
sol_z      = 0.02;
sol_length = 0.10;
sol_peak_B = 0.30;       % moderately strong (LCLS solenoid is ~0.2-0.3 T)
sol_sigma  = 0.025;
sol_z_peak = sol_length / 2;

% --- Geometry: tracks bunch over a short fixed window ---
% Bunch travels at v = beta*c ~ 0.995c. In 1 ns, travels ~0.3 m.
% Use a static box; particles past geom_hi[2] just stop seeing SC.
margin     = 1.5e-3;
geom_lo    = [-margin, -margin, -2.0e-3];
geom_hi    = [ margin,  margin,  0.20];
geom_ncell = [48, 48, 192];

dt         = 0.5e-12;     % 0.5 ps
n_steps    = 1000;        % 0.5 ns total -> bunch travels ~150 mm

% Output every 25 steps -> 40 dumps over the run.
dump_every = 25;

% --- Build TimeTrack ---
tt = TimeTrack();
tt.lattice = { ...
    timetracking.SolenoidAnalytic('name','sol', 'z', sol_z, ...
        'length', sol_length, 'peak_Bz', sol_peak_B, ...
        'sigma', sol_sigma, 'z_peak', sol_z_peak), ...
    timetracking.BeamMonitor('name','mon', 'dump_every', dump_every) };
tt.beam = timetracking.SeedBeam('seed_file', seed_path);
tt.geom_lo    = geom_lo;
tt.geom_hi    = geom_hi;
tt.geom_ncell = geom_ncell;
tt.dt         = dt;
tt.n_steps    = n_steps;
tt.enable_space_charge = true;

fprintf('Phase 4B SC + solenoid demo:\n');
fprintf('  bunch:   %d macros, Q = %.1f pC, gamma_in = %.1f\n', n_macros, total_Q*1e12, gamma_in);
fprintf('  spot:    %.2f mm RMS x %.2f mm RMS z, MTE = %.1f eV\n', spot_RMS*1e3, bunch_RMS_z*1e3, mte_eV);
fprintf('  solenoid: peak %.2f T, length %.0f mm at z = %.0f mm\n', sol_peak_B, sol_length*1e3, sol_z*1e3);
fprintf('  dt = %.1f ps, n_steps = %d, total run %.2f ns\n', dt*1e12, n_steps, dt*n_steps*1e9);

t_start = tic;
tt.run();
elapsed = toc(t_start);
fprintf('  RUN TIME: %.1f s\n', elapsed);

bunches = tt.readDumps();
fprintf('  dumps: %d\n', numel(bunches));

% --- Post-process: emittance vs t ---
fprintf('\nEmittance vs t (normalised RMS):\n');
fprintf('  step   t[ps]   z_mean[mm]   gamma   sx[um]  sy[um]    eps_nx[um.mrad]  eps_ny\n');
fprintf('  ------------------------------------------------------------------------\n');

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
    yp = py ./ max(pz, 1e-30);
    sx = std(xx); sy = std(yy);
    spx = std(xp); spy = std(yp);
    cxxp = mean((xx - mean(xx)) .* (xp - mean(xp)));
    cyyp = mean((yy - mean(yy)) .* (yp - mean(yp)));
    bg_mean = mean(bg);
    eps_nx = bg_mean * sqrt(max(sx^2 * spx^2 - cxxp^2, 0));
    eps_ny = bg_mean * sqrt(max(sy^2 * spy^2 - cyyp^2, 0));
    fprintf('  %5d  %6.1f  %9.3f   %5.2f  %6.1f  %6.1f    %8.3f         %.3f\n', ...
            b.step, b.time*1e12, mean(zz)*1e3, mean(g), ...
            sx*1e6, sy*1e6, eps_nx*1e6, eps_ny*1e6);
    last = struct('eps_nx', eps_nx, 'eps_ny', eps_ny, 'gamma', mean(g), ...
                  'sx', sx, 'sy', sy, 'z_mean', mean(zz), 'time', b.time);
end

fprintf('\n=== Final state ===\n');
if ~isempty(fieldnames(last))
    fprintf('  t = %.2f ns, z_mean = %.1f mm, gamma = %.2f (KE = %.2f MeV)\n', ...
            last.time*1e9, last.z_mean*1e3, last.gamma, (last.gamma-1)*0.511);
    fprintf('  RMS spot = (%.3f, %.3f) mm\n', last.sx*1e3, last.sy*1e3);
    fprintf('  norm eps = (%.3f, %.3f) um.mrad\n', last.eps_nx*1e6, last.eps_ny*1e6);
end

fprintf('\nLCLS reference (Akre et al. 2008 PRSTAB):\n');
fprintf('  Compensated injector eps_n ~ 0.4-0.6 um.mrad at 100 pC.\n');
fprintf('  This demo skips the cathode-emission + gun acceleration and\n');
fprintf('  starts from a relativistic bunch -- so absolute numbers will\n');
fprintf('  differ from the published gun-exit emittances. Watch the trend\n');
fprintf('  vs t for the SC + solenoid focusing pattern.\n');
end
