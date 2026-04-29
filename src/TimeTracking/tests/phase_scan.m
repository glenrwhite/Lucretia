function phase_scan()
% PHASE_SCAN  Sweep gun_phase_deg, find the phase that maximises exit gamma.
%
% Resolves the Phase 4D open question: which sign of the omega*Tini phase
% shift in readImpactTLattice is correct? ImpactT's GUN line specifies
% phase_imp = 304.668 deg with Tini = -13.06 ps; our run starts at t=0,
% so the phase delta is omega*Tini = 2*pi*2.856e9 * (-13.06e-12) = -0.234
% rad = -13.42 deg. That puts the predicted "best" phase for our
% simulation at one of:
%
%   case +shift  (phase_LT = phase_imp + omega*Tini): 304.668 - 13.42 = 291.24 deg
%   case -shift  (phase_LT = phase_imp - omega*Tini): 304.668 + 13.42 = 318.10 deg
%   case no shift                                    : 304.668 deg
%
% This scan picks the phase that gives the highest exit gamma (= most
% effective gun acceleration) for a single test particle, with no SC and
% no cathode emission to keep things fast.

impact_dir = '/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT';
gun_path   = fullfile(impact_dir, 'rfdata201');

% Read gun field zmin so we can place the seeded particle at the cathode
fid = fopen(gun_path, 'r');  hh = textscan(fid, '%f', 4);  fclose(fid);
gun_zmin = hh{1}(2);   % field intrinsic zmin
gun_zmax = hh{1}(3);

% Seed a single electron at the cathode plate (= gun_z + zmin) with a
% small forward velocity (uz = 1e7 m/s ~ 0.03 c, gamma ~ 1.0006). Just
% enough to put the particle on the +z side of the cathode without
% needing image-charge dynamics to extract it.
me = 9.1093837015e-31;  qe = 1.602176634e-19;  c = 299792458.0;
uz_seed = 1.0e7;
gun_z_edge = 0.0;     % from ImpactT.in GUN V1
cathode_z  = gun_z_edge + gun_zmin;

% Fine scan around the predicted peak (~300 deg).
phases = [0:30:269, 270:2:330, 345];
gamma_exit = zeros(size(phases));
ke_exit_MeV = zeros(size(phases));

% Run config
margin = 5e-3;
dt = 1e-13;                  % 0.1 ps -- tight for gun phase sensitivity
n_steps = 5000;              % 0.5 ns -> particle should exit gun in ~ 1 ns
                             % so this captures it before exit; we read the
                             % final state regardless.

fprintf('Phase scan: %d phases from 0 to 345 deg (15 deg steps)\n', numel(phases));
fprintf('  gun field zmin=%.4f m, zmax=%.4f m, cathode at z=%.4f m\n', ...
        gun_zmin, gun_zmax, cathode_z);
fprintf('  seed uz = %g m/s (gamma ~ %.4f)\n', uz_seed, sqrt(1+(uz_seed/c)^2));
fprintf('  dt = %.1f ps, n_steps = %d (total %.2f ns)\n\n', ...
        dt*1e12, n_steps, dt*n_steps*1e9);

t_start = tic;
for k = 1:numel(phases)
    tt = TimeTrack();
    tt.lattice = { ...
        timetracking.ImpactTField('name','GUN','z',gun_z_edge, ...
            'path', gun_path, 'scale_E', 4.75282e7, ...
            'f_RF', 2.856e9, 'phase_deg', phases(k)), ...
        timetracking.BeamMonitor('name','mon','dump_every', n_steps) };
    tt.beam = timetracking.SeedBeam('n_particles', 1, ...
        'x', 0, 'y', 0, 'z', cathode_z + 1e-6, ...
        'ux', 0, 'uy', 0, 'uz', uz_seed);
    tt.geom_lo = [-margin, -margin, cathode_z - 5e-3];
    tt.geom_hi = [ margin,  margin, gun_z_edge + gun_zmax + 5e-3];
    tt.geom_ncell = [16, 16, 64];
    tt.dt      = dt;
    tt.n_steps = n_steps;
    tt.enable_space_charge = false;
    tt.run();

    bunches = tt.readDumps();
    if isempty(bunches)
        gamma_exit(k) = NaN;  ke_exit_MeV(k) = NaN;
        continue;
    end
    last = bunches{end};
    if numel(last.x) < 1
        gamma_exit(k) = NaN;  ke_exit_MeV(k) = NaN;
        continue;
    end
    px = double(last.px(1)); py = double(last.py(1)); pz = double(last.pz(1));
    p2 = px*px + py*py + pz*pz;
    gamma_exit(k) = sqrt(1 + p2/(me*c)^2);
    ke_exit_MeV(k) = (gamma_exit(k) - 1) * 0.511;
    fprintf('  phase = %3d deg: gamma_exit = %5.2f, KE = %5.3f MeV\n', ...
            phases(k), gamma_exit(k), ke_exit_MeV(k));
end
fprintf('\nScan elapsed: %.1f s\n', toc(t_start));

% --- Identify peak ---
[~, i_peak] = max(gamma_exit);
phase_peak = phases(i_peak);
fprintf('\n=== Result ===\n');
fprintf('  Peak gamma exit  = %.2f at phase = %d deg\n', ...
        gamma_exit(i_peak), phase_peak);

% --- Compare to expected shifts ---
omega_Tini_deg = 360 * 2.856e9 * (-13.06e-12);    % deg
fprintf('  ImpactT phase_imp        = 304.668 deg (Tini = -13.06 ps)\n');
fprintf('  omega*Tini               = %+.2f deg\n', omega_Tini_deg);
fprintf('  Predicted shifted phases:\n');
fprintf('    no shift               = 304.67 deg\n');
fprintf('    + omega*Tini           = %.2f deg\n', 304.668 + omega_Tini_deg);
fprintf('    - omega*Tini           = %.2f deg\n', 304.668 - omega_Tini_deg);

% Distance of measured peak to each candidate (modulo 360)
function d = wrap_dist(a, b)
    d = abs(mod(a - b + 180, 360) - 180);
end
candidates = struct( ...
    'no_shift',   304.668, ...
    'plus_shift', mod(304.668 + omega_Tini_deg, 360), ...
    'minus_shift', mod(304.668 - omega_Tini_deg, 360));

fprintf('\n  Distance from measured peak (%d deg) to each candidate:\n', phase_peak);
fnames = fieldnames(candidates);
for i = 1:numel(fnames)
    nm = fnames{i};  cand = candidates.(nm);
    fprintf('    %-12s candidate = %.2f deg, |diff| = %.2f deg\n', ...
            nm, cand, wrap_dist(phase_peak, cand));
end
end
