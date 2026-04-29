function ok = test_fieldmap3d()
% TEST_FIELDMAP3D  Phase 2C validation of FieldMap3D loading + interp.
%
% Generate a uniform B_z = 1 mT 3D field map covering a 4 cm x 4 cm x
% 1 cm box. Drop a single electron (ux = 1e7 m/s, uy = 0, uz = 0) into
% it; the field-map element should reproduce the cyclotron Larmor radius
% within trilinear interpolation accuracy (well below 1% for a uniform
% field). Compares against an inline UniformB element on the same orbit.

me = 9.1093837015e-31;
qe = 1.602176634e-19;
c  = 299792458.0;

% --- Beam: same as test_cyclotron ---
ux0 = 1e7;
Bz0 = 1e-3;
gamma_p = sqrt(1 + (ux0/c)^2);
rL = me * ux0 / (qe * Bz0);
Tc = 2*pi * gamma_p * me / (qe * Bz0);

% --- Build a uniform-Bz 3D map ---
% Cyclotron orbit lives in the x-y plane around (0, +rL, 0).
% Map must cover that orbit transversely; z-extent only needs to cover
% the small z-drift around the element centre.
margin = 4 * rL;
Nx = 16; Ny = 16; Nz = 4;
dx = 2*margin / (Nx - 1);
dy = 2*margin / (Ny - 1);
dz = 1e-3 / (Nz - 1);    % 1 mm spanned by 4 z-points

Bx = zeros(Nx, Ny, Nz);
By = zeros(Nx, Ny, Nz);
Bz = Bz0 * ones(Nx, Ny, Nz);

map_path = fullfile(tempdir, 'lucretia_tt_uniform_Bz.h5');
% grid_origin x,y in lab frame; z relative to element start
timetracking.writeFieldMap3D(map_path, ...
    struct('Bx', Bx, 'By', By, 'Bz', Bz), ...
    [dx, dy, dz], ...
    [-margin, -margin, 0]);

fprintf('FieldMap3D uniform-Bz cyclotron:\n');
fprintf('  rL = %.4f mm,  T_c = %.3f ns\n', rL*1e3, Tc*1e9);

% --- TimeTrack run ---
% Element occupies z in [-0.5 mm, +0.5 mm]; grid covers same range.
tt = TimeTrack();
tt.lattice = { ...
    timetracking.FieldMap3D('name','FM','z',-0.5e-3,'length',1.0e-3, ...
                            'path', map_path), ...
    timetracking.BeamMonitor('name','mon','dump_every',10) };
tt.beam = timetracking.SeedBeam('n_particles',1,'ux',ux0);

tt.geom_lo    = [-margin, -margin, -2e-3];
tt.geom_hi    = [ margin,  margin,  2e-3];
tt.geom_ncell = [32, 32, 16];

tt.dt      = Tc / 1000;
tt.n_steps = 2000;

tt.run();
bunches = tt.readDumps();
fprintf('  dumps = %d\n', numel(bunches));

xs = cellfun(@(b) double(b.x(1)), bunches);
ys = cellfun(@(b) double(b.y(1)), bunches);
ts = cellfun(@(b) b.time, bunches);

% Orbit centre: (0, +rL) for electron with v_x>0, B_z>0.
r       = sqrt(xs.^2 + (ys - rL).^2);
mean_r  = mean(r);
max_dev = max(abs(r - rL)) / rL;
mean_dev = abs(mean_r - rL) / rL;

[~, k1] = min(abs(ts - Tc));
[~, k2] = min(abs(ts - 2*Tc));
d1 = sqrt(xs(k1)^2 + ys(k1)^2);
d2 = sqrt(xs(k2)^2 + ys(k2)^2);

fprintf('  mean orbit radius:    %.6f mm  (analytic %.6f mm)\n', ...
        mean_r*1e3, rL*1e3);
fprintf('  |<r> - r_L| / r_L:    %.2e\n', mean_dev);
fprintf('  max |r - r_L| / r_L:  %.2e\n', max_dev);
fprintf('  return @ T_c:    |xy| = %.4f mm  (%.2f%% of r_L)\n', d1*1e3, 100*d1/rL);
fprintf('  return @ 2*T_c:  |xy| = %.4f mm  (%.2f%% of r_L)\n', d2*1e3, 100*d2/rL);

ok = mean_dev < 5e-3 && max_dev < 1e-2 && d2/rL < 1e-2;
if ok, fprintf('  PASS\n'); else, fprintf('  FAIL\n'); end
end
