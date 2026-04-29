function ok = test_mpi_smoke()
% TEST_MPI_SMOKE  Phase 6 step 2: verify mpirun -np N gives the same
% physics as single-rank for a couple of representative tests.
%
% Tests (each run twice, np=1 then np=2):
%   - cyclotron     : tracker correctness (no SC, no emission, no h5 reads)
%   - uniform_sphere: SC path (deposit + IGF + gather), exercises the
%                     multi-box code paths added in Phase 6
%   - freezeplane   : Phase 5 handoff round-trip; exercises the rank-0
%                     MPI gather in BeamIO + freezePlaneOut on a real bunch
%
% Requires src/TimeTracking/lucretia-tt_mpi (build with `build_tt cpu-mpi`).
% Requires mpirun on PATH or at /opt/homebrew/bin/mpirun.

ok_cyc = run_one('cyclotron',      @run_cyclotron);
ok_sph = run_one('uniform-sphere', @run_uniform_sphere);
ok_fp  = run_one('freezeplane',    @run_freezeplane);

ok = ok_cyc && ok_sph && ok_fp;
if ok
    fprintf('\n=== test_mpi_smoke: PASS ===\n');
else
    fprintf('\n=== test_mpi_smoke: FAIL ===\n');
end
end


function ok = run_one(name, fn)
fprintf('\n=== %s ===\n', name);
fprintf('--- np=1 ---\n');
[r1, sx1] = fn(1);
fprintf('  N=%d  sigma_x = %.6f mm\n', r1, sx1*1e3);
fprintf('--- np=2 ---\n');
[r2, sx2] = fn(2);
fprintf('  N=%d  sigma_x = %.6f mm\n', r2, sx2*1e3);

% Pass criterion: same N, sigma_x within 0.5% (deposit/gather order
% across multiple boxes is not bitwise identical, but should be very
% close when the physics is well-resolved).
ok_N  = (r1 == r2);
rel   = abs(sx2 - sx1) / max(sx1, 1e-30);
ok_sx = (rel < 5e-3);
ok    = ok_N && ok_sx;
fprintf('  -> N match: %d, sigma_x rel diff: %.2e -> %s\n', ...
        ok_N, rel, ternary(ok, 'PASS', 'FAIL'));
end


function [N, sx] = run_cyclotron(nranks)
% Minimal cyclotron: 1 particle in uniform Bz. Same setup as test_cyclotron
% but trimmed to one period for speed.
me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
ux0  = 1e7;
Bz   = 1e-3;
gamma_in = sqrt(1 + (ux0/299792458)^2);
T_c  = 2*pi*gamma_in*me/(qe*Bz);

tt = TimeTrack();
tt.mpi_nranks = nranks;
tt.lattice = { ...
    timetracking.UniformB('name','B','Bx',0,'By',0,'Bz',Bz), ...
    timetracking.BeamMonitor('name','m','dump_every',100) };
tt.beam = struct('n_particles',1,'x',0,'y',0,'z',0,'ux',ux0,'uy',0,'uz',0);
tt.geom_lo    = [-0.1, -0.1, -0.1];
tt.geom_hi    = [ 0.1,  0.1,  0.1];
tt.geom_ncell = [16, 16, 16];
tt.dt = T_c/1000;
tt.n_steps = 1000;
tt.run();
b = tt.readDumps();
last = b{end};
N    = numel(last.x);
% Single particle: report orbit radius as "sigma_x"
sx   = sqrt(double(last.x).^2 + double(last.y).^2);
end


function [N, sx] = run_uniform_sphere(nranks)
% Subset of test_uniform_sphere: 500 macros, 1 nC, 0.5 mm sphere.
me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
gamma_in = 1.0;

N  = 500;
R  = 0.5e-3;
Q  = 1e-9;

rng(7);
% Uniform-density sphere in (x,y,z)
u = rand(N,1);
r = R * u.^(1/3);
ph = 2*pi*rand(N,1);
ct = 2*rand(N,1)-1; st = sqrt(1-ct.^2);
x = r.*st.*cos(ph);  y = r.*st.*sin(ph);  z = r.*ct;
ux = zeros(N,1); uy = zeros(N,1); uz = zeros(N,1);
w_per = Q / (N*qe);

seed = fullfile(tempdir, 'mpi_smoke_seed.h5');
timetracking.writeBunchSeed(seed, x, y, z, ux, uy, uz, w_per);

tt = TimeTrack();
tt.mpi_nranks = nranks;
tt.lattice = { timetracking.BeamMonitor('name','m','dump_every',5) };
tt.beam = timetracking.SeedBeam('seed_file', seed);
tt.geom_lo    = [-2e-3, -2e-3, -2e-3];
tt.geom_hi    = [ 2e-3,  2e-3,  2e-3];
tt.geom_ncell = [32, 32, 32];
tt.dt = 5e-15;
tt.n_steps = 30;
tt.enable_space_charge = true;
tt.run();
b  = tt.readDumps();
last = b{end};
N    = numel(last.x);
sx   = std(double(last.x));
end


function [N, sx] = run_freezeplane(nranks)
% Phase 5 round-trip: seed a bunch, run a 1 ns drift, dump, freezePlaneOut.
% Exercises the BeamIO MPI gather (per-rank counts > 0 for both ranks)
% on a meaningful bunch (1000 macros / 1 nC).
me   = 9.1093837015e-31;
qe   = 1.602176634e-19;
c    = 299792458.0;
N    = 1000;  gamma_in = 50.0;
beta = sqrt(1 - 1/gamma_in^2);
uz0  = gamma_in * beta * c;

rng(11);
x = 0.3e-3*randn(N,1); y = 0.3e-3*randn(N,1); z = 0.3e-3*randn(N,1);
sigma_u_th = sqrt(0.5 * qe / me);
ux = sigma_u_th*randn(N,1); uy = sigma_u_th*randn(N,1); uz = uz0*ones(N,1);
w_per = 100e-12 / (N * qe);

seed = fullfile(tempdir, 'mpi_smoke_fp_seed.h5');
timetracking.writeBunchSeed(seed, x, y, z, ux, uy, uz, w_per);

tt = TimeTrack();
tt.mpi_nranks = nranks;
tt.lattice = { ...
    timetracking.Drift3D('name','D1','length',1.0), ...
    timetracking.BeamMonitor('name','m','dump_every',1000) };
tt.beam = timetracking.SeedBeam('seed_file', seed);
tt.geom_lo    = [-3e-3, -3e-3, -2e-3];
tt.geom_hi    = [ 3e-3,  3e-3,  3e-1];
tt.geom_ncell = [16, 16, 64];
tt.dt = 1e-12;  tt.n_steps = 1000;
tt.enable_space_charge = false;
tt.run();

beam = tt.freezePlaneOut('last');
N    = numel(beam.Bunch.Q);
sx   = std(beam.Bunch.x(1, :));
end


function v = ternary(cond, a, b)
if cond, v = a; else, v = b; end
end
