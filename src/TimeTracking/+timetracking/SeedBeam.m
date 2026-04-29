function s = SeedBeam(varargin)
% SEEDBEAM  Initial-bunch spec for TimeTrack.
%
% Two modes:
%
%  1. Simple colocated: N electrons placed at (x,y,z) with proper
%     velocity (ux,uy,uz) [m/s]. Charge/mass default to electron values.
%       s = timetracking.SeedBeam('n_particles', 1, 'ux', 1e7)
%
%  2. Custom from HDF5 seed file (use timetracking.writeBunchSeed to make):
%       s = timetracking.SeedBeam('seed_file', '/tmp/bunch.h5')

p = inputParser;
p.addParameter('n_particles', 1,    @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('x',           0.0,  @isnumeric);
p.addParameter('y',           0.0,  @isnumeric);
p.addParameter('z',           0.0,  @isnumeric);
p.addParameter('ux',          0.0,  @isnumeric);
p.addParameter('uy',          0.0,  @isnumeric);
p.addParameter('uz',          0.0,  @isnumeric);
p.addParameter('weight',      1.0,  @isnumeric);
p.addParameter('seed_file',   '',   @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
r = p.Results;

s = struct('n_particles', r.n_particles, ...
           'x',  r.x,  'y',  r.y,  'z',  r.z, ...
           'ux', r.ux, 'uy', r.uy, 'uz', r.uz, ...
           'weight',    r.weight, ...
           'seed_file', char(r.seed_file));
end
