function s = Collimator(varargin)
% COLLIMATOR  Spec a rectangular aperture collimator (mirror of ImpactT type-(-11)).
%
% s = timetracking.Collimator('z_start', 0, 'z_end', 1.5, ...
%                              'xmin', -3e-3, 'xmax', 3e-3, ...
%                              'ymin', -3e-3, 'ymax', 3e-3, ...
%                              'kill_backward', false, ...
%                              'name', 'collimator')
%
% Active over [z_start, z_end] in lab z. Particles with x outside
% [xmin, xmax] OR y outside [ymin, ymax] are KILLED (alive=0, parked
% at z=-1e9 to keep them out of bunch stats).
%
% kill_backward=true also kills particles with pz<0 anywhere in the
% z range (mirrors ImpactT's stop_bkw flag, encoded in the type-(-11)
% line as a negative dz).
%
% Applied at the end of every integration step in the C++ TrackingLoop.

p = inputParser;
p.addParameter('name',          'COLL', @(x) ischar(x) || isstring(x));
p.addParameter('z_start',       0.0,    @isnumeric);
p.addParameter('z_end',         0.0,    @isnumeric);
p.addParameter('xmin',         -1e30,   @isnumeric);
p.addParameter('xmax',          1e30,   @isnumeric);
p.addParameter('ymin',         -1e30,   @isnumeric);
p.addParameter('ymax',          1e30,   @isnumeric);
p.addParameter('kill_backward', false,  @(x) islogical(x) || isnumeric(x));
p.parse(varargin{:});
r = p.Results;

s = struct('type',          'collimator', ...
           'name',          char(r.name), ...
           'z_start',       r.z_start, ...
           'z_end',         r.z_end, ...
           'xmin',          r.xmin, ...
           'xmax',          r.xmax, ...
           'ymin',          r.ymin, ...
           'ymax',          r.ymax, ...
           'kill_backward', double(logical(r.kill_backward)));
end
