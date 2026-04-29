function s = QuadAnalytic(varargin)
% QUADANALYTIC  Spec a hard-edge magnetic quadrupole.
%
% s = timetracking.QuadAnalytic('z', 0.5, 'length', 0.1, 'gradient', 5.0)
%
% Within the element: B_x = G*y, B_y = G*x.
% Gradient G in T/m. Hard edge: zero outside [z, z+length]. No fringe.

p = inputParser;
p.addParameter('name',     'Q',  @(x) ischar(x) || isstring(x));
p.addParameter('z',        0.0,  @isnumeric);
p.addParameter('length',   0.1,  @(x) isnumeric(x) && x > 0);
p.addParameter('gradient', 0.0,  @isnumeric);
p.parse(varargin{:});
r = p.Results;

s = struct('type',     'quad', ...
           'name',     char(r.name), ...
           'z',        r.z, ...
           'length',   r.length, ...
           'gradient', r.gradient);
end
