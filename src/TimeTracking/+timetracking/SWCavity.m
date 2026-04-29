function s = SWCavity(varargin)
% SWCAVITY  Spec an analytic N-cell pi-mode standing-wave cavity.
%
% s = timetracking.SWCavity('z', 0, 'length', 0.1, 'n_cells', 1, ...
%                           'f_RF', 2.856e9, 'peak_field', 100e6, 'phase', 0)
%
% On-axis E_z = peak_field * sin(N*pi*z_local/length) * cos(omega*t + phase),
% paraxial 3D extension via Maxwell. phase=0 puts the field at peak at t=0.

p = inputParser;
p.addParameter('name',       'SW',        @(x) ischar(x) || isstring(x));
p.addParameter('z',          0.0,         @isnumeric);
p.addParameter('length',     0.1,         @(x) isnumeric(x) && x > 0);
p.addParameter('n_cells',    1,           @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('f_RF',       2.856e9,     @isnumeric);
p.addParameter('peak_field', 0.0,         @isnumeric);
p.addParameter('phase',      0.0,         @isnumeric);
p.parse(varargin{:});
r = p.Results;

s = struct('type',       'sw_cavity', ...
           'name',       char(r.name), ...
           'z',          r.z, ...
           'length',     r.length, ...
           'n_cells',    r.n_cells, ...
           'f_RF',       r.f_RF, ...
           'peak_field', r.peak_field, ...
           'phase',      r.phase);
end
