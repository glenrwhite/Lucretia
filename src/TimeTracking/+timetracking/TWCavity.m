function s = TWCavity(varargin)
% TWCAVITY  Spec a forward TM traveling-wave cavity (constant amplitude).
%
% s = timetracking.TWCavity('z', 0, 'length', 0.1, 'f_RF', 2.856e9, ...
%                            'peak_field', 100e6, 'phase', 0, 'v_phase', 3e8)
%
% On-axis E_z = peak_field * cos(omega*t - kz*z_local + phase),
% kz = omega/v_phase. Defaults: f_RF = 2.856 GHz, v_phase = c.

p = inputParser;
p.addParameter('name',       'TW',        @(x) ischar(x) || isstring(x));
p.addParameter('z',          0.0,         @isnumeric);
p.addParameter('length',     0.1,         @(x) isnumeric(x) && x > 0);
p.addParameter('f_RF',       2.856e9,     @isnumeric);
p.addParameter('peak_field', 0.0,         @isnumeric);
p.addParameter('phase',      0.0,         @isnumeric);
p.addParameter('v_phase',    299792458.0, @(x) isnumeric(x) && x > 0);
p.parse(varargin{:});
r = p.Results;

s = struct('type',       'tw_cavity', ...
           'name',       char(r.name), ...
           'z',          r.z, ...
           'length',     r.length, ...
           'f_RF',       r.f_RF, ...
           'peak_field', r.peak_field, ...
           'phase',      r.phase, ...
           'v_phase',    r.v_phase);
end
