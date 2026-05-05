function s = ImpactTField(varargin)
% IMPACTTFIELD  Spec an element loaded from an ImpactT rfdata field map.
%
% s = timetracking.ImpactTField('z', 0, 'path', '.../rfdata201', ...
%                                'scale_E', 4.75282e7, ...
%                                'f_RF', 2.856e9, 'phase_deg', 304.668)
%
% s = timetracking.ImpactTField('z', 0.0422, 'path', '.../rfdata102', ...
%                                'scale_B', 0.219982)              % DC solenoid
%
% Parameters:
%   z          - lab-frame z that corresponds to the field map's
%                intrinsic z = 0 (ImpactT type-105 "zedge").
%   length     - m, ImpactT type-105 V2 "Blength". The "real used"
%                field range in lab z is [z, z+length]; outside that
%                range the field is zero. The file's intrinsic
%                [zmin, zmax] may extend beyond this -- the negative-z
%                portion of many ImpactT field maps is Fourier padding,
%                not a physical extent.
%   path       - path to the rfdata file
%   scale_E    - V/m, multiplies the on-axis E_z Fourier sum
%   scale_B    - T,   multiplies the on-axis B_z Fourier sum
%   f_RF       - Hz, RF frequency (default 0 -> static field)
%   phase_deg  - degrees, RF phase offset (default 0). Converted to
%                radians and passed to the C++ kernel.
%
% Convention: E_z = scale_E * F_E(z) * cos(2*pi*f_RF*t + phase_rad).
% Same time factor multiplies B_z; for f_RF = 0 it reduces to a
% constant cos(phase_rad).

p = inputParser;
p.addParameter('name',      'IT',  @(x) ischar(x) || isstring(x));
p.addParameter('z',         0.0,   @isnumeric);
p.addParameter('length',    0.0,   @isnumeric);
p.addParameter('path',      '',    @(x) ischar(x) || isstring(x));
p.addParameter('scale_E',   0.0,   @isnumeric);
p.addParameter('scale_B',   0.0,   @isnumeric);
p.addParameter('f_RF',      0.0,   @isnumeric);
p.addParameter('phase_deg', 0.0,   @isnumeric);
p.parse(varargin{:});
r = p.Results;

if isempty(r.path)
    error('timetracking:ImpactTField:noPath', 'path is required');
end
if r.length <= 0
    error('timetracking:ImpactTField:noLength', ...
          'length (ImpactT Blength) must be > 0; got %g', r.length);
end

s = struct('type',    'impact_t_field', ...
           'name',    char(r.name), ...
           'z',       r.z, ...
           'length',  r.length, ...
           'path',    char(r.path), ...
           'scale_E', r.scale_E, ...
           'scale_B', r.scale_B, ...
           'f_RF',    r.f_RF, ...
           'phase',   r.phase_deg * pi / 180);   % rad
end
