function s = SolenoidAnalytic(varargin)
% SOLENOIDANALYTIC  Spec a solenoid with Gaussian on-axis B_z profile.
%
% s = timetracking.SolenoidAnalytic('z', 0.5, 'length', 0.2, ...
%                                    'peak_Bz', 0.1, 'sigma', 0.03, ...
%                                    'z_peak', 0.1)
%
% On-axis B_z = peak_Bz * exp(-((z_local - z_peak)/sigma)^2). Paraxial
% B_r = -(r/2) dB_z/dz. z_peak is measured from element start.

p = inputParser;
p.addParameter('name',    'SOL',  @(x) ischar(x) || isstring(x));
p.addParameter('z',       0.0,    @isnumeric);
p.addParameter('length',  0.2,    @(x) isnumeric(x) && x > 0);
p.addParameter('peak_Bz', 0.0,    @isnumeric);
p.addParameter('sigma',   0.03,   @(x) isnumeric(x) && x > 0);
p.addParameter('z_peak',  -1,     @isnumeric);  % -1 => length/2
p.parse(varargin{:});
r = p.Results;

if r.z_peak < 0
    z_peak = r.length / 2;
else
    z_peak = r.z_peak;
end

s = struct('type',    'solenoid_analytic', ...
           'name',    char(r.name), ...
           'z',       r.z, ...
           'length',  r.length, ...
           'peak_Bz', r.peak_Bz, ...
           'sigma',   r.sigma, ...
           'z_peak',  z_peak);
end
