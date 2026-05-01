function s = WakeField(varargin)
% WAKEFIELD  Spec a short-range wakefield element (Bane analytic).
%
% s = timetracking.WakeField('z_start', 1.93, 'z_end', 4.97, ...
%                             'iris_a', 0.0116, 'gap_g', 0.0292, ...
%                             'period_L', 0.035, ...
%                             'longitudinal', true, 'transverse', true, ...
%                             'n_slices', 200, 'name', 'wakefield_L0A')
%
% Active when any particle z lies in [z_start, z_end]. At each timestep
% the binner sorts live particles into n_slices longitudinal bins,
% computes the slice charge sum and dipole moments (sum q*x, sum q*y),
% allreduce-s across MPI ranks, convolves causally with Bane's analytic
% wake formulas (longitudinal monopole + transverse dipole), and applies
% the resulting per-particle impulse to each particle's px/py/pz.
%
% Wake parameters describe an iris-loaded periodic structure cell:
%   iris_a   - iris radius a (m)
%   gap_g    - cell gap g (m)
%   period_L - cell period L (m)
%
% For the LCLS/SLAC S-band TW structure (3 GHz, 2*pi/3 mode) typical
% values are a~11.6 mm, g~29.2 mm, L~35 mm (a is averaged over the
% structure if your iris tapers).
%
% longitudinal/transverse (default both true) toggle each contribution
% independently, matching the ImpactT type-(-6) v(5)/v(6) flags.

p = inputParser;
p.addParameter('name',         'WAKE',  @(x) ischar(x) || isstring(x));
p.addParameter('z_start',      0.0,     @isnumeric);
p.addParameter('z_end',        0.0,     @isnumeric);
p.addParameter('iris_a',       0.0,     @isnumeric);
p.addParameter('gap_g',        0.0,     @isnumeric);
p.addParameter('period_L',     0.0,     @isnumeric);
p.addParameter('longitudinal', true,    @(x) islogical(x) || isnumeric(x));
p.addParameter('transverse',   true,    @(x) islogical(x) || isnumeric(x));
p.addParameter('n_slices',     200,     @(x) isnumeric(x) && isscalar(x));
p.parse(varargin{:});
r = p.Results;

s = struct('type',         'wakefield', ...
           'name',         char(r.name), ...
           'z_start',      r.z_start, ...
           'z_end',        r.z_end, ...
           'iris_a',       r.iris_a, ...
           'gap_g',        r.gap_g, ...
           'period_L',     r.period_L, ...
           'longitudinal', double(logical(r.longitudinal)), ...
           'transverse',   double(logical(r.transverse)), ...
           'n_slices',     round(r.n_slices));
end
