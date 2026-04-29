function s = Solenoid3D(varargin)
% SOLENOID3D  Spec a solenoid loaded from a 2D B_z(r,z) HDF5 map.
%
% s = timetracking.Solenoid3D('z', 0.5, 'length', 0.2, ...
%                              'path', 'sol_map.h5')
%
% The HDF5 file (lucretia-tt schema) must contain
%   /Bz             - (Nz, Nr) double
%   /grid_spacing   = [dr, dz]   (m)
%   /grid_origin    = [r0, z0]   (m, relative to element start z; r0 typically 0)
%
% B_r is computed paraxially from dB_z/dz at gather time.
% Use writeFieldMap2D to generate test files.

p = inputParser;
p.addParameter('name',   'SOL3D', @(x) ischar(x) || isstring(x));
p.addParameter('z',      0.0,     @isnumeric);
p.addParameter('length', 0.2,     @(x) isnumeric(x) && x > 0);
p.addParameter('path',   '',      @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
r = p.Results;

if isempty(r.path)
    error('timetracking:Solenoid3D:noPath', ...
          'path is required (HDF5 file containing the 2D B_z map)');
end

s = struct('type',   'solenoid_3d', ...
           'name',   char(r.name), ...
           'z',      r.z, ...
           'length', r.length, ...
           'path',   char(r.path));
end
