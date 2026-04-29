function s = FieldMap3D(varargin)
% FIELDMAP3D  Spec a 3D Cartesian field-map element loaded from HDF5.
%
% s = timetracking.FieldMap3D('z', 0, 'length', 0.1, ...
%                              'path', 'fields.h5')
% s = timetracking.FieldMap3D('z', 0, 'length', 0.1, ...
%                              'path', 'rf_cavity.h5', ...
%                              'rf_modulated', true, ...
%                              'f_RF', 2.856e9, 'phase', 0)
%
% The HDF5 file (lucretia-tt schema) must contain at least one of
%   /Bx /By /Bz   or   /Ex /Ey /Ez   (datasets shaped (Nz, Ny, Nx))
% plus
%   /grid_spacing = [dx, dy, dz]   (m)
%   /grid_origin  = [x0, y0, z0]   (m, relative to element start z)
%
% Use writeFieldMap3D to generate test files.

p = inputParser;
p.addParameter('name',         'FM3D',  @(x) ischar(x) || isstring(x));
p.addParameter('z',            0.0,     @isnumeric);
p.addParameter('length',       0.1,     @(x) isnumeric(x) && x > 0);
p.addParameter('path',         '',      @(x) ischar(x) || isstring(x));
p.addParameter('rf_modulated', false,   @(x) islogical(x) || isnumeric(x));
p.addParameter('f_RF',         0.0,     @isnumeric);
p.addParameter('phase',        0.0,     @isnumeric);
p.parse(varargin{:});
r = p.Results;

if isempty(r.path)
    error('timetracking:FieldMap3D:noPath', ...
          'path is required (HDF5 file containing the field map)');
end

s = struct('type',         'field_map_3d', ...
           'name',         char(r.name), ...
           'z',            r.z, ...
           'length',       r.length, ...
           'path',         char(r.path), ...
           'rf_modulated', double(logical(r.rf_modulated)), ...
           'f_RF',         r.f_RF, ...
           'phase',        r.phase);
end
