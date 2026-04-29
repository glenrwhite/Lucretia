function writeFieldMap2D(path, Bz, grid_spacing, grid_origin)
% WRITEFIELDMAP2D  Write a 2D RZ B_z(r,z) map in lucretia-tt HDF5 schema.
%
%   writeFieldMap2D(path, Bz, grid_spacing, grid_origin)
%
% Bz           - (Nr, Nz) array in MATLAB column-major; written as (Nz, Nr)
% grid_spacing - [dr, dz] in metres
% grid_origin  - [r0, z0] in metres (relative to element start z)

if exist(path, 'file'), delete(path); end

[Nr, Nz] = size(Bz);

h5create(path, '/Bz', [Nz, Nr], 'Datatype', 'double');
h5write(path, '/Bz', permute(Bz, [2, 1]));

h5create(path, '/grid_spacing', 2, 'Datatype', 'double');
h5write(path, '/grid_spacing', double(grid_spacing(:))');

h5create(path, '/grid_origin', 2, 'Datatype', 'double');
h5write(path, '/grid_origin', double(grid_origin(:))');
end
