function writeFieldMap3D(path, comps, grid_spacing, grid_origin)
% WRITEFIELDMAP3D  Write a 3D Cartesian field map in lucretia-tt HDF5 schema.
%
%   writeFieldMap3D(path, comps, grid_spacing, grid_origin)
%
% comps        - struct with optional fields Bx, By, Bz, Ex, Ey, Ez. Each
%                must be the same size (Nx, Ny, Nz). Missing components
%                are skipped (treated as zero by the C++ reader).
% grid_spacing - [dx, dy, dz] in metres
% grid_origin  - [x0, y0, z0] in metres (relative to the element start z)
%
% Datasets are written as (Nz, Ny, Nx) C-order doubles, which is what
% the C++ reader expects. MATLAB stores 3D arrays in column-major
% (Fortran) order, so we permute (3,2,1) before writing -- HDF5's C
% "row-major" matches MATLAB's permuted column-major.
%
% Recreates the file each call (deletes any existing path).

if exist(path, 'file'), delete(path); end

field_names = {'Bx','By','Bz','Ex','Ey','Ez'};
ref_size = [];
for k = 1:numel(field_names)
    if isfield(comps, field_names{k})
        v = comps.(field_names{k});
        if isempty(ref_size)
            ref_size = size(v);
            if numel(ref_size) < 3, ref_size(end+1:3) = 1; end
        elseif ~isequal(size(v), ref_size)
            error('writeFieldMap3D:shape', ...
                  'Component %s has size [%s], expected [%s]', ...
                  field_names{k}, num2str(size(v)), num2str(ref_size));
        end
    end
end
if isempty(ref_size)
    error('writeFieldMap3D:empty', ...
          'comps has no field components; supply at least one of Bx/By/Bz/Ex/Ey/Ez');
end

Nx = ref_size(1); Ny = ref_size(2); Nz = ref_size(3);

% h5create wants the size in the on-disk order: (Nz, Ny, Nx). MATLAB's
% h5write with chunked layout will permute on the fly when we write a
% (3,2,1)-permuted array.
for k = 1:numel(field_names)
    nm = field_names{k};
    if isfield(comps, nm)
        h5create(path, ['/' nm], [Nz, Ny, Nx], 'Datatype', 'double');
        h5write(path, ['/' nm], permute(comps.(nm), [3, 2, 1]));
    end
end

h5create(path, '/grid_spacing', 3, 'Datatype', 'double');
h5write(path, '/grid_spacing', double(grid_spacing(:))');

h5create(path, '/grid_origin', 3, 'Datatype', 'double');
h5write(path, '/grid_origin', double(grid_origin(:))');
end
