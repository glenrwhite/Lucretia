function writeBunchSeed(path, x, y, z, ux, uy, uz, w)
% WRITEBUNCHSEED  Write a particle seed file for lucretia-tt.
%
%   writeBunchSeed(path, x, y, z, ux, uy, uz)             % w defaults to 1
%   writeBunchSeed(path, x, y, z, ux, uy, uz, w)          % uniform or per-particle weight
%
% All vectors must be length N. Values:
%   x, y, z   - particle positions in metres
%   ux,uy,uz  - proper velocities (gamma * v) in m/s
%   w         - macroparticle weight (multiplicity); scalar or length-N
%
% Pass the resulting path as `tt.beam.seed_file` and lucretia-tt will
% read it instead of the simple n_particles + xyz + uxyz scheme.

if exist(path, 'file'), delete(path); end

x  = x(:);  y  = y(:);  z  = z(:);
ux = ux(:); uy = uy(:); uz = uz(:);
N  = numel(x);
assert(numel(y)==N && numel(z)==N && ...
       numel(ux)==N && numel(uy)==N && numel(uz)==N, ...
       'writeBunchSeed: x/y/z/ux/uy/uz must all be length N');

if nargin < 8 || isempty(w)
    w = 1.0;
end
if isscalar(w)
    w_arr = double(w);
else
    w = w(:);
    assert(numel(w) == N, 'writeBunchSeed: weight must be scalar or length N');
    w_arr = double(w);
end

write_vec = @(name, v) helper_write(path, name, double(v));
write_vec('/x',  x);
write_vec('/y',  y);
write_vec('/z',  z);
write_vec('/ux', ux);
write_vec('/uy', uy);
write_vec('/uz', uz);

if isscalar(w_arr)
    h5create(path, '/w', 1, 'Datatype', 'double');
    h5write(path, '/w', w_arr);
else
    h5create(path, '/w', N, 'Datatype', 'double');
    h5write(path, '/w', w_arr);
end
end


function helper_write(path, name, v)
h5create(path, name, numel(v), 'Datatype', 'double');
h5write(path, name, v);
end
