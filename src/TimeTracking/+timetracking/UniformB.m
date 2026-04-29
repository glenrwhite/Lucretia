function s = UniformB(varargin)
% UNIFORMB  Spec a uniform B field element for lucretia-tt.
%
% s = timetracking.UniformB('Bz', 0.01)
% s = timetracking.UniformB('name','sol1','Bx',0,'By',0,'Bz',0.01)
%
% Returns a struct with type='uniform_b'.

p = inputParser;
p.addParameter('name', 'UB', @(x) ischar(x) || isstring(x));
p.addParameter('Bx', 0.0, @isnumeric);
p.addParameter('By', 0.0, @isnumeric);
p.addParameter('Bz', 0.0, @isnumeric);
p.parse(varargin{:});
r = p.Results;

s = struct('type', 'uniform_b', 'name', char(r.name), ...
           'Bx', r.Bx, 'By', r.By, 'Bz', r.Bz);
end
