function s = Drift3D(varargin)
% DRIFT3D  Spec a vacuum-drift element (no field).
%
% s = timetracking.Drift3D('length', 1.0)

p = inputParser;
p.addParameter('name', 'drift', @(x) ischar(x) || isstring(x));
p.addParameter('length', 1.0, @isnumeric);
p.parse(varargin{:});
r = p.Results;

s = struct('type', 'drift3d', 'name', char(r.name), 'length', r.length);
end
