function s = BeamMonitor(varargin)
% BEAMMONITOR  Spec a beam-monitor (openPMD dump every N steps).
%
% s = timetracking.BeamMonitor('dump_every', 10)
% s = timetracking.BeamMonitor('name','m1','dump_every',1)

p = inputParser;
p.addParameter('name', 'mon', @(x) ischar(x) || isstring(x));
p.addParameter('dump_every', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.parse(varargin{:});
r = p.Results;

s = struct('type', 'beam_monitor', 'name', char(r.name), ...
           'dump_every', r.dump_every);
end
