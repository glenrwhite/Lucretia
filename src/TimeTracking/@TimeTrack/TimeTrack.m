classdef TimeTrack < handle
% TIMETRACK  MATLAB front-end for the lucretia-tt time-based 3D PIC tracker.
%
% Workflow:
%   tt = TimeTrack();
%   tt.lattice = {timetracking.UniformB('Bz', 1e-3), ...
%                 timetracking.BeamMonitor('dump_every', 4)};
%   tt.beam    = timetracking.SeedBeam('n_particles', 1, 'ux', 1e7);
%   tt.dt      = 1.7e-10;
%   tt.n_steps = 400;
%   tt.run();
%   bunches = tt.readDumps();   % cell array of per-step structs
%
% TimeTrack writes a ParmParse input file into work_dir, invokes the
% lucretia-tt binary on it, and parses the openPMD HDF5 output.
%
% Set work_dir to control where the run happens; defaults to a tempname
% under the system temp dir. Set binary to override the executable path
% (defaults to which('lucretia-tt')).

properties
    lattice    = {}                                              % cell of element structs
    beam       = struct('n_particles', 1, ...
                        'x', 0, 'y', 0, 'z', 0, ...
                        'ux', 0, 'uy', 0, 'uz', 0)
    dt         = 1e-12                                           % s
    n_steps    = 100
    geom_lo    = [-0.05, -0.05,  0.0]                            % m
    geom_hi    = [ 0.05,  0.05,  1.0]                            % m
    geom_ncell = [32, 32, 32]
    output_dir = 'diags'                                         % relative to work_dir
    work_dir   = ''                                              % '' -> tempname
    binary     = ''                                              % '' -> which('lucretia-tt')
    enable_space_charge = false                                  % toggles ParmParse space_charge.enabled

    last_input_file = ''
    last_log        = ''
    last_status     = -1
end

methods
    function obj = TimeTrack(lattice, beam)
        if nargin >= 1 && ~isempty(lattice), obj.lattice = lattice; end
        if nargin >= 2 && ~isempty(beam),    obj.beam    = beam;    end
    end

    function run(obj)
    % Resolve binary, write input file, invoke lucretia-tt.
        if isempty(obj.binary)
            % Look next to TimeTrack.m: src/TimeTracking/lucretia-tt
            ttDir   = fileparts(fileparts(mfilename('fullpath')));
            cand    = fullfile(ttDir, 'lucretia-tt');
            if exist(cand, 'file') == 2
                obj.binary = cand;
            else
                obj.binary = which('lucretia-tt');
            end
            if isempty(obj.binary) || exist(obj.binary, 'file') ~= 2
                error('TimeTrack:noBinary', ...
                    ['lucretia-tt binary not found at %s nor on path. ' ...
                     'Build with `build_tt cpu` or set tt.binary.'], cand);
            end
        end
        if isempty(obj.work_dir)
            obj.work_dir = tempname;
        end
        if ~exist(obj.work_dir, 'dir')
            mkdir(obj.work_dir);
        end

        in_file = fullfile(obj.work_dir, 'lucretia_tt.in');
        writeParmParseInput(obj, in_file);
        obj.last_input_file = in_file;

        [obj.last_status, obj.last_log] = invokeBinary(obj, in_file);
        if obj.last_status ~= 0
            error('TimeTrack:run', ...
                  'lucretia-tt failed (status %d). Log:\n%s', ...
                  obj.last_status, obj.last_log);
        end
    end

    function bunches = readDumps(obj)
    % Parse all openPMD dumps under work_dir/output_dir into a cell array
    % of structs (sorted by step), each containing x,y,z,px,py,pz,q,w,id,
    % time, and step.
        diag_dir = fullfile(obj.work_dir, obj.output_dir);
        bunches  = readBeamOpenPMD(diag_dir);
    end
end
end
