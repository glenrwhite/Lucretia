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
    dt           = 1e-12                                         % s, initial dt
    n_steps      = 100
    dt_change_t  = []                                            % s, time at which to switch dt (empty = no switch)
    dt_after     = []                                            % s, dt to use after dt_change_t (empty = same as dt)
    t_start      = []                                            % s, initial sim time (empty = 0)
    geom_lo    = [-0.05, -0.05,  0.0]                            % m
    geom_hi    = [ 0.05,  0.05,  1.0]                            % m
    geom_ncell = [32, 32, 32]
    output_dir = 'diags'                                         % relative to work_dir
    work_dir   = ''                                              % '' -> tempname
    binary     = ''                                              % '' -> which('lucretia-tt')
    enable_space_charge = false                                  % toggles ParmParse space_charge.enabled
    sc_comoving         = false                                  % SC mesh follows the bunch in z

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

    function beam = freezePlaneOut(obj, dump_step)
    % FREEZEPLANEOUT  Convert a captured time-based dump into a Lucretia
    % Beam struct ready for downstream s-based tracking via TrackThru.
    %
    %   B = tt.freezePlaneOut('last')          % use the last dump
    %   B = tt.freezePlaneOut(step_id)         % use a specific dump step
    %
    % Returns a Lucretia-format Beam:
    %   B.BunchInterval = 0
    %   B.Bunch.x = 6 x N matrix:
    %       row 1: x   (m)
    %       row 2: x' = px / pz   (rad, small-angle)
    %       row 3: y   (m)
    %       row 4: y' = py / pz
    %       row 5: z   (m, relative to bunch z-centroid)
    %       row 6: P   (GeV/c, total momentum |p|*c in GeV)
    %   B.Bunch.Q = 1 x N: physical charge per macroparticle (C)
    %   B.Bunch.stop = 1 x N: zeros (no particles dead initially)
    %
    % Note: rows 1..5 are taken at the dump TIME (not at a specific s).
    % If the bunch is short relative to its transit length scale (so
    % all particles are near the same z when the dump fires) this is a
    % good approximation to the s-based capture that a true z-triggered
    % freeze plane would produce. The TIMER-style readout (BeamMonitor +
    % freezePlaneOut(step)) is the simplest path -- a true z-triggered
    % FreezePlane element with per-particle drift-back is a future
    % refinement.
        bunches = obj.readDumps();
        if nargin < 2 || (ischar(dump_step) || isstring(dump_step))
            % default 'last'
            target = bunches{end};
        else
            steps  = cellfun(@(b) b.step, bunches);
            idx    = find(steps == dump_step, 1);
            if isempty(idx)
                error('TimeTrack:freezePlaneOut:noStep', ...
                      'No dump at step %d (have %s)', ...
                      dump_step, mat2str(steps));
            end
            target = bunches{idx};
        end

        % Constants
        me_kg = 9.1093837015e-31;
        c     = 299792458.0;
        % m_e * c^2 in GeV
        me_GeV_c2 = me_kg * c^2 / 1.602176634e-19 / 1e9;   % ~ 0.000510999 GeV

        N = numel(target.x);
        if N == 0
            error('TimeTrack:freezePlaneOut:emptyBunch', ...
                  'Dump at step %d has no particles', target.step);
        end

        px = double(target.px(:));         % momentum, kg.m/s
        py = double(target.py(:));
        pz = double(target.pz(:));
        xx = double(target.x(:));
        yy = double(target.y(:));
        zz = double(target.z(:));

        % Lucretia row 6 = total momentum P in GeV/c.
        % |p| in kg.m/s -> GeV/c via |p|*c / 1.602e-19 / 1e9.
        p2  = px.*px + py.*py + pz.*pz;
        Pgev = sqrt(p2) * c / 1.602176634e-19 / 1e9;

        % Slopes (small-angle)
        xp = px ./ max(pz, 1e-30);
        yp = py ./ max(pz, 1e-30);

        % Centred z (longitudinal position relative to bunch centroid)
        z_rel = zz - mean(zz);

        bunch       = struct();
        bunch.x     = [xx.'; xp.'; yy.'; yp.'; z_rel.'; Pgev.'];
        bunch.Q     = abs(double(target.q(:)).' .* double(target.w(:)).');  % C
        bunch.stop  = zeros(1, N);

        beam.BunchInterval = 0;
        beam.Bunch         = bunch;

        amrex_unused = me_GeV_c2;       %#ok<NASGU>  % kept for unit reference
    end
end
end
