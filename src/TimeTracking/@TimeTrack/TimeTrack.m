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
    dt_change_z  = []                                            % m, lab z at which to switch dt when bunch centroid crosses (mirrors ImpactT type-(-4) element)
    dt_after     = []                                            % s, dt to use after dt_change_t / dt_change_z (empty = same as dt)
    t_start      = []                                            % s, initial sim time (empty = 0)
    behind_cathode_z        = []                                 % m, ImpactT-style behind-cathode drift turns ON below this z
    behind_cathode_betazini = []                                 % dimensionless, universal v/c for behind-cathode drift
    use_centroid_phase      = []                                 % logical: use bunch z-centroid/c instead of wallclock t for cos argument (off by default; ImpactT actually uses wallclock)
    centroid_t_offset       = []                                 % s, additive offset for centroid mode
    use_midstep_field       = []                                 % logical: gather external field at midstep particle position (subsumed by use_dkd_integrator when set)
    use_particle_phase      = []                                 % logical: per-particle phase t_eff = z_particle/c (test mode for chirp investigation)
    use_dkd_integrator      = true                               % logical: ImpactT-style drift-kick-drift + first-order emission (default ON; cleanest cross-code calibration)
    disable_self_force      = []                                 % logical (diagnostic): skip the SC self-force LUT subtraction
    sc_use_b_field          = []                                 % logical: apply SC B field explicitly via Boris (matches ImpactT). Captures non-synchronous v×B coupling. Default off (1/gamma^2 boost shortcut).
    self_force_direct       = []                                 % logical: compute self-force directly each step (no LUT). Costs ~7x per particle but eliminates LUT-rebuild noise with adaptive mesh.
    n_emission_steps        = 0                                  % ImpactT Nemission: # fine steps during cathode emission (0=off). When >0, uses 3-phase dt schedule.
    t_emission              = 0                                  % ImpactT Temission: emission window duration (s). dt_emission = t_emission / n_emission_steps.
    particle_trace_ids      = []                                 % global particle indices to trace (write pos+mom every particle_trace_interval steps)
    particle_trace_interval = 10                                 % write every N steps (default 10)
    particle_trace_path     = '/tmp/lt_particle_trace.bin'       % trajectory trace output path
    field_audit_ids         = []                                 % task #19 audit: 0-based partcl.data row indices to dump (id, x,y,z, ux,uy,uz, Ex,Ey,Ez, Bx,By,Bz) as CSV
    field_audit_interval    = 0                                  % write every N steps (0 = off)
    field_audit_file        = '/tmp/lt_field_audit.csv'          % CSV output path for the field audit
    dump_kicks_at_steps     = []                                 % vector of int step indices at which to dump per-particle SC kicks (DKD path only). Files at /tmp/lt_part_kicks_step<N>.bin.
    dump_kicks_at_times     = []                                 % vector of double simulation times (s) at which to dump per-particle SC kicks (each fires at first step with t>=target). Useful for cross-code comparison when dt schedules differ.
    dump_kicks_path_prefix  = ''                                 % prefix for the dump files; empty -> /tmp/lt_part_kicks_step
    trace_file              = ''                                 % path: per-step bunch-mean diagnostics (CSV) for cross-code comparison
    trace_every             = 1                                  % step interval for trace rows (1 = every step)
    geom_lo    = [-0.05, -0.05,  0.0]                            % m
    geom_hi    = [ 0.05,  0.05,  1.0]                            % m
    geom_ncell = [32, 32, 32]
    output_dir = 'diags'                                         % relative to work_dir
    work_dir   = ''                                              % '' -> tempname
    binary     = ''                                              % '' -> which('lucretia-tt')
    enable_space_charge = false                                  % toggles ParmParse space_charge.enabled
    sc_comoving         = false                                  % SC mesh follows the bunch in z
    sc_adaptive         = false                                  % SC mesh resizes EVERY axis to track current bunch sigmas (supersedes sc_comoving)
    sc_exact_range      = false                                  % SC mesh resized to EXACT alive-particle min/max each step (zero padding, ImpactT-style; supersedes sc_adaptive)
    sc_outlier_kill_sigma  = 0                                   % when sc_exact_range=true: KILL particles whose offset from bunch centroid > N*sigma in any axis (mark alive=0; persistent). Effective cap = max(N*sigma, abs_floor). Mitigates 1-2 outlier particles dragging the mesh wider and over-coarsening cells, without leaving outliers half-gathered. 0 = no kill. ~4-6 recommended.
    sc_outlier_kill_floor_xy = 5e-3                              % m: absolute floor on the kill cap in x,y (avoids cascade kill at near-zero sigma during early emission)
    sc_outlier_kill_floor_z  = 1e-2                              % m: absolute floor on the kill cap in z
    sc_outlier_kill_verbose = false                              % when sc_outlier_kill_sigma>0: print kill counts per solve to stdout
    sc_hybrid_z_adaptive = false                                 % SC mesh: STATIC xy + ADAPTIVE z (resize z each step to track sigma_z, keep xy frozen at initial RealBox)
    sc_pad_factor       = 5.0                                    % adaptive: half-extent = max(pad_factor * sigma, min_pad)
    sc_resize_hyst      = 2.0                                    % adaptive: resize triggers when new pad > hyst*current OR < (1/hyst)*current. Larger = fewer resizes (less noise but coarser tracking).
    sc_cent_drift_threshold = 0.5                                % adaptive: resize when |centroid_drift| > frac*half_extent on ANY axis. Default 0.5 fires every ~3 steps for relativistic bunch (z drift dominates). Raise to 0.9-0.95 to reduce field-reset noise.
    sc_integer_cell_shift   = false                              % adaptive: when only the centroid trigger fires, snap the new box center to (cur_center + N*dx) so particles' fractional cell positions are preserved -> zero discretization noise from that resize.
    sc_min_pad_xy       = 1e-3                                   % adaptive: min transverse half-extent (m)
    sc_min_pad_z        = 1e-3                                   % adaptive: min longitudinal half-extent (m)
    sc_image_plane      = false                                  % cathode image-charge handling: deposit mirror charges on the SC mesh so the IGF Poisson satisfies phi=0 on the cathode plane
    sc_image_plane_z    = 0.0                                    % m: lab-frame cathode plane (mirror axis)
    sc_image_cutoff     = 0.05                                   % m: image deposit only fires for particles within this distance of the cathode (matches IMPACT-T's Zimage)
    sc_rho_smooth_passes = 0                                     % # of binomial (1,2,1)/4 smoother passes applied to deposited rho before IGF solve (0 = off; 1-2 cuts CIC noise without losing physical signal)
    sc_green_cache_tol   = 0                                     % relative tolerance (e.g. 0.01 = 1%) for the IGF Green-function cache. 0 = exact match required (rebuild every step on gamma-stretched z mesh; ~30% of total runtime). >0 = skip setGreensFunction when cell_size and z_shift drift by less than this fraction. ~20-25% total runtime saving at tol=0.01.
    sc_shape_order      = 1                                      % particle deposit/gather shape: 1 = CIC (linear, 8-node), 2 = TSC (quadratic, 27-node, smoother field gradients at ~3x cost)
    sc_verbose          = 0                                      % space_charge.verbose: 0 = quiet, 1 = print recenter / per-step diagnostics
    sc_diag_resize_jump = 0                                      % space_charge.diag_resize_jump: 1 = print per-particle dE statistics on each mesh resize (diagnostic of field-discontinuity noise); 0 = off
    sc_dump_field_at_step = 0                                    % space_charge.dump_field_at_step: write rho/phi/Ex/Ey/Ez mesh values to a binary file at this solve count (default 0 = off; cross-code SC field comparison harness)
    sc_dump_field_path  = ''                                     % path for sc_dump_field; empty -> /tmp/sc_field_dump_step<N>.bin
    enable_slice_sc     = false                                  % toggles ParmParse space_charge.slice_enabled (1D longitudinal slice SC, mesh-free in z)
    slice_sc_n          = 256                                    % # of z-slices for the slice SC bunch density estimator
    slice_sc_radius_factor = 2.0                                 % bunch_radius (for the disk-stack formula) = factor * sigma_xy
    slice_sc_gamma_off  = []                                     % if set, slice SC disabled when bunch mean gamma >= this value (lets 3D mesh handle longitudinal at high energy where slice over-counts vs ImpactT)
    slice_sc_profile    = 0                                      % slice SC transverse profile: 0 = uniform disk (default), 1 = Gaussian disk (analytic 2D Gaussian sheet on-axis E_z, uses sigma_xy directly)
    target              = 'cpu'                                  % 'cpu' | 'gpu'  -- combined with mpi_nranks selects the binary
    mpi_nranks          = 1                                      % >1 -> mpirun -np N (requires MPI-enabled binary)
    mpi_binary          = ''                                     % deprecated (use target+mpi_nranks); '' -> autodetect
    mpirun_bin          = ''                                     % '' -> /opt/homebrew/bin/mpirun, /usr/local/bin/mpirun, or PATH
    mpirun_extra_args   = ''                                     % e.g. '--bind-to socket --map-by socket' for NUMA-aware placement
    verbose_run         = false                                  % stream lucretia-tt stdout to Command Window during the run

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
    %
    % Binary selection precedence:
    %   1. obj.binary (if set) -- explicit override.
    %   2. obj.mpi_binary (if mpi_nranks>1, deprecated) -- legacy override.
    %   3. (target, mpi_nranks) -> staged binary name next to TimeTrack.m:
    %        cpu  + 1  -> lucretia-tt
    %        cpu  + N  -> lucretia-tt_mpi
    %        gpu  + 1  -> lucretia-tt_gpu
    %        gpu  + N  -> lucretia-tt_gpu_mpi
    %   4. which('lucretia-tt') -- last-ditch path search (cpu+1 only).
        ttDir   = fileparts(fileparts(mfilename('fullpath')));
        nranks  = 1;
        if isprop(obj, 'mpi_nranks') && ~isempty(obj.mpi_nranks)
            nranks = max(1, round(double(obj.mpi_nranks)));
        end
        target = 'cpu';
        if isprop(obj, 'target') && ~isempty(obj.target)
            target = lower(strtrim(obj.target));
        end
        if ~ismember(target, {'cpu', 'gpu'})
            error('TimeTrack:badTarget', ...
                  'tt.target must be ''cpu'' or ''gpu'' (got ''%s'')', target);
        end

        if isempty(obj.binary)
            % (2) legacy mpi_binary override (only applies when mpi_nranks>1)
            if nranks > 1 && isprop(obj, 'mpi_binary') && ~isempty(obj.mpi_binary) ...
               && exist(obj.mpi_binary, 'file') == 2
                obj.binary = obj.mpi_binary;
            else
                % (3) systematic name based on (target, nranks)
                binName = pick_binary_name(target, nranks);
                cand = fullfile(ttDir, binName);
                if exist(cand, 'file') == 2
                    obj.binary = cand;
                elseif strcmp(binName, 'lucretia-tt')
                    % (4) last-ditch: PATH search for the cpu+1 binary
                    obj.binary = which('lucretia-tt');
                end
                if isempty(obj.binary) || exist(obj.binary, 'file') ~= 2
                    build_arg = pick_build_arg(target, nranks);
                    error('TimeTrack:noBinary', ...
                        ['Binary not found: %s\n' ...
                         '  expected at: %s\n' ...
                         '  build with:  build_tt %s\n' ...
                         '  or set tt.binary explicitly.'], ...
                         binName, cand, build_arg);
                end
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
    %   B.Bunch.stop = 1 x N: 0 = alive, 1 = killed upstream (collimator
    %                  etc.). Dead particles remain in the matrix at their
    %                  park location (z_rel = -1e9) so particle-ID
    %                  alignment is preserved; downstream code filters
    %                  via stop == 0. lt's on-disk signal for "dead" is
    %                  z < -1e6 (Collimator.cpp parks at z = -1e9; the
    %                  in-memory IntSoA::alive flag is not dumped).
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

        % Identify dead/collimated particles via lt's park sentinel
        % (Collimator.cpp parks killed particles at z = -1e9; the
        % in-memory IntSoA::alive flag is not written to dumps).
        dead = zz < -1.0e6;
        live = ~dead;

        % Lucretia row 6 = total momentum P in GeV/c.
        % |p| in kg.m/s -> GeV/c via |p|*c / 1.602e-19 / 1e9.
        p2  = px.*px + py.*py + pz.*pz;
        Pgev = sqrt(p2) * c / 1.602176634e-19 / 1e9;

        % Slopes (small-angle). Skip dead particles -- their pz is stale
        % from before the kill and dividing by it injects garbage into
        % rows 2/4 that would corrupt downstream beamFit / emittance code
        % that doesn't gate on stop.
        xp = zeros(N, 1);
        yp = zeros(N, 1);
        xp(live) = px(live) ./ max(pz(live), 1e-30);
        yp(live) = py(live) ./ max(pz(live), 1e-30);

        % Centred z (relative to ALIVE-particle centroid). Including the
        % parked-dead particles in mean(zz) shifts the centroid by ~1e9
        % per dead particle and corrupts z_rel for the live bunch.
        if any(live)
            z_centroid = mean(zz(live));
        else
            z_centroid = 0;
        end
        z_rel       = zz - z_centroid;
        z_rel(dead) = -1.0e9;          % keep park sentinel visible to downstream

        bunch       = struct();
        bunch.x     = [xx.'; xp.'; yy.'; yp.'; z_rel.'; Pgev.'];
        bunch.Q     = abs(double(target.q(:)).' .* double(target.w(:)).');  % C
        % Lucretia convention: stop == 0 alive, > 0 killed. lt dumps don't
        % carry the killing-element index, so use 1 as a generic marker
        % (matches CheckBeamMomenta.m).
        bunch.stop       = zeros(1, N);
        bunch.stop(dead) = 1;

        beam.BunchInterval = 0;
        beam.Bunch         = bunch;

        amrex_unused = me_GeV_c2;       %#ok<NASGU>  % kept for unit reference
    end
end
end
