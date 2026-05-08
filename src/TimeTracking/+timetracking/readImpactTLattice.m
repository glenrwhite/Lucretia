function [lattice, beam, geom, tracking, info] = readImpactTLattice(impactt_dir, varargin)
% READIMPACTTLATTICE  Parse an ImpactT.in file into a lucretia-tt setup.
%
%   [lattice, beam, geom, tracking, info] = readImpactTLattice(dir)
%   [lattice, beam, geom, tracking, info] = readImpactTLattice(dir, 'n_macros', 5000)
%
% Returns:
%   lattice  : cell array of element specs (compatible with TimeTrack.lattice)
%   beam     : struct usable as TimeTrack.beam (auto-emits via CathodeSource
%              built into lattice; SeedBeam fallback with n_particles=0)
%   geom     : struct .lo .hi .ncell suggesting a SC mesh sized to the lattice
%   tracking : struct .dt .n_steps copied from the ImpactT header
%   info     : struct with parsed values + warnings about unsupported elements
%
% Recognises ImpactT element types:
%   105  -> ImpactTField (RF cavity or DC solenoid based on scale_E/scale_B)
%   1    -> QuadAnalytic (skipped silently if gradient = 0)
%   -2/-4/-5/-6/-11/-99  -> control elements, skipped with a note in info.warnings

p = inputParser;
p.addParameter('n_macros', 2000, @(x) isnumeric(x) && isscalar(x));
p.addParameter('mte_eV',   0.5,  @isnumeric);
p.parse(varargin{:});
opts = p.Results;

in_path = fullfile(impactt_dir, 'ImpactT.in');
fid = fopen(in_path, 'r');
if fid < 0, error('readImpactTLattice:noFile', 'Cannot open %s', in_path); end
cleanup = onCleanup(@() fclose(fid));

% --- Read header (first 9 non-comment, non-empty lines) ---
header_vals = cell(9, 1);
header_idx = 0;
in_lattice = false;
lat_lines = {};
while true
    tline = fgetl(fid);
    if ~ischar(tline), break; end
    s = strtrim(tline);
    if isempty(s), continue; end
    if startsWith(s, '!')
        if contains(lower(s), 'lattice'), in_lattice = true; end
        continue;
    end
    % strip trailing comment after /!
    pos = strfind(s, '/!');
    if ~isempty(pos), s = strtrim(s(1:pos(1)-1)); end

    if ~in_lattice && header_idx < 9
        header_idx = header_idx + 1;
        header_vals{header_idx} = sscanf(s, '%f');
    elseif in_lattice
        % grab name from the original line if present
        nm = '';
        name_pos = strfind(tline, 'name:');
        if ~isempty(name_pos)
            nm = strtrim(tline(name_pos(1)+5:end));
        end
        lat_lines{end+1} = struct('vals', sscanf(s, '%f'), 'name', nm); %#ok<AGROW>
    end
end

if header_idx < 9
    error('readImpactTLattice:header', ...
          'Expected 9 header lines in %s, only found %d', in_path, header_idx);
end

% Header lines (per ImpactT manual order):
%  1: Npcol Nprow
%  2: Dt Ntstep Nbunch
%  3: Dim Np Flagmap Flagerr Flagdiag Flagimg Zimage
%  4: Nx Ny Nz Flagbc Xrad Yrad Perdlen
%  5: Flagdist Rstartflg Flagsbstp Nemission Temission
%  6: sigx sigpx muxpx xscale pxscale xmu1 xmu2
%  7: sigy sigpy muxpy yscale pyscale ymu1 ymu2
%  8: sigz sigpz muxpz zscale pzscale zmu1 zmu2
%  9: Bcurr Bkenergy Bmass Bcharge Bfreq Tini

h2 = header_vals{2};   dt0 = h2(1);  n_steps0 = round(h2(2));
h4 = header_vals{4};   ncellH = h4(1:3)';  xrad = h4(5);  yrad = h4(6);
h5 = header_vals{5};   t_em_full = h5(5);  flagdist = h5(1);  Nemission = round(h5(4));

% ImpactT semantics: Temission (h5(5)) is the upper bound on the
% emission window; the actual emission rate has Nemission particles
% per ImpactT timestep at the initial dt = h2(1). Effective window is
% min(Temission, Nemission * dt_initial). Verified empirically (Phase
% 4D-followup): the narrow Nemission*dt_initial window gives a peak
% gamma matching the IT plateau to 2% on the LCLS gun, while the wider
% Temission window underestimates by ~25%.
t_em_narrow = Nemission * h2(1);
if t_em_narrow > 0 && t_em_narrow < t_em_full
    t_em = t_em_narrow;
else
    t_em = t_em_full;
end
h6 = header_vals{6};   sigx = h6(1);
h7 = header_vals{7};   sigy = h7(1);
h8 = header_vals{8};   sigz = h8(1);  %#ok<NASGU>
h9 = header_vals{9};   Bcurr = h9(1);  Bfreq = h9(5);  Tini = 0;
if numel(h9) >= 6, Tini = h9(6); end

% NOTE on time references: ImpactT's "Tini" header (h9(6), typically
% negative -- e.g. -13.06 ps for LCLS) only sets when the FIRST particle
% is emitted relative to the simulation t=0; it does NOT shift the RF
% reference. The field cos(omega t + phase) uses the same t=0 in both
% codes, so phase_imp passes through to lucretia-tt unchanged. Verified
% empirically by phase_scan.m (peak exit gamma matches phase_imp =
% 304.668 deg to within ~1 deg, the scan resolution).

% Total bunch charge per ImpactT convention: Q = Bcurr / Bfreq (where
% Bfreq is the bunch repetition / scaling frequency, typically the RF
% frequency). For LCLS (Bcurr = 2.91 A, Bfreq = 2.856 GHz) this gives
% Q = 1.02 nC. The earlier "Bcurr * Temission" formula (~100 pC) was
% wrong -- it interpreted Bcurr as the average emission-current.
total_charge = abs(Bcurr) / max(Bfreq, eps);   % C
pulse_shape  = 'flat_top';          % flagdist 16 = uniform-in-t
if flagdist ~= 16
    pulse_shape = 'gaussian';       % fallback
end

% --- Parse lattice elements ---
warnings = {};
lattice = {};
elem_z_max = 0;          % furthest downstream lab z (for geom suggestion)
gun_z_edge = NaN;
gun_field_zmin = NaN;
gun_path = '';
% Variable-dt schedule extracted from any type-(-4) "change_timestep"
% control elements. ImpactT specifies these as z-based switches; for
% lucretia-tt (which uses t-based dt switching) we estimate the time
% the bunch centroid reaches each z and translate.
dt_schedule = struct('z', {}, 'dt', {});
% End-of-run z plane extracted from a type-(-99) STOP element. Used to
% size n_steps so the bunch reaches the ImpactT exit plane (rather than
% running the lucretia-tt-default 18000 steps blindly).
z_stop      = NaN;

for k = 1:numel(lat_lines)
    v = lat_lines{k}.vals;
    nm = lat_lines{k}.name;
    if isempty(nm), nm = sprintf('e%d', k); end
    if numel(v) < 4, continue; end
    L = v(1);  Bnpstp = round(v(4));

    switch Bnpstp
    case 105
        % SolRF: solenoid (B), RF cavity (E), or both, via rfdata file.
        % ImpactT manual: "V1: zedge, the real used field range in z is
        % [zedge, zedge+Blength]." So:
        %   v(1) = Blength (lattice-line length used for field gating)
        %   v(5) = zedge   (lab z corresponding to file z = 0 AND start
        %                   of the "real used" range)
        %
        % scale_B position depends on how many misalignment slots the
        % ImpactT line carries before the trailing B-field scale:
        %   RF cavity:   6 misal + optional scale_B at v(17) (often 0.0)
        %   Solenoid:    5 misal + scale_B at v(end), e.g. the LCLS-style
        %                sprintf("... 105 z 0.0 0.0 0.0 102 0.15 mx my mzx mzy mzz scale_B")
        z_edge   = v(5);
        scale_E  = v(6);
        f_RF     = v(7);
        phase_dg = v(8);
        file_id  = round(v(9));
        scale_B  = 0.0;
        if scale_E == 0 && numel(v) >= 11
            % Pure solenoid: last token is the field scale (Tesla scale).
            scale_B = v(end);
        elseif numel(v) >= 17
            % RF cavity with explicit B scale (rare; usually 0).
            scale_B = v(17);
        end
        path = fullfile(impactt_dir, sprintf('rfdata%d', file_id));
        lattice{end+1} = timetracking.ImpactTField('name', nm, ...
            'z', z_edge, 'length', L, 'path', path, ...
            'scale_E', scale_E, 'scale_B', scale_B, ...
            'f_RF', f_RF, 'phase_deg', phase_dg);  %#ok<AGROW>
        % Track first gun-like element (RF, scale_E > 0)
        if scale_E > 0 && (isnan(gun_z_edge) || z_edge < gun_z_edge)
            gun_z_edge = z_edge;
            gun_path   = path;
        end
        elem_z_max = max(elem_z_max, z_edge + L);
    case 1
        % Quadrupole
        z_edge   = v(5);
        gradient = v(6);                      % T/m (best-guess from V2 slot)
        if gradient ~= 0
            lattice{end+1} = timetracking.QuadAnalytic('name', nm, ...
                'z', z_edge, 'length', L, 'gradient', gradient); %#ok<AGROW>
        end
        elem_z_max = max(elem_z_max, z_edge + L);
    case -4
        % Change-timestep control element. ImpactT format:
        %   "0 0 0 -4 0.0 0.0 z_switch new_dt /!name:..."
        % i.e. v(7)=z_switch (m), v(8)=new_dt (s).
        if numel(v) >= 8
            dt_schedule(end+1).z  = v(7); %#ok<AGROW>
            dt_schedule(end).dt   = v(8);
        else
            warnings{end+1} = sprintf( ...
                'skipping %s (type -4 with too few fields)', nm); %#ok<AGROW>
        end
    case -99
        % End-of-run plane: ImpactT terminates tracking when the bunch
        % reaches z_stop. lucretia-tt has no z-trigger yet (run length
        % is set by n_steps), so we extract z_stop here and use it
        % below to compute n_steps that gets the bunch to that plane.
        if numel(v) >= 5
            z_stop_now = v(5);
            % Take the latest (largest z) STOP element if multiple are
            % present in the lattice -- matches ImpactT's "first stop wins"
            % semantics if the user only intends one, and falls back
            % gracefully for files with multiple commented-in stops.
            if isnan(z_stop) || z_stop_now > z_stop
                z_stop = z_stop_now;
            end
        else
            warnings{end+1} = sprintf( ...
                'skipping %s (type -99 with too few fields)', nm); %#ok<AGROW>
        end
    case -11
        % Collimator / backward-particle filter. ImpactT format:
        %   "0 0 0 -11 v0 z_trigger xmin xmax ymin ymax /!name:..."
        % per IMPACT-T's AccSimulator.f90 parsing (line 1053-1058):
        %   v(6)=z_trigger -> tcol  (the z position at which the
        %                            collimator fires ONCE, like a thin
        %                            physical aperture at that z)
        %   v(7..10)       = xmin, xmax, ymin, ymax aperture
        % Imp's trigger: when bunch centroid crosses tcol, lostXY checks
        % ALL particles' (x,y) and kills those outside the aperture. NOT
        % a continuous z-range cut. v(5) appears unused for the active
        % collimator; for stop_bkw type elements v(6) is set negative as
        % a flag (per LCLS deck convention) and stop_bkw kills backward-
        % traveling particles. Older lt parser treated v(5)=z_start +
        % v(6)=dz as a continuous z-range aperture, which mass-killed
        % particles through the entire 1.5-m drift -- bug fixed here.
        if numel(v) >= 10
            z_trigger = v(6);
            xmin      = v(7);
            xmax      = v(8);
            ymin      = v(9);
            ymax      = v(10);
            if z_trigger < 0
                % stop_bkw: kill backward particles in ALL z. No aperture.
                lattice{end+1} = timetracking.Collimator('name', nm, ...
                    'z_start',       -1, ...
                    'z_end',         elem_z_max + 100, ...
                    'kill_backward', true); %#ok<AGROW>
            else
                % Thin aperture at z=z_trigger. Fire ONCE when the bunch
                % centroid first crosses z_trigger; kill ALL alive
                % particles outside the aperture regardless of their own
                % z. Matches IMPACT-T's lostXY trigger semantics
                % exactly (AccSimulator.f90:1669 + BeamBunch.f90:1304-1314).
                lattice{end+1} = timetracking.Collimator('name', nm, ...
                    'fire_once', true, ...
                    'z_target',  z_trigger, ...
                    'xmin',    xmin,    'xmax',  xmax, ...
                    'ymin',    ymin,    'ymax',  ymax); %#ok<AGROW>
            end
        else
            warnings{end+1} = sprintf( ...
                'skipping %s (type -11 with too few fields)', nm); %#ok<AGROW>
        end
    case -6
        % Wakefield element. ImpactT format:
        %   "0 -1 0 -6 long_on trans_on z_start z_end iris_a gap_g period_L /!name:..."
        % i.e. v(5)=long_on, v(6)=trans_on, v(7..8)=z_start/z_end,
        % v(9..11)=iris_a/gap_g/period_L (all m).
        if numel(v) >= 11
            lattice{end+1} = timetracking.WakeField('name', nm, ...
                'z_start',      v(7), ...
                'z_end',        v(8), ...
                'iris_a',       v(9), ...
                'gap_g',        v(10), ...
                'period_L',     v(11), ...
                'longitudinal', v(5) ~= 0, ...
                'transverse',   v(6) ~= 0); %#ok<AGROW>
        else
            warnings{end+1} = sprintf( ...
                'skipping %s (type -6 wakefield with too few fields)', nm); %#ok<AGROW>
        end
    case {-2, -5}
        warnings{end+1} = sprintf('skipping %s (type %d, control element)', ...
                                  nm, Bnpstp); %#ok<AGROW>
    otherwise
        warnings{end+1} = sprintf('skipping %s (type %d, unsupported)', ...
                                  nm, Bnpstp); %#ok<AGROW>
    end
end

% --- Cathode-at-z=0 normalization ---
%
% By convention (matching ImpactT's image-charge model: "the cathode is
% always assumed to be at z = 0"), we translate the entire lattice so
% the cathode (= first RF cavity's zedge) lands at lucretia-tt z = 0.
% All downstream zedge/z_start/z_end/dt-switch/z_stop values are shifted
% by -gun_z_edge. For a properly-written ImpactT.in where the gun is
% already at zedge=0 (e.g. the LCLS lattice), this is a no-op.
%
% Rationale:
%   - Cathode at z=0 means image-charge math is unambiguous.
%   - Element zedge values become "distance downstream of cathode," which
%     is the natural mental model for photoinjector layouts.
%   - Removes a class of subtle bugs around what z=0 means.
%
% The original ImpactT-frame cathode position is recorded in
% info.z_shift so users can map results back to ImpactT coordinates if
% needed (z_impactT = z_lucretia + info.z_shift).
if isnan(gun_z_edge)
    warnings{end+1} = 'no gun found; cathode placed at z=0';
    z_shift   = 0.0;
    cathode_z = 0.0;
else
    z_shift = gun_z_edge;     % positive = ImpactT placed cathode this far downstream
    cathode_z = 0.0;
    try
        ffid = fopen(gun_path, 'r');
        h = textscan(ffid, '%f', 4);
        fclose(ffid);
        gun_field_zmin = h{1}(2);            % kept for info struct
    catch
        gun_field_zmin = NaN;
        warnings{end+1} = sprintf('cannot read gun rfdata %s for diagnostic zmin', gun_path);
    end
end

% Apply the shift to every element's z-bearing field (in-place on the
% parsed lattice cell array).
if z_shift ~= 0
    for k = 1:numel(lattice)
        e = lattice{k};
        if isfield(e, 'z'),       e.z       = e.z       - z_shift; end
        if isfield(e, 'z_start'), e.z_start = e.z_start - z_shift; end
        if isfield(e, 'z_end'),   e.z_end   = e.z_end   - z_shift; end
        lattice{k} = e;
    end
    for k = 1:numel(dt_schedule)
        dt_schedule(k).z = dt_schedule(k).z - z_shift;
    end
    if ~isnan(z_stop),     z_stop     = z_stop     - z_shift; end
    if ~isnan(elem_z_max), elem_z_max = elem_z_max - z_shift; end
    gun_z_edge = 0.0;
    fprintf(['readImpactTLattice: cathode-at-z=0 normalization: shifted ', ...
             'all element z by %+.6f m (original ImpactT cathode at lab ', ...
             'z=%+.6f m).\n'], -z_shift, z_shift);
end

% Insert CathodeSource at the FRONT of the lattice. Emission window
% is centered on Tini (matches ImpactT's RF reference), so the wall-
% clock instant of each emission matches ImpactT, and the bunch sees
% the same RF phase history during transit. The simulation begins at
% Tini - half_emission_window (set below in tracking.t_start), so the
% LEADING half of the emission pulse is also captured -- otherwise we
% would emit only the trailing half (~50% loss).
cathode = timetracking.CathodeSource('name', 'cat', ...
    'z_cathode', cathode_z, 'image_charge', true, ...
    'n_macroparticles_total', opts.n_macros, ...
    'total_charge', total_charge, ...
    'pulse_shape', pulse_shape, ...
    'pulse_duration', t_em, 'pulse_t0', Tini, ...
    'transverse_profile', 'gaussian', ...
    'spot_size', max(sigx, sigy), 'mte', opts.mte_eV);

% BeamMonitor is appended AFTER tracking.n_steps is computed (below) so
% dump_every can be sized to the actual run length, not the lattice
% header's Ntstep (which is typically a huge upper bound -- e.g. 500000
% for LCLS, vs an actual ~10000 steps to reach z_stop). Sizing on the
% header value gave dump_every=5000 with only 9758 actual steps,
% producing a single dump at step 5000 (mid-injector, BEFORE L0A) and
% missing the post-L0A bunch entirely. Now we target ~100 dumps over
% the actual run.
lattice = [{cathode}, lattice];

% --- Append a global ±xrad/±yrad transverse aperture (mirrors IMPACT-T's
%     lostREC_BeamBunch which is called every step from AccSimulator.f90
%     line 1696 and kills any particle outside [-xrad, xrad] in x or
%     [-yrad, yrad] in y). Without this, lt keeps wide-angle halo particles
%     that imp continuously prunes -- causing per-step bunch divergence
%     in the gun + L0A regions even when SC physics matches. ---
if xrad > 0 && yrad > 0
    lattice{end+1} = timetracking.Collimator('name', 'global_xrad_aperture', ...
        'z_start',  -1, ...
        'z_end',    elem_z_max + 100, ...
        'xmin',    -xrad, 'xmax', xrad, ...
        'ymin',    -yrad, 'ymax', yrad);
end

% --- Beam (no seed; emission from cathode) ---
beam = timetracking.SeedBeam('n_particles', 0);

% --- Geometry suggestion ---
margin_xy = max(2 * max(xrad, yrad), 5e-3);
margin_z_low  = cathode_z - 5e-3;
margin_z_high = elem_z_max + 0.1;
geom.lo    = [-margin_xy, -margin_xy, margin_z_low];
geom.hi    = [ margin_xy,  margin_xy, margin_z_high];
geom.ncell = ncellH;       % from ImpactT header (Nx Ny Nz)

% --- Tracking suggestion ---
% Honour ImpactT's variable-dt schedule when present. ImpactT typically
% starts with a small dt (~1e-13 s) for the cathode/gun region and
% bumps to a larger dt (~4e-12 s) past the gun for speed. We translate
% z_switch -> t_switch using a half-c estimate for the cathode->gun-exit
% transit (avg beta ~ 0.5 from beta=0 at cathode to beta~0.997 just
% past gun), then add Tini for the wall-clock origin.
% Use the END-of-run z plane from the type-(-99) STOP element when
% present (matches ImpactT semantics); fall back to the furthest-
% downstream lattice element edge.
z_target = elem_z_max;
if ~isnan(z_stop)
    z_target = z_stop;
end

if isempty(dt_schedule)
    tracking.dt          = max(dt0, 0.5e-12);
    % Steps to reach z_target at average c (relativistic post-cathode).
    c_light              = 299792458.0;
    n_to_target          = ceil((z_target / (0.5 * c_light)) / tracking.dt);
    tracking.n_steps     = max(1000, n_to_target + 200);
    tracking.dt_change   = [];
    tracking.dt_change_z = [];
    tracking.dt_after    = [];
else
    tracking.dt = dt0;
    if numel(dt_schedule) > 1
        warnings{end+1} = sprintf( ...
            'multiple dt-change schedules (%d); using only the first (z=%g, dt=%g)', ...
            numel(dt_schedule), dt_schedule(1).z, dt_schedule(1).dt);
    end
    z_switch  = dt_schedule(1).z;
    dt_after  = dt_schedule(1).dt;
    c_light   = 299792458.0;
    beta_avg  = 0.5;                          % cathode -> gun exit average
    t_to_z    = z_switch / (beta_avg * c_light);
    % Expose BOTH triggers; the caller picks one. dt_change_z fires when
    % the bunch z-centroid passes z_switch (mirrors ImpactT exactly);
    % dt_change is the rough wall-clock equivalent for backward-compat.
    tracking.dt_change   = Tini + t_to_z;
    tracking.dt_change_z = z_switch;
    tracking.dt_after    = dt_after;
    % n_steps suggestion: enough small-dt steps to reach the switch,
    % plus enough big-dt steps to reach z_target at c after that.
    n_pre  = ceil(t_to_z / tracking.dt);
    z_post = max(z_target - z_switch, 0);
    n_post = ceil((z_post / c_light) / tracking.dt_after);
    tracking.n_steps = n_pre + n_post + 200;  % small slack
end
% Add the steps needed to cover the pre-emission window we just shifted
% t_start back by (so the bunch still reaches z_target at the end of
% the run). Computed once we know the pulse_shape below.
% Start tracking BEFORE the leading edge of the cathode emission pulse
% so all particles get emitted (otherwise the half of the pulse with
% t < Tini gets dropped because tracking has not started yet). Per-shape
% half-window:
%   gaussian       FWHM/2 + 2*sigma  ~  3*sigma   (covers ~99.7% of pulse)
%   super_gaussian 4*sigma                          (CathodeSource SG support)
%   flat_top       full_width / 2
% Then dt_change_t and n_steps below also shift to absorb the extra
% pre-emission time, so the same z_target is still reached.
switch lower(pulse_shape)
    case 'gaussian'
        emission_half_window = 1.5 * t_em;     % FWHM/2 + 2*sigma ~ 1.5*FWHM
    case 'super_gaussian'
        emission_half_window = 4.0 * t_em;     % CathodeSource SG support is ±4*sigma
    case 'flat_top'
        emission_half_window = 0.5 * t_em;
    otherwise
        emission_half_window = 0.5 * t_em;
end
tracking.t_start = Tini - emission_half_window;
% Bump n_steps to cover the pre-emission window with the SMALL dt
% (so the bunch still reaches z_target at the end of the run despite
% the earlier t_start).
tracking.n_steps = tracking.n_steps + ceil(emission_half_window / tracking.dt);

% Append BeamMonitor with dump_every sized to the actual run so we get
% ~100 dumps total (last dump close to the end of the run, capturing
% the post-L0A bunch at the exit plane).
mon = timetracking.BeamMonitor('name', 'mon', ...
                               'dump_every', max(1, round(tracking.n_steps / 100)));
lattice = [lattice, {mon}];

% --- Info / warnings ---
info.warnings    = warnings;
info.cathode_z   = cathode_z;     % always 0 after normalization
info.gun_z_edge  = gun_z_edge;    % always 0 after normalization
info.gun_path    = gun_path;
info.total_charge = total_charge;
info.t_emission   = t_em;
info.RF_freq      = Bfreq;
info.z_stop       = z_stop;       % NaN if no -99 element seen, in lucretia-tt frame
info.z_shift      = z_shift;      % z_impactT = z_lucretia + z_shift

if ~isempty(warnings)
    fprintf('readImpactTLattice: %d warnings:\n', numel(warnings));
    for i = 1:numel(warnings), fprintf('  - %s\n', warnings{i}); end
end
end
