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

for k = 1:numel(lat_lines)
    v = lat_lines{k}.vals;
    nm = lat_lines{k}.name;
    if isempty(nm), nm = sprintf('e%d', k); end
    if numel(v) < 4, continue; end
    L = v(1);  Bnpstp = round(v(4));

    switch Bnpstp
    case 105
        % RF cavity / solenoid via rfdata file.
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
            'z', z_edge, 'path', path, ...
            'scale_E', scale_E, 'scale_B', scale_B, ...
            'f_RF', f_RF, 'phase_deg', phase_dg);  %#ok<AGROW>
        % Track first gun-like element (RF, scale_E > 0)
        if scale_E > 0 && (isnan(gun_z_edge) || z_edge < gun_z_edge)
            gun_z_edge = z_edge;
            gun_path   = path;
        end
        elem_z_max = max(elem_z_max, z_edge + max(L, 1.5));  % field map can extend
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
    case {-2, -5, -6, -11, -99}
        warnings{end+1} = sprintf('skipping %s (type %d, control element)', ...
                                  nm, Bnpstp); %#ok<AGROW>
    otherwise
        warnings{end+1} = sprintf('skipping %s (type %d, unsupported)', ...
                                  nm, Bnpstp); %#ok<AGROW>
    end
end

% --- Build cathode source from beam params + first gun ---
if isnan(gun_z_edge)
    warnings{end+1} = 'no gun found; cathode placed at z=0';
    cathode_z = 0.0;
else
    % Read first 4 lines of gun rfdata to get zmin
    try
        ffid = fopen(gun_path, 'r');
        h = textscan(ffid, '%f', 4);
        fclose(ffid);
        gun_field_zmin = h{1}(2);
        cathode_z = gun_z_edge + gun_field_zmin;
    catch
        warnings{end+1} = sprintf('cannot read gun rfdata %s; cathode at z=0', gun_path);
        cathode_z = 0.0;
    end
end

% Insert CathodeSource at the FRONT of the lattice. Emission window
% starts at Tini in lab time so that the wall-clock instant at which
% each particle is emitted matches ImpactT, and the bunch sees the same
% RF phase history during transit. tracking.t_start (set below) makes
% the simulation actually begin at Tini so that t = pulse_t0 is the
% first emission step.
cathode = timetracking.CathodeSource('name', 'cat', ...
    'z_cathode', cathode_z, 'image_charge', true, ...
    'n_macroparticles_total', opts.n_macros, ...
    'total_charge', total_charge, ...
    'pulse_shape', pulse_shape, ...
    'pulse_duration', t_em, 'pulse_t0', Tini, ...
    'transverse_profile', 'gaussian', ...
    'spot_size', max(sigx, sigy), 'mte', opts.mte_eV);

% Always add a beam monitor at the end, dump_every chosen later in tracking
mon = timetracking.BeamMonitor('name', 'mon', 'dump_every', max(1, round(n_steps0/100)));

lattice = [{cathode}, lattice, {mon}];

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
if isempty(dt_schedule)
    tracking.dt        = max(dt0, 0.5e-12);
    tracking.n_steps   = max(1000, round(n_steps0 / 100));
    tracking.dt_change = [];
    tracking.dt_after  = [];
else
    % Use ImpactT's initial dt (much smaller than the ImpactT n_steps
    % we'd back out, so use the actual dt0).
    tracking.dt = dt0;
    % Take the first switch (most common ImpactT use); if multiple are
    % present we warn and use the first.
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
    tracking.dt_change = Tini + t_to_z;
    tracking.dt_after  = dt_after;
    % n_steps suggestion: enough small-dt steps to reach the switch,
    % plus enough big-dt steps to traverse the rest of the lattice at c.
    n_pre  = ceil(t_to_z / tracking.dt);
    z_post = max(elem_z_max - z_switch, 0);
    n_post = ceil((z_post / c_light) / tracking.dt_after);
    tracking.n_steps = n_pre + n_post + 200;  % small slack
end
tracking.t_start = Tini;     % match ImpactT's wall-clock start

% --- Info / warnings ---
info.warnings    = warnings;
info.cathode_z   = cathode_z;
info.gun_z_edge  = gun_z_edge;
info.gun_path    = gun_path;
info.total_charge = total_charge;
info.t_emission   = t_em;
info.RF_freq      = Bfreq;

if ~isempty(warnings)
    fprintf('readImpactTLattice: %d warnings:\n', numel(warnings));
    for i = 1:numel(warnings), fprintf('  - %s\n', warnings{i}); end
end
end
