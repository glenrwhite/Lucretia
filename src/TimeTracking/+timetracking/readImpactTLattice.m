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
h5 = header_vals{5};   t_em = h5(5);  flagdist = h5(1);
h6 = header_vals{6};   sigx = h6(1);
h7 = header_vals{7};   sigy = h7(1);
h8 = header_vals{8};   sigz = h8(1);  %#ok<NASGU>
h9 = header_vals{9};   Bcurr = h9(1);  Bfreq = h9(5);  Tini = 0;
if numel(h9) >= 6, Tini = h9(6); end

% ImpactT time reference: simulation starts at t = Tini (header h9(6),
% typically negative -- e.g. -13 ps for LCLS). Our run starts at t = 0.
% To make the RF phase that any element sees identical at the matching
% wall-clock instant, we shift every RF element's phase by omega*Tini:
%   IT field at T_imp:  cos(omega T_imp + phase_imp)
%   LT field at LT_t:   cos(omega LT_t  + phase_LT)
%   With LT_t = T_imp - Tini -> phase_LT = phase_imp + omega*Tini
phase_shift_rad = 2*pi * Bfreq * Tini;
phase_shift_deg = phase_shift_rad * 180 / pi;

total_charge = abs(Bcurr) * t_em;   % C (current in A * emission time in s)
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

for k = 1:numel(lat_lines)
    v = lat_lines{k}.vals;
    nm = lat_lines{k}.name;
    if isempty(nm), nm = sprintf('e%d', k); end
    if numel(v) < 4, continue; end
    L = v(1);  Bnpstp = round(v(4));

    switch Bnpstp
    case 105
        % RF cavity / solenoid via rfdata file
        z_edge   = v(5);
        scale_E  = v(6);
        f_RF     = v(7);
        phase_dg = v(8);
        file_id  = round(v(9));
        scale_B  = 0.0;
        if numel(v) >= 17
            scale_B = v(17);
        end
        path = fullfile(impactt_dir, sprintf('rfdata%d', file_id));
        % Shift phase to compensate for ImpactT's Tini origin.
        phase_shifted = phase_dg + phase_shift_deg;
        lattice{end+1} = timetracking.ImpactTField('name', nm, ...
            'z', z_edge, 'path', path, ...
            'scale_E', scale_E, 'scale_B', scale_B, ...
            'f_RF', f_RF, 'phase_deg', phase_shifted);  %#ok<AGROW>
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
    case {-2, -4, -5, -6, -11, -99}
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

% Insert CathodeSource at the FRONT of the lattice
cathode = timetracking.CathodeSource('name', 'cat', ...
    'z_cathode', cathode_z, 'image_charge', true, ...
    'n_macroparticles_total', opts.n_macros, ...
    'total_charge', total_charge, ...
    'pulse_shape', pulse_shape, ...
    'pulse_duration', t_em, 'pulse_t0', 0.0, ...
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
tracking.dt      = max(dt0, 0.5e-12);   % bump up if ImpactT used very small dt
tracking.n_steps = max(1000, round(n_steps0 / 100));   % scale down

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
