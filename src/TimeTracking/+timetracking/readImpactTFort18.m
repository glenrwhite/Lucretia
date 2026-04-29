function ref = readImpactTFort18(impactt_dir)
% READIMPACTTFORT18  Read ImpactT diagnostic outputs (fort.18, fort.24-26).
%
%   ref = readImpactTFort18(dir)
%
% Returns a struct with arrays sampled per ImpactT diagnostic step:
%   .t       (s)        time
%   .z_mean  (m)        bunch centroid z
%   .gamma   (-)        average Lorentz factor
%   .sigma_x (m)        RMS x
%   .sigma_y (m)        RMS y
%   .sigma_z (m)        RMS z
%   .eps_nx  (m.rad)    normalised x emittance
%   .eps_ny  (m.rad)    normalised y emittance
%   .eps_nz  (m.rad)    normalised z emittance
%
% Column conventions follow ReadImpactTData.m's parseEData/parsePosData
% layout. Files must already be present (i.e. ImpactT has been run).

paths.f18 = fullfile(impactt_dir, 'fort.18');
paths.f24 = fullfile(impactt_dir, 'fort.24');
paths.f25 = fullfile(impactt_dir, 'fort.25');
paths.f26 = fullfile(impactt_dir, 'fort.26');

for fld = fieldnames(paths)'
    if ~isfile(paths.(fld{1}))
        error('readImpactTFort18:missing', '%s not found', paths.(fld{1}));
    end
end

% fort.18: 7 columns. col1=time, col2=z_mean, col3=gamma_avg
d = textscan(fopen(paths.f18, 'r'), '%f');  d = d{1};
n = numel(d) / 7;
ref.t      = d(1:7:end);
ref.z_mean = d(2:7:end);
ref.gamma  = d(3:7:end);

% fort.24: 8 columns. col4=RMS x, col8=emit_x
d = textscan(fopen(paths.f24, 'r'), '%f');  d = d{1};
ref.sigma_x = d(4:8:end);
ref.eps_nx  = d(8:8:end);

% fort.25: 8 columns. col4=RMS y, col8=emit_y
d = textscan(fopen(paths.f25, 'r'), '%f');  d = d{1};
ref.sigma_y = d(4:8:end);
ref.eps_ny  = d(8:8:end);

% fort.26: 7 columns. col3=RMS z, col7=emit_z
d = textscan(fopen(paths.f26, 'r'), '%f');  d = d{1};
ref.sigma_z = d(3:7:end);
ref.eps_nz  = d(7:7:end);

% close any stray fids
fids = openedFiles();
for f = fids(:).', fclose(f); end

% Sanity-trim arrays to common length
nmin = min([numel(ref.t), numel(ref.z_mean), numel(ref.gamma), ...
            numel(ref.sigma_x), numel(ref.sigma_y), numel(ref.sigma_z), ...
            numel(ref.eps_nx),  numel(ref.eps_ny),  numel(ref.eps_nz)]);
fnames = fieldnames(ref);
for k = 1:numel(fnames)
    ref.(fnames{k}) = ref.(fnames{k})(1:nmin);
end

fprintf('readImpactTFort18: loaded %d diagnostic samples from %s\n', nmin, impactt_dir);
fprintf('  t span:       %.2f ps -> %.2f ns\n', ref.t(1)*1e12, ref.t(end)*1e9);
fprintf('  z span:       %.4f mm -> %.2f m\n',  ref.z_mean(1)*1e3, ref.z_mean(end));
fprintf('  gamma span:   %.3f -> %.2f\n',       ref.gamma(1), ref.gamma(end));
fprintf('  eps_nx span:  %.3g -> %.3g m.rad\n', ref.eps_nx(1), ref.eps_nx(end));
end
