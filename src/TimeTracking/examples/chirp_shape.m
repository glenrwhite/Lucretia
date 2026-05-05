function chirp_shape(work_dir, impactt_dir)
% Bin particles by z; report mean gamma per bin. Reveals chirp shape
% (linear vs S-curve) that summary statistics hide.
if nargin < 1, work_dir = '/tmp/partcl_seed_run_paraxial'; end
if nargin < 2, impactt_dir = '/tmp/impactT_noSC'; end

tt = TimeTrack();
tt.work_dir   = char(work_dir);
tt.output_dir = 'diags';
bunches = tt.readDumps();
b = bunches{end};
keep = double(b.z) > -1.0;
zz = double(b.z(keep));
px = double(b.px(keep));
py = double(b.py(keep));
pz = double(b.pz(keep));
me_kg = 9.1093837015e-31;
c     = 299792458.0;
gam = sqrt(1 + (px.^2 + py.^2 + pz.^2)/(me_kg*c)^2);

z_mean = mean(zz);
zr = zz - z_mean;     % centered z

% Bin by zr quantiles
n_bins = 21;
qbnd = linspace(0, 1, n_bins+1);
fprintf('LTT (work_dir=%s):\n', char(work_dir));
fprintf('  bins: %d  N=%d  z range=[%g, %g] mm  g range=[%g, %g]\n', ...
        n_bins, numel(zz), min(zr)*1e3, max(zr)*1e3, min(gam), max(gam));
fprintf('  %-8s %-12s %-12s\n', 'qbin', 'z_centred[mm]', '<gamma>');
for k = 1:n_bins
    lo = quantile(zr, qbnd(k));
    hi = quantile(zr, qbnd(k+1));
    if k < n_bins, sel = zr >= lo & zr < hi; else sel = zr >= lo; end
    if any(sel)
        fprintf('  %-8d %-12.4f %-12.4f\n', k, mean(zr(sel))*1e3, mean(gam(sel)));
    end
end

% Same for ImpactT fort.101
fort101 = fullfile(char(impactt_dir), 'fort.101');
if exist(fort101, 'file')
    A = readmatrix(fort101, 'FileType', 'text');
    zi  = A(:,5);
    pxi = A(:,2);  pyi = A(:,4);  pzi = A(:,6);
    gi  = sqrt(1 + pxi.^2 + pyi.^2 + pzi.^2);
    zir = zi - mean(zi);
    fprintf('\nImpactT (fort.101 z=%g m):\n', mean(zi));
    fprintf('  N=%d  z range=[%g, %g] mm  g range=[%g, %g]\n', ...
            numel(zi), min(zir)*1e3, max(zir)*1e3, min(gi), max(gi));
    fprintf('  %-8s %-12s %-12s\n', 'qbin', 'z_centred[mm]', '<gamma>');
    for k = 1:n_bins
        lo = quantile(zir, qbnd(k));
        hi = quantile(zir, qbnd(k+1));
        if k < n_bins, sel = zir >= lo & zir < hi; else sel = zir >= lo; end
        if any(sel)
            fprintf('  %-8d %-12.4f %-12.4f\n', k, mean(zir(sel))*1e3, mean(gi(sel)));
        end
    end
end
end
