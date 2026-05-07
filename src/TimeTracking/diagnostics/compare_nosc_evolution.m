function compare_nosc_evolution()
% Compare lt no-SC vs imp no-SC at gun-region intermediate dumps.
% Tests user's hypothesis that we previously matched gun-region.
cd('/Users/glenwhite/code/Lucretia');
addpath('src/TimeTracking');
addpath('/tmp/claude');

% lt no-SC final particles
bunches = readBeamOpenPMD('/tmp/bench_full_nosc/diags');
fprintf('lt no-SC has %d intermediate dumps\n', numel(bunches));

% Read imp no-SC fort.18 (time-evolution stats)
ref_imp_nosc = readImpactTFort18Local(fullfile(getenv('TMPDIR'), 'imp_nosc'));
ref_imp_sc   = readImpactTFort18Local('/Users/glenwhite/Documents/GitHub/Lattices/common/ImpactT');
fprintf('imp no-SC fort.18 has %d entries; imp WITH SC has %d\n', numel(ref_imp_nosc.t), numel(ref_imp_sc.t));

me = 9.1093837015e-31; c = 299792458;

% Comparison table at each lt dump time
fprintf('\nGun-region comparison: lt no-SC vs imp NO-SC vs imp WITH SC\n');
fprintf('%-9s | %-7s %-7s %-7s | %-7s %-7s %-7s\n', ...
    't[ns]', 'lt-sx', 'imp-sx-NoSC', 'imp-sx-SC', 'lt-gam', 'imp-gam-NoSC', 'imp-gam-SC');
fprintf('%s\n', repmat('-', 1, 100));
for k = 1:numel(bunches)
    b = bunches{k};
    if numel(b.x) < 10, continue; end
    if b.time > 1e-9, break; end   % gun region only (first 1 ns)
    xx = double(b.x); yy = double(b.y); zz = double(b.z);
    px = double(b.px); py = double(b.py); pz = double(b.pz);
    alive = (zz > -1e6);
    xx = xx(alive); yy = yy(alive);
    px = px(alive); py = py(alive); pz = pz(alive);
    if numel(xx) < 5, continue; end
    g = sqrt(1 + (px.^2+py.^2+pz.^2)/(me*c)^2);
    sx = std(xx);
    g_mean = mean(g);
    tk = b.time;
    if tk < ref_imp_nosc.t(1) || tk > ref_imp_nosc.t(end), continue; end
    g_imp_nosc = interp1(ref_imp_nosc.t, ref_imp_nosc.gamma,   tk, 'linear');
    sx_imp_nosc = interp1(ref_imp_nosc.t, ref_imp_nosc.sigma_x, tk, 'linear');
    g_imp_sc = interp1(ref_imp_sc.t, ref_imp_sc.gamma,   tk, 'linear');
    sx_imp_sc = interp1(ref_imp_sc.t, ref_imp_sc.sigma_x, tk, 'linear');
    fprintf('%-9.3f | %-7.3f %-13.3f %-9.3f | %-7.2f %-13.2f %-9.2f\n', ...
        tk*1e9, sx*1e3, sx_imp_nosc*1e3, sx_imp_sc*1e3, ...
        g_mean, g_imp_nosc, g_imp_sc);
end
end


function ref = readImpactTFort18Local(impactt_dir)
% Mirror of timetracking.readImpactTFort18: per its source, fort.18 has 7
% cols (col1=t, col3=gamma); fort.24 has 8 cols (col4=sigma_x, col8=eps_nx).
fid = fopen(fullfile(impactt_dir, 'fort.18'), 'r');
d = textscan(fid, '%f');  d = d{1};
fclose(fid);
ref.t      = d(1:7:end);
ref.z_mean = d(2:7:end);
ref.gamma  = d(3:7:end);

fid = fopen(fullfile(impactt_dir, 'fort.24'), 'r');
d = textscan(fid, '%f');  d = d{1};
fclose(fid);
ref.sigma_x = d(4:8:end);
ref.eps_nx  = d(8:8:end);
end
