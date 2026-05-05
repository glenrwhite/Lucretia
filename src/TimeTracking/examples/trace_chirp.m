function trace_chirp(work_dir)
% TRACE_CHIRP  Plot (id, z, gamma) for a lucretia-tt run to see emission-time
% vs final state. Lower particle IDs = earlier-emitted (CathodeSource adds
% them in order). Helps diagnose the chirp sign reversal.

if nargin < 1, work_dir = '/tmp/cathode_imptT_run_matched'; end

tt = TimeTrack();
tt.work_dir   = char(work_dir);
tt.output_dir = 'diags';
bunches = tt.readDumps();

b = bunches{end};
keep = double(b.z) > -1.0;     % filter parked dead particles
xx = double(b.x(keep));
zz = double(b.z(keep));
px = double(b.px(keep));
py = double(b.py(keep));
pz = double(b.pz(keep));
id = double(b.id(keep));

me_kg = 9.1093837015e-31;
c     = 299792458.0;
gam = sqrt(1 + (px.^2 + py.^2 + pz.^2)/(me_kg*c)^2);

% Sort by id
[id_sorted, idx] = sort(id);
z_sorted = zz(idx);
g_sorted = gam(idx);

% Stats
N = numel(id_sorted);
fprintf('Last dump: N=%d (alive)\n', N);
fprintf('  id range: [%d .. %d]\n', min(id_sorted), max(id_sorted));
fprintf('  z range:  [%g .. %g] mm\n', min(z_sorted)*1e3, max(z_sorted)*1e3);
fprintf('  g range:  [%g .. %g]\n',    min(g_sorted), max(g_sorted));

% Bin particles by ID into 10 quantiles. Each bin = particles emitted in
% one slice of the emission window (approximately).
n_bins = 10;
qbnd = round(linspace(1, N, n_bins+1));
fprintf('\n%-8s %-12s %-12s %-12s %-12s %-12s\n', ...
        'bin', 'id_range', '<z> [mm]', 'sigma_z [mm]', '<gamma>', 'sigma_g');
for k = 1:n_bins
    a = qbnd(k); b_ = qbnd(k+1);
    z_k = z_sorted(a:b_);
    g_k = g_sorted(a:b_);
    fprintf('%-8d %-12s %-12.4f %-12.4f %-12.4f %-12.6f\n', k, ...
            sprintf('[%d,%d]', id_sorted(a), id_sorted(b_)), ...
            mean(z_k)*1e3, std(z_k)*1e3, mean(g_k), std(g_k));
end

% Also fit a line: gamma vs id
p = polyfit(id_sorted, g_sorted, 1);
fprintf('\nLinear fit: gamma = %g * id + %g\n', p(1), p(2));
fprintf('  slope sign: %s   (positive = later-emitted has higher gamma)\n', ...
        char(48 + (p(1) > 0)*1));

figure('Position', [100, 100, 1100, 480]);
subplot(1,2,1);
scatter(id_sorted, z_sorted*1e3, 4, '.');
xlabel('Particle ID (lower = earlier emission)');
ylabel('z [mm] in last dump');
title('Particle ID vs z — head/tail vs emission order');
grid on;

subplot(1,2,2);
scatter(id_sorted, g_sorted, 4, '.');
xlabel('Particle ID (lower = earlier emission)');
ylabel('gamma in last dump');
title('Particle ID vs gamma — chirp by emission order');
grid on;

print('-dpng', '-r150', 'trace_chirp.png');
fprintf('\nSaved -> trace_chirp.png\n');
end
