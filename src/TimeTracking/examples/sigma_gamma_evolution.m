function sigma_gamma_evolution(work_dir)
% Plot sigma_gamma vs z across all dumps to see when chirp develops.
if nargin < 1, work_dir = '/tmp/partcl_seed_run_centroid_frac'; end

tt = TimeTrack();
tt.work_dir   = char(work_dir);
tt.output_dir = 'diags';
bunches = tt.readDumps();

n = numel(bunches);
me_kg = 9.1093837015e-31;
c     = 299792458.0;

z_avg     = zeros(1, n);
sigma_z   = zeros(1, n);
gamma_avg = zeros(1, n);
sigma_g   = zeros(1, n);

for k = 1:n
    b = bunches{k};
    keep = double(b.z) > -1.0;
    zz = double(b.z(keep));
    px = double(b.px(keep));
    py = double(b.py(keep));
    pz = double(b.pz(keep));
    p2 = px.^2 + py.^2 + pz.^2;
    g  = sqrt(1 + p2/(me_kg*c)^2);
    z_avg(k)     = mean(zz);
    sigma_z(k)   = std(zz);
    gamma_avg(k) = mean(g);
    sigma_g(k)   = std(g);
end

fprintf('%-4s %-12s %-12s %-12s %-12s\n', 'k', '<z>[m]', 'sigma_z[mm]', '<gamma>', 'sigma_gamma');
for k = 1:n
    if mod(k, max(1, round(n/30))) == 0 || k == 1 || k == n
        fprintf('%-4d %-12.4f %-12.4f %-12.4f %-12.6f\n', ...
                k, z_avg(k), sigma_z(k)*1e3, gamma_avg(k), sigma_g(k));
    end
end

% Find sigma_gamma at gun exit (z ≈ 0.15m)
[~, k_ge] = min(abs(z_avg - 0.15));
fprintf('\nAt gun exit (z=%g m, dump %d): sigma_gamma = %g\n', ...
        z_avg(k_ge), k_ge, sigma_g(k_ge));
[~, k_e] = min(abs(z_avg - 0.20));
fprintf('At z=0.20 m (dump %d): sigma_gamma = %g\n', k_e, sigma_g(k_e));
fprintf('Final (z=%g m): sigma_gamma = %g\n', z_avg(end), sigma_g(end));
end
