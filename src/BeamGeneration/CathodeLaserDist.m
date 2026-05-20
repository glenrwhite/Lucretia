classdef CathodeLaserDist < handle
  % CATHODELASERDIST  Cold-start cathode beam distribution generator.
  %
  % Generates the initial particle distribution at a photocathode for
  % photoinjector simulation. Native MATLAB replacement for distgen
  % (https://github.com/ColwynGulliford/distgen) that uses a Hammersley
  % low-discrepancy sequence to avoid the macroparticle clumping that
  % drives micro-bunch instabilities in PIC simulation. Supports both
  % ImpactT (partcl.data) and lucretia-tt (HDF5 seed) output formats.
  %
  % Distributions supported (independently for transverse and longitudinal):
  %   'gaussian'        - standard Normal, sigma controls RMS
  %   'super_gaussian'  - exp(-(|x/(sqrt(2)*sigma)|^p)) with p = 1/alpha
  %                       alpha=1 -> Gaussian, alpha->0 -> uniform-with-edges,
  %                       alpha=0.5 -> mildly flat-topped (matches distgen)
  %   'uniform'         - flat distribution in [-W, W] (transverse: r in [0, R])
  %
  % Temporal distribution can carry a 'slope_fraction' (linear ramp on
  % top of the chosen profile, matching distgen's deformable distribution).
  %
  % Cathode emission adds thermal momenta with mean transverse energy MTE,
  % each px,py component Gaussian with sigma_p = sqrt(m_e*MTE), pz half-
  % Maxwell with a min-pz cutoff to avoid trapped particles.
  %
  % Usage:
  %   c = CathodeLaserDist();
  %   c.n_particles = 1e5;
  %   c.sigma_xy    = 475e-6;
  %   c.sigma_t     = 7e-12;
  %   c.r_type      = 'super_gaussian';   c.r_alpha = 0.5;
  %   c.t_type      = 'super_gaussian';   c.t_alpha = 0.5;  c.t_slope = -0.14;
  %   c.Q_total     = 1e-9;               % 1 nC
  %   Beam = c.generate();
  %   c.writePartclData('partcl.data');               % for ImpactT
  %   c.writeBunchSeedH5('seed.h5', 'work_dir', '.'); % for lucretia-tt
  %
  % The Beam struct returned by generate() matches distgen.beamgen output
  % field-for-field (x, y, z, px, py, pz, t) so it can be used as a drop-
  % in replacement.

  properties
    n_particles = 1e4

    % Total bunch charge (C). Per-macroparticle weight is computed from
    % this and n_particles.
    Q_total     = 1e-9

    % --- Transverse (radial) distribution ---
    r_type      = 'gaussian'   % 'gaussian' | 'super_gaussian' | 'uniform'
    sigma_xy    = 475e-6       % m, RMS radius (Gaussian/SG); for 'uniform' this is the disc radius
    r_alpha     = 0.5          % super-Gaussian shape param; ignored for other types
    r_truncate  = 1.3          % cut radial distribution at r_truncate * sigma_xy (Gaussian/SG only)

    % --- Longitudinal (temporal) distribution ---
    t_type      = 'super_gaussian' % 'gaussian' | 'super_gaussian' | 'uniform'
    sigma_t     = 7e-12        % s, RMS time (Gaussian/SG); half-width for 'uniform'
    t_alpha     = 0.5          % super-Gaussian shape parameter
    t_slope     = -0.142581    % slope fraction (linear ramp on top of profile)

    % --- Cathode emission ---
    MTE         = 0.414        % mean transverse energy at cathode (eV)
    pz_min      = 2e-5         % min pz/(m*c) cutoff to avoid trapped particles

    % --- Sampling ---
    sampling    = 'hammersley' % 'hammersley' (cold-start) | 'random' (Box-Muller)
    rng_seed    = []           % empty = use current rng state, scalar = rng(seed)
  end

  properties(SetAccess = private)
    Beam                       % Last-generated beam struct (distgen-compatible)
  end

  methods
    function obj = CathodeLaserDist()
    end

    function Beam = generate(obj)
    % Generate the cathode distribution. Returns a struct with fields
    %   .x, .y     transverse position (m)
    %   .z         longitudinal position (m), z = -t * c (head at z=0)
    %   .px, .py   normalised momentum p_xy / (m_e * c), dimensionless
    %   .pz        normalised momentum p_z  / (m_e * c)
    %   .t         emission time (s) -- positive numbers = later emission
    %   .Q_per     per-macroparticle charge (C)

      if ~isempty(obj.rng_seed)
        rng(double(obj.rng_seed));
      end

      N = round(obj.n_particles);

      % --- 5D low-discrepancy sample [u1..u5], one row per particle ---
      switch lower(obj.sampling)
        case 'hammersley'
          U = obj.hammersley(N, 5);
        case 'random'
          U = rand(N, 5);
        otherwise
          error('CathodeLaserDist:badSampling', ...
                'sampling must be ''hammersley'' or ''random''');
      end

      % --- Transverse: dims 1,2 -> (r, phi) ---
      r   = obj.sample_radial(U(:,1));
      phi = 2*pi*U(:,2);
      x   = r .* cos(phi);
      y   = r .* sin(phi);

      % --- Temporal: dim 3 -> emission time (s) ---
      t = obj.sample_temporal(U(:,3));

      % --- Cathode thermal momentum: dims 4,5 -> Gaussian (px, py) ---
      % MTE = <p_perp^2/(2 m_e)> = m_e c^2 * (1/2) * <(px/(m_e c))^2 + ...>
      % so each component sigma in normalised units = sqrt(MTE / (m_e c^2))
      % m_e c^2 in eV = 0.51099906e-3 GeV * 1e9 = 510999.06 eV
      mec2_eV = physConsts.emass * 1e9;
      sigma_p_norm = sqrt(obj.MTE / mec2_eV);
      px = sigma_p_norm .* obj.gauss_inv(U(:,4));
      py = sigma_p_norm .* obj.gauss_inv(U(:,5));

      % --- Longitudinal momentum at cathode (half-Maxwell, then cutoff) ---
      % Distgen models this as the cathode emitting only forward; here we
      % use the same |p_perp| magnitude rolled onto +z (gives correct MTE
      % balance) and floor at pz_min to avoid sub-cutoff trapping.
      pz = abs(sigma_p_norm .* randn(N,1));
      pz(pz < obj.pz_min) = pz(pz < obj.pz_min) + obj.pz_min;

      % --- Reference KE for time -> z conversion (matches distgen) ---
      % distgen uses KE = 1 eV (essentially zero, gives beta ~ 0 so z ~ 0)
      % but for reproducibility we keep the same convention.
      KE_eV  = 1.0;
      gamma  = 1 + KE_eV / mec2_eV;
      beta   = sqrt(1 - gamma^-2);
      z      = t .* beta .* physConsts.clight;
      z      = z - max(z);   % head of bunch at z = 0

      % --- Pack ---
      Beam.x     = x;
      Beam.y     = y;
      Beam.z     = z;
      Beam.px    = px;
      Beam.py    = py;
      Beam.pz    = pz;
      Beam.t     = t;
      Beam.Q_per = obj.Q_total / N;
      obj.Beam   = Beam;
    end

    function writePartclData(obj, path)
    % Write ImpactT-format partcl.data (text, one header line + N rows of
    % "x px y py z pz" with px/py/pz normalised to m_e*c). Matches the
    % output of distgen.m beamgen(true).
      if isempty(obj.Beam), obj.generate(); end
      B = obj.Beam;
      fid = fopen(path, 'w');
      cleanupFid = onCleanup(@() fclose(fid));
      fprintf(fid, '%d\n', numel(B.x));
      fprintf(fid, '%g %g %g %g %g %g\n', ...
              [B.x(:) B.px(:) B.y(:) B.py(:) B.z(:) B.pz(:)]');
    end

    function cs = toCathodeSource(obj, z_cathode, pulse_t0)
    % Return a configured timetracking.CathodeSource element struct that
    % reproduces this CathodeLaserDist's emission profile inside lucretia-tt
    % proper (i.e. as a CathodeSource element in the lattice, with
    % image-charge causality and time-staggered emission), rather than as
    % a pre-generated SeedBeam.
    %
    %   z_cathode : lab-frame z of the cathode plane (m)
    %   pulse_t0  : ImpactT-equivalent leading-edge / Tini reference (s).
    %               Internally we shift this forward by the pulse half-window
    %               so lt's CathodeSource (which centers the CDF on its own
    %               pulse_t0) reproduces ImpactT's behind-cathode model
    %               (head emerges at t=Tini, tail later via drift). Without
    %               this shift, lt's bunch was centered ON Tini -- 4*sigma
    %               earlier than imp's bunch, putting it in a different RF
    %               phase region and causing 6× longitudinal mismatch on
    %               long-Gaussian setups (Substrate, FWHM=23 ps).
    %
    % Distribution-shape mapping (CathodeSource now supports all three
    % temporal and transverse shapes that CathodeLaserDist does):
    %   gaussian       -> CathodeSource 'gaussian' (pulse_duration = FWHM)
    %   uniform        -> CathodeSource 'flat_top' / 'uniform_disk'
    %   super_gaussian -> CathodeSource 'super_gaussian' with alpha + slope
    %                     pulse_duration / spot_size = sigma directly
    % All four super-Gaussian extras (pulse_alpha, pulse_slope,
    % transverse_alpha, transverse_truncate) are forwarded.
        if nargin < 2 || isempty(z_cathode), z_cathode = 0.0;        end
        if nargin < 3 || isempty(pulse_t0),  pulse_t0  = 0.0;        end

        % Temporal pulse: shape + size convention + leading-edge -> centre offset.
        % ImpactT pulse_t0 is the leading edge of the pulse (CathodeLaserDist
        % does z = z - max(z) so the head crosses cathode at t=Tini). lt's
        % CathodeSource samples symmetrically around its pulse_t0, so we
        % advance pulse_t0 by the pulse's half-window so the centroids match.
        switch lower(obj.t_type)
            case 'gaussian'
                t_pulse_shape = 'gaussian';
                t_dur         = 2 * sqrt(2*log(2)) * obj.sigma_t;   % FWHM
                t_offset      = 4 * obj.sigma_t;                    % half-window for typical N (Hammersley max ~ 4 sigma)
            case 'super_gaussian'
                t_pulse_shape = 'super_gaussian';
                t_dur         = obj.sigma_t;                        % sigma
                t_offset      = 4 * obj.sigma_t;                    % CathodeSource SG support is ±4*sigma
            case 'uniform'
                t_pulse_shape = 'flat_top';
                t_dur         = 2 * sqrt(3) * obj.sigma_t;          % full width
                t_offset      = sqrt(3) * obj.sigma_t;              % half-width
            otherwise
                error('CathodeLaserDist:toCathodeSource:badTType', ...
                      'unrecognised t_type ''%s''', obj.t_type);
        end
        pulse_t0 = pulse_t0 + t_offset;

        % Transverse: spot + profile
        switch lower(obj.r_type)
            case 'gaussian'
                r_profile = 'gaussian';
                spot      = obj.sigma_xy;                           % per-component RMS
            case 'super_gaussian'
                r_profile = 'super_gaussian';
                spot      = obj.sigma_xy;                           % sigma
            case 'uniform'
                r_profile = 'uniform_disk';
                spot      = obj.sigma_xy;                           % disc radius
            otherwise
                error('CathodeLaserDist:toCathodeSource:badRType', ...
                      'unrecognised r_type ''%s''', obj.r_type);
        end

        cs = timetracking.CathodeSource('name', 'cat', ...
            'z_cathode',              z_cathode, ...
            'image_charge',           true, ...
            'n_macroparticles_total', round(obj.n_particles), ...
            'total_charge',           abs(obj.Q_total), ...
            'pulse_shape',            t_pulse_shape, ...
            'pulse_duration',         t_dur, ...
            'pulse_t0',               pulse_t0, ...
            'transverse_profile',     r_profile, ...
            'spot_size',              spot, ...
            'mte',                    obj.MTE, ...
            'pulse_alpha',            obj.t_alpha, ...
            'pulse_slope',            obj.t_slope, ...
            'transverse_alpha',       obj.r_alpha, ...
            'transverse_truncate',    obj.r_truncate);
    end

    function writeBunchSeedH5(obj, path)
    % Write lucretia-tt HDF5 seed (positions + proper velocities + per-
    % particle weight). Wraps timetracking.writeBunchSeed.
    %
    % Conversion from distgen-format Beam (px = p/(m_e c)) to lucretia-tt
    % seed (ux = gamma*v in m/s):
    %   u_i = (p_i / m_e) = (m_e*c) * (px_norm) / m_e = c * px_norm
    % i.e. just multiply normalised momentum by c.
      if isempty(obj.Beam), obj.generate(); end
      B = obj.Beam;
      c  = physConsts.clight;
      ux = c * B.px(:);
      uy = c * B.py(:);
      uz = c * B.pz(:);
      % Per-macroparticle weight = number of real electrons it represents.
      w_per = (obj.Q_total / numel(B.x)) / physConsts.eQ;
      addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                       'TimeTracking'));    % ensure +timetracking on path
      timetracking.writeBunchSeed(path, B.x, B.y, B.z, ux, uy, uz, w_per);
    end
  end


  methods(Access = private)
    function r = sample_radial(obj, u)
    % Map uniform [0,1] samples to a radial distribution.
      sigma = obj.sigma_xy;
      switch lower(obj.r_type)
        case 'gaussian'
          % radial-Gaussian inverse CDF, with truncation
          % CDF(r) = 1 - exp(-r^2 / (2 sigma^2))
          if obj.r_truncate > 0 && obj.r_truncate < Inf
            R   = obj.r_truncate * sigma;
            cdf_R = 1 - exp(-R^2 / (2*sigma^2));
            r = sigma * sqrt(-2 * log(1 - u .* cdf_R));
          else
            r = sigma * sqrt(-2 * log(1 - u));
          end

        case 'super_gaussian'
          % radial-Super-Gaussian: P(r) ~ r * exp(-(r/(sqrt(2)*sigma*c))^p)
          % where p = 1/alpha, c chosen so RMS(r)^2/2 = sigma^2.
          % We sample by inversion via tabulated CDF.
          alpha = max(obj.r_alpha, 1e-3);
          p     = 1.0 / alpha;
          R_max = obj.r_truncate * sigma;
          r     = obj.invert_cdf_radial_sg(u, sigma, p, R_max);

        case 'uniform'
          % radial uniform on disc r in [0, sigma_xy], P(r) = 2r/R^2.
          r = sigma * sqrt(u);

        otherwise
          error('CathodeLaserDist:badRType', ...
                'r_type must be ''gaussian'', ''super_gaussian'', or ''uniform''');
      end
    end

    function t = sample_temporal(obj, u)
    % Map uniform [0,1] samples to a temporal distribution.
      sigma = obj.sigma_t;
      switch lower(obj.t_type)
        case 'gaussian'
          t = sigma * obj.gauss_inv(u);

        case 'super_gaussian'
          alpha = max(obj.t_alpha, 1e-3);
          p     = 1.0 / alpha;
          % half-width to truncate at: 4 sigma is well past any meaningful
          % super-Gaussian content
          T_max = 4 * sigma;
          t     = obj.invert_cdf_temporal_sg(u, sigma, p, T_max, obj.t_slope);

        case 'uniform'
          % uniform on [-sigma_t*sqrt(3), +sigma_t*sqrt(3)] so RMS = sigma_t
          W = sigma * sqrt(3);
          t = (2*u - 1) * W;

        otherwise
          error('CathodeLaserDist:badTType', ...
                't_type must be ''gaussian'', ''super_gaussian'', or ''uniform''');
      end
    end
  end


  methods(Static, Access = private)
    function H = hammersley(N, D)
    % HAMMERSLEY  N x D Hammersley low-discrepancy sequence in [0,1)^D.
    % Dim 1 is i/N; subsequent dims are van der Corput sequences in
    % bases (2, 3, 5, 7, 11, 13, ...).
      primes_list = [2 3 5 7 11 13 17 19 23 29 31];
      assert(D <= length(primes_list)+1, 'hammersley: D too large');
      H = zeros(N, D);
      H(:,1) = ((1:N).' - 0.5) / N;
      for d = 2:D
        b = primes_list(d-1);
        H(:,d) = CathodeLaserDist.van_der_corput(N, b);
      end
    end

    function v = van_der_corput(N, b)
    % VAN_DER_CORPUT  N-point van der Corput sequence in base b.
      v = zeros(N, 1);
      for i = 1:N
        x = 0;
        f = 1.0 / b;
        k = i;
        while k > 0
          x = x + f * mod(k, b);
          k = floor(k / b);
          f = f / b;
        end
        v(i) = x;
      end
    end

    function z = gauss_inv(u)
    % Inverse Normal CDF via erfinv. u in (0,1) -> z ~ N(0,1).
      uu = min(max(u, 1e-12), 1 - 1e-12);
      z  = sqrt(2) * erfinv(2*uu - 1);
    end

    function r = invert_cdf_radial_sg(u, sigma, p, R_max)
    % Tabulated inverse-CDF sampler for the 2D radial super-Gaussian
    %   f(r) ~ r * exp(-(r/(sqrt(2)*sigma))^p)    (intensity)
    % Returns r values in [0, R_max].
      n_table = 4096;
      r_grid  = linspace(0, R_max, n_table).';
      pdf     = r_grid .* exp(- (r_grid ./ (sqrt(2) * sigma)).^p);
      cdf     = cumsum(pdf);
      cdf     = cdf / cdf(end);
      % Use unique to make interp1 happy in edge cases
      [cdf_u, ia] = unique(cdf, 'stable');
      r = interp1(cdf_u, r_grid(ia), u, 'linear', 0);
    end

    function t = invert_cdf_temporal_sg(u, sigma, p, T_max, slope)
    % Tabulated inverse-CDF sampler for the 1D temporal super-Gaussian
    %   f(t) ~ exp(-(|t|/(sqrt(2)*sigma))^p) * (1 + slope * t / T_max)
    % over t in [-T_max, +T_max]. The slope term is the deformable
    % distribution's linear ramp; clipped non-negative if it would go
    % below zero in the tails.
      n_table = 4096;
      t_grid  = linspace(-T_max, T_max, n_table).';
      pdf     = exp(- (abs(t_grid) ./ (sqrt(2) * sigma)).^p);
      pdf     = pdf .* max(1 + slope .* t_grid ./ T_max, 0);
      cdf     = cumsum(pdf);
      cdf     = cdf / cdf(end);
      [cdf_u, ia] = unique(cdf, 'stable');
      t = interp1(cdf_u, t_grid(ia), u, 'linear', 0);
    end
  end
end
