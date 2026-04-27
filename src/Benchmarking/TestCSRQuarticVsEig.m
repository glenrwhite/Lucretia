function out = TestCSRQuarticVsEig( varargin )
% TESTCSRQUARTICVSEIG  Self-test: verify the vectorised Newton quartic
% solver in applyCSR.m's drift-CSR path agrees with the original
% eig-of-companion-matrix loop.
%
% Runs both implementations on synthetic CSR drift parameters covering
% a range of (R, X, dsmax, nbin, bw) typical of light-source linacs and
% bunch-compressor lattices, and reports the per-bin root mismatch and
% the resulting ZINT mismatch.
%
% Pass true to plot the worst-case ZINT comparison.
%
%   out = TestCSRQuarticVsEig
%   out = TestCSRQuarticVsEig(true)
%
% out is a struct with fields:
%   maxRelErr_root :  max relative error in the per-bin root psi
%   maxRelErr_zint :  max relative error in ZINT
%   nCases         :  number of synthetic test cases run
%
% Tolerances expected:
%   - root match to ~1e-12 relative (Newton converges past machine eps)
%   - ZINT match to ~1e-12 relative (driven by root match)
%
% See also: applyCSR.

doPlot = nargin >= 1 && varargin{1};

rng(20260427) ;   % reproducible cases

% Synthetic parameter grid covering plausible CSR drift situations.
caseGrid = [
  % R [m]    X (=driftL/R)   dsmax_factor   nbin   bw [m]
       1.0      0.0          1.0           128    1e-4
       1.0      0.5          1.0           128    1e-4
       2.0      1.0          0.5           128    1e-5
      10.0      0.1          2.0           128    1e-4
      10.0      5.0          1.0           128    1e-3
       0.5      0.05         3.0            64    1e-5
       5.0      0.0          0.5           256    5e-5
      20.0     10.0          1.0           128    2e-4
] ;

maxRelErr_root = 0 ;
maxRelErr_zint = 0 ;
worst.case = [] ;
worst.psi_old = [] ; worst.psi_new = [] ;
worst.zint_old = [] ; worst.zint_new = [] ;

fprintf(['Case  R[m]    X       nbin  bw[m]    K     ', ...
         'maxRootRelErr  maxZINTRelErr\n']) ;
for ic = 1:size(caseGrid,1)
  R    = caseGrid(ic,1) ;
  X    = caseGrid(ic,2) ;
  fac  = caseGrid(ic,3) ;
  nbin = caseGrid(ic,4) ;
  bw   = caseGrid(ic,5) ;
  PHI  = 0.05 ;                          % typical bend angle [rad]
  % Pick dsmax so K varies nontrivially
  dsmax = fac * ((R*PHI^3)/24)*((PHI+4*X)/(PHI+X+1e-12)) ;
  if dsmax <= bw
    fprintf('  %2d  skip (dsmax < bw)\n', ic) ;
    continue
  end
  z = bw * ((1:nbin) - (nbin/2)) ;       % uniform bin centres
  % Synthesize a smooth dq (Gaussian-like) so the wake sum is nontrivial
  dq = exp(-0.5*((z - mean(z))/(0.1*(max(z)-min(z)))).^2) - 0.5 ;
  dq = dq / sum(abs(dq)) ;

  [psi_old, ZINT_old] = drift_csr_old(z, dq, bw, R, X, dsmax, nbin) ;
  [psi_new, ZINT_new] = drift_csr_new(z, dq, bw, R, X, dsmax, nbin) ;

  % Compare only where both are valid (nan in old, nonzero/nonzero in new)
  validPsi = ~isnan(psi_old) & psi_old > 0 ;
  if any(validPsi)
    rerr_p = max(abs(psi_old(validPsi) - psi_new(validPsi)) ...
                 ./ max(abs(psi_old(validPsi)), 1e-30)) ;
  else
    rerr_p = 0 ;
  end
  K = min(floor(dsmax/bw), nbin-1) ;
  validZ = false(nbin,1) ;
  validZ(2:end) = true ;          % is=1 always zero in both
  rerr_z = max(abs(ZINT_old(validZ) - ZINT_new(validZ)) ...
               ./ max(abs(ZINT_old(validZ)), 1e-30)) ;

  fprintf('  %2d  %5.2g  %5.2g  %4d  %.0e   %3d   %.2e   %.2e\n', ...
    ic, R, X, nbin, bw, K, rerr_p, rerr_z) ;

  if rerr_z > maxRelErr_zint
    maxRelErr_zint = rerr_z ;
    worst.case = ic ;
    worst.psi_old  = psi_old ;
    worst.psi_new  = psi_new ;
    worst.zint_old = ZINT_old ;
    worst.zint_new = ZINT_new ;
  end
  maxRelErr_root = max(maxRelErr_root, rerr_p) ;
end

fprintf('\n=== Summary ===\n') ;
fprintf('  Cases run:                 %d\n', size(caseGrid,1)) ;
fprintf('  Max relative root error:   %.2e\n', maxRelErr_root) ;
fprintf('  Max relative ZINT error:   %.2e\n', maxRelErr_zint) ;
if maxRelErr_zint < 1e-10
  fprintf('  PASS (< 1e-10)\n') ;
else
  fprintf('  *** FAIL (> 1e-10) ***\n') ;
end

if doPlot && ~isempty(worst.case)
  figure ;
  subplot(2,1,1)
  plot(worst.psi_old, '-x') ; hold on ; plot(worst.psi_new, '-o') ;
  xlabel('bin') ; ylabel('psi') ; legend('eig (old)','Newton (new)') ;
  title(sprintf('Case %d: per-bin root', worst.case)) ;
  subplot(2,1,2)
  plot(worst.zint_old, '-x') ; hold on ; plot(worst.zint_new, '-o') ;
  xlabel('bin') ; ylabel('ZINT') ; legend('old','new') ;
end

out = struct('maxRelErr_root', maxRelErr_root, ...
             'maxRelErr_zint', maxRelErr_zint, ...
             'nCases',         size(caseGrid,1) ) ;
end


function [psi_out, ZINT] = drift_csr_old( z, dq, bw, R, X, dsmax, nbin )
% Reference: original eig-of-companion-matrix loop, copied verbatim.
ZINT = zeros(nbin,1,'like',z) ;
psi  = nan(1,nbin) ;
psi_out = nan(nbin,1) ;
if dsmax > diff(z(1:2))
  for is = 1:nbin
    isp = z >= (z(is)-dsmax) & (1:length(z)) < is ;
    if any(isp)
      ds = z(is) - z(isp) ;
      a  = 24*ds(1)/R ; b = 4*X ; C = [-1 -b 0 a a*X] ;
      a  = diag(ones(1,3,'like',X), -1) ;
      d  = C(2:end)./C(1) ;
      a(1,:) = -d ;
      rpsi = eig(a) ;
      psi(1) = max(real(rpsi(imag(rpsi)==0))) ;
      psi_out(is) = psi(1) ;
      psi_eval = psi(~isnan(psi) & isp) ;
      ZINT(is) = sum( (1./(psi_eval+2.*X)).*dq(isp).*bw ) ;
      psi = circshift(psi, 1) ;
    end
  end
end
end


function [psi_out, ZINT] = drift_csr_new( z, dq, bw, R, X, dsmax, nbin )
% Refactored: vectorised Newton + filter().  Mirrors the in-tree
% applyCSR.m drift-CSR block.
ZINT = zeros(nbin,1,'like',z) ;
psi_out = zeros(nbin,1) ;
if dsmax > bw
  K       = min(floor(dsmax/bw), nbin-1) ;
  k_vec   = (1:nbin)' ;
  ds1_vec = bw * min(k_vec - 1, K) ;
  a_vec   = 24 * ds1_vec / R ;
  bcoef   = 4 * X ;
  a_safe  = max(a_vec, eps) ;
  X_safe  = max(X, eps) ;
  psi     = max( (2*a_safe).^(1/3), (2*a_safe*X_safe).^(1/4) ) ;
  for iter = 1:16
    fp  = psi.^4 + bcoef*psi.^3 - a_vec.*psi - a_vec*X ;
    fpp = 4*psi.^3 + 3*bcoef*psi.^2 - a_vec ;
    psi = psi - fp ./ fpp ;
  end
  psi(1)  = 0 ;
  psi_out = psi ;
  kernel        = zeros(K+1, 1) ;
  kernel(2:K+1) = 1 ./ (psi(2:K+1) + 2*X) ;
  ZINT_row      = filter(kernel, 1, dq) ;
  ZINT          = bw * ZINT_row(:) ;
end
end
