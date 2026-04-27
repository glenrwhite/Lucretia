function M = CWIGGLERStruc( L, P, BMax, Periods, Name, varargin )
%
% CWIGGLERSTRUC Create a Lucretia canonical wiggler / undulator
%               (CWIGGLER) data structure, modelled after Elegant's
%               CWIGGLER element (canonical integration code of Y. Wu,
%               Duke University).
%
% M = CWIGGLERStruc( L, P, BMax, Periods, Name, ...optional name/value pairs )
%
%     L:        Total wiggler length [m].
%     P:        Reference momentum [GeV/c].
%     BMax:     Peak on-axis magnetic field [T].
%     Periods:  Number of full wiggler periods (integer).
%     Name:     Element name (char).
%
% Optional name/value pairs (defaults match Elegant where sensible):
%
%     'Helical'           (0)         0 = planar, 1 = helical.  Helical
%                                     adds a hardcoded second component
%                                     90 deg out of phase, giving the
%                                     standard rotating-field undulator.
%     'Vertical'          (0)         If 1 (planar only), wiggle in y
%                                     instead of x.  Implemented via
%                                     Tilt = pi/2 in the element offsets.
%     'StepsPerPeriod'    (12)        Symplectic-integrator sub-steps per
%                                     wiggler period.  MUST be a
%                                     multiple of 4 (the same constraint
%                                     Elegant imposes).
%     'IntegrationOrder'  (4)         2 (leapfrog) or 4 (Yoshida).
%     'Harmonics'         ([])        Nx5 matrix of field harmonics:
%                                       [Cmn KxOverKw KyOverKw KzOverKw Phase]
%                                     matching Elegant's BX_FILE/BY_FILE
%                                     SDDS column names.  Empty -> ideal
%                                     sinusoidal field (single harmonic
%                                     with kx=0, ky=ku, kz=ku, C=1, phi=0).
%                                     Each row must satisfy the Maxwell
%                                     constraint  ky^2 = kx^2 + kz^2.
%     'PoleFactor1/2/3'   (1, 1, 1)   End-pole strength taper, applied to
%                                     poles 1/2/3 from each end.
%     'SynchRad'          (0)         Apply classical SR per integration
%                                     step.  Element-level flag,
%                                     independent of the SynRad TrackFlag.
%     'ISR'               (0)         Apply quantum (incoherent) SR per
%                                     step.  Element-level flag.
%     'aper'              ([])        Circular half-aperture [m] for the
%                                     exit-face check (only consulted
%                                     when the Aper TrackFlag is set).
%
% References:
%   Elegant CWIGGLER documentation, APS / M. Borland.
%   Y. Wu et al, "Symplectic models for general insertion devices",
%       Phys. Rev. ST Accel. Beams 1, 044001 (1998).
%
% Version date: 27-Apr-2026.
%

p = inputParser ;
p.addParameter('Helical',          0)    ;
p.addParameter('Vertical',         0)    ;
p.addParameter('StepsPerPeriod',   12)   ;
p.addParameter('IntegrationOrder', 4)    ;
p.addParameter('Harmonics',        [])   ;
p.addParameter('PoleFactor1',      1.0)  ;
p.addParameter('PoleFactor2',      1.0)  ;
p.addParameter('PoleFactor3',      1.0)  ;
p.addParameter('SynchRad',         0)    ;
p.addParameter('ISR',              0)    ;
p.addParameter('aper',             [])   ;
p.parse(varargin{:}) ;
opt = p.Results ;

if mod(opt.StepsPerPeriod, 4) ~= 0
  error('CWIGGLERStruc:badSteps', ...
        'StepsPerPeriod (%d) must be a positive multiple of 4', ...
        opt.StepsPerPeriod) ;
end
if ~ismember(opt.IntegrationOrder, [2 4])
  error('CWIGGLERStruc:badOrder', ...
        'IntegrationOrder must be 2 or 4 (got %g)', opt.IntegrationOrder) ;
end
if opt.Helical && opt.Vertical
  warning('CWIGGLERStruc:helicalVertical', ...
          'Vertical=1 is ignored when Helical=1; the helical field is symmetric.') ;
  opt.Vertical = 0 ;
end

% Validate the Harmonics matrix.
H = opt.Harmonics ;
if ~isempty(H)
  if size(H, 2) ~= 5
    error('CWIGGLERStruc:badHarmonics', ...
          'Harmonics must be an Nx5 matrix [Cmn KxN KyN KzN Phase]') ;
  end
  tol = 1e-6 ;
  for ih = 1:size(H, 1)
    kxn = H(ih, 2) ; kyn = H(ih, 3) ; kzn = H(ih, 4) ;
    lhs = kyn * kyn ;
    rhs = kxn * kxn + kzn * kzn ;
    if abs(lhs - rhs) > tol * max(1, abs(rhs))
      error('CWIGGLERStruc:badHarmonics', ...
            ['Harmonic row %d violates the Maxwell constraint ' ...
             'ky^2 = kx^2 + kz^2 (%g vs %g)'], ih, lhs, rhs) ;
    end
    if kyn == 0
      error('CWIGGLERStruc:badHarmonics', ...
            ['Harmonic row %d has KyOverKw = 0; the horizontal-pole ' ...
             'field formula is singular here.  Use Vertical=1 for ' ...
             'a vertically deflecting wiggler instead.'], ih) ;
    end
  end
end

M.Name    = Name ;
M.S       = 0 ;
M.P       = P ;
M.Class   = 'CWIGGLER' ;
M.L       = L ;
M.BMax    = BMax ;
M.Periods = Periods ;

M.Helical          = opt.Helical ;
M.Vertical         = opt.Vertical ;
M.StepsPerPeriod   = opt.StepsPerPeriod ;
M.IntegrationOrder = opt.IntegrationOrder ;
M.Harmonics        = H ;
M.PoleFactor1      = opt.PoleFactor1 ;
M.PoleFactor2      = opt.PoleFactor2 ;
M.PoleFactor3      = opt.PoleFactor3 ;
M.SynchRad         = opt.SynchRad ;
M.ISR              = opt.ISR ;

M.Offset = [0 0 0 0 0 0] ;
M.Girder = 0 ;
% Vertical=1 is implemented as a 90-degree rotation about the beam axis.
if opt.Vertical
  M.Tilt = pi/2 ;
else
  M.Tilt = 0 ;
end
if ~isempty(opt.aper)
  M.aper = opt.aper ;
end

M.TrackFlag.ZMotion      = 0 ;
M.TrackFlag.LorentzDelay = 0 ;
M.TrackFlag.Aper         = 0 ;
M.TrackFlag.Split        = 0 ;
