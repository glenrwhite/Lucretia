function M = LSRMDLTRStruc( len, P, Bu, Periods, Name, varargin )
%
% LSRMDLTRSTRUC Create a Lucretia laser-seeded relativistic modulator
%               (LSRMDLTR) data structure.
%
% M = LSRMDLTRStruc( L, P, Bu, Periods, Name, ...optional name/value pairs )
%
%     L:        Undulator length [m].
%     P:        Reference momentum [GeV/c].
%     Bu:       Peak undulator magnetic field [T].
%     Periods:  Number of undulator periods (integer).
%     Name:     Element name (char).
%
% Optional name/value pairs (defaults match the pLucretia LSRMdltr
% element):
%
%     'LaserWavelength' (0)        Laser wavelength [m].  0 -> compute
%                                  from the resonance condition
%                                  lambda_r = lambda_u/(2 gamma^2)*(1+K^2/2).
%     'LaserPeakPower'  (0)        Peak laser power [W].  0 -> no laser.
%     'LaserW0'         (1)        Gaussian beam waist (1/e^2 intensity
%                                  radius) [m].
%     'LaserPhase'      (0)        Laser phase offset [rad].
%     'LaserX0','LaserY0','LaserZ0' (0)  Laser beam centre offset [m].
%     'LaserTilt'       (0)        Polarisation rotation [rad].
%     'LaserM','LaserN' (0)        Hermite-Gauss mode indices, 0..4.
%     'NSteps'          (100)      Number of fixed-step RK4 sub-steps.
%     'FieldExpansion'  ('leading')  'ideal'|'exact'|'leading'  or
%                                    0|1|2.  See lsrmdltr.py for the
%                                    field models.
%     'PoleFactor1','PoleFactor2','PoleFactor3' (0.155715, 0.380687,
%                                  0.802829)  End-pole field tapers.
%     'Helical'         (0)        0 = planar, 1 = helical.
%     'SynchRad'        (0)        Apply classical SR per RK4 step.
%     'ISR'             (0)        Apply quantum SR (incoherent) per step.
%     'aper'            ([])       Element half-aperture [m] (scalar,
%                                  circular).  Empty -> no aperture
%                                  check.
%
% The LSRMDLTR tracker is non-splittable (the RK4 integration carries
% state through the entire element); the Split TrackFlag has no effect
% on this element.  The element-level SynchRad and ISR flags above are
% used INSTEAD of the standard SynRad TrackFlag.
%
% Version date: 24-Apr-2026.
%

p = inputParser ;
p.addParameter('LaserWavelength', 0)    ;
p.addParameter('LaserPeakPower',  0)    ;
p.addParameter('LaserW0',         1)    ;
p.addParameter('LaserPhase',      0)    ;
p.addParameter('LaserX0',         0)    ;
p.addParameter('LaserY0',         0)    ;
p.addParameter('LaserZ0',         0)    ;
p.addParameter('LaserTilt',       0)    ;
p.addParameter('LaserM',          0)    ;
p.addParameter('LaserN',          0)    ;
p.addParameter('NSteps',          100)  ;
p.addParameter('FieldExpansion',  'leading') ;
p.addParameter('PoleFactor1',     0.1557150345503998) ;
p.addParameter('PoleFactor2',     0.3806870122885810) ;
p.addParameter('PoleFactor3',     0.8028293373481792) ;
p.addParameter('Helical',         0)    ;
p.addParameter('SynchRad',        0)    ;
p.addParameter('ISR',             0)    ;
p.addParameter('aper',            [])   ;
p.parse(varargin{:}) ;
opt = p.Results ;

% Convert FieldExpansion string -> numeric (0=ideal, 1=exact, 2=leading).
fe = opt.FieldExpansion ;
if ischar(fe) || isstring(fe)
  switch lower(char(fe))
    case 'ideal',                   fe = 0 ;
    case 'exact',                   fe = 1 ;
    case {'leading','leading terms'}, fe = 2 ;
    otherwise
      error('LSRMDLTRStruc:badFieldExpansion', ...
            'FieldExpansion must be ''ideal''/''exact''/''leading'' or 0/1/2') ;
  end
elseif ~ismember(fe, [0 1 2])
  error('LSRMDLTRStruc:badFieldExpansion', ...
        'FieldExpansion numeric value must be 0, 1, or 2') ;
end

M.Name    = Name ;
M.S       = 0 ;
M.P       = P ;
M.Class   = 'LSRMDLTR' ;
M.L       = len ;
M.Bu      = Bu ;
M.Periods = Periods ;

M.LaserWavelength = opt.LaserWavelength ;
M.LaserPeakPower  = opt.LaserPeakPower  ;
M.LaserW0         = opt.LaserW0         ;
M.LaserPhase      = opt.LaserPhase      ;
M.LaserX0         = opt.LaserX0         ;
M.LaserY0         = opt.LaserY0         ;
M.LaserZ0         = opt.LaserZ0         ;
M.LaserTilt       = opt.LaserTilt       ;
M.LaserM          = opt.LaserM          ;
M.LaserN          = opt.LaserN          ;
M.NSteps          = opt.NSteps          ;
M.FieldExpansion  = fe                  ;
M.PoleFactor1     = opt.PoleFactor1     ;
M.PoleFactor2     = opt.PoleFactor2     ;
M.PoleFactor3     = opt.PoleFactor3     ;
M.Helical         = opt.Helical         ;
M.SynchRad        = opt.SynchRad        ;
M.ISR             = opt.ISR             ;

M.Offset = [0 0 0 0 0 0] ;
M.Girder = 0 ;
if ~isempty(opt.aper)
  M.aper = opt.aper ;
end

M.TrackFlag.ZMotion      = 0 ;
M.TrackFlag.LorentzDelay = 0 ;
M.TrackFlag.Aper         = 0 ;
M.TrackFlag.Split        = 0 ;
