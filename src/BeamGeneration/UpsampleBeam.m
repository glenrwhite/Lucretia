function BeamOut = UpsampleBeam(BeamIn,Nmacro,Bandwidth,BunchID)
%UPSAMPLEBEAM Add macro particles to provided Lucretia beam particles
%BeamOut = UpsampleBeam(BeamIn,nmacro [,BunchID])
% [REQUIRES >R2023b and Statistics toolbox]
%  BeamIn : Lucretia Beam structure
%  Nmacro : Final # macro particles for beam Bunches to contain (must be greater than original)
%  Bandwidth (optional) : BW to use, pass either scalar (for all dims) or [1x6] vector (default=0.1)
%  BunchID (optional) : Bunch # to convert, 0=all [default]

if ~exist('BunchID','var') || BunchID==0
  BunchID = 1:length(BeamIn.Bunch) ;
end

if ~exist('Bandwidth','var') || isempty(Bandwidth)
  Bandwidth=ones(1,6).*0.1;
elseif length(Bandwidth)==1
  Bandwidth=ones(1,6).*Bandwidth;
end

BeamOut=BeamIn;
for id=BunchID
  Qp=sum(BeamIn.Bunch(id).Q)/length(BeamIn.Bunch(id).Q);
  X = BeamIn.Bunch(id).x(:,~BeamIn.Bunch(id).stop) ;
  nadd = Nmacro - length(X(1,:)) ;
  if nadd<=0
    error('Need to ask for more macro particles than provided in BeamIn');
  end
  nadd=9;
  gx1 = linspace(min(X(1,:)),max(X(1,:)),nadd) ;
  gx2 = linspace(min(X(2,:)),max(X(2,:)),nadd) ;
  gx3 = linspace(min(X(3,:)),max(X(3,:)),nadd) ;
  gx4 = linspace(min(X(4,:)),max(X(4,:)),nadd) ;
  gx5 = linspace(min(X(5,:)),max(X(5,:)),nadd) ;
  gx6 = linspace(min(X(6,:)),max(X(6,:)),nadd) ;
  [x1,x2,x3,x4,x5,x6]=ndgrid(gx1,gx2,gx3,gx4,gx5,gx6);
  xi = [x1(:) x2(:) x3(:) x4(:) x5(:) x6(:)];
  iv = randperm(length(X),min([1e4,length(X)]));
  tic
  f = mvksdensity(X(:,iv)',xi,'Bandwidth',range(xi).*Bandwidth);
  toc
  BeamOut.Bunch(id).stop = [BeamOut.Bunch(id).stop zeros(1,length(f))] ;
  BeamOut.Bunch(id).Q = [BeamOut.Bunch(id).Q ones(1,length(f)).*Qp] ;
  BeamOut.Bunch(id).x = [BeamOut.Bunch(id).x f(1:nadd,:)'] ;
end