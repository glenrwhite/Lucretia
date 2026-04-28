function build(varargin)
% Build Lucretia mex files
% - See below for available build targets
% build cpu - build Lucretia mex libraries for cpu target, no GEANT4
%    support (default)
% build cpu-g4  - build Lucretia mex libraries for cpu target, with GEANT4
%    support. Requires GEANT4 (v.10+) installed
%    (with "geant4-config" script on the system search path)
% build gpu - build Lucretia mex libraries for gpu target, no GEANT4
%    support. This requires an NVidia graphics card with compute capability
%    2.0+ and the latest cuda drivers installed. Also requires the Matlab
%    parallel comupting toolbox.
%
% WARNING: When changing compile targets, issue a 'clean' command before
% re-building
% ===========
% Commands
% - can be issued as standalone arguments or added to above compile targets
% -----------
% install - move built mex libraries into correct folders
% clean - clear all build object files and libraries
% cleanall - as above but also clear GEANT4 library in g4track if built
% ===========
% Options
% - Provide arguments below in addition to build target arguments to modify
%   the build
% -----------
% mlrand : use Matlabs default random number generator instead of C
% compiler standard rand() function. e.g. for synchrotron radiation
% calculation related functions. Caution: This is MUCH slower.
% omp    : enable OpenMP parallelisation of per-ray tracking loops in the
%   compiled mex files (Drift, QSOS, Mult, SBend, RF, BPM, INST,
%   LSRMDLTR, CWIGGLER). Does NOT require the Parallel Computing Toolbox
%   at runtime. On macOS, prefers MATLAB's bundled libomp (R2024b+,
%   under matlabroot/bin/<arch>/libomp.dylib) so the mex shares MATLAB's
%   own OpenMP runtime; falls back to Homebrew libomp with a warning if
%   MATLAB does not ship it. The header omp.h is taken from MATLAB
%   Coder's bundle if present, else from Homebrew. Set OMP_NUM_THREADS
%   in the environment to tune the worker count.
% fast   : enable aggressive compiler optimisations (-O3 -march=native
%   -ffast-math -funroll-loops -flto on Linux/macOS, /O2 /Oi /Ot /fp:fast
%   /GL on Windows MSVC). Combines safely with 'omp'. Adds compile time
%   but typically buys 1.5-3x runtime in tracking-heavy workloads. The
%   fast-math flags allow associative re-ordering of floating-point sums,
%   so reductions (BPM moments, etc.) may differ at the last few bits
%   from the standard build.

% verbose : echo all build command output
% =======

% If cleaning, deal with that now
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'clean')
      delete(sprintf('*.%s',mexext));
      if ispc
        delete('*.obj');
      else
        delete('*.o');
      end
    end
    if strcmp(varargin{iarg},'cleanall')
      delete(sprintf('*.%s',mexext));
      if ispc
        delete('*.obj')
        delete('g4track/*.lib');
      else
        delete('*.o');
        delete('g4track/*.a');
      end
    end
  end
end

% - Set defaults and parse input arguments
target='none';
randFunc='c';
useOMP=false;
useFast=false;
if nargin>0
  for iarg=1:nargin
    switch lower(varargin{iarg})
      case {'cpu','gpu','cpu-g4'}
        target=lower(varargin{iarg});
      case 'mlrand'
        randFunc='matlab';
      case 'omp'
        useOMP=true;
      case 'fast'
        useFast=true;
    end
  end
end

% Set compiler flags and perform initialization actions
FLAGS = '-DLUCETIA_DPREC -largeArrayDims';
LIBS = [] ;
if strcmp(randFunc,'matlab')
  FLAGS = [FLAGS ' -DLUCRETIA_MLRAND'] ;
end

if strcmp(target,'gpu')
  CC='mexcuda -dynamic';
elseif strcmp(target,'cpu-g4')
  CC='mex CC=c++';
else
  if ismac
%     CC='mex CC=clang++';
    CC='mex';
  else
    CC='mex CC=c++';
  end
end
% Verbose output?
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'verbose')
      FLAGS=[FLAGS ' -v'];
    end
  end
end
% Aggressive optimisation flags -- enables auto-vectorisation, link-time
% optimisation, loop unrolling, and the unsafe-but-usually-fine
% fast-math relaxations (associative reordering, denormals-as-zero,
% no-NaN-checking).  Works alongside 'omp'.  Skipped on cpu-g4 because
% G4 builds with C++ which has slightly different flag plumbing -- add
% by hand if you need it.
if useFast
  if strcmp(target,'gpu')
    error('Build:fast:gpuConflict', ...
          'The ''fast'' option is for the cpu/cpu-g4 targets only.')
  end
  if ispc
    fastFlags = ' COMPFLAGS=''$COMPFLAGS /O2 /Oi /Ot /fp:fast /GL''' ;
  else
    fastFlags = [' CFLAGS=''$CFLAGS -O3 -march=native -ffast-math' ...
                 ' -funroll-loops -flto'' ' ...
                 'LDFLAGS=''$LDFLAGS -flto'''] ;
  end
  FLAGS = [FLAGS fastFlags] ;
end

% OpenMP parallelisation of per-ray tracking loops.  See LucretiaCommon.c
% for which kernels are parallelised; currently Drift, QSOS (Quad/Sext/
% Octu/Plens/Solenoid), Mult, SBend, RF (LCAV/TCAV per-slice ray loop),
% BPM and INST.
if useOMP
  if strcmp(target,'gpu')
    error('Build:omp:gpuConflict', ...
          'The ''omp'' option is not supported with the gpu target.')
  end
  if ismac
    % MATLAB ships its own LLVM libomp under bin/<arch>/libomp.dylib
    % (install_name = @rpath/libomp.dylib).  If we link against
    % Homebrew's libomp at its absolute path, two libomps end up loaded
    % in the same process -- the mex sees the Homebrew path while MATLAB
    % has already loaded its own copy via @rpath.  The duplicate
    % runtime causes pthread_mutex_init to fail with EINVAL on first
    % parallel region.  Prefer MATLAB's bundled libomp when present
    % (R2024b+); link with -L pointing at bin/<arch> so the recorded
    % dependency is @rpath/libomp.dylib and dyld resolves it to the
    % already-loaded copy.  Falls back to Homebrew (with a warning)
    % only if MATLAB's libomp is missing.
    arch = computer('arch') ;                              % 'maca64' / 'maci64'
    matlabOmpLibDir = fullfile(matlabroot, 'bin', arch) ;
    matlabOmpLib    = fullfile(matlabOmpLibDir, 'libomp.dylib') ;
    matlabOmpInc    = fullfile(matlabroot, 'toolbox', 'eml', ...
                               'externalDependency', 'omp', arch, ...
                               'include') ;
    matlabOmpHdr    = fullfile(matlabOmpInc, 'omp.h') ;

    if exist(matlabOmpLib,'file')
      % Header preference: MATLAB's (Coder) > Homebrew's.  MATLAB's
      % libomp is ABI-compatible with both, so a Homebrew header is
      % a fine fallback.
      if exist(matlabOmpHdr,'file')
        ompIncFlag = sprintf('-I%s', matlabOmpInc) ;
      elseif exist('/opt/homebrew/opt/libomp/include/omp.h','file')
        ompIncFlag = '-I/opt/homebrew/opt/libomp/include' ;
      elseif exist('/usr/local/opt/libomp/include/omp.h','file')
        ompIncFlag = '-I/usr/local/opt/libomp/include' ;
      else
        error('Build:omp:headerMissing', ...
          ['OpenMP libomp.dylib found in MATLAB but omp.h is not.\n' ...
           'Either install MATLAB Coder (which bundles omp.h) or\n' ...
           'install Homebrew libomp for its header:\n' ...
           '    brew install libomp\n' ...
           'and re-run the build.'])
      end
      FLAGS = [FLAGS sprintf( ...
        [' CFLAGS=''$CFLAGS -Xpreprocessor -fopenmp %s''' ...
         ' LDFLAGS=''$LDFLAGS -L%s -lomp'''], ...
        ompIncFlag, matlabOmpLibDir)];
    else
      % MATLAB doesn't ship libomp -- fall back to Homebrew with a
      % warning.  This path is the source of the
      % `pthread_mutex_init: EINVAL` error if MATLAB later loads its
      % own libomp anyway, so warn explicitly.
      if exist('/opt/homebrew/opt/libomp','dir')
        ompPrefix = '/opt/homebrew/opt/libomp' ;
      elseif exist('/usr/local/opt/libomp','dir')
        ompPrefix = '/usr/local/opt/libomp' ;
      else
        error('Build:omp:libompMissing', ...
          ['OpenMP requested but libomp was not found in either\n' ...
           '    %s\n  or\n    /opt/homebrew/opt/libomp\n' ...
           'Install with:  brew install libomp'], matlabOmpLib)
      end
      warning('Build:omp:homebrewFallback', ...
        ['MATLAB does not ship libomp at %s; falling back to %s.\n' ...
         'If tracking aborts with `OMP: Error #179: ' ...
         'pthread_mutex_init failed`, the duplicate-runtime conflict ' ...
         'is the cause; an updated MATLAB (R2024b+) ships libomp.'], ...
        matlabOmpLib, ompPrefix) ;
      FLAGS = [FLAGS sprintf( ...
        [' CFLAGS=''$CFLAGS -Xpreprocessor -fopenmp -I%s/include''' ...
         ' LDFLAGS=''$LDFLAGS -L%s/lib -lomp'''], ...
        ompPrefix, ompPrefix)];
    end
  elseif ispc
    FLAGS = [FLAGS ' COMPFLAGS=''$COMPFLAGS /openmp'''];
  else  % assume Linux gcc/clang
    FLAGS = [FLAGS ...
      ' CFLAGS=''$CFLAGS -fopenmp'' LDFLAGS=''$LDFLAGS -fopenmp'''];
  end
end

if strcmp(target,'cpu-g4')
  [stat,G4LIBS] = system('geant4-config --libs');
  %G4LIBS='-L/var/geant4/bin/../lib64 -lG4Tree -lG4FR -lG4GMocren -lG4visHepRep -lG4RayTracer -lG4VRML -lG4vis_management -lG4modeling -lG4interfaces -lG4analysis -lG4error_propagation -lG4readout -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms -lG4global -lG4clhep -lG4zlib';
  if stat
    error('Error getting geant4 library flags- check ''geant4-config'' on search path')
  end
  [stat,G4FLAGS] = system('geant4-config --cflags');
  if stat
    error('Error getting geant4 compile flags- check ''geant4-config'' on search path')
  end
  G4FLAGS=strrep(G4FLAGS,sprintf('\n'),''); %#ok<SPRINTFN>
  G4LIBS=strrep(G4LIBS,sprintf('\n'),''); %#ok<SPRINTFN>
  G4FLAGS=strrep(G4FLAGS,sprintf('\r'),'');
  G4LIBS=strrep(G4LIBS,sprintf('\r'),'');
  FLAGS=[FLAGS ' -DLUCRETIA_G4TRACK' sprintf(' CXXFLAGS=''$CXXFLAGS %s'' CFLAGS=''$CFLAGS %s''',G4FLAGS,G4FLAGS)];
  % LIBS=[LIBS ' -Lg4track/ -lg4track -L/home/glen/xerces-c-3.0.1/lib -lxerces-c ' G4LIBS];
  LIBS=[LIBS ' -Lg4track/ -lg4track ' G4LIBS];
end

% Dependencies
dep.LucretiaCommon={'LucretiaCommon.h' 'LucretiaMatlab.h' 'LucretiaDictionary.h' 'LucretiaPhysics.h' 'LucretiaGlobalAccess.h' 'LucretiaVersionProto.h' 'LucretiaCuda.h'};
dep.LucretiaPhysics={'LucretiaCommon.h' 'LucretiaPhysics.h' 'LucretiaGlobalAccess.h' 'LucretiaVersionProto.h'};
dep.LucretiaMatlab={'LucretiaGlobalAccess.h' 'LucretiaVersionProto.h' 'LucretiaMatlab.h' 'LucretiaCuda.h'};
dep.LucretiaMatlabErrMsg={'LucretiaGlobalAccess.h'};
dep.GetRmats={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h'};
dep.GetTwiss={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h' 'LucretiaPhysics.h'};
dep.RmatAtoB={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h'};
dep.TrackThru={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h' 'LucretiaCuda.h'};
dep.VerifyLattice={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h'};
dep.SeedLucretiaRng={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaGlobalAccess.h' 'LucretiaCommon.h'};
% - target dependent dependencies
if strcmp(target,'cpu-g4')
  if ispc
    dep.TrackThru{end+1}='g4track/libg4track.lib';
    dep.LucretiaCommon{end+1}='g4track/libg4track.lib';
  else
    dep.TrackThru{end+1}='g4track/libg4track.a';
    dep.LucretiaCommon{end+1}='g4track/libg4track.a';
  end
end

% Form build & install lists
if strcmp(target,'gpu')
  blist={'TrackThru' 'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg'};
  ilist.exe={'TrackThru'};
  ilist.dir={'Tracking'};
else
  blist={'GetRmats' 'GetTwiss' 'RmatAtoB' 'TrackThru' 'VerifyLattice' 'SeedLucretiaRng' 'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg'};
  ilist.exe={'GetRmats' 'GetTwiss' 'RmatAtoB' 'TrackThru' 'VerifyLattice' 'SeedLucretiaRng'};
  ilist.dir={'RMatrix'  'Twiss'    'RMatrix'  'Tracking'  'LatticeVerification' 'Tracking'};
end
  
% Perform the build
anybuild=false;
if ~strcmp(target,'none')
  for ib=1:length(blist)
    doThisBuild=false;
    % Check if any dependenices need building first
    for idep=1:length(dep.(blist{ib}))
      if isempty(regexp(dep.(blist{ib}){idep},'\.','once')) && checkBuild(dep.(blist{ib}){idep},dep)
        doBuild(dep.(blist{ib}){idep},CC,FLAGS,LIBS,ilist,dep,target);
        doThisBuild=true;
        anybuild=true;
      end
    end
    % Build this item if needed
    if doThisBuild || checkBuild(blist{ib},dep)
      doBuild(blist{ib},CC,FLAGS,LIBS,ilist,dep,target);
      anybuild=true;
    end
  end
  if ~anybuild
    disp('Build up to date, nothing done.')
  end
end

% Install files
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'install')
      for il=1:length(ilist.exe)
        if exist(sprintf('%s.%s',ilist.exe{il},mexext),'file')
          if strcmp(target,'gpu') && strcmp(ilist.exe{il},'TrackThru')
            movefile(sprintf('%s.%s',ilist.exe{il},mexext),fullfile('..',ilist.dir{il},sprintf('%s_gpu.%s',ilist.exe{il},mexext)));
          elseif strcmp(target,'cpu-g4')
            movefile(sprintf('%s.%s',ilist.exe{il},mexext),fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)));
            copyfile(fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)),fullfile('..',ilist.dir{il},sprintf('%s_g4.%s',ilist.exe{il},mexext)))
          else
            movefile(sprintf('%s.%s',ilist.exe{il},mexext),fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)));
            copyfile(fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)),fullfile('..',ilist.dir{il},sprintf('%s_cpu.%s',ilist.exe{il},mexext)))
          end
        end
      end
    end
  end
end

% After a successful OMP build, print OMP_NUM_THREADS recommendations
% based on the detected CPU / NUMA topology of the current machine.
if useOMP && ~strcmp(target,'none')
  printOmpHints() ;
end

% =========================================================================
% perform build
function doBuild(bname,CC,FLAGS,LIBS,ilist,dep,target)
olist=[];
if ispc
  oext='obj';
else
  oext='o';
end
for idep=1:length(dep.(bname))
  if isempty(regexp(dep.(bname){idep},'\.','once'))
    olist=[olist sprintf('%s.%s',dep.(bname){idep},oext) ' '];
  end
end
fext='c';
if strcmp(target,'gpu')
  copyfile(sprintf('%s.c',bname),sprintf(sprintf('%s.cu',bname)));
  fext='cu';
end
if ismember(bname,ilist.exe)
  bcmd=[CC ' ' FLAGS sprintf(' %s.%s ',bname,fext) olist LIBS];
else
  bcmd=[CC ' ' FLAGS sprintf(' -c %s.%s ',bname,fext) olist LIBS];
end
try
  eval(bcmd);
catch ME
  fprintf('Error building: %s\n',bname)
  fprintf('(%s)\n',bcmd);
  if strcmp(target,'gpu')
    delete(sprintf('%s.cu',bname));
  end
  rethrow(ME)
end
if strcmp(target,'gpu')
  delete(sprintf('%s.cu',bname));
end

% return true if need to build
function dobuild=checkBuild(name,dep)
dobuild=false;
if ispc
  oext='obj';
else
  oext='o';
end
% Check source and header exists and add to
% build list if file not compiled or source/header newer than compiled file
if ~exist(sprintf('%s.c',name),'file')
  error('Missing dependenices for %s',name)
end
if ~exist(sprintf('%s.%s',name,oext),'file') && ~exist(fullfile('.',sprintf('%s.%s',name,mexext)),'file')
  dobuild=true;
else
  f=dir(sprintf('%s.%s',name,oext));
  if isempty(f)
    f=dir(sprintf('%s.%s',name,mexext));
  end
  odate=f.datenum;
  f=dir(sprintf('%s.c',name)); fdate(1)=f.datenum;
  for idep=1:length(dep.(name))
    if ~isempty(regexp(dep.(name){idep},'\.','once'))
      f=dir(dep.(name){idep}); fdate(end+1)=f.datenum;
    end
  end
  if any(fdate>odate)
    dobuild=true;
  end
end


% =========================================================================
% OMP hint printer -- detect CPU / NUMA topology and suggest the right
% OMP_NUM_THREADS, OMP_PROC_BIND, and numactl command for this machine.
function printOmpHints()
fprintf('\n') ;
fprintf('==========================================================\n') ;
fprintf(' OMP build complete -- recommended runtime settings\n') ;
fprintf('==========================================================\n') ;

if ismac
  % macOS: use physical cores, not logical (HT doubles the count but
  % not the memory bandwidth available to the OMP workers).
  [~, physStr] = system('sysctl -n hw.physicalcpu 2>/dev/null') ;
  nPhys = str2double(strtrim(physStr)) ;
  [~, logStr]  = system('sysctl -n hw.logicalcpu 2>/dev/null') ;
  nLog  = str2double(strtrim(logStr)) ;
  fprintf('Platform:   macOS  (%d physical cores, %d logical)\n', nPhys, nLog) ;
  fprintf('\nBest setting (stay on physical cores; HT rarely helps\n') ;
  fprintf('for memory-bandwidth-bound tracking):\n\n') ;
  fprintf('  export OMP_NUM_THREADS=%d\n', nPhys) ;
  fprintf('  export OMP_PROC_BIND=close\n') ;
  fprintf('  export OMP_PLACES=cores\n') ;

elseif isunix
  [~, lscpuOut] = system('lscpu 2>/dev/null') ;
  nSockets     = parseTopologyInt(lscpuOut, 'Socket\(s\)') ;
  coresPerSock = parseTopologyInt(lscpuOut, 'Core\(s\) per socket') ;
  nNuma        = parseTopologyInt(lscpuOut, 'NUMA node\(s\)') ;
  nCpuTotal    = parseTopologyInt(lscpuOut, '^CPU\(s\)') ;

  fprintf('Platform:   Linux\n') ;
  fprintf('Sockets:    %d  |  Cores/socket: %d  |  NUMA nodes: %d  |  Total CPUs (incl HT): %d\n', ...
          nSockets, coresPerSock, nNuma, nCpuTotal) ;

  if nNuma > 0 && nSockets > 0 && nNuma > nSockets
    % Sub-NUMA Clustering (SNC) -- most important case for large servers.
    sncFactor    = nNuma / nSockets ;
    coresPerNuma = round(coresPerSock / sncFactor) ;
    numaOneSocket = strjoin(arrayfun(@num2str, 0:sncFactor-1, ...
                                     'UniformOutput', false), ',') ;
    fprintf('\n*** Sub-NUMA Clustering (SNC=%d) detected ***\n', sncFactor) ;
    fprintf('Each socket is divided into %d NUMA domains of %d physical\n', ...
            sncFactor, coresPerNuma) ;
    fprintf('cores each.  Crossing domain boundaries costs 2x memory\n') ;
    fprintf('latency -- this is why using all %d cores can be SLOWER\n', ...
            nCpuTotal) ;
    fprintf('than using far fewer.\n') ;
    fprintf('\nStart here (one NUMA node, fastest for a single job):\n\n') ;
    fprintf('  export OMP_NUM_THREADS=%d\n', coresPerNuma) ;
    fprintf('  export OMP_PROC_BIND=close\n') ;
    fprintf('  export OMP_PLACES=cores\n') ;
    fprintf('  numactl --localalloc --cpunodebind=0 matlab\n') ;
    fprintf('\nScale to one full socket (%d cores) with interleaved memory:\n\n', ...
            coresPerSock) ;
    fprintf('  export OMP_NUM_THREADS=%d\n', coresPerSock) ;
    fprintf('  numactl --interleave=%s --cpunodebind=%s matlab\n', ...
            numaOneSocket, numaOneSocket) ;
    fprintf('\nBeyond one socket: adds NUMA penalty, rarely worth it\n') ;
    fprintf('unless numactl --interleave=all is used for ALL sockets.\n') ;

  elseif nSockets > 0 && coresPerSock > 0
    % Standard NUMA -- one memory domain per socket.
    fprintf('\nBest setting (physical cores on one socket, local memory):\n\n') ;
    fprintf('  export OMP_NUM_THREADS=%d\n', coresPerSock) ;
    fprintf('  export OMP_PROC_BIND=close\n') ;
    fprintf('  export OMP_PLACES=cores\n') ;
    fprintf('  numactl --localalloc --cpunodebind=0 matlab\n') ;
    if nSockets > 1
      numaAll = strjoin(arrayfun(@num2str, 0:nSockets-1, ...
                                 'UniformOutput', false), ',') ;
      fprintf('\nTo span all %d sockets with interleaved memory:\n\n', nSockets) ;
      fprintf('  export OMP_NUM_THREADS=%d\n', nSockets * coresPerSock) ;
      fprintf('  numactl --interleave=%s matlab\n', numaAll) ;
    end
  else
    fprintf('\nCould not determine core/NUMA topology; run:\n') ;
    fprintf('  lscpu | grep -E "Socket|Core|NUMA"\n') ;
    fprintf('and set OMP_NUM_THREADS to the physical-core count of one NUMA node.\n') ;
  end

  fprintf('\nTo install numactl if not already present:\n') ;
  fprintf('  sudo apt install numactl  (Ubuntu/Debian)\n') ;
  fprintf('  sudo yum install numactl  (RHEL/CentOS)\n') ;

else  % Windows
  fprintf('Platform:   Windows\n') ;
  fprintf('\nSet OMP_NUM_THREADS to the number of physical processor cores.\n') ;
  fprintf('Check: Task Manager > Performance > CPU > Cores.\n') ;
  fprintf('  set OMP_NUM_THREADS=<N>\n') ;
  fprintf('  matlab\n') ;
end

fprintf('\nTo verify at runtime:\n') ;
fprintf('  SeedLucretiaRng(42)  %% optional, for reproducible SR\n') ;
fprintf('  tic; I.track(1,1); toc\n') ;
fprintf('  %% Sweep OMP_NUM_THREADS = [1 4 8 N] to find the optimum.\n') ;
fprintf('==========================================================\n\n') ;
end


% =========================================================================
function n = parseTopologyInt( txt, pattern )
% Extract the first integer after `pattern:` from lscpu output.
n = 0 ;
tok = regexp(txt, [pattern '\s*:\s*([0-9]+)'], 'tokens', 'once') ;
if ~isempty(tok)
  n = str2double(tok{1}) ;
end
end

