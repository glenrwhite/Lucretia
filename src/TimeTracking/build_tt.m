function build_tt(varargin)
% BUILD_TT  Build the lucretia-tt standalone time-based 3D PIC tracker.
%
% lucretia-tt is a separate C++/CUDA executable that lives in
% Lucretia/external/lucretia-tt/.  It links against AMReX + ABLASTR (lifted
% from BLAST-LBNL).  This wrapper is the single MATLAB entry point for
% configuring and building it from the Lucretia source tree.
%
% Usage:
%   build_tt              % default target: 'cpu'
%   build_tt cpu          % single-node CPU OpenMP, no MPI (macOS-friendly)
%   build_tt cpu-mpi      % CPU OpenMP + MPI
%   build_tt gpu          % CUDA single-GPU (Linux only)
%   build_tt gpu-mpi      % CUDA + MPI multi-GPU (Linux only)
%
% Modifiers (any combination):
%   clean      - rm -rf build/ before configuring
%   debug      - CMAKE_BUILD_TYPE=Debug (default Release)
%   nofft      - LucretiaTT_FFT=OFF (skip FFT-based Poisson; for early bring-up)
%   noopmd     - LucretiaTT_OPENPMD=OFF (skip openPMD I/O; for early bring-up)
%   opmd-mpi   - LucretiaTT_OPENPMD_MPI=ON (use parallel HDF5; requires hdf5-mpi)
%                Default OFF: openPMD-api uses serial HDF5 and BeamIO MPI-gathers
%                to rank 0 before writing -- works on every platform without
%                hdf5-mpi installed. Set this for high-frequency / large dumps
%                on Linux production boxes where collective parallel I/O matters.
%   verbose    - cmake --verbose
%
% Examples:
%   build_tt cpu nofft noopmd          % minimal CPU build, fastest path to a binary
%   build_tt cpu-mpi clean             % full CPU+MPI rebuild from scratch
%   build_tt gpu-mpi                   % production target on the deployment box
%   build_tt gpu-mpi opmd-mpi clean    % production w/ collective parallel HDF5
%
% Also invokable as `build tt cpu`, `build tt clean`, etc. via build.m.

target   = 'cpu' ;
doClean  = false ;
debug    = false ;
nofft    = false ;
noopmd   = false ;
opmdMPI  = false ;
verbose  = false ;
for iarg = 1:nargin
  switch lower(varargin{iarg})
    case {'cpu','cpu-mpi','gpu','gpu-mpi'}
      target = lower(varargin{iarg}) ;
    case 'clean',    doClean  = true ;
    case 'debug',    debug    = true ;
    case 'nofft',    nofft    = true ;
    case 'noopmd',   noopmd   = true ;
    case {'opmd-mpi','openpmd-mpi'}
                     opmdMPI  = true ;
    case 'verbose',  verbose  = true ;
    otherwise
      warning('build_tt:unknownArg', 'Ignoring unknown argument: %s', ...
              varargin{iarg}) ;
  end
end

% Resolve paths.  This file lives at src/TimeTracking/build_tt.m; lucretia-tt
% source is at external/lucretia-tt/ (two levels up + over).
% Per-target build dirs and binary names so cpu and gpu builds can coexist:
%   cpu      -> build/         lucretia-tt
%   cpu-mpi  -> build_mpi/     lucretia-tt_mpi
%   gpu      -> build_gpu/     lucretia-tt_gpu
%   gpu-mpi  -> build_gpu_mpi/ lucretia-tt_gpu_mpi
thisDir   = fileparts(mfilename('fullpath')) ;
projRoot  = fileparts(fileparts(thisDir)) ;                        % Lucretia/
ttRoot    = fullfile(projRoot, 'external', 'lucretia-tt') ;
isMPI     = endsWith(target, '-mpi') ;
isGPU     = startsWith(target, 'gpu') ;
suffix    = '' ;
if isGPU, suffix = [suffix '_gpu'] ; end
if isMPI, suffix = [suffix '_mpi'] ; end
if isempty(suffix)
  buildDir = fullfile(ttRoot, 'build') ;
  binName  = 'lucretia-tt' ;
else
  buildDir = fullfile(ttRoot, ['build' suffix]) ;
  binName  = ['lucretia-tt' suffix] ;
end

if ~exist(ttRoot, 'dir')
  error('build_tt:missing', 'lucretia-tt source not found at %s', ttRoot) ;
end

% Map target -> CMake compute/MPI options
switch target
  case 'cpu',     opts = '-DLucretiaTT_COMPUTE=OMP  -DLucretiaTT_MPI=OFF' ;
  case 'cpu-mpi', opts = '-DLucretiaTT_COMPUTE=OMP  -DLucretiaTT_MPI=ON'  ;
  case 'gpu',     opts = '-DLucretiaTT_COMPUTE=CUDA -DLucretiaTT_MPI=OFF' ;
  case 'gpu-mpi', opts = '-DLucretiaTT_COMPUTE=CUDA -DLucretiaTT_MPI=ON'  ;
end

opts = [opts sprintf(' -DCMAKE_BUILD_TYPE=%s',       ternary(debug,'Debug','Release'))] ;
opts = [opts sprintf(' -DLucretiaTT_FFT=%s',         ternary(nofft,  'OFF','ON'))] ;
opts = [opts sprintf(' -DLucretiaTT_OPENPMD=%s',     ternary(noopmd, 'OFF','ON'))] ;
opts = [opts sprintf(' -DLucretiaTT_OPENPMD_MPI=%s', ternary(opmdMPI,'ON', 'OFF'))] ;

if opmdMPI && ~isMPI
  warning('build_tt:opmdMpiNoMpi', ...
    'opmd-mpi requested but target is %s -- collective parallel I/O needs an MPI build. Add cpu-mpi or gpu-mpi.', target) ;
end

% macOS: point CMake at Homebrew libomp (AppleClang ships no native OpenMP).
% On Linux gcc/clang find OpenMP without hints.
if ismac
  if exist('/opt/homebrew/opt/libomp', 'dir')
    opts = [opts ' -DOpenMP_ROOT=/opt/homebrew/opt/libomp'] ;
  elseif exist('/usr/local/opt/libomp', 'dir')
    opts = [opts ' -DOpenMP_ROOT=/usr/local/opt/libomp'] ;
  else
    error('build_tt:noLibomp', ...
      ['libomp not found in /opt/homebrew/opt/libomp or /usr/local/opt/libomp.\n' ...
       'Install with:  brew install libomp']) ;
  end
end

% macOS + MPI: route compilers through Homebrew's mpicxx wrapper so that
% MPI is found without hand-tuning include/lib paths.
if ismac && isMPI
  if exist('/opt/homebrew/bin/mpicxx', 'file')
    opts = [opts ' -DCMAKE_C_COMPILER=/opt/homebrew/bin/mpicc' ...
                 ' -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/mpicxx'] ;
  elseif exist('/usr/local/bin/mpicxx', 'file')
    opts = [opts ' -DCMAKE_C_COMPILER=/usr/local/bin/mpicc' ...
                 ' -DCMAKE_CXX_COMPILER=/usr/local/bin/mpicxx'] ;
  else
    error('build_tt:noMPIWrappers', ...
      ['MPI compiler wrappers (mpicxx) not found in /opt/homebrew/bin or /usr/local/bin.\n' ...
       'Install with:  brew install open-mpi']) ;
  end
end

% Locate cmake.  Homebrew puts it in /opt/homebrew/bin on Apple Silicon.
cmakeBin = locateBin('cmake', ...
  {'/opt/homebrew/bin/cmake', '/usr/local/bin/cmake'}) ;

% On Linux, MATLAB ships its own libstdc++/libcurl/libjsoncpp under
% sys/os/glnxa64 and prepends them to LD_LIBRARY_PATH, breaking system
% cmake/gcc which need the host's newer libstdc++. Strip MATLAB-rooted
% entries from LD_LIBRARY_PATH for spawned subprocesses; keep everything
% else (CUDA paths, custom HDF5, etc.) intact. No-op on Mac.
envP = timetracking.linux_matlab_env_prefix() ;

% Clean
if doClean && exist(buildDir, 'dir')
  fprintf('build_tt: removing %s\n', buildDir) ;
  rmdir(buildDir, 's') ;
end

% Configure
fprintf('build_tt: configuring (%s)\n', target) ;
cfgCmd = sprintf('%s"%s" -S "%s" -B "%s" %s', envP, cmakeBin, ttRoot, buildDir, opts) ;
fprintf('  $ %s\n', cfgCmd) ;
stat = system(cfgCmd) ;
if stat ~= 0
  error('build_tt:configure', 'cmake configure failed (status %d)', stat) ;
end

% Build
fprintf('\nbuild_tt: compiling (this fetches AMReX+ABLASTR on first run; ~15-25 min)\n') ;
buildCmd = sprintf('%s"%s" --build "%s" -j', envP, cmakeBin, buildDir) ;
if verbose
  buildCmd = [buildCmd ' --verbose'] ;
end
fprintf('  $ %s\n', buildCmd) ;
stat = system(buildCmd) ;
if stat ~= 0
  error('build_tt:build', 'cmake --build failed (status %d)', stat) ;
end

% Stage binary into src/TimeTracking/ for which()-based discovery from MATLAB.
% Non-MPI build -> lucretia-tt;  MPI build -> lucretia-tt_mpi (so the two
% can coexist and TimeTrack can pick by mpi_nranks).
binSrc = fullfile(buildDir, 'bin', 'lucretia-tt') ;
binDst = fullfile(thisDir, binName) ;
if exist(binSrc, 'file')
  copyfile(binSrc, binDst) ;
  fileattrib(binDst, '+x') ;
  fprintf('\nbuild_tt: built %s\n', binDst) ;
else
  warning('build_tt:noBinary', ...
    'cmake build returned 0 but binary not at %s', binSrc) ;
end

end


function bin = locateBin(name, candidates)
% Find an executable: prefer PATH (`command -v`), fall back to candidates.
[stat, out] = system(sprintf('command -v %s', name)) ;
if stat == 0
  bin = strtrim(out) ;
  return ;
end
for ic = 1:numel(candidates)
  if exist(candidates{ic}, 'file')
    bin = candidates{ic} ;
    return ;
  end
end
error('build_tt:notFound', ...
      '%s not found on PATH or in candidates: %s', ...
      name, strjoin(candidates, ', ')) ;
end


function v = ternary(cond, a, b)
if cond, v = a ; else, v = b ; end
end
