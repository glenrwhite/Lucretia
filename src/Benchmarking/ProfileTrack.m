function out = ProfileTrack( testFcn, varargin )
% PROFILETRACK  Profile a Lucretia tracking call at MATLAB and C levels.
%
%   out = ProfileTrack(@() I.track(1,1))
%   out = ProfileTrack(@() I.track(1,1), 'name', value, ...)
%
% Runs `testFcn()` (a no-argument function handle) under both:
%
%   1. The MATLAB profiler (profile on / profile off) -- shows time per
%      MATLAB function, including TrackThru / GetTwiss as a single
%      opaque blob each.  Useful for finding hot spots in the
%      MATLAB-side wrappers, lattice loops, BPM data unpacking, etc.
%
%   2. The macOS `sample` tool, attached to the matlab process during
%      the run -- shows C-level call stacks inside the mex files.
%      Useful for finding which Lucretia C functions consume the
%      most time, and whether time is spent in serial sections (CSR
%      binning, wakefield convolution) vs the OpenMP per-ray loops.
%
% Both reports land in `outputDir` (default: a fresh temp directory).
% The function returns a struct with paths and a brief textual summary.
%
% Optional name/value pairs:
%
%   'cSampleSeconds'   (60)   How long to attach the C-level sampler
%                             for.  Sampling captures call stacks at
%                             ~10 ms intervals; 60 s is enough for a
%                             reasonable statistical picture and adds
%                             very little overhead to the run.  If
%                             your test is shorter than this, set the
%                             value to slightly less than the expected
%                             runtime.
%   'cSampleDelaySec'  (5)    Delay before starting the C-level
%                             sampler, to skip the initial MATLAB-side
%                             setup.  Increase if your test does heavy
%                             prep work before tracking begins.
%   'outputDir'        ([])   Directory to save the reports.  Default
%                             is a fresh subdirectory under the system
%                             temp directory.
%   'matlabReport'     (true) Generate the MATLAB HTML profile report
%                             via profsave.  Open
%                             out.matlabReportIndex in a browser to
%                             explore.
%   'skipCSample'      (false) Skip the C-level sampler entirely.
%                              Useful in environments without `sample`
%                              on the PATH (non-macOS, sandboxed shells,
%                              etc.).
%
% Example workflow:
%
%   load injector
%   out = ProfileTrack(@() I.track(1,1));
%   web(out.matlabReportIndex);            % browse MATLAB profile
%   edit(out.cSampleReportPath);           % view C-level call tree
%
% After looking at the reports, the typical questions are:
%   - "Where do the unaccounted seconds go?" -> compare wall time vs
%     sum-of-mex time in the MATLAB report; the difference is
%     MATLAB-side overhead (loops, struct reshaping, etc.).
%   - "Is the OMP scaling limited by serial sections?" -> in the C
%     sample report, look for time in CSR/LSC functions
%     (ProcLSC, BinRays, ConvolveSRWFWithBeam) and in
%     TrackBunchThruRF's wakefield-prep code -- those run serially.
%   - "Which tracker dominates?" -> grep the C report for
%     TrackBunchThru* function names; sample counts are proportional
%     to time.
%
% Requirements:
%
%   - macOS for the C-level sampler (`sample` is a built-in tool).
%     On Linux you can substitute `perf record -g -p $PID sleep 60`
%     and post-process with `perf report` -- the MATLAB-side profile
%     still works without modification.
%
% See also: profile, profview, profsave.

p = inputParser ;
p.addParameter('cSampleSeconds',  60) ;
p.addParameter('cSampleDelaySec', 5)  ;
p.addParameter('outputDir',       []) ;
p.addParameter('matlabReport',    true) ;
p.addParameter('skipCSample',     false) ;
p.parse(varargin{:}) ;
opt = p.Results ;

if ~isa(testFcn, 'function_handle')
  error('ProfileTrack:badInput', ...
        'First argument must be a no-argument function handle, e.g. @() I.track(1,1)') ;
end

% Output directory
if isempty(opt.outputDir)
  outDir = fullfile(tempdir, sprintf('lucretia_profile_%s', ...
                                     datestr(now,'yyyymmdd_HHMMSS'))) ;
else
  outDir = opt.outputDir ;
end
if ~exist(outDir, 'dir'), mkdir(outDir) ; end

cSampleReportPath = fullfile(outDir, 'c_sample_report.txt') ;
matlabReportDir   = fullfile(outDir, 'matlab_profile') ;

out = struct() ;
out.outputDir          = outDir ;
out.cSampleReportPath  = cSampleReportPath ;
out.matlabReportDir    = matlabReportDir ;
out.matlabReportIndex  = fullfile(matlabReportDir, 'file0.html') ;

fprintf('=== ProfileTrack ===\n') ;
fprintf('Output directory:  %s\n', outDir) ;

% ---- launch the C-level sampler in the background ----
% `sample` runs as a subprocess; it grabs the matlab PID via feature.
sampleStarted = false ;
samplePid     = '' ;
if ~opt.skipCSample
  if ~ismac
    warning('ProfileTrack:notMac', ...
      'C-level sampler uses the macOS `sample` tool; skipping on non-macOS.  Use perf or VTune instead.') ;
  else
    matPid = feature('getpid') ;
    delaySec = max(0, opt.cSampleDelaySec) ;
    durSec   = max(1, opt.cSampleSeconds) ;
    % `sample` invocation:
    %   - delay D seconds before starting (skip MATLAB startup / lattice load)
    %   - then sample for N seconds at default 10 ms intervals
    %   - write the call-tree text report to our file
    %   - run in background; capture sample's PID so we can wait on it
    samplePidFile = fullfile(outDir, 'sample.pid') ;
    sampleCmd = sprintf( ...
      '( sleep %d && sample %d %d -file %s -mayDie ) & echo $! > %s', ...
      delaySec, matPid, durSec, ...
      escapeForShell(cSampleReportPath), ...
      escapeForShell(samplePidFile)) ;
    [s, msg] = system(sampleCmd) ;
    if s ~= 0
      warning('ProfileTrack:sampleStartFailed', ...
        'Failed to launch `sample`:\n%s', msg) ;
    else
      try
        samplePid = strtrim(fileread(samplePidFile)) ;
        sampleStarted = true ;
        fprintf('C-level sampler:   sample %d %d (PID %s, delay %ds)\n', ...
                matPid, durSec, samplePid, delaySec) ;
        fprintf('                     -> %s\n', cSampleReportPath) ;
      catch
        warning('ProfileTrack:samplePidUnknown', ...
          'Could not read sampler PID from %s', samplePidFile) ;
      end
    end
  end
end

% ---- MATLAB profiler ----
profile clear ;
profile on -history -timer real ;
fprintf('MATLAB profiler:   on (real-time clock)\n') ;
fprintf('Running test...\n') ;

t0 = tic ;
try
  testFcn() ;
catch ME
  profile off ;
  if sampleStarted
    system(sprintf('kill -TERM %s 2>/dev/null', samplePid)) ;
  end
  rethrow(ME) ;
end
out.wallSeconds = toc(t0) ;
profile off ;

fprintf('Wall time:         %.2f s\n', out.wallSeconds) ;

% ---- MATLAB profile report ----
if opt.matlabReport
  if ~exist(matlabReportDir, 'dir'), mkdir(matlabReportDir) ; end
  profsave(profile('info'), matlabReportDir) ;
  fprintf('MATLAB report:     %s\n', out.matlabReportIndex) ;
end

% ---- wait for the sampler to complete ----
if sampleStarted
  fprintf('Waiting for `sample` to finish writing %s...\n', ...
          cSampleReportPath) ;
  % Poll for the sampler's exit (the file appears at the end).
  expected = max(opt.cSampleSeconds + opt.cSampleDelaySec + 5, ...
                 ceil(out.wallSeconds) + 5) ;
  waited = 0 ;
  while waited < expected
    pidStillAlive = system(sprintf('kill -0 %s 2>/dev/null', samplePid)) == 0 ;
    if ~pidStillAlive, break ; end
    pause(1) ;
    waited = waited + 1 ;
  end
  if exist(cSampleReportPath, 'file')
    di = dir(cSampleReportPath) ;
    fprintf('C-level report:    %s  (%d bytes)\n', ...
            cSampleReportPath, di.bytes) ;
  else
    fprintf('C-level report:    *** sampler did not write a file ***\n') ;
  end
end

fprintf('\nDone.\n') ;
fprintf('  Open MATLAB report:  web(''%s'')\n', out.matlabReportIndex) ;
if sampleStarted
  fprintf('  View C-level report: edit(''%s'')\n', out.cSampleReportPath) ;
  fprintf('\nQuick triage tips:\n') ;
  fprintf('  - Search the C report for "TrackBunchThru" -- the highest\n') ;
  fprintf('    sample counts identify which tracker dominates.\n') ;
  fprintf('  - "ProcLSC", "BinRays", "ConvolveSRWFWithBeam" indicate\n') ;
  fprintf('    serial collective stages limiting OMP scaling.\n') ;
  fprintf('  - "GetDatabaseParameters" / "GetElemNumericPar" with high\n') ;
  fprintf('    counts means MATLAB<->C dispatch overhead.\n') ;
end

end


function s = escapeForShell( raw )
% Naive shell-quoting for paths with spaces.  Adequate for our use.
s = ['"' strrep(raw, '"', '\"') '"'] ;
end
