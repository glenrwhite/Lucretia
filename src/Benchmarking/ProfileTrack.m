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
%   2. A C-level sampler attached to the matlab process during the run:
%        macOS:  the built-in `sample` tool (call-tree text report).
%        Linux:  `perf record` + `perf report` (requires
%                linux-tools-$(uname -r) and
%                kernel.perf_event_paranoid <= 1).
%      Shows which Lucretia C functions consume the most time and whether
%      the bottleneck is in serial collective stages (CSR/LSC binning,
%      wakefield convolution) vs the OpenMP per-ray loops.
%
% Both reports land in `outputDir` (default: a fresh temp directory).
% The function returns a struct with paths and a brief textual summary.
%
% Optional name/value pairs:
%
%   'cSampleSeconds'   (60)   Duration of the C-level sampler.  60 s
%                             gives ~6000 macOS / ~6000 Linux samples --
%                             enough for good statistics.  Set lower for
%                             short-running tests.
%   'cSampleDelaySec'  (5)    Seconds to wait before starting the sampler,
%                             skipping MATLAB-side lattice load.
%   'perfFreqHz'       (99)   Linux only: perf sampling frequency in Hz.
%                             99 Hz avoids lockstep with typical 100 Hz
%                             kernel timers and adds negligible overhead.
%   'outputDir'        ([])   Directory for reports (default: tempdir).
%   'matlabReport'     (true) Generate the MATLAB HTML profile report.
%   'skipCSample'      (false) Skip C-level profiling entirely (useful
%                              when neither `sample` nor `perf` is
%                              available, or for a quick MATLAB-only run).
%
% Example workflow:
%
%   load injector
%   out = ProfileTrack(@() I.track(1,1));
%   web(out.matlabReportIndex);             % browse MATLAB profile
%   edit(out.cSampleReportPath);            % view C-level call tree
%
%   % On a multi-core / NUMA Linux server, also run:
%   out = ProfileTrack(@() I.track(1,1), 'cSampleSeconds', 120);
%
% Quick triage in the C-level report:
%   - Search for "TrackBunchThru" -- highest counts = dominant tracker.
%   - "ProcLSC", "BinRays", "ConvolveSRWFWithBeam" = serial stages
%     that cap OMP scaling regardless of thread count.
%   - "__kmp_wait" / "futex" with high counts = threads blocked at OMP
%     barriers, often a sign of too many threads for the work size or
%     NUMA memory contention.
%   - "GetDatabaseParameters" / "GetElemNumericPar" high counts = MATLAB
%     <-> C dispatch overhead (many short elements).
%
% Linux / NUMA scaling note:
%   If performance on a 144-core server is WORSE than on 10 cores, the
%   usual suspects are:
%     1. OMP thread overhead: each `parallel for` forks N threads.  With
%        N=144 and short per-element work, the fork overhead dominates.
%        Fix: OMP_NUM_THREADS=<cores per socket> (not total cores).
%     2. NUMA: thread 0 allocated the bunch arrays on socket 0; thread 100
%        running on socket 1 pays 2x the memory latency for every read.
%        Fix: numactl --localalloc --cpunodebind=0 matlab
%     3. OMP_PROC_BIND: threads migrating between cores thrash L1/L2.
%        Fix: OMP_PROC_BIND=close OMP_PLACES=cores
%
% Requirements:
%   macOS  -- `sample` (part of Developer Tools, on default PATH).
%   Linux  -- `perf` from linux-tools-$(uname -r):
%               sudo apt install linux-tools-common linux-tools-$(uname -r)
%               sudo sysctl -w kernel.perf_event_paranoid=1
%
% See also: profile, profview, profsave.

p = inputParser ;
p.addParameter('cSampleSeconds',  60) ;
p.addParameter('cSampleDelaySec', 5)  ;
p.addParameter('perfFreqHz',      99) ;
p.addParameter('outputDir',       []) ;
p.addParameter('matlabReport',    true) ;
p.addParameter('skipCSample',     false) ;
p.parse(varargin{:}) ;
opt = p.Results ;

if ~isa(testFcn, 'function_handle')
  error('ProfileTrack:badInput', ...
        'First argument must be a no-argument function handle, e.g. @() I.track(1,1)') ;
end

% ---- output directory ----
if isempty(opt.outputDir)
  outDir = fullfile(tempdir, sprintf('lucretia_profile_%s', ...
                                     char(datetime('now','Format','yyyyMMdd_HHmmss')))) ;
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

% ---- detect platform and select C-level sampler ----
isLinux = isunix && ~ismac ;
sampleStarted = false ;
samplePid     = '' ;

if ~opt.skipCSample
  if ismac
    sampleStarted = startMacSampler(opt, outDir, cSampleReportPath) ;
    if sampleStarted
      % Read PID from the side file the macOS launcher wrote.
      pidFile = fullfile(outDir, 'sample.pid') ;
      try
        samplePid = strtrim(fileread(pidFile)) ;
      catch
      end
    end
  elseif isLinux
    [sampleStarted, samplePid] = startLinuxPerf(opt, outDir, cSampleReportPath) ;
  else
    warning('ProfileTrack:notSupported', ...
      'C-level profiling is not implemented for this platform (Windows).  Use VTune or AMD uProf instead.') ;
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
  if sampleStarted && ~isempty(samplePid)
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

% ---- wait for C-level sampler ----
if sampleStarted
  fprintf('Waiting for C-level sampler to finish...\n') ;
  expected = max(opt.cSampleSeconds + opt.cSampleDelaySec + 30, ...
                 ceil(out.wallSeconds) + 10) ;
  waited = 0 ;
  while waited < expected
    if ~isempty(samplePid)
      alive = system(sprintf('kill -0 %s 2>/dev/null', samplePid)) == 0 ;
    else
      alive = false ;
    end
    if ~alive, break ; end
    pause(1) ; waited = waited + 1 ;
  end
  if exist(cSampleReportPath, 'file')
    di = dir(cSampleReportPath) ;
    fprintf('C-level report:    %s  (%d bytes)\n', cSampleReportPath, di.bytes) ;
  else
    fprintf('C-level report:    *** sampler did not produce a file ***\n') ;
    if isLinux
      fprintf('  -> Check perf is installed and perf_event_paranoid <= 1:\n') ;
      fprintf('     sudo apt install linux-tools-common linux-tools-$(uname -r)\n') ;
      fprintf('     sudo sysctl -w kernel.perf_event_paranoid=1\n') ;
    end
  end
end

% ---- NUMA hint for Linux ----
if isLinux
  printNumaHints() ;
end

fprintf('\nDone.\n') ;
fprintf('  Open MATLAB report:  web(''%s'')\n', out.matlabReportIndex) ;
if sampleStarted && exist(cSampleReportPath, 'file')
  fprintf('  View C-level report: edit(''%s'')\n', out.cSampleReportPath) ;
  fprintf('\nQuick triage tips:\n') ;
  fprintf('  - "TrackBunchThru*" counts: which tracker dominates.\n') ;
  fprintf('  - "ProcLSC" / "BinRays" / "ConvolveSRWFWithBeam": serial ceiling.\n') ;
  fprintf('  - "GetDatabaseParameters": MATLAB<->C dispatch overhead.\n') ;
  if isLinux
    fprintf('  - "futex" / "sched_yield": thread blocking -- consider\n') ;
    fprintf('    reducing OMP_NUM_THREADS to cores-per-socket.\n') ;
  else
    fprintf('  - "__kmp_wait" / "kmp_flag_64::wait": OMP thread barrier waits.\n') ;
  end
end

end


% =========================================================================
function started = startMacSampler( opt, outDir, reportPath )
started = false ;
matPid    = feature('getpid') ;
delaySec  = max(0, opt.cSampleDelaySec) ;
durSec    = max(1, opt.cSampleSeconds) ;
pidFile   = fullfile(outDir, 'sample.pid') ;
cmd = sprintf( ...
  '( sleep %d && sample %d %d -file %s -mayDie ) & echo $! > %s', ...
  delaySec, matPid, durSec, escapeShell(reportPath), escapeShell(pidFile)) ;
[s, msg] = system(cmd) ;
if s ~= 0
  warning('ProfileTrack:sampleFailed', 'Failed to launch `sample`:\n%s', msg) ;
  return
end
started = true ;
try
  pid = strtrim(fileread(pidFile)) ;
catch
  pid = '?' ;
end
fprintf('C-level sampler:   sample %d %ds (PID %s, delay %ds)\n', ...
        matPid, durSec, pid, delaySec) ;
fprintf('                   -> %s\n', reportPath) ;
end


% =========================================================================
function [started, bgPid] = startLinuxPerf( opt, outDir, reportPath )
started = false ; bgPid = '' ;
% Check perf is available.
[rc, ~] = system('which perf 2>/dev/null') ;
if rc ~= 0
  warning('ProfileTrack:noperf', ...
    ['`perf` not found.  Install it with:\n' ...
     '  sudo apt install linux-tools-common linux-tools-$(uname -r)\n' ...
     'and set:\n' ...
     '  sudo sysctl -w kernel.perf_event_paranoid=1']) ;
  return
end
% Check paranoid setting.
[~, paranoidStr] = system('cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null') ;
paranoid = str2double(strtrim(paranoidStr)) ;
if isnan(paranoid) || paranoid > 1
  warning('ProfileTrack:perfParanoid', ...
    ['perf_event_paranoid=%s -- perf may not be able to sample the process.\n' ...
     'Run:  sudo sysctl -w kernel.perf_event_paranoid=1'], strtrim(paranoidStr)) ;
end

% Detect whether hardware PMU cycle counters are available.
% On VMs / containers the hypervisor often does not expose the CPU's
% hardware PMU, so `perf record` (which samples on cycles by default)
% silently records nothing.  Fall back to the `cpu-clock` software
% timer, which is always available and measures wall time per sample
% slot -- actually MORE useful for finding tracking bottlenecks.
[rc2, ~] = system('perf stat -e cycles echo x 2>&1 | grep -q "<not supported>"') ;
hwPmuAvail = (rc2 ~= 0) ;   % grep returns 0 if the string was found
if hwPmuAvail
  eventFlag = '' ;           % default: hardware cycles
  envNote   = 'hardware PMU' ;
else
  eventFlag = '-e cpu-clock' ;   % software fallback for VMs
  envNote   = 'software cpu-clock (VM/container: hardware PMU not available)' ;
end

matPid   = feature('getpid') ;
delaySec = max(0, opt.cSampleDelaySec) ;
durSec   = max(1, opt.cSampleSeconds) ;
freqHz   = max(1, opt.perfFreqHz) ;
pidFile  = fullfile(outDir, 'perf.pid') ;
dataFile = fullfile(outDir, 'perf.data') ;

% Build the record + report chain.
% `perf record -p PID -- sleep DURATION` records the target PID for
% exactly DURATION seconds (the `sleep` child is the recording timer).
recordCmd = sprintf('perf record %s -F %d -g -p %d -o %s -- sleep %d', ...
                    eventFlag, freqHz, matPid, escapeShell(dataFile), durSec) ;
% Flat report sorted by self-time -- the most useful view for finding
% hot tracking kernels.
reportCmd = sprintf( ...
  'perf report --no-pager --stdio --no-children --call-graph flat,0.001 -i %s > %s 2>&1', ...
  escapeShell(dataFile), escapeShell(reportPath)) ;
bgCmd = sprintf('( sleep %d && %s && %s ) & echo $! > %s', ...
                delaySec, recordCmd, reportCmd, escapeShell(pidFile)) ;

[s, msg] = system(bgCmd) ;
if s ~= 0
  warning('ProfileTrack:perfFailed', 'Failed to launch perf:\n%s', msg) ;
  return
end
started = true ;
try
  bgPid = strtrim(fileread(pidFile)) ;
catch
  bgPid = '?' ;
end
fprintf('C-level sampler:   perf record %s -F %d -p %d for %ds (PID %s, delay %ds)\n', ...
        eventFlag, freqHz, matPid, durSec, bgPid, delaySec) ;
fprintf('                   (%s)\n', envNote) ;
fprintf('                   -> %s\n', reportPath) ;
end


% =========================================================================
function printNumaHints()
% Print NUMA topology so it lands in the session log next to the profile.
fprintf('\n--- NUMA / CPU topology ---\n') ;
[~, out] = system('lscpu 2>/dev/null | grep -E "^CPU.s.|^Thread|^Core|^Socket|NUMA node"') ;
if ~isempty(strtrim(out))
  fprintf('%s', out) ;
end
[~, out] = system('numactl --hardware 2>/dev/null | head -12') ;
if ~isempty(strtrim(out))
  fprintf('%s', out) ;
end
nThreads = getenv('OMP_NUM_THREADS') ;
if isempty(nThreads), nThreads = '(unset -- defaulting to all cores)' ; end
fprintf('OMP_NUM_THREADS:   %s\n', nThreads) ;
nBind = getenv('OMP_PROC_BIND') ;
if isempty(nBind), nBind = '(unset)' ; end
fprintf('OMP_PROC_BIND:     %s\n', nBind) ;
fprintf('---------------------------\n') ;
end


% =========================================================================
function s = escapeShell( raw )
s = ['"' strrep(raw, '"', '\"') '"'] ;
end
