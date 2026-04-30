function envPrefix = linux_matlab_env_prefix()
% LINUX_MATLAB_ENV_PREFIX  Shell prefix that strips MATLAB's bundled libs.
%
% MATLAB on Linux prepends its own /sys/os/glnxa64/ to LD_LIBRARY_PATH so
% it can ship its own libstdc++ / libcurl / libjsoncpp / etc. for internal
% use. Those bundled libs are usually OLDER than what the host distro
% provides, which breaks anything system-built that we spawn via system():
%   * cmake (linked against the host's newer libstdc++)
%   * gcc / make / nvcc (same)
%   * lucretia-tt itself (built with system libstdc++)
%
% This function returns a shell prefix string that, when prepended to the
% command, removes only MATLAB-rooted entries from LD_LIBRARY_PATH while
% keeping every other entry (e.g. CUDA, Spack, custom HDF5) intact.
%
% On macOS or non-MATLAB environments where this doesn't apply, returns
% an empty string -- safe to always prepend.

envPrefix = '';
if ~isunix || ismac
    return
end

mlRoot = matlabroot();
oldPath = getenv('LD_LIBRARY_PATH');
if isempty(oldPath) || ~contains(oldPath, mlRoot)
    return
end

parts = strsplit(oldPath, ':');
keep  = parts(~startsWith(parts, mlRoot));
newPath = strjoin(keep, ':');
envPrefix = sprintf('LD_LIBRARY_PATH="%s" ', newPath);
end
