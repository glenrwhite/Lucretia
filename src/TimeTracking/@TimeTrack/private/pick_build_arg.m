function arg = pick_build_arg(target, nranks)
% Inverse of pick_binary_name: (target, nranks) -> the build_tt argument
% that produces the corresponding binary. Used in error messages so the
% user gets a copy-paste fix.
%
%   ('cpu', 1) -> 'cpu'
%   ('cpu', N) -> 'cpu-mpi'
%   ('gpu', 1) -> 'gpu'
%   ('gpu', N) -> 'gpu-mpi'

arg = target;
if nranks > 1
    arg = [arg '-mpi'];
end
end
