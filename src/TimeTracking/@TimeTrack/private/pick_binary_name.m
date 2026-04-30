function name = pick_binary_name(target, nranks)
% Stage-name resolver for the lucretia-tt binary, matching build_tt.m's
% staging convention. (target, nranks) -> filename under src/TimeTracking/.
%
%   ('cpu', 1) -> 'lucretia-tt'
%   ('cpu', N) -> 'lucretia-tt_mpi'
%   ('gpu', 1) -> 'lucretia-tt_gpu'
%   ('gpu', N) -> 'lucretia-tt_gpu_mpi'

name = 'lucretia-tt';
if strcmp(target, 'gpu')
    name = [name '_gpu'];
end
if nranks > 1
    name = [name '_mpi'];
end
end
