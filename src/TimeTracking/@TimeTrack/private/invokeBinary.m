function [status, log] = invokeBinary(obj, in_file)
% Run the lucretia-tt binary on the given input file inside work_dir,
% capturing combined stdout/stderr.

cmd = sprintf('cd "%s" && "%s" "%s" 2>&1', ...
              obj.work_dir, obj.binary, in_file);
[status, log] = system(cmd);
end
