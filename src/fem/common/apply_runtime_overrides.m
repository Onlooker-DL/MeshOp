function cfg = apply_runtime_overrides(cfg)
%APPLY_RUNTIME_OVERRIDES Apply the options set by a FEM experiment runner.

global NOEM_FEM_RUNTIME_OVERRIDES
if isempty(NOEM_FEM_RUNTIME_OVERRIDES)
    return;
end

names = fieldnames(NOEM_FEM_RUNTIME_OVERRIDES);
for k = 1:numel(names)
    name = names{k};
    if ~isfield(cfg,name)
        error('NOEM:UnknownFEMOption', ...
            'Unsupported FEM configuration field: %s',name);
    end
    cfg.(name) = NOEM_FEM_RUNTIME_OVERRIDES.(name);
end
end
