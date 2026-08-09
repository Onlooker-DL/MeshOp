function output = run_reaction_diffusion_accu_fem()
%RUN_REACTION_DIFFUSION_ACCU_FEM Project-level entry point.

cfg = reaction_diffusion_accu_fem_config();
thisDir = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(thisDir)));

setupFile = fullfile(root,'setup.m');
if exist(setupFile,'file')==2
    run(setupFile);
else
    addpath(genpath(fullfile(root,'src')));
end
addpath(fullfile(root,'src','fem'));

operatorDir = fullfile(root,'result','operators','reaction_diffusion', ...
    cfg.operatorName,cfg.operatorExperiment);

if isempty(cfg.predictionMat)
    candidates = { ...
        fullfile(operatorDir,'predictions.mat'), ...
        fullfile(operatorDir, ...
            'fno_reactiondiffusion_3d_score_test_predictions.mat')};
    cfg.predictionMat = first_existing_file(candidates);
else
    cfg.predictionMat = absolute_or_root_path(root,cfg.predictionMat);
end

if isempty(cfg.metricsJson)
    cfg.metricsJson = fullfile(operatorDir,'final_metrics.json');
else
    cfg.metricsJson = absolute_or_root_path(root,cfg.metricsJson);
end

if isempty(cfg.datasetMat)
    cfg.datasetMat = fullfile(root,'data','reaction_diffusion_accu', ...
        'reaction_diffusion_accu_3100.mat');
else
    cfg.datasetMat = absolute_or_root_path(root,cfg.datasetMat);
end

if isempty(cfg.outputTag)
    tag = sprintf('tol_%s',strrep(sprintf('%.0e', ...
        cfg.targetRelativeL2),'-','m'));
else
    tag = cfg.outputTag;
end

cfg.projectRoot = root;
cfg.outDir = fullfile(root,'result','fem','reaction_diffusion_accu', ...
    cfg.operatorName,cfg.operatorExperiment,tag);
cfg.figureDir = fullfile(root,'figures','fem','reaction_diffusion_accu', ...
    cfg.operatorName,cfg.operatorExperiment,tag);

cfg.exportedScoreMaxLevel = cfg.maxGeneration;
cfg.operatorMaxLevel = cfg.maxGeneration;
cfg.operatorRefinementMultiplier = cfg.scoreMultiplier;
cfg.operatorMaxRefineCalls = cfg.maxRefineCalls;
cfg.requireExportedThresholdMatch = ~cfg.allowThresholdOverride;

output = reaction_diffusion_accu_fem_comparison(cfg);
end


function pathOut = first_existing_file(candidates)
for k = 1:numel(candidates)
    if exist(candidates{k},'file')==2
        pathOut = candidates{k};
        return;
    end
end
error(['No prediction MAT was found. Tried:\n  %s'], ...
    strjoin(candidates,sprintf('\n  ')));
end


function pathOut = absolute_or_root_path(root,pathIn)
pathIn = char(string(pathIn));
if is_absolute_path(pathIn)
    pathOut = pathIn;
else
    pathOut = fullfile(root,pathIn);
end
end


function tf = is_absolute_path(p)
if ispc
    tf = ~isempty(regexp(p,'^[A-Za-z]:[\\/]|^\\\\','once'));
else
    tf = startsWith(p,'/');
end
end
