function outputs = run_cd_disk_fem()
%RUN_CD_DISK_FEM The CD-disk FEM experiment entry point.

cfg = cd_disk_fem_config();
root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(root,'src','fem'));
addpath(fullfile(root,'src','fem','cd_disk'));

if isempty(cfg.predictionMat)
    cfg.predictionMat = fullfile(root,'result','operators','cd_disk', ...
        cfg.operatorName,cfg.operatorExperiment,'predictions.mat');
end
if isempty(cfg.datasetMat)
    cfg.datasetMat = fullfile(root,'data','cd_disk',sprintf( ...
        'cd_disk_%d.mat',cfg.datasetCheckpointSamples));
end

if ~exist(cfg.predictionMat,'file')
    error('run_cd_disk_fem:MissingPredictions', ...
        'Predictions file not found:\n%s',cfg.predictionMat);
end
if ~exist(cfg.datasetMat,'file')
    error('run_cd_disk_fem:MissingDataset', ...
        'Dataset file not found:\n%s',cfg.datasetMat);
end

outputs = struct();
if cfg.runBase
    cfg.hybridAFEMCycles = 0;
    outputs.base = cd_disk_fem_comparison(cfg);
end
if cfg.runHybrid
    cfg.hybridAFEMCycles = cfg.hybridAFEMCycles;
    outputs.hybrid = cd_disk_fem_comparison(cfg);
end
end