function cfg = cd_disk_fem_config()
%CD_DISK_FEM_CONFIG Edit this file, then run run_cd_disk_fem.
% All AFEM parameters (initial mesh, theta, NVB, score cap) come from
% cd_config() in src/fem/cd_disk; only paths and switches live
% here, mirroring burgers_fem_config.

cfg.operatorName = 'fno';        % fno, cno, deeponet, pod_deeponet
cfg.operatorExperiment = 'cd_b3000_mse';

% Optional full paths; leave empty for canonical project paths.
cfg.predictionMat = '';
cfg.datasetMat = '';
cfg.datasetCheckpointSamples = 3100;

cfg.numSamples = 100;            % tests evaluated (unless sampleIds set)
cfg.sampleIds = [];              % e.g. [1 4 69]
cfg.plotSampleIds = [1 3 5 7 9];

% ---- method switches (mirror burgers) ----
cfg.runFNOSeededAFEM = true;
cfg.runTargetSeededAFEM = true;
cfg.runStandardAFEM = true;
cfg.runTimeMatchedAFEM = false;
cfg.runAccuracyMatchedAFEM = true;
cfg.runFixedRefinementAFEM = true;
cfg.runDOFMatchedUniform = false;
cfg.runFixedUniform = true;
cfg.uniformHmax = 0.01;   % initmesh Hmax for the fixed-uniform row

% runBase = +0 AFEM cycles; runHybrid = +hybridAFEMCycles cycles.
cfg.runBase = true;
cfg.runHybrid = false;
cfg.hybridAFEMCycles = 2;
cfg.fixedAFEMRefineCycles = 12;
cfg.afemMaxCycles = 40;

cfg.saveSampleMeshes = true;

% Plot switches (mirror burgers). Figures go to figures/fem/cd_disk/...
cfg.makeMeshPlots = true;
cfg.makeScorePlots = true;
cfg.makeConvergencePlots = true;
cfg.makeSummaryPlots = true;
cfg.outputTag = '';
end