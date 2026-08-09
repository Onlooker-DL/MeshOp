function cfg = reaction_diffusion_accu_fem_config()
%REACTION_DIFFUSION_ACCU_FEM_CONFIG Edit, then run the FEM entry point.

% Trained operator output.
cfg.operatorName = 'fno';
cfg.operatorExperiment = 'rd_accu_b3000_mse';%'rd_accu_b3000_mse';
cfg.predictionMat = ''; % empty -> resolved from result/operators/...
cfg.metricsJson = '';   % empty -> resolved beside predictions.mat

% Accuracy data set and test selection.
cfg.datasetMat = '';    % empty -> data/reaction_diffusion_accu/..._3100.mat
cfg.numSamples = 100;
cfg.sampleIds = [];     % empty -> deterministic random subset
cfg.sampleSelectionSeed = 20260725;

% Common downstream target. This is intentionally stricter than the
% 4e-2 stopping tolerance used to generate the training labels.
cfg.targetRelativeL2 = 4.0e-2;

% Five independent method switches.
cfg.runFNOSeededAFEM = true;
cfg.runTargetSeededAFEM = true;
cfg.runStandardAFEM = true;
cfg.runAccuracyMatchedUniform = false;
cfg.runFixedUniform = true;

% Accuracy-matched uniform search. Search time is saved separately and is
% excluded from the reported MethodTimeSec. The first tested grid reaching
% the target is rebuilt and solved once for the reported time.
cfg.uniformSearchCells = 16:4:80;

% Fixed-size uniform baseline. All test samples use the same grid.
% When fixedUniformCells is empty, the smallest structured grid with
% (n-1)^3 >= fixedUniformTargetDOF is used. 250000 -> n=64 -> 250047 DOF.
cfg.fixedUniformTargetDOF = 300000;%300000;2571353
cfg.fixedUniformCells = [];
cfg.newtonInitializationMode = 'coarse_interpolation'; % zero; initial_condition; coarse_interpolation
% Same PDE, GRF, AFEM and reference settings as the accuracy generator.
cfg.epsilon = 2.0e-2;
cfg.initialCells = 16;
cfg.theta = 0.80;
cfg.maxGeneration = 12;
cfg.generationThreshold = 0.50;
cfg.allowThresholdOverride = false;
cfg.scoreMultiplier = 1.0;
cfg.afemMaxCycles = 30;
cfg.maxRefineCalls = 30;
cfg.maxFreeDOF = 50000000;
cfg.maxElementsCase = 350000000;

% Optional manual timing override. NaN reads final_metrics.json.
cfg.operatorInferenceTimePerSample = NaN;

% Output and diagnostics.
cfg.outputTag = '';
cfg.makePlots = true;
cfg.figureVisible = 'off';
cfg.figureResolution = 180;
cfg.saveSelectedMeshes = true;
end
