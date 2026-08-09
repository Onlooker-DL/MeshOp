function cfg = burgers_fem_config()
%BURGERS_FEM_CONFIG Edit this file, then run run_burgers_fem.
%
% The FEM/Newton solver, AFEM estimator and Fourier reference reconstruct
% the PDE inputs from the exact coefficients saved in burgers_5100.mat.

cfg.operatorName = 'fno';%pod_deeponet,deeponet,fno,cno
cfg.operatorExperiment = 'b3000_mse';

% Optional full paths. Leave empty to use canonical project paths.
cfg.predictionMat = '';
cfg.datasetMat = '';
cfg.datasetCheckpointSamples = 5100;
cfg.newtonInitializationMode = 'coarse_interpolation';

% Read the training-set size from the operator export metadata when empty.
cfg.trainSamples = [];
cfg.numSamples = 100;

% Optional test-sample selection. Leave empty to evaluate a deterministic
% random subset of cfg.numSamples tests; set e.g. [1 69] to evaluate only
% those tests. plotSampleIds controls which tests produce mesh/error/score
% figures (empty = first cfg.numPlotSamples successful cases).
cfg.sampleIds = [];
cfg.plotSampleIds = [1 4 69];

% -------------------------------------------------------------------------
% METHOD SWITCHES
cfg.runFNOSeededAFEM = true;
cfg.runTargetSeededAFEM = true;

% Standard AFEM master switch and independent row switches.
cfg.runStandardAFEM = true;
cfg.runTimeMatchedAFEM = false;
cfg.runAccuracyMatchedAFEM = false;
cfg.runFixedRefinementAFEM = true;

cfg.runDOFMatchedUniform = false;
cfg.runFixedUniform = true;
cfg.fixedUniformFreeDOF = 650^2;%650^2;250^2

cfg.operatorInitialGrid = 32;
cfg.afemInitialGrid = 32;
cfg.operatorScoreSafetyBias = 0.0;

% Standard score-to-generation rule g(s)=floor(s+0.5).
cfg.generationThreshold = 0.50;
cfg.scoreMultiplier = 1;
cfg.maxGeneration = ceil(12*cfg.scoreMultiplier);
cfg.maxElementsCase = 3000000;
cfg.maxRefineCalls = 80;

% Exact PDE-input reconstruction.
cfg.useExactFourierInputs = true;
cfg.exactInputAbsTolerance = 5.0e-6;
cfg.exactInputRelTolerance = 5.0e-6;

% Standard AFEM controls.
cfg.fixedAFEMRefineCycles = 12;
cfg.afemMaxCycles = 40;

% runBase performs +0 AFEM; runHybrid performs +hybridAFEMCycles AFEM.
cfg.runBase = true;
cfg.runHybrid = false;
cfg.hybridAFEMCycles = 2;
cfg.outputTag = '';

cfg.numPlotSamples = 3;
cfg.makeMeshPlots = true;
cfg.makeScorePlots = true;
cfg.makePointwisePlots = true;
cfg.makeAFEMConvergencePlots = true;
cfg.saveSampleMeshes = true;
end
