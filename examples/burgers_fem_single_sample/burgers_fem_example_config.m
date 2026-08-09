function cfg = burgers_fem_example_config()
%BURGERS_FEM_EXAMPLE_CONFIG Config for the one-sample Burgers FEM example.
%
% Edit this file, then run run_burgers_fem_example.
%
% The example solves ONE Burgers test instance (Test 001, dataset id 5001)
% with two meshes: the FNO-predicted refinement field and the target
% refinement field. All other baselines are disabled so the run stays fast.
%
% It requires:
%   - a trained operator export at
%       result/operators/burgers/fno/b3000_mse/predictions.mat
%   - the data file data/burgers/burgers_5100.mat

cfg.operatorName = 'fno';
cfg.operatorExperiment = 'b3000_mse';

% Optional full paths. Leave empty to use canonical project paths.
cfg.predictionMat = '';
cfg.datasetMat = '';
cfg.datasetCheckpointSamples = 5100;
cfg.newtonInitializationMode = 'coarse_interpolation';

% Read the training-set size from the operator export metadata when empty.
cfg.trainSamples = [];
cfg.numSamples = 1;

% Evaluate exactly one test: Test 001 (dataset id 5001), and plot it.
cfg.sampleIds = 1;
cfg.plotSampleIds = 1;

% -------------------------------------------------------------------------
% METHOD SWITCHES: only the two seeded AFEM variants are enabled.
cfg.runFNOSeededAFEM = true;
cfg.runTargetSeededAFEM = true;

% Standard AFEM master switch and independent row switches (all off).
cfg.runStandardAFEM = false;
cfg.runTimeMatchedAFEM = false;
cfg.runAccuracyMatchedAFEM = false;
cfg.runFixedRefinementAFEM = false;

cfg.runDOFMatchedUniform = false;
cfg.runFixedUniform = false;
cfg.fixedUniformFreeDOF = 650^2;

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

% Standard AFEM controls (kept for compatibility, not used here).
cfg.fixedAFEMRefineCycles = 12;
cfg.afemMaxCycles = 40;

% runBase performs +0 AFEM; runHybrid performs +hybridAFEMCycles AFEM.
cfg.runBase = true;
cfg.runHybrid = false;
cfg.hybridAFEMCycles = 2;

% Separate output tag so the example never overwrites the paper results.
cfg.outputTag = 'example_1sample';

cfg.numPlotSamples = 1;
cfg.makeMeshPlots = true;
cfg.makeScorePlots = false;
cfg.makePointwisePlots = false;
cfg.makeAFEMConvergencePlots = false;
cfg.saveSampleMeshes = true;
end