function cfg = cd_config()
%CD_CONFIG Shared configuration for the CD-disk experiment.
%
%   Single source of truth for both data generation and FEM comparison:
%   same initial mesh setting, same theta (Dorfler), same NVB settings,
%   same polar query grid, same score maximum and same spectral reference
%   resolutions.  Modify here, both stages stay consistent.

cfg.problem = 'cd_disk';

% PDE
cfg.epsilon = 1.0e-3;
cfg.betaMagnitude = 1.0;

% Localized GRF forcing (K=16, covariance 625 P_16(-Delta+25I)^(-2)P_16)
cfg.grfK = 16;
cfg.grfPower = 2.0;
cfg.grfKappa2 = 25.0;
cfg.grfAmplitude = 25.0;
cfg.forcingCenterMin = -0.50;
cfg.forcingCenterMax =  0.50;
cfg.forcingWidthMin = 0.05;
cfg.forcingWidthMax = 0.15;

% Initial disk mesh: identical to one_sample (initmesh, Hmax=0.1).
cfg.HmaxInitial = 0.1;
% Uniform-row mesh control (not the shared initial mesh).
cfg.initialBoundaryPoints = 126;
cfg.initialInteriorRings = 10;
cfg.initialRingAngles = 96;

% Main AFEM design (SAME for data generation and comparison).
cfg.theta = 0.80;
cfg.mainAFEMCycles = 12;
cfg.hybridAFEMCycles = 2;
cfg.fixedAFEMRefineCycles = 12;
cfg.afemMaxCycles = 40;
cfg.nvbMaxCompletionSteps = 1000;
cfg.maxElements = 2.0e6;

% Score / polar query grid (nr=64 clustered toward r=1, ntheta=128).
cfg.nr = 64;
cfg.ntheta = 128;
cfg.scoreMaximum = 12;
cfg.generationThreshold = 0.50;
cfg.scoreMultiplier = 1.0;
cfg.scoreRefineMaxCalls = 80;

% Spectral reference resolutions.
cfg.specCoarse.NrIntervals = 127;
cfg.specCoarse.M = 64;
cfg.specFine.NrIntervals = 255;
cfg.specFine.M = 128;
cfg.specInterpolationNtheta = 512;
cfg.specCheckGrid = 121;

% Method switches for the FEM comparison (mirror burgers).
cfg.runFNOSeededAFEM = true;
cfg.runTargetSeededAFEM = true;
cfg.runStandardAFEM = true;
cfg.runTimeMatchedAFEM = false;
cfg.runAccuracyMatchedAFEM = false;
cfg.runFixedRefinementAFEM = true;
cfg.runDOFMatchedUniform = false;
cfg.runFixedUniform = true;
cfg.uniformHmax = 0.05;   % initmesh Hmax for the fixed-uniform row
cfg.runBase = true;
cfg.runHybrid = false;

% Operator/experiment names used for paths.
cfg.operatorName = 'fno';
cfg.operatorExperiment = 'cd_b3000_mse';
cfg.numSamples = 100;
cfg.sampleIds = [];
cfg.plotSampleIds = [1 4 69];
cfg.saveSampleMeshes = true;
end