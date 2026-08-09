function result = burgers_fem_comparison( ...
        trainSamples,afemInitialGrid,numSamples,operatorScoreSafetyBias,experimentName)
%BURGERS_FEM_COMPARISON
% VERSION: BURGERS_INDEPENDENT_AFEM_SWITCHES_20260806_V4
% Switchable downstream comparison with up to seven reported methods:
%
%   1. FNO predicted-score FEM + cfg.hybridAFEMCycles AFEM corrections;
%   2. ground-truth target-score FEM + the same AFEM corrections;
%   3. standard AFEM matched to the FNO-hybrid online time;
%   4. standard AFEM matched to the FNO-hybrid space-time relative L2 error;
%   5. standard AFEM after cfg.fixedAFEMRefineCycles refinements;
%   6. the smallest uniform FEM with free DOF strictly above FNO-hybrid DOF;
%   7. a fixed-free-DOF uniform FEM.
%
% Method switches:
%   cfg.runFNOSeededAFEM
%   cfg.runTargetSeededAFEM
%   cfg.runStandardAFEM              (master switch)
%   cfg.runTimeMatchedAFEM           (standard-AFEM row switch)
%   cfg.runAccuracyMatchedAFEM       (standard-AFEM row switch)
%   cfg.runFixedRefinementAFEM       (standard-AFEM row switch)
%   cfg.runDOFMatchedUniform
%   cfg.runFixedUniform
%
% The three standard-AFEM outputs share one refinement trajectory but are
% independently selectable. Disabled criteria neither appear in the output
% nor force the trajectory to continue. Time- and accuracy-matching depend
% on the FNO-hybrid result; fixed-refinement AFEM does not. Therefore, when
% either matching row is enabled while cfg.runFNOSeededAFEM is false, the
% FNO hybrid is computed internally as a dependency but is not reported.
%
% There is deliberately NO standard-AFEM DOF-matched selection. The fixed
% AFEM stage remains the cycle-count baseline.
%
% The same residual estimator, Dörfler parameter theta, conforming periodic
% newest-vertex bisection, Newton solver and timing convention are used by
% the hybrid correction cycles and the standard-AFEM trajectory.
%
% Newton initialization is selected by cfg.newtonInitializationMode:
%   'zero'                 : zero on free DOFs; exact u0 on t=tmin;
%   'initial_condition'    : u0(x) extended through the time slab;
%   'coarse_interpolation' : solve the common 32x32 base from u0(x), then
%                            interpolate it to every enabled score/uniform
%                            target mesh. The coarse solve cost is included
%                            in each corresponding standalone method time.
%
% Every later AFEM refinement always interpolates the previous FEM solution
% to the refined mesh. Spectral-reference construction, error evaluation,
% plotting and offline FNO training are excluded from method times. A finite
% stalled/nonconverged Newton solve is retained and audited.
if nargin < 1 || isempty(trainSamples)
    trainSamples = 2000;
end
if nargin < 2 || isempty(afemInitialGrid)
    afemInitialGrid = 32;
end
if nargin < 3 || isempty(numSamples)
    numSamples = 100;
end
if nargin < 4 || isempty(operatorScoreSafetyBias)
    operatorScoreSafetyBias = 0.0;
end
if nargin < 5 || isempty(experimentName), experimentName = 'weighted'; end
experimentName = char(string(experimentName));

validateattributes(trainSamples,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',1}, ...
    mfilename,'trainSamples',1);
validateattributes(afemInitialGrid,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',2}, ...
    mfilename,'afemInitialGrid',2);
validateattributes(numSamples,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',1,'<=',100}, ...
    mfilename,'numSamples',3);
validateattributes(operatorScoreSafetyBias,{'numeric'}, ...
    {'scalar','real','finite'}, ...
    mfilename,'operatorScoreSafetyBias',4);

trainSamples = double(trainSamples);
afemInitialGrid = double(afemInitialGrid);
numSamples = double(numSamples);
operatorScoreSafetyBias = double(operatorScoreSafetyBias);

clc; close all;

%% ========================================================================
% Configuration
% =========================================================================

% Project root for the burgers_e12 data and 12-refine FNO models.
cfg.projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cfg.dataDir = fullfile(cfg.projectRoot,'data','burgers');
cfg.resultDir = fullfile(cfg.projectRoot,'result','burgers_fno',experimentName);

% Original MATLAB dataset containing the exact random Fourier coefficients.
% The FEM/Newton solver, estimator and spectral reference reconstruct u0 and
% f from these coefficients rather than interpolating exported sensor values.
cfg.datasetMat = fullfile(cfg.dataDir,'burgers_5100.mat');
cfg.useExactFourierInputs = true;
cfg.exactInputAbsTolerance = 5.0e-6;
cfg.exactInputRelTolerance = 5.0e-6;

cfg.trainSamples = trainSamples;

% Folder created by train_fno2d_burgers_score_12refine_*.py.
cfg.predictionFolder = cfg.resultDir;

% Set this to a full MAT-file path only when the folder/file name differs
% from the convention above. Normally leave it empty.
cfg.predictionMatManual = '';

if isempty(cfg.predictionMatManual)
    cfg.predictionMat = fullfile( ...
        cfg.predictionFolder, ...
        'fno_burgers_continuous_score_test_predictions.mat');
else
    cfg.predictionMat = cfg.predictionMatManual;
end

cfg.outDir = fullfile(cfg.projectRoot,'result','burgers_fem',experimentName,'hybrid');
cfg.figureDir = fullfile(cfg.projectRoot,'figures','burgers_fem',experimentName,'hybrid');

% Statistical evaluation set.
% Leave sampleIds empty to draw a reproducible random subset from the
% exported test split. All six methods are evaluated on exactly these cases.
cfg.numSamples = numSamples;
cfg.sampleIds = [];
cfg.sampleSelectionSeed = 20260713;
cfg.operatorName = 'fno';

% Only these randomly selected cases produce the original per-sample plots.
% The other evaluation cases still contribute to every error/time statistic.
cfg.numPlotSamples = min(3,numSamples);
cfg.plotSampleIds = [];
cfg.plotSelectionSeed = 20260714;

% PDE and space-time domain.
cfg.nu   = 5.0e-3;
cfg.xmin = -1.0;
cfg.xmax =  1.0;
cfg.tmin =  0.0;
cfg.tmax =  1.0;

% Data-distribution metadata. The prediction MAT supplies score tensors and
% exported sensor arrays for consistency checks. The authoritative PDE inputs
% are reconstructed from the exact random Fourier coefficients in datasetMat.
cfg.grfPower = 2.5;
cfg.trainingUnderWeightAlpha = 2.0; % metadata for the trained FNO loss

% Continuous-score definition and operator mesh.
% Raw score metadata: the prediction data were generated relative to 32x32.
cfg.scoreReferenceGrid = 32;
cfg.exportedScoreMaxLevel = 12;

% Original score-reference mesh: operator/target methods start from 32x32.
cfg.operatorInitialGrid = 32;

% Original direct refinement map:
%   raw score 1 -> generation 1,
%   raw score 2 -> generation 2.
cfg.operatorRefinementMultiplier = 1.0;

% The operator base and score-reference grid are both 32x32, so the
% generation offset is exactly zero. Keep this false.
cfg.operatorApplyReferenceGridOffset = false;

% The current 0,...,12 raw range becomes 0,...,12 generations.
% The optional safety bias is applied only to the FNO-predicted score;
% the ground-truth target-score baseline remains unbiased.
cfg.operatorMaxLevel = 12;
cfg.operatorScoreSafetyBias = operatorScoreSafetyBias; % FNO prediction only
cfg.targetScoreSafetyBias = 0.00;              % oracle target remains unbiased
cfg.generationThreshold = 0.50;
cfg.operatorQueryPattern = 'seven'; % 'four' or 'seven'
cfg.operatorQueryChunkElements = 200000; % identical queries, fewer chunks
cfg.operatorMaxRefineCalls = 80;

% Unified score-to-generation rule:
%   effectiveScore = multiplier*predictedScore
%                    + optional reference-grid offset + safety bias;
%   effectiveScore is clipped to [0,12];
%   fraction < 0.50 is rounded down;
%   fraction >= 0.50 is rounded up;
%   refine K exactly when generation(K) < desiredGeneration(K).
%
% Examples:
%   -0.2 -> 0, 0.49 -> 0, 0.50 -> 1,
%    1.49 -> 1, 1.50 -> 2, 11.49 -> 11, 11.50 -> 12, 13.0 -> 12.

% Self-contained periodic NVB.
cfg.refinementBackend = 'local_nvb';
cfg.nvbMaxCompletionSteps = 500;
cfg.periodicCompletionMax = 200;
cfg.meshQualityWarningAngle = 4.0;

% Standard AFEM and first-strictly-above DOF selection.
% Default is 32x32. Change it either in the function call or on this line.
cfg.afemInitialGrid = afemInitialGrid;
cfg.theta = 0.8;
cfg.hybridAFEMCycles = 2;

% -------------------------------------------------------------------------
% INDEPENDENT METHOD SWITCHES
%
% runStandardAFEM is a master switch. Its three child switches independently
% control the time-matched, accuracy-matched and fixed-refinement AFEM rows.
% The enabled child criteria share one standard-AFEM trajectory. A disabled
% criterion is not reported and does not extend the trajectory.
cfg.runFNOSeededAFEM = true;
cfg.runTargetSeededAFEM = true;
cfg.runStandardAFEM = true;
cfg.runTimeMatchedAFEM = true;
cfg.runAccuracyMatchedAFEM = true;
cfg.runFixedRefinementAFEM = true;
cfg.runDOFMatchedUniform = true;
cfg.runFixedUniform = true;

% Fixed-free-DOF uniform baseline. For this periodic space-time mesh,
% freeDOF = n^2 exactly, so the requested value must be a perfect square.
% The default 65536 corresponds to a 256x256 uniform space-time grid.
cfg.fixedUniformFreeDOF = 65536;

% Stage 0 is the solved initial mesh. Stage N is after N complete
% ESTIMATE--MARK--REFINE--SOLVE cycles. The labels use 12 refinements.
cfg.fixedAFEMRefineCycles = 12;
cfg.afemMaxCycles = 40;

% Online FNO inference cost.
% When true, the code reads test_inference_time_sec from the exported MAT
% file and divides it by the number of exported test samples.
cfg.useExportedInferenceTime = true;
cfg.operatorInferenceTimePerSample = NaN;

% target_score_test is already stored in the prediction MAT file. Therefore
% its original offline data-generation cost is unavailable here and is not
% included in target-score FEM time.
cfg.targetScoreAccessTimePerSample = 0.0;

% Safety and performance.
cfg.maxElementsCase = 3000000;
cfg.errorChunkElements = 300000; % identical quadrature, fewer chunks

% -------------------------------------------------------------------------
% FAST-EXACT implementation controls.
% These do not change the seven-point query, configured threshold, periodic NVB,
% residual estimator, Dörfler marking, Newton equations/tolerances, or the
% finite-last-iterate nonconvergence policy. They only remove repeated work.
% Time-matched AFEM may select a different stage because wall-clock times
% become smaller; the numerical refinement trajectory itself should not.
% -------------------------------------------------------------------------
cfg.fastUseOperatorWorkQueue = true;
cfg.fastTrackNVBGeneration = true;
cfg.fastPackedNVBEdges = true;
cfg.fastBatchPeriodicRequests = true;
cfg.fastUseDirectDOFCount = true;
cfg.fastSkipIntermediateMeshQuality = true;

% Keep the original per-call validity checks for AFEM refinements.
% The score-mesh compiler disables only these repeated checks locally and
% still performs the same full validity/periodicity check on its final mesh.
cfg.fastCheckEveryNVBCall = true;
cfg.fastFinalNVBValidityCheck = true;
cfg.fastCheckNVBSubcalls = false;

% Expensive one-sample audits; false for production timing.
cfg.fastAuditOperatorMesh = false;
cfg.fastAuditNVB = false;
cfg.fastAuditPeriodicSideEdges = false;
cfg.fastAuditDOFCount = false;

% Damped Newton.
cfg.newtonMaxIt = 40;
cfg.newtonTol = 1e-6;
cfg.newtonRelTol = 1e-6;
cfg.newtonLineSearch = true;
cfg.newtonLineSearchMax = 16;
cfg.newtonRegularizationTrials = 5;
cfg.newtonRegularizationBase = 1e-12;
cfg.newtonMaxUpdateFactor = 2.0;
cfg.newtonVerbose = false;

% -------------------------------------------------------------------------
% USER-SELECTABLE NEWTON INITIALIZATION
% Change only the following line to compare the three policies.
%
%   'zero'
%       Independent first solves start from zero on every free space-time
%       DOF. The prescribed trace u(x,tmin)=u0(x) is still imposed exactly.
%
%   'initial_condition'
%       Independent first solves start from u^(0)(x,t)=u0(x) over the whole
%       space-time slab. No preliminary coarse FEM solve is performed.
%
%   'coarse_interpolation'
%       First solve the operator 32x32 base mesh from u0(x), then interpolate
%       its converged FEM solution to the FNO, target-score and uniform meshes.
%       The base-solve time is included in those three method times.
%
% The same selection is applied to every enabled score/uniform method and
% standard-AFEM stage 0. Subsequent AFEM refinements always use the previous
% FEM solution interpolated to the refined mesh.
cfg.newtonInitializationMode = 'coarse_interpolation'; % zero; initial_condition; coarse_interpolation

% Keep results from different initialization experiments in separate folders:
%   .../hybrid/init_zero
%   .../hybrid/init_initial_condition
%   .../hybrid/init_coarse_interpolation
cfg.appendInitializationModeToOutput = true;

% Periodic Fourier ETDRK4 reference.
cfg.NxRefSpectral = 512;
cfg.NtRefSpectral = 201;
cfg.refDt = 2.5e-4;
cfg.dealiasFraction = 2/3;
cfg.compareGridX = 2001;
cfg.compareGridT = 241;

% Output and plotting.
cfg.figureVisible = 'off';
cfg.figureResolution = 220;
cfg.plotMaxElements = 180000;
cfg.makeMeshPlots = true;
cfg.makeScorePlots = true;
cfg.makePointwisePlots = true;   % plotting only; no numerical effect
cfg.makeAFEMConvergencePlots = true;
cfg.saveSampleMeshes = false;      % large MAT files; set true when needed
cfg.savePointwiseErrorArrays = false;
cfg.pointwiseErrorFaceColor = 'interp';
cfg.pointwiseShowMeshEdges = false;

cfg = apply_runtime_overrides(cfg);

% Operator display name used in console output and figure titles.
switch lower(cfg.operatorName)
    case 'fno'
        operatorLabel = 'FNO';
    case 'deeponet'
        operatorLabel = 'DeepONet';
    case 'pod_deeponet'
        operatorLabel = 'POD-DeepONet';
    case 'transolver'
        operatorLabel = 'Transolver';
    case 'cno'
        operatorLabel = 'CNO';
    otherwise
        operatorLabel = upper(cfg.operatorName);
end

cfg.newtonInitializationMode = normalize_newton_initialization_mode( ...
    cfg.newtonInitializationMode);

methodSwitchNames = { ...
    'runFNOSeededAFEM','runTargetSeededAFEM','runStandardAFEM', ...
    'runTimeMatchedAFEM','runAccuracyMatchedAFEM', ...
    'runFixedRefinementAFEM','runDOFMatchedUniform','runFixedUniform'};
for switchId = 1:numel(methodSwitchNames)
    switchName = methodSwitchNames{switchId};
    if ~isfield(cfg,switchName)
        error('Missing method switch cfg.%s.',switchName);
    end
    validateattributes(cfg.(switchName),{'logical','numeric'},{'scalar'});
    cfg.(switchName) = logical(cfg.(switchName));
end

% The master switch gates all three standard-AFEM row switches.
cfg.runTimeMatchedAFEM = ...
    cfg.runStandardAFEM && cfg.runTimeMatchedAFEM;
cfg.runAccuracyMatchedAFEM = ...
    cfg.runStandardAFEM && cfg.runAccuracyMatchedAFEM;
cfg.runFixedRefinementAFEM = ...
    cfg.runStandardAFEM && cfg.runFixedRefinementAFEM;
runAnyStandardAFEM = ...
    cfg.runTimeMatchedAFEM || cfg.runAccuracyMatchedAFEM || ...
    cfg.runFixedRefinementAFEM;

if ~(cfg.runFNOSeededAFEM || cfg.runTargetSeededAFEM || ...
        runAnyStandardAFEM || cfg.runDOFMatchedUniform || ...
        cfg.runFixedUniform)
    error('At least one reported method must be enabled.');
end
validateattributes(cfg.fixedUniformFreeDOF,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',4});
cfg.fixedUniformFreeDOF = double(cfg.fixedUniformFreeDOF);
cfg.fixedUniformGridN = ...
    uniform_grid_from_fixed_free_dof(cfg.fixedUniformFreeDOF);

validateattributes(cfg.appendInitializationModeToOutput, ...
    {'logical','numeric'},{'scalar'});
cfg.appendInitializationModeToOutput = ...
    logical(cfg.appendInitializationModeToOutput);
if cfg.appendInitializationModeToOutput
    initializationFolder = ['init_' cfg.newtonInitializationMode];
    cfg.outDir = fullfile(cfg.outDir,initializationFolder);
    cfg.figureDir = fullfile(cfg.figureDir,initializationFolder);
end

validateattributes(cfg.fixedAFEMRefineCycles,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',0});
validateattributes(cfg.afemMaxCycles,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',0});
if cfg.fixedAFEMRefineCycles > cfg.afemMaxCycles
    error(['cfg.fixedAFEMRefineCycles=%d exceeds cfg.afemMaxCycles=%d. ', ...
           'Increase afemMaxCycles or reduce fixedAFEMRefineCycles.'], ...
        cfg.fixedAFEMRefineCycles,cfg.afemMaxCycles);
end

if ~exist(cfg.predictionMat,'file')
    error('Prediction MAT file not found:\n%s',cfg.predictionMat);
end
if cfg.useExactFourierInputs && ~exist(cfg.datasetMat,'file')
    error(['Original Burgers MATLAB dataset not found:\n%s\n', ...
           'Set cfg.datasetMat to the burgers_5100.mat checkpoint.'], ...
        cfg.datasetMat);
end

if ~exist(cfg.outDir,'dir')
    mkdir(cfg.outDir);
end
if ~exist(cfg.figureDir,'dir')
    mkdir(cfg.figureDir);
end

if ~isfinite(cfg.operatorRefinementMultiplier) || ...
        cfg.operatorRefinementMultiplier<=0
    error('cfg.operatorRefinementMultiplier must be positive.');
end
if ~isfinite(cfg.generationThreshold) || ...
        cfg.generationThreshold<=0 || cfg.generationThreshold>=1
    error('cfg.generationThreshold must lie strictly between 0 and 1.');
end
if cfg.operatorApplyReferenceGridOffset
    gridRatio = cfg.scoreReferenceGrid/cfg.operatorInitialGrid;
    gridOffset = 2*log2(gridRatio);
    if abs(gridOffset-round(gridOffset))>1e-12
        error(['scoreReferenceGrid/operatorInitialGrid must be a ', ...
               'power-of-two ratio when the offset is enabled.']);
    end
else
    gridOffset = 0;
end

mappedLevelOne = min(max( ...
    cfg.operatorRefinementMultiplier + gridOffset + ...
    cfg.operatorScoreSafetyBias,0),cfg.operatorMaxLevel);
mappedLevelTwo = min(max( ...
    2*cfg.operatorRefinementMultiplier + gridOffset + ...
    cfg.operatorScoreSafetyBias,0),cfg.operatorMaxLevel);
if abs(cfg.operatorScoreSafetyBias)<=1e-12 && ...
        abs(cfg.operatorRefinementMultiplier-1)<=1e-12 && ...
        (abs(mappedLevelOne-1)>1e-12 || abs(mappedLevelTwo-2)>1e-12)
    warning(['Current zero-bias settings map raw levels 1 and 2 to ', ...
             '%.3f and %.3f, not exactly 1 and 2.'], ...
             mappedLevelOne,mappedLevelTwo);
end

% Random streams are initialized explicitly during sample selection.

%% ========================================================================
% Load deterministic continuous-score predictions
% =========================================================================

D = load(cfg.predictionMat);
requiredFields = {'pred_score','target_score_test','U0_test','F_test', ...
    'x_input','query_x','query_t'};
for k = 1:numel(requiredFields)
    if ~isfield(D,requiredFields{k})
        error('Field "%s" is missing from %s.', ...
            requiredFields{k},cfg.predictionMat);
    end
end

xInput = double(D.x_input(:));
queryX = double(D.query_x(:));
queryT = double(D.query_t(:));

if numel(queryX)~=101 || numel(queryT)~=101
    error(['This comparison file expects a 101x101 prediction grid, ' ...
           'but query_x/query_t have lengths %d and %d.'], ...
        numel(queryX),numel(queryT));
end
if isfield(D,'max_score')
    exportedMaxScore = round(double(D.max_score(1)));
    if exportedMaxScore~=cfg.exportedScoreMaxLevel
        error(['Prediction file max_score=%d, whereas ' ...
               'cfg.exportedScoreMaxLevel=%d.'], ...
            exportedMaxScore,cfg.exportedScoreMaxLevel);
    end
end

U0test = double(D.U0_test);
Ftest = double(D.F_test);
if size(U0test,1)~=numel(xInput) && size(U0test,2)==numel(xInput)
    U0test = U0test.';
end
if size(U0test,1)~=numel(xInput)
    error('U0_test shape %s is incompatible with x_input length %d.', ...
        mat2str(size(U0test)),numel(xInput));
end

nTest = size(U0test,2);
if size(Ftest,1)~=numel(xInput) && size(Ftest,2)==numel(xInput); Ftest=Ftest.'; end
if ~isequal(size(Ftest),size(U0test)); error('F_test must have the same shape as U0_test.'); end

% Load only compact exact coefficients; the large target tensors are not read.
BurgersExact = struct();
nDatasetSamples = NaN;
if cfg.useExactFourierInputs
    exactFields = { ...
        'sine_amplitude','sine_shift', ...
        'grf_xi_cos','grf_xi_sin','grf_modes','grf_spectral_std', ...
        'forcing_xi_cos','forcing_xi_sin', ...
        'forcing_modes','forcing_spectral_std'};
    fileInfo = whos('-file',cfg.datasetMat);
    availableFields = {fileInfo.name};
    missingExact = setdiff(exactFields,availableFields);
    if ~isempty(missingExact)
        error('Dataset %s is missing exact-input fields: %s', ...
            cfg.datasetMat,strjoin(missingExact,', '));
    end
    BurgersExact = load(cfg.datasetMat,exactFields{:});
    nDatasetSamples = numel(BurgersExact.sine_amplitude);
    validate_exact_burgers_dataset(BurgersExact,nDatasetSamples);
end

if cfg.useExportedInferenceTime && ...
        (~isfinite(cfg.operatorInferenceTimePerSample) || ...
         cfg.operatorInferenceTimePerSample<0)
    if isfield(D,'test_inference_time_sec')
        totalInferenceTime = double(D.test_inference_time_sec(1));
        cfg.operatorInferenceTimePerSample = ...
            totalInferenceTime/max(nTest,1);
    else
        warning(['test_inference_time_sec is not present in the MAT file; ' ...
                 'the reported operator inference time is set to zero.']);
        cfg.operatorInferenceTimePerSample = 0.0;
    end
end
if ~isfinite(cfg.operatorInferenceTimePerSample) || ...
        cfg.operatorInferenceTimePerSample<0
    cfg.operatorInferenceTimePerSample = 0.0;
end

if isfield(D,'training_time_sec')
    cfg.exportedFNOTrainingTimeSec = double(D.training_time_sec(1));
else
    cfg.exportedFNOTrainingTimeSec = NaN;
end

% Build a candidate pool. In the default mode the entire exported test set
% is placed in a deterministic random order. Failed cases are replaced by
% later candidates, so numSamples refers to successful complete comparisons.
if isempty(cfg.sampleIds)
    requestedCases = min(max(round(cfg.numSamples),1),nTest);
    rng(cfg.sampleSelectionSeed,'twister');
    candidateSampleIds = randperm(nTest);
else
    candidateSampleIds = unique(round(cfg.sampleIds(:).'),'stable');
    if any(candidateSampleIds<1) || any(candidateSampleIds>nTest)
        error('cfg.sampleIds contains an index outside 1,...,%d.',nTest);
    end
    requestedCases = min(max(round(cfg.numSamples),1), ...
        numel(candidateSampleIds));
end

if isempty(candidateSampleIds)
    error('No candidate evaluation samples were selected.');
end

candidateSourceIds = arrayfun( ...
    @(id)get_source_index(D,id,nTest,nDatasetSamples),candidateSampleIds);

candidatePlanTable = table( ...
    (1:numel(candidateSampleIds)).', ...
    candidateSampleIds(:),candidateSourceIds(:), ...
    'VariableNames',{'CandidateOrder','TestID','SourceID'});

candidatePlanCsv = fullfile(cfg.outDir, ...
    'candidate_test_case_order.csv');
writetable(candidatePlanTable,candidatePlanCsv);

failureCsv = fullfile(cfg.outDir,'failed_cases.csv');
failureMat = fullfile(cfg.outDir,'failed_cases.mat');

fprintf('============================================================\n');
fprintf('Periodic Burgers switchable-method comparison (FAST-EXACT)\n');
fprintf('Prediction       : deterministic continuous score, %dx%d\n', ...
    numel(queryX),numel(queryT));
fprintf('Raw score basis  : %dx%d, raw max level %d\n', ...
    cfg.scoreReferenceGrid,cfg.scoreReferenceGrid, ...
    cfg.exportedScoreMaxLevel);
fprintf(['Operator start   : %dx%d, score multiplier %.3f, ', ...
         'generation offset %+d, max generation %d\n'], ...
    cfg.operatorInitialGrid,cfg.operatorInitialGrid, ...
    cfg.operatorRefinementMultiplier,round(gridOffset), ...
    cfg.operatorMaxLevel);
fprintf('Raw level map    : 1 -> %.1f generations, 2 -> %.1f generations\n', ...
    mappedLevelOne,mappedLevelTwo);
fprintf(['Operator query   : %s, %s safety bias %.3f, ', ...
         'target safety bias %.3f, generation threshold %.3f\n'], ...
    cfg.operatorQueryPattern,operatorLabel,cfg.operatorScoreSafetyBias, ...
    cfg.targetScoreSafetyBias,cfg.generationThreshold);
fprintf('Data metadata    : grfPower %.2f, training under-weight alpha %.1f\n', ...
    cfg.grfPower,cfg.trainingUnderWeightAlpha);
fprintf('AFEM start       : %dx%d, theta %.3f\n', ...
    cfg.afemInitialGrid,cfg.afemInitialGrid,cfg.theta);
fprintf('Hybrid correction: exactly %d AFEM cycles after each score mesh\n', ...
    cfg.hybridAFEMCycles);
fprintf(['Standard AFEM   : master=%d, time-match=%d, error-match=%d, ', ...
         'fixed-%d=%d\n'],cfg.runStandardAFEM,cfg.runTimeMatchedAFEM, ...
    cfg.runAccuracyMatchedAFEM,cfg.fixedAFEMRefineCycles, ...
    cfg.runFixedRefinementAFEM);
fprintf('%s infer/sample : %.6e s\n', ...
    operatorLabel,cfg.operatorInferenceTimePerSample);
if isfinite(cfg.exportedFNOTrainingTimeSec)
    fprintf('Offline %s train: %.3f s (reported only, excluded below)\n', ...
        operatorLabel,cfg.exportedFNOTrainingTimeSec);
end
fprintf('Training samples : %d\n',cfg.trainSamples);
fprintf('Prediction folder: %s\n',cfg.predictionFolder);
fprintf('Prediction file  : %s\n',cfg.predictionMat);
if cfg.useExactFourierInputs
    fprintf('Exact PDE inputs : Fourier coefficients from %s\n',cfg.datasetMat);
else
    fprintf('PDE inputs       : exported sensor interpolation (legacy mode)\n');
end
fprintf('Fixed AFEM stage : %d refinement cycles\n', ...
    cfg.fixedAFEMRefineCycles);
fprintf('Fixed uniform    : requested DOF=%d, grid=%dx%d\n', ...
    cfg.fixedUniformFreeDOF,cfg.fixedUniformGridN,cfg.fixedUniformGridN);
fprintf(['Method switches  : %s=%d, target=%d, standard-master=%d, ', ...
         'time-AFEM=%d, accuracy-AFEM=%d, fixed-AFEM=%d, ', ...
         'DOF-uniform=%d, fixed-uniform=%d\n'], ...
    operatorLabel,cfg.runFNOSeededAFEM,cfg.runTargetSeededAFEM, ...
    cfg.runStandardAFEM,cfg.runTimeMatchedAFEM, ...
    cfg.runAccuracyMatchedAFEM,cfg.runFixedRefinementAFEM, ...
    cfg.runDOFMatchedUniform,cfg.runFixedUniform);
fprintf('Newton init mode : %s\n',cfg.newtonInitializationMode);
fprintf('Output directory : %s\n',cfg.outDir);
fprintf('Requested success: %d\n',requestedCases);
fprintf('Candidate pool   : %d exported test samples\n', ...
    numel(candidateSampleIds));
fprintf('Default plots    : first %d successful cases\n', ...
    min(cfg.numPlotSamples,requestedCases));
fprintf('============================================================\n\n');

%% ========================================================================
% Fixed mesh templates (built once). Their construction was not included in
% the original reported method times, so this changes only wall-clock cost.
% MATLAB copy-on-write preserves exactly the original node/element ordering.
% =========================================================================
[operatorBaseNodeTemplate,operatorBaseElemTemplate] = ...
    make_uniform_spacetime_mesh(cfg.operatorInitialGrid,cfg);
operatorBaseElemTemplate = nvb_label_initial_mesh( ...
    operatorBaseNodeTemplate,operatorBaseElemTemplate);

% This fixed operator-base mesh is used by both the FNO-score and
% target-score compilers for every sample.  Its quality is therefore
% computed once and reused verbatim; no mesh or numerical rule is changed.
cfg.operatorBaseQualityTemplate = mesh_angle_quality( ...
    operatorBaseNodeTemplate,operatorBaseElemTemplate);

if cfg.afemInitialGrid==cfg.operatorInitialGrid
    afemBaseNodeTemplate = operatorBaseNodeTemplate;
    afemBaseElemTemplate = operatorBaseElemTemplate;
else
    [afemBaseNodeTemplate,afemBaseElemTemplate] = ...
        make_uniform_spacetime_mesh(cfg.afemInitialGrid,cfg);
    afemBaseElemTemplate = nvb_label_initial_mesh( ...
        afemBaseNodeTemplate,afemBaseElemTemplate);
end

%% ========================================================================
% Main loop
% =========================================================================

nCases = requestedCases;
successfulCases = 0;
successfulSampleIds = zeros(1,requestedCases);
successfulPlotIds = zeros(1,0);

sampleResults = cell(requestedCases,1);
summaryRows = cell(0,23);
directRows = cell(0,20);

failureRows = cell(0,7);
failureVarNames = { ...
    'CandidateOrder','IntendedCase','TestID','SourceID', ...
    'Stage','Identifier','Message'};

for candidatePosition = 1:numel(candidateSampleIds)
    if successfulCases>=requestedCases
        break;
    end

    testId = candidateSampleIds(candidatePosition);
    sourceId = get_source_index(D,testId,nTest,nDatasetSamples);
    s = successfulCases+1;

    if isempty(cfg.plotSampleIds)
        makePlotsForThisSample = s<=cfg.numPlotSamples;
    else
        makePlotsForThisSample = ismember(testId,cfg.plotSampleIds);
    end

    currentStage = 'case_initialization';
    summaryCountBefore = size(summaryRows,1);
    directCountBefore = size(directRows,1);

    try

    predScore = get_score_sample( ...
        D.pred_score,testId,nTest,numel(queryX),numel(queryT));
    predScore = min(max(double(predScore),0),cfg.exportedScoreMaxLevel);

    targetScore = get_score_sample( ...
        D.target_score_test,testId,nTest,numel(queryX),numel(queryT));
    targetScore = min(max(double(targetScore),0),cfg.exportedScoreMaxLevel);

    scoreDifference = predScore-targetScore;
    scoreMAE = mean(abs(scoreDifference(:)));
    scoreRMSE = sqrt(mean(scoreDifference(:).^2));

    predEffectiveScore = ...
        cfg.operatorRefinementMultiplier*predScore + ...
        gridOffset + cfg.operatorScoreSafetyBias;
    targetEffectiveScore = ...
        cfg.operatorRefinementMultiplier*targetScore + ...
        gridOffset + cfg.targetScoreSafetyBias;

    predGeneration = score_to_generation_threshold03( ...
        predEffectiveScore,cfg.operatorMaxLevel, ...
        cfg.generationThreshold);
    targetGeneration = score_to_generation_threshold03( ...
        targetEffectiveScore,cfg.operatorMaxLevel, ...
        cfg.generationThreshold);
    underGenerationRate = mean( ...
        predGeneration(:)<targetGeneration(:));
    exactGenerationRate = mean( ...
        predGeneration(:)==targetGeneration(:));
    overGenerationRate = mean( ...
        predGeneration(:)>targetGeneration(:));

    exportedU0 = U0test(:,testId);
    exportedF = Ftest(:,testId);

    if cfg.useExactFourierInputs
        [u0fun,ffun,inputMeta] = make_exact_burgers_fourier_functions( ...
            BurgersExact,sourceId,cfg);
        u0vec = u0fun(xInput);
        fvec = ffun(xInput);

        inputMeta.u0SensorMaxAbsMismatch = ...
            max(abs(u0vec(:)-exportedU0(:)));
        inputMeta.forcingSensorMaxAbsMismatch = ...
            max(abs(fvec(:)-exportedF(:)));

        % The exported U0_test is reconstructed from the same Fourier
        % coefficients, so it should agree with the exact initial condition.
        u0Tolerance = cfg.exactInputAbsTolerance + ...
            cfg.exactInputRelTolerance*max(1,max(abs(exportedU0)));

        if inputMeta.u0SensorMaxAbsMismatch > u0Tolerance
            error('NOEM:ExactInputMappingMismatch', ...
                ['Exact Fourier initial condition does not match the exported ', ...
                'U0_test sensors for test ID %d, source ID %d. ', ...
                'max|du0|=%.3e, tolerance=%.3e. ', ...
                'This indicates an incorrect sample/index mapping.'], ...
                testId,sourceId, ...
                inputMeta.u0SensorMaxAbsMismatch,u0Tolerance);
        end

        % F_test is only an exported finite-sensor representation. It can differ
        % slightly from the exact Fourier forcing because the original forcing was
        % sampled on a fixed grid and stored in single precision. The FEM, AFEM
        % estimator, and spectral reference continue to use the exact Fourier ffun.
        forcingExportAuditTolerance = 2.0e-3;

        if inputMeta.forcingSensorMaxAbsMismatch > ...
                forcingExportAuditTolerance
            warning('NOEM:ExportedForcingMismatch', ...
                ['Exported F_test differs noticeably from the exact Fourier ', ...
                'forcing for test ID %d, source ID %d: max|df|=%.3e. ', ...
                'The exact Fourier forcing is still used by the FEM.'], ...
                testId,sourceId, ...
                inputMeta.forcingSensorMaxAbsMismatch);
        end
    else
        % Legacy fallback: periodic PCHIP interpolation of exported sensors.
        u0vec = exportedU0;
        fvec = exportedF;
        u0fun = make_discrete_u0_function(xInput,u0vec,cfg);
        ffun = make_discrete_u0_function(xInput,fvec,cfg);
        inputMeta = struct( ...
            'sourceId',sourceId, ...
            'representation','periodic-pchip-exported-sensors', ...
            'u0SensorMaxAbsMismatch',0, ...
            'forcingSensorMaxAbsMismatch',0);
    end
    cfg.forcingFun = ffun;

    fprintf('\n============================================================\n');
    fprintf('Sample %d/%d: test ID=%d, source ID=%d, plots=%d\n', ...
        s,nCases,testId,sourceId,makePlotsForThisSample);
    fprintf(['Exact-input audit: max|u0_Fourier-u0_export|=%.3e, ', ...
             'max|f_Fourier-f_export|=%.3e\n'], ...
        inputMeta.u0SensorMaxAbsMismatch, ...
        inputMeta.forcingSensorMaxAbsMismatch);
    fprintf(['Predicted score: min %.3f, mean %.3f, max %.3f, ' ...
             'P95 %.3f, P99 %.3f\n'], ...
        min(predScore(:)),mean(predScore(:)),max(predScore(:)), ...
        simple_percentile(predScore(:),95), ...
        simple_percentile(predScore(:),99));
    fprintf(['Target score   : min %.3f, mean %.3f, max %.3f, ' ...
             'P95 %.3f, P99 %.3f\n'], ...
        min(targetScore(:)),mean(targetScore(:)),max(targetScore(:)), ...
        simple_percentile(targetScore(:),95), ...
        simple_percentile(targetScore(:),99));
    fprintf(['Score comparison: MAE=%.3e, RMSE=%.3e, ' ...
             'under/exact/over generation=%.2f%% / %.2f%% / %.2f%%\n'], ...
        scoreMAE,scoreRMSE,100*underGenerationRate, ...
        100*exactGenerationRate,100*overGenerationRate);
    fprintf('============================================================\n');

    % High-resolution reference; excluded from every method time.
    currentStage = 'spectral_reference';
    refTimer = tic;
    ref = solve_burgers_fourier_reference(u0fun,cfg);
    referenceTime = toc(refTimer);
    Fref = make_ref_interpolant(ref); % reuse exact same interpolant
    fprintf('Spectral reference time = %.3f s\n',referenceTime);

    % Determine internal dependencies. Time-matched AFEM,
    % accuracy-matched AFEM and DOF-matched uniform FEM use the FNO hybrid
    % as a target. Fixed-refinement AFEM is independent of the FNO result.
    needFNOReference = cfg.runFNOSeededAFEM || ...
        cfg.runTimeMatchedAFEM || cfg.runAccuracyMatchedAFEM || ...
        cfg.runDOFMatchedUniform;
    needTargetMethod = cfg.runTargetSeededAFEM;
    needStandardAFEM = runAnyStandardAFEM;
    needDOFMatchedUniform = cfg.runDOFMatchedUniform;
    needFixedUniform = cfg.runFixedUniform;

    if needFNOReference && ~cfg.runFNOSeededAFEM
        fprintf(['%s hybrid will be computed only as an internal matching ', ...
                 'dependency and omitted from reported methods.\n'], ...
            operatorLabel);
    end

    initializationMode = cfg.newtonInitializationMode;
    useCoarseInterpolation = strcmp( ...
        initializationMode,'coarse_interpolation');

    % Initialize labels and optional outputs so all packaging paths are safe.
    operatorInitializationLabel = 'not_run';
    targetInitializationLabel = 'not_run';
    dofUniformInitializationLabel = 'not_run';
    fixedUniformInitializationLabel = 'not_run';
    afemBaseInitializationMode = 'not_run';

    operatorHybrid = [];
    targetHybrid = [];
    afemSelected = [];
    afemHistory = struct();
    operatorCorrectionHistory = struct();
    targetCorrectionHistory = struct();
    operatorMeshStats = struct();
    targetMeshStats = struct();

    % The operator-base topology is also the common coarse-start topology.
    operatorBaseNode = operatorBaseNodeTemplate;
    operatorBaseElem = operatorBaseElemTemplate;
    operatorBaseDOF = count_free_dofs(operatorBaseNode,cfg);
    operatorBaseU = [];
    operatorBaseNewton = [];
    operatorBaseSolveTime = 0.0;
    operatorBaseErr = [];
    operatorBaseWasSolved = false;

    needCommonCoarseSolve = useCoarseInterpolation && ( ...
        needFNOReference || needTargetMethod || ...
        needDOFMatchedUniform || needFixedUniform || needStandardAFEM);

    if needCommonCoarseSolve
        currentStage = 'operator_base_solve_for_coarse_interpolation';
        operatorBaseTimer = tic;
        [operatorBaseU,operatorBaseNewton] = solve_burgers_newton_st( ...
            operatorBaseNode,operatorBaseElem,u0fun,cfg, ...
            'initial_condition');
        operatorBaseSolveTime = toc(operatorBaseTimer);
        operatorBaseErr = compute_fem_vs_reference_error( ...
            operatorBaseNode,operatorBaseElem,operatorBaseU,ref,cfg,Fref);
        operatorBaseWasSolved = true;

        fprintf(['Operator base %dx%d for interpolation: DOF=%d, ', ...
                 'elems=%d, ST rel=%.3e, solve time=%.3f s\n'], ...
            cfg.operatorInitialGrid,cfg.operatorInitialGrid, ...
            operatorBaseDOF,size(operatorBaseElem,1), ...
            operatorBaseErr.spaceTimeL2Rel,operatorBaseSolveTime);
    else
        fprintf(['Operator base %dx%d: geometry template only; ', ...
                 'no preliminary FEM solve in mode %s.\n'], ...
            cfg.operatorInitialGrid,cfg.operatorInitialGrid, ...
            initializationMode);
    end

    % Standard-AFEM stage 0 is created only when that method group is on.
    afemBaseNode = [];
    afemBaseElem = [];
    afemBaseU = [];
    afemBaseNewton = [];
    afemBaseSolveTime = NaN;
    afemBaseDOF = NaN;
    afemBaseErr = [];

    if needStandardAFEM
        if useCoarseInterpolation && ...
                cfg.afemInitialGrid==cfg.operatorInitialGrid
            afemBaseNode = operatorBaseNode;
            afemBaseElem = operatorBaseElem;
            afemBaseU = operatorBaseU;
            afemBaseNewton = operatorBaseNewton;
            afemBaseSolveTime = operatorBaseSolveTime;
            afemBaseDOF = operatorBaseDOF;
            afemBaseErr = operatorBaseErr;
            afemBaseInitializationMode = 'initial_condition';
        else
            afemBaseNode = afemBaseNodeTemplate;
            afemBaseElem = afemBaseElemTemplate;
            if useCoarseInterpolation
                afemBaseInitializationMode = 'initial_condition';
            else
                afemBaseInitializationMode = initializationMode;
            end

            currentStage = ['afem_base_solve_' afemBaseInitializationMode];
            afemBaseTimer = tic;
            [afemBaseU,afemBaseNewton] = solve_burgers_newton_st( ...
                afemBaseNode,afemBaseElem,u0fun,cfg, ...
                afemBaseInitializationMode);
            afemBaseSolveTime = toc(afemBaseTimer);
            afemBaseDOF = count_free_dofs(afemBaseNode,cfg);
            afemBaseErr = compute_fem_vs_reference_error( ...
                afemBaseNode,afemBaseElem,afemBaseU,ref,cfg,Fref);
        end

        fprintf(['AFEM base %dx%d: init=%s, DOF=%d, elems=%d, ', ...
                 'ST rel=%.3e, solve time=%.3f s\n'], ...
            cfg.afemInitialGrid,cfg.afemInitialGrid, ...
            afemBaseInitializationMode,afemBaseDOF,size(afemBaseElem,1), ...
            afemBaseErr.spaceTimeL2Rel,afemBaseSolveTime);
    end

    %% FNO predicted-score FEM followed by fixed AFEM corrections.
    if needFNOReference
        currentStage = 'fno_score_mesh';
        operatorMeshTimer = tic;
        [operatorSeedNode,operatorSeedElem,operatorMeshStats] = ...
            build_continuous_score_operator_mesh_fast_exact( ...
                predScore,queryX,queryT,cfg, ...
                operatorBaseNodeTemplate,operatorBaseElemTemplate);
        operatorMeshTime = toc(operatorMeshTimer);

        currentStage = ['fno_seed_fem_solve_' initializationMode];
        if useCoarseInterpolation
            operatorInitialArgument = interpolate_to_new_nodes( ...
                operatorBaseNode,operatorBaseElem,operatorBaseU, ...
                operatorSeedNode);
            operatorInitializationLabel = ...
                'coarse_solution_interpolation';
        else
            operatorInitialArgument = initializationMode;
            operatorInitializationLabel = initializationMode;
        end

        operatorSolveTimer = tic;
        [operatorSeedU,operatorSeedNewton] = solve_burgers_newton_st( ...
            operatorSeedNode,operatorSeedElem,u0fun,cfg, ...
            operatorInitialArgument);
        operatorSeedNewton.initializationMode = ...
            operatorInitializationLabel;
        operatorSolveTime = toc(operatorSolveTimer);

        operatorSeedTime = cfg.operatorInferenceTimePerSample + ...
            operatorBaseSolveTime + operatorMeshTime + operatorSolveTime;
        operatorSeedDOF = count_free_dofs(operatorSeedNode,cfg);
        operatorSeedErr = compute_fem_vs_reference_error( ...
            operatorSeedNode,operatorSeedElem,operatorSeedU,ref,cfg,Fref);
        operatorSeedQuality = operatorMeshStats.finalQuality;

        fprintf(['%s score seed: init=%s, DOF=%d, elems=%d, ', ...
                 'ST rel=%.3e, time=%.3f s ', ...
                 '(base %.3f + infer %.6f + mesh %.3f + solve %.3f)\n'], ...
            operatorLabel,operatorInitializationLabel,operatorSeedDOF, ...
            size(operatorSeedElem,1),operatorSeedErr.spaceTimeL2Rel, ...
            operatorSeedTime,operatorBaseSolveTime, ...
            cfg.operatorInferenceTimePerSample,operatorMeshTime, ...
            operatorSolveTime);

        currentStage = sprintf( ...
            'fno_%d_afem_corrections',cfg.hybridAFEMCycles);
        [operatorHybrid,operatorCorrectionHistory] = ...
            run_fixed_afem_cycles_fast_exact( ...
                operatorSeedNode,operatorSeedElem,operatorSeedU, ...
                operatorSeedTime,operatorSeedErr,operatorSeedNewton, ...
                u0fun,ref,cfg,cfg.hybridAFEMCycles, ...
                sprintf('%s hybrid',operatorLabel),Fref);

        operatorNode = operatorHybrid.node;
        operatorElem = operatorHybrid.elem;
        operatorU = operatorHybrid.u;
        operatorDOF = operatorHybrid.dof;
        operatorTime = operatorHybrid.timeSec;
        operatorErr = operatorHybrid.error;
        operatorNewton = operatorHybrid.newtonInfo;
        operatorQuality = mesh_angle_quality(operatorNode,operatorElem);

        fprintf(['%s predicted-score FEM + %d AFEM: reported=%d, ', ...
                 'DOF=%d, elems=%d, ST rel=%.3e, final rel=%.3e, ', ...
                 'total time=%.3f s\n'], ...
            operatorLabel,cfg.hybridAFEMCycles,cfg.runFNOSeededAFEM,operatorDOF, ...
            size(operatorElem,1),operatorErr.spaceTimeL2Rel, ...
            operatorErr.finalTimeL2Rel,operatorTime);
    end

    %% Ground-truth target-score FEM followed by the same corrections.
    if needTargetMethod
        targetCfg = cfg;
        targetCfg.operatorScoreSafetyBias = cfg.targetScoreSafetyBias;

        currentStage = 'target_score_mesh';
        targetMeshTimer = tic;
        [targetSeedNode,targetSeedElem,targetMeshStats] = ...
            build_continuous_score_operator_mesh_fast_exact( ...
                targetScore,queryX,queryT,targetCfg, ...
                operatorBaseNodeTemplate,operatorBaseElemTemplate);
        targetMeshTime = toc(targetMeshTimer);

        currentStage = ['target_seed_fem_solve_' initializationMode];
        if useCoarseInterpolation
            targetInitialArgument = interpolate_to_new_nodes( ...
                operatorBaseNode,operatorBaseElem,operatorBaseU, ...
                targetSeedNode);
            targetInitializationLabel = ...
                'coarse_solution_interpolation';
        else
            targetInitialArgument = initializationMode;
            targetInitializationLabel = initializationMode;
        end

        targetSolveTimer = tic;
        [targetSeedU,targetSeedNewton] = solve_burgers_newton_st( ...
            targetSeedNode,targetSeedElem,u0fun,cfg,targetInitialArgument);
        targetSeedNewton.initializationMode = targetInitializationLabel;
        targetSolveTime = toc(targetSolveTimer);

        targetSeedTime = cfg.targetScoreAccessTimePerSample + ...
            operatorBaseSolveTime + targetMeshTime + targetSolveTime;
        targetSeedDOF = count_free_dofs(targetSeedNode,cfg);
        targetSeedErr = compute_fem_vs_reference_error( ...
            targetSeedNode,targetSeedElem,targetSeedU,ref,cfg,Fref);
        targetSeedQuality = targetMeshStats.finalQuality;

        fprintf(['Target-score seed: init=%s, DOF=%d, elems=%d, ', ...
                 'ST rel=%.3e, time=%.3f s\n'], ...
            targetInitializationLabel,targetSeedDOF, ...
            size(targetSeedElem,1),targetSeedErr.spaceTimeL2Rel, ...
            targetSeedTime);

        currentStage = sprintf( ...
            'target_%d_afem_corrections',cfg.hybridAFEMCycles);
        [targetHybrid,targetCorrectionHistory] = ...
            run_fixed_afem_cycles_fast_exact( ...
                targetSeedNode,targetSeedElem,targetSeedU, ...
                targetSeedTime,targetSeedErr,targetSeedNewton, ...
                u0fun,ref,cfg,cfg.hybridAFEMCycles, ...
                'Target-score hybrid',Fref);

        targetNode = targetHybrid.node;
        targetElem = targetHybrid.elem;
        targetU = targetHybrid.u;
        targetDOF = targetHybrid.dof;
        targetTime = targetHybrid.timeSec;
        targetErr = targetHybrid.error;
        targetNewton = targetHybrid.newtonInfo;
        targetQuality = mesh_angle_quality(targetNode,targetElem);

        fprintf(['Ground-truth target-score FEM + %d AFEM: ', ...
                 'DOF=%d, elems=%d, ST rel=%.3e, final rel=%.3e, ', ...
                 'total time=%.3f s\n'], ...
            cfg.hybridAFEMCycles,targetDOF,size(targetElem,1), ...
            targetErr.spaceTimeL2Rel,targetErr.finalTimeL2Rel,targetTime);

        if cfg.runFNOSeededAFEM
            fprintf(['Direct hybrid %s/target ratios: DOF %.4f, ', ...
                     'ST-error %.4f, final-error %.4f, online-time %.4f\n'], ...
                operatorLabel,operatorDOF/max(targetDOF,1), ...
                operatorErr.spaceTimeL2Rel/ ...
                    max(targetErr.spaceTimeL2Rel,eps), ...
                operatorErr.finalTimeL2Rel/ ...
                    max(targetErr.finalTimeL2Rel,eps), ...
                operatorTime/max(targetTime,eps));
        end
    end

    %% Standard-AFEM trajectory with independently enabled criteria.
    standardTargetDOF = NaN;
    standardTargetTime = NaN;
    standardTargetError = NaN;
    timeAFEM = [];
    accuracyAFEM = [];
    fixedAFEM = [];
    timeAFEMQuality = [];
    accuracyAFEMQuality = [];
    fixedQuality = [];

    if needFNOReference
        standardTargetDOF = operatorDOF;
    end
    if cfg.runTimeMatchedAFEM
        standardTargetTime = operatorTime;
    end
    if cfg.runAccuracyMatchedAFEM
        standardTargetError = operatorErr.spaceTimeL2Rel;
    end

    if needStandardAFEM
        currentStage = 'standard_afem_enabled_criteria';
        [afemSelected,afemHistory] = ...
            run_afem_for_two_targets_and_fixed_fast_exact( ...
                afemBaseNode,afemBaseElem,afemBaseU, ...
                afemBaseSolveTime,afemBaseErr,afemBaseNewton, ...
                standardTargetTime,standardTargetError, ...
                u0fun,ref,cfg,Fref);

        if cfg.runTimeMatchedAFEM
            timeAFEM = afemSelected.time;
            timeAFEMQuality = mesh_angle_quality( ...
                timeAFEM.node,timeAFEM.elem);
            fprintf(['Time-matched AFEM: stage=%d, time=%.3f s ', ...
                     '(%s target %.3f, gap %.2f%%), DOF=%d, ', ...
                     'ST rel=%.3e\n'], ...
                operatorLabel,timeAFEM.stage,timeAFEM.timeSec,standardTargetTime, ...
                100*timeAFEM.matchRelativeGap,timeAFEM.dof, ...
                timeAFEM.error.spaceTimeL2Rel);
        end

        if cfg.runAccuracyMatchedAFEM
            accuracyAFEM = afemSelected.accuracy;
            accuracyAFEMQuality = mesh_angle_quality( ...
                accuracyAFEM.node,accuracyAFEM.elem);
            fprintf(['Accuracy-matched AFEM: stage=%d, ST rel=%.3e ', ...
                     '(%s target %.3e, gap %.2f%%), DOF=%d, ', ...
                     'time=%.3f s\n'], ...
                operatorLabel,accuracyAFEM.stage,accuracyAFEM.error.spaceTimeL2Rel, ...
                standardTargetError, ...
                100*accuracyAFEM.matchRelativeGap,accuracyAFEM.dof, ...
                accuracyAFEM.timeSec);
        end

        if cfg.runFixedRefinementAFEM
            fixedAFEM = afemSelected.fixed;
            fixedQuality = mesh_angle_quality( ...
                fixedAFEM.node,fixedAFEM.elem);
            fprintf(['Fixed-%d-refine AFEM: stage=%d, DOF=%d, elems=%d, ', ...
                     'ST rel=%.3e, final rel=%.3e, time=%.3f s\n'], ...
                cfg.fixedAFEMRefineCycles,fixedAFEM.stage,fixedAFEM.dof, ...
                size(fixedAFEM.elem,1),fixedAFEM.error.spaceTimeL2Rel, ...
                fixedAFEM.error.finalTimeL2Rel,fixedAFEM.timeSec);
        end
    end

    %% Smallest uniform FEM whose free DOF is strictly above FNO-hybrid DOF.
    if needDOFMatchedUniform
        currentStage = 'uniform_fem_strictly_above_fno_hybrid_dof';
        dofUniformN = choose_uniform_grid_strictly_above_dof( ...
            operatorDOF,cfg);

        dofUniformMeshTimer = tic;
        [dofUniformNode,dofUniformElem] = ...
            make_uniform_spacetime_mesh(dofUniformN,cfg);
        dofUniformMeshTime = toc(dofUniformMeshTimer);

        if useCoarseInterpolation
            dofUniformInitialArgument = interpolate_to_new_nodes( ...
                operatorBaseNode,operatorBaseElem,operatorBaseU, ...
                dofUniformNode);
            dofUniformInitializationLabel = ...
                'coarse_solution_interpolation';
        else
            dofUniformInitialArgument = initializationMode;
            dofUniformInitializationLabel = initializationMode;
        end

        dofUniformSolveTimer = tic;
        [dofUniformU,dofUniformNewton] = solve_burgers_newton_st( ...
            dofUniformNode,dofUniformElem,u0fun,cfg, ...
            dofUniformInitialArgument);
        dofUniformNewton.initializationMode = ...
            dofUniformInitializationLabel;
        dofUniformSolveTime = toc(dofUniformSolveTimer);

        dofUniformTime = operatorBaseSolveTime + ...
            dofUniformMeshTime + dofUniformSolveTime;
        dofUniformDOF = count_free_dofs(dofUniformNode,cfg);
        dofUniformErr = compute_fem_vs_reference_error( ...
            dofUniformNode,dofUniformElem,dofUniformU,ref,cfg,Fref);
        dofUniformQuality = ...
            mesh_angle_quality(dofUniformNode,dofUniformElem);

        fprintf(['Smallest strictly-above uniform %dx%d: init=%s, ', ...
                 'DOF=%d (%s %d, excess %d, overshoot %.2f%%), ', ...
                 'ST rel=%.3e, final rel=%.3e, time=%.3f s\n'], ...
            dofUniformN,dofUniformN,dofUniformInitializationLabel,operatorLabel, ...
            dofUniformDOF,operatorDOF,dofUniformDOF-operatorDOF, ...
            100*(dofUniformDOF-operatorDOF)/max(operatorDOF,1), ...
            dofUniformErr.spaceTimeL2Rel, ...
            dofUniformErr.finalTimeL2Rel,dofUniformTime);
    end

    %% Fixed-free-DOF uniform FEM.
    if needFixedUniform
        currentStage = 'fixed_free_dof_uniform_fem';
        fixedUniformN = cfg.fixedUniformGridN;

        fixedUniformMeshTimer = tic;
        [fixedUniformNode,fixedUniformElem] = ...
            make_uniform_spacetime_mesh(fixedUniformN,cfg);
        fixedUniformMeshTime = toc(fixedUniformMeshTimer);

        if useCoarseInterpolation
            fixedUniformInitialArgument = interpolate_to_new_nodes( ...
                operatorBaseNode,operatorBaseElem,operatorBaseU, ...
                fixedUniformNode);
            fixedUniformInitializationLabel = ...
                'coarse_solution_interpolation';
        else
            fixedUniformInitialArgument = initializationMode;
            fixedUniformInitializationLabel = initializationMode;
        end

        fixedUniformSolveTimer = tic;
        [fixedUniformU,fixedUniformNewton] = solve_burgers_newton_st( ...
            fixedUniformNode,fixedUniformElem,u0fun,cfg, ...
            fixedUniformInitialArgument);
        fixedUniformNewton.initializationMode = ...
            fixedUniformInitializationLabel;
        fixedUniformSolveTime = toc(fixedUniformSolveTimer);

        fixedUniformTime = operatorBaseSolveTime + ...
            fixedUniformMeshTime + fixedUniformSolveTime;
        fixedUniformDOF = count_free_dofs(fixedUniformNode,cfg);
        fixedUniformErr = compute_fem_vs_reference_error( ...
            fixedUniformNode,fixedUniformElem,fixedUniformU,ref,cfg,Fref);
        fixedUniformQuality = ...
            mesh_angle_quality(fixedUniformNode,fixedUniformElem);

        if fixedUniformDOF~=cfg.fixedUniformFreeDOF
            error(['Fixed uniform mesh produced %d free DOFs instead of ', ...
                   'the configured %d.'], ...
                fixedUniformDOF,cfg.fixedUniformFreeDOF);
        end

        fprintf(['Fixed uniform %dx%d: init=%s, DOF=%d, elems=%d, ', ...
                 'ST rel=%.3e, final rel=%.3e, time=%.3f s\n'], ...
            fixedUniformN,fixedUniformN,fixedUniformInitializationLabel, ...
            fixedUniformDOF,size(fixedUniformElem,1), ...
            fixedUniformErr.spaceTimeL2Rel, ...
            fixedUniformErr.finalTimeL2Rel,fixedUniformTime);
    end

    % Build dynamic plotting/reporting collections from enabled rows only.
    methodNames = {};
    methodNodes = {};
    methodElems = {};
    methodU = {};
    methodDOF = zeros(1,0);
    methodTime = zeros(1,0);
    methodErr = {};
    methodQuality = {};

    if cfg.runFNOSeededAFEM
        methodNames{end+1} = sprintf( ...
            '%s predicted-score FEM + %d AFEM', ...
            operatorLabel,cfg.hybridAFEMCycles);
        methodNodes{end+1} = operatorNode;
        methodElems{end+1} = operatorElem;
        methodU{end+1} = operatorU;
        methodDOF(end+1) = operatorDOF;
        methodTime(end+1) = operatorTime;
        methodErr{end+1} = operatorErr;
        methodQuality{end+1} = operatorQuality;
    end
    if cfg.runTargetSeededAFEM
        methodNames{end+1} = sprintf( ...
            'Ground-truth score FEM + %d AFEM',cfg.hybridAFEMCycles);
        methodNodes{end+1} = targetNode;
        methodElems{end+1} = targetElem;
        methodU{end+1} = targetU;
        methodDOF(end+1) = targetDOF;
        methodTime(end+1) = targetTime;
        methodErr{end+1} = targetErr;
        methodQuality{end+1} = targetQuality;
    end
    if cfg.runTimeMatchedAFEM
        methodNames{end+1} = 'Time-matched AFEM';
        methodNodes{end+1} = timeAFEM.node;
        methodElems{end+1} = timeAFEM.elem;
        methodU{end+1} = timeAFEM.u;
        methodDOF(end+1) = timeAFEM.dof;
        methodTime(end+1) = timeAFEM.timeSec;
        methodErr{end+1} = timeAFEM.error;
        methodQuality{end+1} = timeAFEMQuality;
    end
    if cfg.runAccuracyMatchedAFEM
        methodNames{end+1} = 'Accuracy-matched AFEM';
        methodNodes{end+1} = accuracyAFEM.node;
        methodElems{end+1} = accuracyAFEM.elem;
        methodU{end+1} = accuracyAFEM.u;
        methodDOF(end+1) = accuracyAFEM.dof;
        methodTime(end+1) = accuracyAFEM.timeSec;
        methodErr{end+1} = accuracyAFEM.error;
        methodQuality{end+1} = accuracyAFEMQuality;
    end
    if cfg.runFixedRefinementAFEM
        methodNames{end+1} = sprintf( ...
            'Fixed-%d-refine AFEM',cfg.fixedAFEMRefineCycles);
        methodNodes{end+1} = fixedAFEM.node;
        methodElems{end+1} = fixedAFEM.elem;
        methodU{end+1} = fixedAFEM.u;
        methodDOF(end+1) = fixedAFEM.dof;
        methodTime(end+1) = fixedAFEM.timeSec;
        methodErr{end+1} = fixedAFEM.error;
        methodQuality{end+1} = fixedQuality;
    end
    if cfg.runDOFMatchedUniform
        methodNames{end+1} = sprintf( ...
            'Smallest-strictly-above-%s-DOF uniform FEM',operatorLabel);
        methodNodes{end+1} = dofUniformNode;
        methodElems{end+1} = dofUniformElem;
        methodU{end+1} = dofUniformU;
        methodDOF(end+1) = dofUniformDOF;
        methodTime(end+1) = dofUniformTime;
        methodErr{end+1} = dofUniformErr;
        methodQuality{end+1} = dofUniformQuality;
    end
    if cfg.runFixedUniform
        methodNames{end+1} = sprintf( ...
            'Fixed-%d-DOF uniform FEM',cfg.fixedUniformFreeDOF);
        methodNodes{end+1} = fixedUniformNode;
        methodElems{end+1} = fixedUniformElem;
        methodU{end+1} = fixedUniformU;
        methodDOF(end+1) = fixedUniformDOF;
        methodTime(end+1) = fixedUniformTime;
        methodErr{end+1} = fixedUniformErr;
        methodQuality{end+1} = fixedUniformQuality;
    end

    nReportedMethods = numel(methodNames);
    if nReportedMethods<1
        error('No reported method remains after applying the switches.');
    end

    currentStage = 'plotting_and_packaging';
    caseTag = sprintf('sample_%03d_test_%04d_source_%05d', ...
        s,testId,sourceId);

    if makePlotsForThisSample && cfg.makeMeshPlots
        plot_six_meshes_generic( ...
            methodNodes,methodElems,methodDOF,methodQuality, ...
            methodNames,caseTag,cfg);
    end

    if makePlotsForThisSample && cfg.makeScorePlots
        if needFNOReference && needTargetMethod
            plot_fno_target_score_comparison( ...
                predScore,targetScore,queryX,queryT,u0vec,xInput, ...
                operatorSeedNode,operatorSeedElem,operatorMeshStats, ...
                targetSeedNode,targetSeedElem,targetMeshStats,caseTag,cfg);
        else
            fprintf(['  Score/seed comparison plot skipped because both ', ...
                     '%s and target-score meshes were not computed.\n'], ...
                operatorLabel);
        end
    end

    pointwiseErrorData = [];
    if makePlotsForThisSample && cfg.makePointwisePlots
        pointwiseErrorData = plot_six_pointwise_absolute_error( ...
            methodNodes,methodElems,methodU,ref,Fref,methodNames, ...
            methodErr,methodTime,methodDOF,caseTag,cfg);
    end

    if makePlotsForThisSample && cfg.makeAFEMConvergencePlots && ...
            needStandardAFEM
        plot_afem_selected_stage_convergence( ...
            afemHistory,standardTargetDOF,standardTargetTime, ...
            standardTargetError,afemSelected,caseTag,cfg);
    end

    % Per-sample result.
    sampleResult = struct();
    sampleResult.testId = testId;
    sampleResult.sourceId = sourceId;
    sampleResult.wasPlotted = makePlotsForThisSample;
    sampleResult.xInput = xInput;
    sampleResult.u0 = u0vec;
    sampleResult.forcing = fvec;
    sampleResult.inputReconstruction = inputMeta;
    sampleResult.queryX = queryX;
    sampleResult.queryT = queryT;
    sampleResult.predictedScore = single(predScore);
    sampleResult.targetScore = single(targetScore);
    sampleResult.scoreComparison = struct( ...
        'mae',scoreMAE, ...
        'rmse',scoreRMSE, ...
        'underGenerationRate',underGenerationRate, ...
        'exactGenerationRate',exactGenerationRate, ...
        'overGenerationRate',overGenerationRate);
    sampleResult.referenceTime = referenceTime;
    sampleResult.methodSwitches = struct( ...
        'runFNOSeededAFEM',cfg.runFNOSeededAFEM, ...
        'runTargetSeededAFEM',cfg.runTargetSeededAFEM, ...
        'runStandardAFEM',cfg.runStandardAFEM, ...
        'runTimeMatchedAFEM',cfg.runTimeMatchedAFEM, ...
        'runAccuracyMatchedAFEM',cfg.runAccuracyMatchedAFEM, ...
        'runFixedRefinementAFEM',cfg.runFixedRefinementAFEM, ...
        'runDOFMatchedUniform',cfg.runDOFMatchedUniform, ...
        'runFixedUniform',cfg.runFixedUniform);
    sampleResult.internalDependencies = struct( ...
        'fnoComputed',needFNOReference, ...
        'fnoReported',cfg.runFNOSeededAFEM, ...
        'fnoUsedForTimeMatchedAFEM',cfg.runTimeMatchedAFEM, ...
        'fnoUsedForAccuracyMatchedAFEM',cfg.runAccuracyMatchedAFEM, ...
        'fnoUsedForDOFUniform',cfg.runDOFMatchedUniform);
    sampleResult.newtonInitializationPolicy = struct( ...
        'selectedMode',initializationMode, ...
        'fnoSeed',operatorInitializationLabel, ...
        'targetSeed',targetInitializationLabel, ...
        'dofMatchedUniform',dofUniformInitializationLabel, ...
        'fixedUniform',fixedUniformInitializationLabel, ...
        'afemStage0',afemBaseInitializationMode, ...
        'laterAFEM','previous_afem_solution_interpolation');
    sampleResult.operatorBase = struct( ...
        'gridN',cfg.operatorInitialGrid, ...
        'dof',operatorBaseDOF, ...
        'elements',size(operatorBaseElem,1), ...
        'solveTime',operatorBaseSolveTime, ...
        'error',operatorBaseErr, ...
        'newtonInfo',operatorBaseNewton, ...
        'wasSolved',operatorBaseWasSolved);

    if needStandardAFEM
        sampleResult.afemBase = struct( ...
            'gridN',cfg.afemInitialGrid, ...
            'dof',afemBaseDOF, ...
            'elements',size(afemBaseElem,1), ...
            'solveTime',afemBaseSolveTime, ...
            'error',afemBaseErr, ...
            'newtonInfo',afemBaseNewton);
        sampleResult.baseSolveTime = afemBaseSolveTime;
        sampleResult.baseNewton = afemBaseNewton;
    end

    if needFNOReference
        sampleResult.operatorSeed = pack_method_result( ...
            operatorSeedNode,operatorSeedElem,operatorSeedU, ...
            operatorSeedDOF,operatorSeedTime,operatorSeedErr, ...
            operatorSeedQuality,operatorSeedNewton);
        sampleResult.operatorSeed.meshBuildTime = operatorMeshTime;
        sampleResult.operatorSeed.solveTime = operatorSolveTime;
        sampleResult.operatorSeed.inferenceTime = ...
            cfg.operatorInferenceTimePerSample;
        sampleResult.operatorSeed.initializationMode = ...
            operatorInitializationLabel;

        sampleResult.operatorHybrid = pack_method_result( ...
            operatorNode,operatorElem,operatorU,operatorDOF, ...
            operatorTime,operatorErr,operatorQuality,operatorNewton);
        sampleResult.operatorHybrid.reported = cfg.runFNOSeededAFEM;
        sampleResult.operatorHybrid.seedTime = operatorSeedTime;
        sampleResult.operatorHybrid.correctionTime = ...
            operatorHybrid.correctionTimeSec;
        sampleResult.operatorHybrid.correctionCycles = ...
            cfg.hybridAFEMCycles;
        sampleResult.operatorHybrid.correctionHistory = ...
            operatorCorrectionHistory;
    end

    if needTargetMethod
        sampleResult.targetSeed = pack_method_result( ...
            targetSeedNode,targetSeedElem,targetSeedU,targetSeedDOF, ...
            targetSeedTime,targetSeedErr,targetSeedQuality,targetSeedNewton);
        sampleResult.targetSeed.meshBuildTime = targetMeshTime;
        sampleResult.targetSeed.solveTime = targetSolveTime;
        sampleResult.targetSeed.scoreAccessTime = ...
            cfg.targetScoreAccessTimePerSample;
        sampleResult.targetSeed.initializationMode = ...
            targetInitializationLabel;

        sampleResult.targetHybrid = pack_method_result( ...
            targetNode,targetElem,targetU,targetDOF, ...
            targetTime,targetErr,targetQuality,targetNewton);
        sampleResult.targetHybrid.seedTime = targetSeedTime;
        sampleResult.targetHybrid.correctionTime = ...
            targetHybrid.correctionTimeSec;
        sampleResult.targetHybrid.correctionCycles = ...
            cfg.hybridAFEMCycles;
        sampleResult.targetHybrid.correctionHistory = ...
            targetCorrectionHistory;
    end

    if cfg.runTimeMatchedAFEM
        sampleResult.afemTime = pack_method_result( ...
            timeAFEM.node,timeAFEM.elem,timeAFEM.u,timeAFEM.dof, ...
            timeAFEM.timeSec,timeAFEM.error,timeAFEMQuality, ...
            timeAFEM.newtonInfo);
        sampleResult.afemTime.selection = ...
            strip_afem_mesh_fields(timeAFEM);
    end

    if cfg.runAccuracyMatchedAFEM
        sampleResult.afemAccuracy = pack_method_result( ...
            accuracyAFEM.node,accuracyAFEM.elem,accuracyAFEM.u, ...
            accuracyAFEM.dof,accuracyAFEM.timeSec, ...
            accuracyAFEM.error,accuracyAFEMQuality, ...
            accuracyAFEM.newtonInfo);
        sampleResult.afemAccuracy.selection = ...
            strip_afem_mesh_fields(accuracyAFEM);
    end

    if cfg.runFixedRefinementAFEM
        sampleResult.afemFixed = pack_method_result( ...
            fixedAFEM.node,fixedAFEM.elem,fixedAFEM.u,fixedAFEM.dof, ...
            fixedAFEM.timeSec,fixedAFEM.error,fixedQuality, ...
            fixedAFEM.newtonInfo);
        sampleResult.afemFixed.selection = ...
            strip_afem_mesh_fields(fixedAFEM);
        sampleResult.afemFixed.refineCycles = ...
            cfg.fixedAFEMRefineCycles;
    end

    if needDOFMatchedUniform
        sampleResult.uniform = pack_method_result( ...
            dofUniformNode,dofUniformElem,dofUniformU,dofUniformDOF, ...
            dofUniformTime,dofUniformErr,dofUniformQuality, ...
            dofUniformNewton);
        sampleResult.uniform.gridN = dofUniformN;
        sampleResult.uniform.meshTime = dofUniformMeshTime;
        sampleResult.uniform.initializationMode = ...
            dofUniformInitializationLabel;
        sampleResult.uniform.requestedInitializationMode = ...
            initializationMode;
        sampleResult.uniform.coarseBaseSolveTimeIncluded = ...
            useCoarseInterpolation;
        sampleResult.uniform.matchTargetFNOFreeDOF = operatorDOF;
    end

    if needFixedUniform
        sampleResult.fixedUniform = pack_method_result( ...
            fixedUniformNode,fixedUniformElem,fixedUniformU, ...
            fixedUniformDOF,fixedUniformTime,fixedUniformErr, ...
            fixedUniformQuality,fixedUniformNewton);
        sampleResult.fixedUniform.gridN = fixedUniformN;
        sampleResult.fixedUniform.meshTime = fixedUniformMeshTime;
        sampleResult.fixedUniform.configuredFreeDOF = ...
            cfg.fixedUniformFreeDOF;
        sampleResult.fixedUniform.initializationMode = ...
            fixedUniformInitializationLabel;
        sampleResult.fixedUniform.requestedInitializationMode = ...
            initializationMode;
        sampleResult.fixedUniform.coarseBaseSolveTimeIncluded = ...
            useCoarseInterpolation;
    end

    sampleResult.afemHistory = afemHistory;
    if needFNOReference
        sampleResult.operatorCorrectionHistory = ...
            operatorCorrectionHistory;
        sampleResult.operatorMeshStats = operatorMeshStats;
    end
    if needTargetMethod
        sampleResult.targetCorrectionHistory = ...
            targetCorrectionHistory;
        sampleResult.targetMeshStats = targetMeshStats;
    end
    if cfg.savePointwiseErrorArrays && ~isempty(pointwiseErrorData)
        sampleResult.pointwiseError = pointwiseErrorData;
    end

    sampleResults{s} = make_lightweight_six_method_result(sampleResult);

    if makePlotsForThisSample && cfg.saveSampleMeshes
        sampleFile = fullfile(cfg.outDir, ...
            [caseTag '_enabled_method_comparison.mat']);
        save(sampleFile,'sampleResult','ref','cfg','-v7.3');
    end

    % Summary rows for enabled reported methods only.
    if cfg.runFNOSeededAFEM
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId, ...
            sprintf('%s predicted-score FEM + %d AFEM', ...
                operatorLabel,cfg.hybridAFEMCycles), ...
            'fno_predicted_score_plus_fixed_afem', ...
            NaN,NaN,NaN,operatorDOF,size(operatorNode,1), ...
            size(operatorElem,1),operatorErr,operatorTime, ...
            operatorQuality,referenceTime,operatorNewton); %#ok<AGROW>
    end

    if cfg.runTargetSeededAFEM
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId, ...
            sprintf('Ground-truth score FEM + %d AFEM', ...
                cfg.hybridAFEMCycles), ...
            'ground_truth_score_plus_fixed_afem', ...
            NaN,NaN,NaN,targetDOF,size(targetNode,1), ...
            size(targetElem,1),targetErr,targetTime,targetQuality, ...
            referenceTime,targetNewton); %#ok<AGROW>
    end

    if cfg.runTimeMatchedAFEM
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId,'Time-matched AFEM','time', ...
            standardTargetTime,timeAFEM.timeSec,timeAFEM.stage, ...
            timeAFEM.dof,size(timeAFEM.node,1),size(timeAFEM.elem,1), ...
            timeAFEM.error,timeAFEM.timeSec,timeAFEMQuality, ...
            referenceTime,timeAFEM.newtonInfo); %#ok<AGROW>
    end

    if cfg.runAccuracyMatchedAFEM
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId,'Accuracy-matched AFEM','accuracy', ...
            standardTargetError,accuracyAFEM.error.spaceTimeL2Rel, ...
            accuracyAFEM.stage,accuracyAFEM.dof, ...
            size(accuracyAFEM.node,1),size(accuracyAFEM.elem,1), ...
            accuracyAFEM.error,accuracyAFEM.timeSec, ...
            accuracyAFEMQuality,referenceTime, ...
            accuracyAFEM.newtonInfo); %#ok<AGROW>
    end

    if cfg.runFixedRefinementAFEM
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId, ...
            sprintf('Fixed-%d-refine AFEM',cfg.fixedAFEMRefineCycles), ...
            'fixed_refine_cycles',cfg.fixedAFEMRefineCycles, ...
            fixedAFEM.stage,fixedAFEM.stage,fixedAFEM.dof, ...
            size(fixedAFEM.node,1),size(fixedAFEM.elem,1), ...
            fixedAFEM.error,fixedAFEM.timeSec,fixedQuality, ...
            referenceTime,fixedAFEM.newtonInfo); %#ok<AGROW>
    end

    if cfg.runDOFMatchedUniform
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId, ...
            sprintf('Smallest-strictly-above-%s-DOF uniform FEM', ...
                operatorLabel), ...
            'uniform_dof_strictly_above',operatorDOF,dofUniformDOF,NaN, ...
            dofUniformDOF,size(dofUniformNode,1), ...
            size(dofUniformElem,1),dofUniformErr,dofUniformTime, ...
            dofUniformQuality,referenceTime,dofUniformNewton); %#ok<AGROW>
    end

    if cfg.runFixedUniform
        summaryRows(end+1,:) = make_extended_summary_row( ...
            s,testId,sourceId, ...
            sprintf('Fixed-%d-DOF uniform FEM', ...
                cfg.fixedUniformFreeDOF), ...
            'fixed_uniform_free_dof',cfg.fixedUniformFreeDOF, ...
            fixedUniformDOF,NaN,fixedUniformDOF, ...
            size(fixedUniformNode,1),size(fixedUniformElem,1), ...
            fixedUniformErr,fixedUniformTime,fixedUniformQuality, ...
            referenceTime,fixedUniformNewton); %#ok<AGROW>
    end

    % Direct FNO-versus-target comparison is meaningful only when both rows
    % are explicitly enabled.
    if cfg.runFNOSeededAFEM && cfg.runTargetSeededAFEM
        directRows(end+1,:) = { ...
            s,testId,sourceId,scoreMAE,scoreRMSE, ...
            underGenerationRate,exactGenerationRate,overGenerationRate, ...
            operatorDOF,targetDOF,operatorDOF/max(targetDOF,1), ...
            operatorErr.spaceTimeL2Rel,targetErr.spaceTimeL2Rel, ...
            operatorErr.spaceTimeL2Rel/max(targetErr.spaceTimeL2Rel,eps), ...
            operatorErr.finalTimeL2Rel,targetErr.finalTimeL2Rel, ...
            operatorErr.finalTimeL2Rel/max(targetErr.finalTimeL2Rel,eps), ...
            operatorTime,targetTime, ...
            operatorTime/max(targetTime,eps)}; %#ok<AGROW>
    end

        % Commit this case after every required method, statistic and
        % result-packaging step has completed. Finite Newton
        % nonconvergence is retained and is not a fatal case failure.
        successfulCases = successfulCases+1;
        successfulSampleIds(successfulCases) = testId;
        if makePlotsForThisSample
            successfulPlotIds(end+1) = testId; %#ok<AGROW>
        end

        fprintf(['\nCompleted comparison %d/%d: ', ...
                 'test ID=%d, source ID=%d.\n'], ...
            successfulCases,requestedCases,testId,sourceId);

    catch ME

        % An exact-input mapping mismatch is a global dataset/configuration
        % error, not an individual FEM failure. Stop immediately.
        if strcmp(ME.identifier,'NOEM:ExactInputMappingMismatch')
            rethrow(ME);
        end
    
        % Remove any partially appended rows from the rejected case.
        summaryRows = summaryRows(1:summaryCountBefore,:);
        directRows = directRows(1:directCountBefore,:);
        sampleResults{s} = [];

        failureRows(end+1,:) = { ...
            candidatePosition,s,testId,sourceId,currentStage, ...
            ME.identifier,ME.message}; %#ok<AGROW>

        failureTable = cell2table( ...
            failureRows,'VariableNames',failureVarNames);
        writetable(failureTable,failureCsv);
        save(failureMat,'failureTable','cfg','-v7.3');

        fprintf(2,['\nRejected candidate test ID=%d, source ID=%d, ', ...
                   'stage=%s.\nReason: %s\n'], ...
            testId,sourceId,currentStage,ME.message);
        fprintf(2,'The next candidate test case will be attempted.\n');
        continue;
    end
end

nCases = successfulCases;
sampleIds = successfulSampleIds(1:nCases);
plotSampleIds = successfulPlotIds;

if nCases<requestedCases
    warning(['Only %d of the requested %d complete comparisons succeeded. ', ...
             'The exported test pool was exhausted.'], ...
        nCases,requestedCases);
end
if nCases<1
    error(['No complete comparison succeeded. Inspect %s for the ', ...
           'failure stages and messages.'],failureCsv);
end

sampleResults = sampleResults(1:nCases);

selectionSourceIds = arrayfun( ...
    @(id)get_source_index(D,id,nTest,nDatasetSamples),sampleIds);
selectionTable = table( ...
    (1:nCases).',sampleIds(:),selectionSourceIds(:), ...
    ismember(sampleIds(:),plotSampleIds(:)), ...
    'VariableNames',{'Case','TestID','SourceID','WasPlotted'});

selectionCsv = fullfile(cfg.outDir,sprintf( ...
    'successful_%d_comparison_cases.csv',nCases));
writetable(selectionTable,selectionCsv);
save(fullfile(cfg.outDir,'successful_sample_selection.mat'), ...
    'sampleIds','plotSampleIds','selectionTable', ...
    'candidatePlanTable','cfg');

if isempty(failureRows)
    failureTable = cell2table( ...
        cell(0,numel(failureVarNames)), ...
        'VariableNames',failureVarNames);
    writetable(failureTable,failureCsv);
    save(failureMat,'failureTable','cfg','-v7.3');
else
    failureTable = cell2table( ...
        failureRows,'VariableNames',failureVarNames);
end


%% ========================================================================
% Save global summary
% =========================================================================

varNames = { ...
    'Case','TestID','SourceID','Method','MatchType', ...
    'TargetValue','AchievedValue','MatchRelativeGap','SelectedStage', ...
    'FreeDOF','Nodes','Elements','SpaceTimeRelL2', ...
    'FinalTimeRelL2','MethodTimeSec','MinAngleDeg', ...
    'P05AngleDeg','ReferenceTimeSec', ...
    'NewtonConverged','NewtonIterations','NewtonFinalResidual', ...
    'NewtonTargetResidual','NewtonExitReason'};

summaryTable = cell2table(summaryRows,'VariableNames',varNames);

% Audit every selected/final method solve that did not satisfy the Newton
% convergence criterion but was intentionally retained.
newtonConvergedColumn = double(summaryTable.NewtonConverged);
newtonNonconvergedMask = ...
    isfinite(newtonConvergedColumn) & newtonConvergedColumn<0.5;
newtonNonconvergedTable = summaryTable(newtonNonconvergedMask,{ ...
    'Case','TestID','SourceID','Method','FreeDOF','MethodTimeSec', ...
    'SpaceTimeRelL2','NewtonIterations','NewtonFinalResidual', ...
    'NewtonTargetResidual','NewtonExitReason'});

directVarNames = { ...
    'Case','TestID','SourceID','ScoreMAE','ScoreRMSE', ...
    'UnderGenerationRate','ExactGenerationRate','OverGenerationRate', ...
    'FNOHybridDOF','TargetHybridDOF', ...
    'FNOHybridDivTargetHybridDOF', ...
    'FNOHybridSpaceTimeRelL2','TargetHybridSpaceTimeRelL2', ...
    'FNOHybridDivTargetHybridSpaceTimeError', ...
    'FNOHybridFinalTimeRelL2','TargetHybridFinalTimeRelL2', ...
    'FNOHybridDivTargetHybridFinalTimeError', ...
    'FNOHybridOnlineTimeSec','TargetHybridOnlineTimeSec', ...
    'FNOHybridDivTargetHybridOnlineTime'};
directComparisonTable = cell2table( ...
    directRows,'VariableNames',directVarNames);

summaryCsv = fullfile(cfg.outDir,sprintf( ...
    'comparison_summary_%d_samples_enabled_methods.csv',nCases));
directCsv = fullfile(cfg.outDir,sprintf( ...
    'fno_hybrid_vs_target_hybrid_%d_samples.csv',nCases));
resultMat = fullfile(cfg.outDir,sprintf( ...
    'comparison_result_%d_samples_enabled_methods.mat',nCases));

writetable(summaryTable,summaryCsv);
writetable(directComparisonTable,directCsv);

methodStatisticsTable = build_method_statistics_table(summaryTable);
statisticsCsv = fullfile(cfg.outDir,sprintf( ...
    'method_statistics_%d_samples.csv',nCases));
writetable(methodStatisticsTable,statisticsCsv);

newtonNonconvergedCsv = fullfile(cfg.outDir,sprintf( ...
    'newton_nonconverged_records_%d_samples.csv',nCases));
writetable(newtonNonconvergedTable,newtonNonconvergedCsv);

fprintf('\nPer-method statistics over %d evaluation samples:\n', ...
    numel(sampleIds));
disp(methodStatisticsTable(:,{ ...
    'Method','Samples','MeanMethodTimeSec', ...
    'MeanSpaceTimeRelL2','BestSpaceTimeRelL2','BestTestID', ...
    'WorstSpaceTimeRelL2','WorstTestID', ...
    'NewtonNonconvergedSamples','NewtonConvergenceRate', ...
    'MaxNewtonFinalResidual'}));

methodTimingTable = build_method_timing_table( ...
    summaryTable,sampleResults);
methodTimingStatisticsTable = build_method_timing_statistics_table( ...
    methodTimingTable,methodStatisticsTable);
timingStatisticsCsv = fullfile(cfg.outDir,sprintf( ...
    'method_timing_statistics_%d_samples.csv',nCases));
writetable(methodTimingStatisticsTable,timingStatisticsCsv);

fprintf('\nPer-method timing breakdown over %d samples (mean sec/sample):\n', ...
    numel(sampleIds));
disp(methodTimingStatisticsTable(:,{ ...
    'Method','Samples','MeanMeshTimeSec','MeanAssemblyTimeSec', ...
    'MeanSolveTimeSec','MeanOtherTimeSec','MeanMethodTimeSec'}));

plot_multi_sample_method_summary( ...
    summaryTable,methodStatisticsTable,numel(sampleIds),cfg);

save(resultMat, ...
    'sampleResults','summaryTable','directComparisonTable', ...
    'methodStatisticsTable','methodTimingTable', ...
    'methodTimingStatisticsTable','newtonNonconvergedTable', ...
    'selectionTable','candidatePlanTable', ...
    'failureTable','cfg','sampleIds','plotSampleIds','-v7.3');

fprintf('\n============================================================\n');
fprintf('All enabled-method comparisons finished (Newton init: %s).\n', ...
    cfg.newtonInitializationMode);
fprintf('Method CSV     : %s\n',summaryCsv);
fprintf('Direct hybrid CSV: %s\n',directCsv);
fprintf('Statistics CSV : %s\n',statisticsCsv);
fprintf('Timing CSV     : %s\n',timingStatisticsCsv);
fprintf('Newton audit   : %s\n',newtonNonconvergedCsv);
fprintf('Result MAT     : %s\n',resultMat);
fprintf('Successful IDs: %s\n',mat2str(sampleIds));
fprintf('Failed cases  : %s\n',failureCsv);
fprintf(['Method times exclude spectral reference, error evaluation, ' ...
         'plotting and offline %s training.\n'],operatorLabel);
fprintf(['Ground-truth target-score FEM time also excludes the offline ' ...
         'cost that originally generated target_score_test.\n']);
fprintf('============================================================\n');

result = struct();
result.sampleResults = sampleResults;
result.summaryTable = summaryTable;
result.directComparisonTable = directComparisonTable;
result.methodStatisticsTable = methodStatisticsTable;
result.methodTimingTable = methodTimingTable;
result.methodTimingStatisticsTable = methodTimingStatisticsTable;
result.newtonNonconvergedTable = newtonNonconvergedTable;
result.selectionTable = selectionTable;
result.candidatePlanTable = candidatePlanTable;
result.failureTable = failureTable;
result.cfg = cfg;
result.methodSwitches = struct( ...
    'runFNOSeededAFEM',cfg.runFNOSeededAFEM, ...
    'runTargetSeededAFEM',cfg.runTargetSeededAFEM, ...
    'runStandardAFEM',cfg.runStandardAFEM, ...
    'runTimeMatchedAFEM',cfg.runTimeMatchedAFEM, ...
    'runAccuracyMatchedAFEM',cfg.runAccuracyMatchedAFEM, ...
    'runFixedRefinementAFEM',cfg.runFixedRefinementAFEM, ...
    'runDOFMatchedUniform',cfg.runDOFMatchedUniform, ...
    'runFixedUniform',cfg.runFixedUniform);
result.sampleIds = sampleIds;
result.plotSampleIds = plotSampleIds;
end



function mode = normalize_newton_initialization_mode(modeInput)
%NORMALIZE_NEWTON_INITIALIZATION_MODE Canonicalize user-facing aliases.
% Supported canonical modes:
%   zero, initial_condition, coarse_interpolation.

    if ~(ischar(modeInput) || (isstring(modeInput) && isscalar(modeInput)))
        error('cfg.newtonInitializationMode must be a character vector or scalar string.');
    end

    token = lower(strtrim(char(string(modeInput))));
    token = strrep(token,'-','_');
    token = strrep(token,' ','_');

    switch token
        case {'zero','0','zero_initialization','zero_init'}
            mode = 'zero';
        case {'initial','initial_value','initial_condition', ...
              'initial_condition_extension','u0','u_0'}
            mode = 'initial_condition';
        case {'interpolation','interpolated','interpolate', ...
              'coarse_interpolation','coarse_solution_interpolation', ...
              'coarse_warm_start','warm_start'}
            mode = 'coarse_interpolation';
        otherwise
            error(['Unsupported cfg.newtonInitializationMode="%s". ', ...
                   'Use ''zero'', ''initial_condition'', or ', ...
                   '''coarse_interpolation''.'],char(string(modeInput)));
    end
end


function T = build_method_statistics_table(summaryTable)
%BUILD_METHOD_STATISTICS_TABLE Aggregate all requested cross-sample metrics.
%
% For each method, report:
%   * mean and standard deviation of method time;
%   * mean, median and standard deviation of global space-time rel-L2 error;
%   * best/worst space-time rel-L2 error and the corresponding case/test IDs;
%   * analogous final-time error summaries;
%   * mean and standard deviation of free DOFs;
%   * Newton convergence count/rate and final-residual diagnostics.

    methodNames = unique(summaryTable.Method,'stable');
    nMethods = numel(methodNames);

    Method = cell(nMethods,1);
    Samples = zeros(nMethods,1);

    MeanMethodTimeSec = zeros(nMethods,1);
    StdMethodTimeSec = zeros(nMethods,1);
    MedianMethodTimeSec = zeros(nMethods,1);
    MinMethodTimeSec = zeros(nMethods,1);
    MaxMethodTimeSec = zeros(nMethods,1);

    MeanSpaceTimeRelL2 = zeros(nMethods,1);
    MedianSpaceTimeRelL2 = zeros(nMethods,1);
    StdSpaceTimeRelL2 = zeros(nMethods,1);
    BestSpaceTimeRelL2 = zeros(nMethods,1);
    BestCase = zeros(nMethods,1);
    BestTestID = zeros(nMethods,1);
    BestSourceID = zeros(nMethods,1);
    WorstSpaceTimeRelL2 = zeros(nMethods,1);
    WorstCase = zeros(nMethods,1);
    WorstTestID = zeros(nMethods,1);
    WorstSourceID = zeros(nMethods,1);

    MeanFinalTimeRelL2 = zeros(nMethods,1);
    MedianFinalTimeRelL2 = zeros(nMethods,1);
    StdFinalTimeRelL2 = zeros(nMethods,1);
    BestFinalTimeRelL2 = zeros(nMethods,1);
    WorstFinalTimeRelL2 = zeros(nMethods,1);

    MeanFreeDOF = zeros(nMethods,1);
    StdFreeDOF = zeros(nMethods,1);

    NewtonConvergedSamples = zeros(nMethods,1);
    NewtonNonconvergedSamples = zeros(nMethods,1);
    NewtonConvergenceRate = zeros(nMethods,1);
    MedianNewtonFinalResidual = NaN(nMethods,1);
    MaxNewtonFinalResidual = NaN(nMethods,1);

    for m = 1:nMethods
        Method{m} = methodNames{m};
        mask = strcmp(summaryTable.Method,methodNames{m});
        rowIds = find(mask);
        Samples(m) = numel(rowIds);

        methodTime = double(summaryTable.MethodTimeSec(mask));
        stError = double(summaryTable.SpaceTimeRelL2(mask));
        finalError = double(summaryTable.FinalTimeRelL2(mask));
        freeDOF = double(summaryTable.FreeDOF(mask));
        newtonConverged = summaryTable.NewtonConverged(mask);
        newtonResidual = summaryTable.NewtonFinalResidual(mask);
        if iscell(newtonConverged)
            newtonConverged = cell2mat(newtonConverged);
        end
        if iscell(newtonResidual)
            newtonResidual = cell2mat(newtonResidual);
        end
        newtonConverged = double(newtonConverged);
        newtonResidual = double(newtonResidual);

        if isempty(stError)
            error('Method "%s" has no summary rows.',methodNames{m});
        end
        if any(~isfinite(stError)) || any(~isfinite(methodTime))
            error('Method "%s" contains non-finite error or time.', ...
                methodNames{m});
        end

        MeanMethodTimeSec(m) = mean(methodTime);
        StdMethodTimeSec(m) = std(methodTime,0);
        MedianMethodTimeSec(m) = median(methodTime);
        MinMethodTimeSec(m) = min(methodTime);
        MaxMethodTimeSec(m) = max(methodTime);

        MeanSpaceTimeRelL2(m) = mean(stError);
        MedianSpaceTimeRelL2(m) = median(stError);
        StdSpaceTimeRelL2(m) = std(stError,0);

        [BestSpaceTimeRelL2(m),bestLocal] = min(stError);
        bestRow = rowIds(bestLocal);
        BestCase(m) = summaryTable.Case(bestRow);
        BestTestID(m) = summaryTable.TestID(bestRow);
        BestSourceID(m) = summaryTable.SourceID(bestRow);

        [WorstSpaceTimeRelL2(m),worstLocal] = max(stError);
        worstRow = rowIds(worstLocal);
        WorstCase(m) = summaryTable.Case(worstRow);
        WorstTestID(m) = summaryTable.TestID(worstRow);
        WorstSourceID(m) = summaryTable.SourceID(worstRow);

        MeanFinalTimeRelL2(m) = mean(finalError);
        MedianFinalTimeRelL2(m) = median(finalError);
        StdFinalTimeRelL2(m) = std(finalError,0);
        BestFinalTimeRelL2(m) = min(finalError);
        WorstFinalTimeRelL2(m) = max(finalError);

        MeanFreeDOF(m) = mean(freeDOF);
        StdFreeDOF(m) = std(freeDOF,0);

        validNewton = isfinite(newtonConverged);
        NewtonConvergedSamples(m) = nnz(newtonConverged(validNewton)>0.5);
        NewtonNonconvergedSamples(m) = nnz(newtonConverged(validNewton)<=0.5);
        if any(validNewton)
            NewtonConvergenceRate(m) = ...
                NewtonConvergedSamples(m)/nnz(validNewton);
        else
            NewtonConvergenceRate(m) = NaN;
        end

        validResidual = isfinite(newtonResidual);
        if any(validResidual)
            MedianNewtonFinalResidual(m) = median(newtonResidual(validResidual));
            MaxNewtonFinalResidual(m) = max(newtonResidual(validResidual));
        end
    end

    T = table( ...
        Method,Samples, ...
        MeanMethodTimeSec,StdMethodTimeSec, ...
        MedianMethodTimeSec,MinMethodTimeSec,MaxMethodTimeSec, ...
        MeanSpaceTimeRelL2,MedianSpaceTimeRelL2,StdSpaceTimeRelL2, ...
        BestSpaceTimeRelL2,BestCase,BestTestID,BestSourceID, ...
        WorstSpaceTimeRelL2,WorstCase,WorstTestID,WorstSourceID, ...
        MeanFinalTimeRelL2,MedianFinalTimeRelL2,StdFinalTimeRelL2, ...
        BestFinalTimeRelL2,WorstFinalTimeRelL2, ...
        MeanFreeDOF,StdFreeDOF, ...
        NewtonConvergedSamples,NewtonNonconvergedSamples, ...
        NewtonConvergenceRate,MedianNewtonFinalResidual, ...
        MaxNewtonFinalResidual);
end


function T = build_method_timing_table(summaryTable,sampleResults)
%BUILD_METHOD_TIMING_TABLE Per-sample wall-clock breakdown by method.
% Reconstructs mesh / matrix-assembly / matrix-solve times for every row of
% summaryTable from the lightweight per-sample results.  The buckets follow
% the MethodTimeSec convention: estimator/marking, score inference and other
% small overheads are excluded (they appear as "other" time in the
% aggregate table).

    nRows = size(summaryTable,1);
    caseCol = double(summaryTable.Case);
    testCol = double(summaryTable.TestID);
    sourceCol = double(summaryTable.SourceID);
    methodCol = summaryTable.Method;
    meshCol = NaN(nRows,1);
    asmCol = NaN(nRows,1);
    solCol = NaN(nRows,1);

    for r = 1:nRows
        ci = caseCol(r);
        if ci<1 || ci>numel(sampleResults)
            continue;
        end
        S = sampleResults{ci};
        if isempty(S) || ~isstruct(S)
            continue;
        end
        b = method_timing_breakdown(S,char(methodCol(r)));
        meshCol(r) = b.meshTimeSec;
        asmCol(r) = b.assemblyTimeSec;
        solCol(r) = b.solveTimeSec;
    end

    T = table(caseCol,testCol,sourceCol,methodCol,meshCol,asmCol,solCol, ...
        'VariableNames',{'Case','TestID','SourceID','Method', ...
        'MeshTimeSec','AssemblyTimeSec','SolveTimeSec'});
end


function T = build_method_timing_statistics_table(timingTable,methodStatsTable)
%BUILD_METHOD_TIMING_STATISTICS_TABLE Mean timing buckets per method.
% MeanOtherTimeSec = MeanMethodTimeSec-(mesh+assembly+solve) and covers
% estimator/marking, score inference and other unclassified work.

    methodNames = unique(cellstr(timingTable.Method),'stable');
    nMethods = numel(methodNames);
    meanMesh = zeros(nMethods,1);
    meanAsm = zeros(nMethods,1);
    meanSol = zeros(nMethods,1);
    meanOther = NaN(nMethods,1);
    meanTotal = NaN(nMethods,1);
    nSamples = zeros(nMethods,1);

    for m = 1:nMethods
        mask = strcmp(cellstr(timingTable.Method),methodNames{m});
        meshV = timingTable.MeshTimeSec(mask);
        asmV = timingTable.AssemblyTimeSec(mask);
        solV = timingTable.SolveTimeSec(mask);
        ok = isfinite(meshV) & isfinite(asmV) & isfinite(solV);
        nSamples(m) = nnz(ok);
        if nSamples(m)>0
            meanMesh(m) = mean(meshV(ok));
            meanAsm(m) = mean(asmV(ok));
            meanSol(m) = mean(solV(ok));
        else
            meanMesh(m) = NaN;
            meanAsm(m) = NaN;
            meanSol(m) = NaN;
        end
    end

    for m = 1:nMethods
        rowMask = strcmp(cellstr(methodStatsTable.Method),methodNames{m});
        if any(rowMask)
            meanTotal(m) = double(methodStatsTable.MeanMethodTimeSec( ...
                find(rowMask,1)));
        else
            meanTotal(m) = NaN;
        end
        if isfinite(meanTotal(m))
            meanOther(m) = meanTotal(m)-meanMesh(m)-meanAsm(m)-meanSol(m);
        else
            meanOther(m) = NaN;
        end
    end

    T = table(methodNames,nSamples,meanMesh,meanAsm,meanSol, ...
        meanOther,meanTotal, ...
        'VariableNames',{'Method','Samples', ...
        'MeanMeshTimeSec','MeanAssemblyTimeSec','MeanSolveTimeSec', ...
        'MeanOtherTimeSec','MeanMethodTimeSec'});
end


function b = method_timing_breakdown(S,methodName)
%METHOD_TIMING_BREAKDOWN Reconstruct one method row's mesh/assembly/solve
% wall-clock seconds from the lightweight per-sample result.

    b = struct('meshTimeSec',NaN,'assemblyTimeSec',NaN,'solveTimeSec',NaN);
    if isempty(S) || ~isstruct(S)
        return;
    end

    [baseAsm,baseSol] = base_solve_breakdown(S,false);

    if contains(methodName,'predicted-score FEM')
        if ~isfield(S,'operatorSeed') || ...
                ~isfield(S,'operatorCorrectionHistory')
            return;
        end
        b.meshTimeSec = get_scalar_time( ...
            S.operatorSeed,'meshBuildTime');
        b.assemblyTimeSec = baseAsm+get_scalar_time( ...
            S.operatorSeed.newtonInfo,'assemblyTimeSec');
        b.solveTimeSec = baseSol+get_scalar_time( ...
            S.operatorSeed.newtonInfo,'solveTimeSec');
        b = add_history_timing_breakdown( ...
            b,S.operatorCorrectionHistory);
    elseif contains(methodName,'Ground-truth score FEM')
        if ~isfield(S,'targetSeed') || ...
                ~isfield(S,'targetCorrectionHistory')
            return;
        end
        b.meshTimeSec = get_scalar_time( ...
            S.targetSeed,'meshBuildTime');
        b.assemblyTimeSec = baseAsm+get_scalar_time( ...
            S.targetSeed.newtonInfo,'assemblyTimeSec');
        b.solveTimeSec = baseSol+get_scalar_time( ...
            S.targetSeed.newtonInfo,'solveTimeSec');
        b = add_history_timing_breakdown( ...
            b,S.targetCorrectionHistory);
    elseif contains(methodName,'Time-matched AFEM') || ...
            contains(methodName,'Accuracy-matched AFEM') || ...
            contains(methodName,'refine AFEM')
        if ~isfield(S,'afemHistory') || ~isfield(S,'afemBase')
            return;
        end
        stage = selected_afem_stage(S,methodName);
        h = S.afemHistory;
        if ~isfield(h,'stage') || ...
                ~isfield(h,'cumulativeRefineTimeSec')
            return;
        end
        idx = find(double(h.stage)==stage,1,'last');
        if isempty(idx)
            idx = numel(h.stage);
        end
        [baseAsm,baseSol] = base_solve_breakdown(S,true);
        b.meshTimeSec = double(h.cumulativeRefineTimeSec(idx));
        b.assemblyTimeSec = baseAsm+ ...
            double(h.cumulativeAssemblyTimeSec(idx));
        b.solveTimeSec = baseSol+ ...
            double(h.cumulativeSolveTimeSec(idx));
    elseif contains(methodName,'Smallest-strictly-above')
        b = uniform_timing_breakdown(S,'uniform',baseAsm,baseSol);
    elseif contains(methodName,'DOF uniform FEM')
        b = uniform_timing_breakdown(S,'fixedUniform',baseAsm,baseSol);
    end
end


function stage = selected_afem_stage(S,methodName)
    stage = NaN;
    if contains(methodName,'Time-matched AFEM') && ...
            isfield(S,'afemTime') && isfield(S.afemTime,'selection')
        stage = double(S.afemTime.selection.stage);
    elseif contains(methodName,'Accuracy-matched AFEM') && ...
            isfield(S,'afemAccuracy') && isfield(S.afemAccuracy,'selection')
        stage = double(S.afemAccuracy.selection.stage);
    elseif contains(methodName,'refine AFEM') && ...
            isfield(S,'afemFixed') && isfield(S.afemFixed,'selection')
        stage = double(S.afemFixed.selection.stage);
    end
    if ~isfinite(stage) && contains(methodName,'refine AFEM') && ...
            isfield(S,'afemFixed') && isfield(S.afemFixed,'refineCycles')
        stage = double(S.afemFixed.refineCycles);
    end
end


function b = uniform_timing_breakdown(S,fname,baseAsm,baseSol)
    b = struct('meshTimeSec',NaN,'assemblyTimeSec',NaN, ...
        'solveTimeSec',NaN);
    if ~isfield(S,fname)
        return;
    end
    U = S.(fname);
    b.meshTimeSec = get_scalar_time(U,'meshTime');
    b.assemblyTimeSec = baseAsm+get_scalar_time( ...
        U.newtonInfo,'assemblyTimeSec');
    b.solveTimeSec = baseSol+get_scalar_time( ...
        U.newtonInfo,'solveTimeSec');
end


function b = add_history_timing_breakdown(b,h)
    if isempty(h) || ~isstruct(h) || ...
            ~isfield(h,'cumulativeRefineTimeSec') || ...
            ~isfield(h,'cumulativeAssemblyTimeSec') || ...
            ~isfield(h,'cumulativeSolveTimeSec')
        return;
    end
    if isempty(h.cumulativeRefineTimeSec)
        return;
    end
    b.meshTimeSec = b.meshTimeSec + ...
        double(h.cumulativeRefineTimeSec(end));
    b.assemblyTimeSec = b.assemblyTimeSec + ...
        double(h.cumulativeAssemblyTimeSec(end));
    b.solveTimeSec = b.solveTimeSec + ...
        double(h.cumulativeSolveTimeSec(end));
end


function [asm,sol] = base_solve_breakdown(S,useAfemBase)
    asm = 0.0;
    sol = 0.0;
    if useAfemBase
        if isfield(S,'afemBase')
            B = S.afemBase;
        else
            B = [];
        end
    else
        if isfield(S,'operatorBase')
            B = S.operatorBase;
        else
            B = [];
        end
    end
    if isempty(B) || ~isstruct(B) || ~isfield(B,'newtonInfo')
        return;
    end
    asm = get_scalar_time(B.newtonInfo,'assemblyTimeSec');
    sol = get_scalar_time(B.newtonInfo,'solveTimeSec');
end


function t = get_scalar_time(X,fieldName)
    t = 0.0;
    if isempty(X) || ~isstruct(X) || ~isfield(X,fieldName)
        return;
    end
    v = double(X.(fieldName));
    if isscalar(v) && isfinite(v) && v>=0
        t = v;
    end
end


function t = newton_bucket_time(ninfo,fieldName)
    t = 0.0;
    if isempty(ninfo) || ~isstruct(ninfo) || ~isfield(ninfo,fieldName)
        return;
    end
    v = double(ninfo.(fieldName));
    if isscalar(v) && isfinite(v) && v>=0
        t = v;
    end
end


function plot_multi_sample_method_summary( ...
        summaryTable,statisticsTable,nCases,cfg)
%PLOT_MULTI_SAMPLE_METHOD_SUMMARY Error/time traces and aggregate statistics.

    methodNames = statisticsTable.Method;
    nMethods = numel(methodNames);
    methodIndex = (1:nMethods).';

    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[40,50,1600,900]);
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    hold on;
    for m = 1:nMethods
        mask = strcmp(summaryTable.Method,methodNames{m});
        cases = summaryTable.Case(mask);
        values = summaryTable.SpaceTimeRelL2(mask);
        [cases,order] = sort(cases);
        values = values(order);
        semilogy(cases,values,'-','LineWidth',1.05);
    end
    grid on; box on;
    xlabel('evaluation case');
    ylabel('space-time relative L^2 error');
    title(sprintf('Per-case relative L^2 error (%d samples)',nCases));
    legend(methodNames,'Interpreter','none','Location','bestoutside');

    nexttile;
    hold on;
    for m = 1:nMethods
        mask = strcmp(summaryTable.Method,methodNames{m});
        cases = summaryTable.Case(mask);
        values = summaryTable.MethodTimeSec(mask);
        [cases,order] = sort(cases);
        values = values(order);
        plot(cases,values,'-','LineWidth',1.05);
    end
    grid on; box on;
    xlabel('evaluation case');
    ylabel('method time (s)');
    title(sprintf('Per-case method time (%d samples)',nCases));

    nexttile;
    meanError = statisticsTable.MeanSpaceTimeRelL2;
    lowerError = meanError-statisticsTable.BestSpaceTimeRelL2;
    upperError = statisticsTable.WorstSpaceTimeRelL2-meanError;
    errorbar(methodIndex,meanError,lowerError,upperError, ...
        'o','LineWidth',1.5,'MarkerSize',8);
    set(gca,'YScale','log');
    set(gca,'XTick',methodIndex,'XTickLabel',methodNames);
    xtickangle(20);
    grid on; box on;
    ylabel('space-time relative L^2 error');
    title('Mean error with best-to-worst range');

    nexttile;
    bar(methodIndex,statisticsTable.MeanMethodTimeSec);
    set(gca,'XTick',methodIndex,'XTickLabel',methodNames);
    xtickangle(20);
    grid on; box on;
    ylabel('mean method time (s)');
    title(sprintf('Mean online method time over %d samples',nCases));

    sgtitle(sprintf( ...
        'Burgers FEM: enabled-method statistics over %d successful test samples', ...
        nCases));

    exportgraphics(fig,fullfile(cfg.figureDir, ...
        sprintf('%d_sample_enabled_method_summary.png',nCases)), ...
        'Resolution',cfg.figureResolution);
    close(fig);
end



%% ========================================================================
% Continuous-score operator-mesh helpers
% =========================================================================

function score = get_score_sample(A,k,nTest,nx,nt)
%GET_SCORE_SAMPLE Return one score sample as nx-by-nt.
%
% The supplied Python exporter writes pred_score and target_score_test as
%
%       score array : nx x nt x nTest
%
% in the MATLAB file. Several alternative layouts are accepted as well.

    sz = size(A);
    if ndims(A)~=3
        error('The score array must be 3-D, got size %s.',mat2str(sz));
    end

    if sz(3)==nTest
        score = squeeze(A(:,:,k));
    elseif sz(1)==nTest
        score = squeeze(A(k,:,:));
    elseif sz(2)==nTest
        score = squeeze(A(:,k,:));
    else
        error(['Cannot identify the sample dimension of the score array ', ...
               'with size %s and nTest=%d.'],mat2str(sz),nTest);
    end

    score = double(score);
    if isequal(size(score),[nx,nt])
        return;
    end
    if isequal(size(score),[nt,nx])
        score = score.';
        return;
    end

    error('One score sample has size %s, expected %dx%d.', ...
        mat2str(size(score)),nx,nt);
end


function Fscore = make_periodic_score_interpolant( ...
        score,queryX,queryT,cfg)
%MAKE_PERIODIC_SCORE_INTERPOLANT
% Build a linear x-periodic interpolant of the deterministic score field.

    queryX = double(queryX(:));
    queryT = double(queryT(:));
    score = double(score);

    if ~isequal(size(score),[numel(queryX),numel(queryT)])
        error('Score array and query coordinates are incompatible.');
    end

    [queryX,ix] = sort(queryX);
    [queryT,it] = sort(queryT);
    score = score(ix,it);

    tol = 100*eps(max(1,max(abs([cfg.xmin,cfg.xmax]))));

    % Remove a duplicated periodic right endpoint if a future exporter saves it.
    if numel(queryX)>=2 && abs(queryX(end)-cfg.xmax)<=tol
        queryX = queryX(1:end-1);
        score = score(1:end-1,:);
    end

    if abs(queryX(1)-cfg.xmin)>100*tol
        error('The first score x-coordinate must coincide with cfg.xmin.');
    end

    xExtended = [queryX;cfg.xmax];
    scoreExtended = [score;score(1,:)];

    try
        Fscore = griddedInterpolant( ...
            {xExtended,queryT},scoreExtended,'linear','nearest');
    catch
        Fscore = griddedInterpolant( ...
            {xExtended,queryT},scoreExtended,'nearest','nearest');
    end
end


function generation = score_to_generation_threshold03( ...
        score,maxLevel,threshold)
%SCORE_TO_GENERATION_THRESHOLD03
% Convert a continuous score to an integer generation using one explicit
% asymmetric rounding rule.
%
% The score is clipped to [0,maxLevel].  In every interval [k,k+1),
% fractional part < threshold maps to k, while fractional part >= threshold
% maps to k+1.  With the main threshold=0.50:
%   0.49 -> 0, 0.50 -> 1, 1.49 -> 1, 1.50 -> 2.
%
% This helper is deliberately used both by diagnostics and by actual mesh
% construction, so the reported under/exact/over rates match the mesh logic.

    validateattributes(maxLevel,{'numeric'}, ...
        {'scalar','real','finite','integer','>=',1});
    validateattributes(threshold,{'numeric'}, ...
        {'scalar','real','finite','>',0,'<',1});

    score = min(max(double(score),0),double(maxLevel));
    generation = floor(score + (1-threshold) + 1.0e-12);
    generation = min(max(generation,0),double(maxLevel));
end


function [node,elem,stats] = build_continuous_score_operator_mesh_fast_exact( ...
        predScore,queryX,queryT,cfg,baseNodeTemplate,baseElemTemplate)
%BUILD_CONTINUOUS_SCORE_OPERATOR_MESH_FAST_EXACT
%
% Same numerical rule as build_continuous_score_operator_mesh, with two
% exact implementation optimizations:
%
%   1. generation(child)=generation(parent)+1 is propagated through every
%      NVB bisection instead of recomputing log2(A0/A_K) on all leaves;
%   2. an unchanged leaf that was already checked and found sufficiently
%      refined is never queried again. Only newly created leaves are queried
%      after the first pass.
%
% A final full-leaf query is still performed to prove that no leaf remains
% below its requested generation and to populate finalEffectiveScore.

    Fscore = make_periodic_score_interpolant( ...
        predScore,queryX,queryT,cfg);

    if nargin>=5 && ~isempty(baseNodeTemplate) && ...
            nargin>=6 && ~isempty(baseElemTemplate)
        node = baseNodeTemplate;
        elem = baseElemTemplate;
        dx = (cfg.xmax-cfg.xmin)/cfg.operatorInitialGrid;
        dt = (cfg.tmax-cfg.tmin)/cfg.operatorInitialGrid;
        A0 = dx*dt/2;
    else
        [node,elem,A0] = make_uniform_spacetime_mesh( ...
            cfg.operatorInitialGrid,cfg);
        elem = nvb_label_initial_mesh(node,elem);
    end

    if cfg.operatorApplyReferenceGridOffset
        generationOffset = 2*log2( ...
            cfg.scoreReferenceGrid/cfg.operatorInitialGrid);
    else
        generationOffset = 0;
    end

    % Reuse the quality of the fixed base template when available.
    % The cached struct was produced by the same mesh_angle_quality call on
    % the same node/element arrays, so all reported quality values are exact.
    if nargin>=5 && ~isempty(baseNodeTemplate) && ...
            nargin>=6 && ~isempty(baseElemTemplate) && ...
            isfield(cfg,'operatorBaseQualityTemplate') && ...
            ~isempty(cfg.operatorBaseQualityTemplate)
        Q0 = cfg.operatorBaseQualityTemplate;
    else
        Q0 = mesh_angle_quality(node,elem);
    end
    % Preserve the original warning/check location and frequency exactly.
    warn_bad_mesh_quality_local(Q0,cfg,'operator initial mesh');

    generation = zeros(size(elem,1),1,'uint8');
    workMask = true(size(elem,1),1);

    % Repeated full-mesh validity checks inside every score-refinement pass
    % are diagnostic only.  Disable them locally for the score compiler; the
    % unchanged final validity and periodicity checks below remain enabled.
    scoreMeshNvbCfg = cfg;
    scoreMeshNvbCfg.fastCheckEveryNVBCall = false;

    stats.referenceGrid = cfg.scoreReferenceGrid;
    stats.initialGrid = cfg.operatorInitialGrid;
    stats.generationOffset = generationOffset;
    stats.refinementMultiplier = cfg.operatorRefinementMultiplier;
    stats.queryPattern = cfg.operatorQueryPattern;
    stats.scoreSafetyBias = cfg.operatorScoreSafetyBias;
    stats.generationThreshold = cfg.generationThreshold;
    stats.pass = 0;
    stats.nodesByPass = size(node,1);
    stats.elemsByPass = size(elem,1);
    stats.freeDOFByPass = count_free_dofs(node,cfg);
    stats.activeByPass = 0;
    stats.queriedByPass = 0;
    stats.savedQueriesByPass = 0;
    stats.markedByPass = 0;
    stats.totalBisectedParentsByPass = 0;
    stats.completionBisectedParentsByPass = 0;
    stats.minScoreByPass = NaN;
    stats.meanScoreByPass = NaN;
    stats.maxScoreByPass = NaN;
    stats.minAngleByPass = Q0.minAngleDeg;
    stats.p05AngleByPass = Q0.p05AngleDeg;

    for pass = 1:cfg.operatorMaxRefineCalls

        fullActiveMask = double(generation)<cfg.operatorMaxLevel;
        fullActiveCount = nnz(fullActiveMask);

        if cfg.fastUseOperatorWorkQueue
            active = find(workMask & fullActiveMask);
        else
            active = find(fullActiveMask);
        end

        if isempty(active)
            break;
        end

        scoreActive = evaluate_score_on_elements_vectorized( ...
            Fscore,node,elem,active,cfg);

        effectiveScore = cfg.operatorRefinementMultiplier*scoreActive + ...
            generationOffset + cfg.operatorScoreSafetyBias;
        effectiveScore = min(max(effectiveScore,0),cfg.operatorMaxLevel);

        desiredGeneration = score_to_generation_threshold03( ...
            effectiveScore,cfg.operatorMaxLevel, ...
            cfg.generationThreshold);

        localGeneration = double(generation(active));
        markedLocal = localGeneration<desiredGeneration;
        markedElem = active(markedLocal);

        stats.pass(end+1,1) = pass; %#ok<AGROW>
        stats.activeByPass(end+1,1) = fullActiveCount; %#ok<AGROW>
        stats.queriedByPass(end+1,1) = numel(active); %#ok<AGROW>
        stats.savedQueriesByPass(end+1,1) = ...
            fullActiveCount-numel(active); %#ok<AGROW>
        stats.markedByPass(end+1,1) = numel(markedElem); %#ok<AGROW>
        stats.minScoreByPass(end+1,1) = min(scoreActive); %#ok<AGROW>
        stats.meanScoreByPass(end+1,1) = mean(scoreActive); %#ok<AGROW>
        stats.maxScoreByPass(end+1,1) = max(scoreActive); %#ok<AGROW>

        if isempty(markedElem)
            stats.nodesByPass(end+1,1) = size(node,1); %#ok<AGROW>
            stats.elemsByPass(end+1,1) = size(elem,1); %#ok<AGROW>
            stats.freeDOFByPass(end+1,1) = ...
                count_free_dofs(node,cfg); %#ok<AGROW>
            stats.totalBisectedParentsByPass(end+1,1) = 0; %#ok<AGROW>
            stats.completionBisectedParentsByPass(end+1,1) = 0; %#ok<AGROW>
            stats.minAngleByPass(end+1,1) = NaN; %#ok<AGROW>
            stats.p05AngleByPass(end+1,1) = NaN; %#ok<AGROW>
            workMask(:) = false;
            break;
        end

        state = struct();
        if cfg.fastTrackNVBGeneration
            state.elementGeneration = generation;
        else
            state.elementGeneration = [];
        end
        state.elementNewLeaf = false(size(elem,1),1);

        [node,elem,rstats,state] = ...
            nvb_refine_conforming_periodic_fast_exact( ...
                node,elem,markedElem,scoreMeshNvbCfg,state);

        if cfg.fastTrackNVBGeneration
            generation = state.elementGeneration;
        else
            generation = uint8( ...
                element_generation_from_area_local(node,elem,A0));
        end

        if cfg.fastUseOperatorWorkQueue
            workMask = state.elementNewLeaf;
        else
            workMask = true(size(elem,1),1);
        end

        if size(elem,1)>cfg.maxElementsCase
            error(['Continuous-score operator mesh exceeded ', ...
                   'cfg.maxElementsCase=%d.'],cfg.maxElementsCase);
        end

        completion = max(0, ...
            rstats.totalBisectedParents-numel(markedElem));

        stats.nodesByPass(end+1,1) = size(node,1); %#ok<AGROW>
        stats.elemsByPass(end+1,1) = size(elem,1); %#ok<AGROW>
        stats.freeDOFByPass(end+1,1) = ...
            count_free_dofs(node,cfg); %#ok<AGROW>
        stats.totalBisectedParentsByPass(end+1,1) = ...
            rstats.totalBisectedParents; %#ok<AGROW>
        stats.completionBisectedParentsByPass(end+1,1) = ...
            completion; %#ok<AGROW>

        if isfield(cfg,'fastSkipIntermediateMeshQuality') && ...
                cfg.fastSkipIntermediateMeshQuality
            stats.minAngleByPass(end+1,1) = NaN; %#ok<AGROW>
            stats.p05AngleByPass(end+1,1) = NaN; %#ok<AGROW>
        else
            Q = mesh_angle_quality(node,elem);
            warn_bad_mesh_quality_local(Q,cfg, ...
                sprintf('operator refine pass %d',pass));
            stats.minAngleByPass(end+1,1) = Q.minAngleDeg; %#ok<AGROW>
            stats.p05AngleByPass(end+1,1) = Q.p05AngleDeg; %#ok<AGROW>
        end

        fprintf(['  Operator pass %02d: active=%d, queried=%d, marked=%d, ' ...
                 'DOF=%d, elems=%d, completion=%d, ' ...
                 'score=[%.2f,%.2f,%.2f]\n'], ...
            pass,fullActiveCount,numel(active),numel(markedElem), ...
            stats.freeDOFByPass(end),size(elem,1),completion, ...
            min(scoreActive),mean(scoreActive),max(scoreActive));
    end

    % Final full scan: verification + finalEffectiveScore metadata.
    active = find(double(generation)<cfg.operatorMaxLevel);
    remaining = 0;
    finalEffectiveScore = zeros(size(elem,1),1);

    if ~isempty(active)
        scoreActive = evaluate_score_on_elements_vectorized( ...
            Fscore,node,elem,active,cfg);

        effectiveScore = min(max( ...
            cfg.operatorRefinementMultiplier*scoreActive + ...
            generationOffset + cfg.operatorScoreSafetyBias,0), ...
            cfg.operatorMaxLevel);

        desiredGeneration = score_to_generation_threshold03( ...
            effectiveScore,cfg.operatorMaxLevel, ...
            cfg.generationThreshold);

        finalEffectiveScore(active) = effectiveScore;
        remaining = nnz(double(generation(active))<desiredGeneration);
    end

    if remaining>0
        error(['FAST operator mesh stopped with %d triangles below the ', ...
               'continuous target. Increase operatorMaxRefineCalls.'], ...
            remaining);
    end

    if cfg.fastFinalNVBValidityCheck
        check_mesh_valid(node,elem,'final FAST continuous-score operator mesh');
        assert_periodic_boundary_match_fast_exact( ...
            node,elem,cfg,'final FAST operator mesh');
    end

    Qfinal = mesh_angle_quality(node,elem);
    warn_bad_mesh_quality_local(Qfinal,cfg,'final operator mesh');

    stats.finalNodes = size(node,1);
    stats.finalElems = size(elem,1);
    stats.finalFreeDOF = count_free_dofs(node,cfg);
    stats.finalGeneration = uint8(min(double(generation),255));
    stats.finalEffectiveScore = single(finalEffectiveScore);
    stats.remainingBelowTarget = remaining;
    stats.finalQuality = Qfinal;
    stats.fastTotalQueriedLeaves = sum(stats.queriedByPass);
    stats.fastTotalFullScanLeaves = sum(stats.activeByPass);
    stats.fastSavedRepeatedQueries = sum(stats.savedQueriesByPass);

    % Optional end-to-end exact mesh audit against the original compiler.
    if isfield(cfg,'fastAuditOperatorMesh') && cfg.fastAuditOperatorMesh
        cfgLegacy = cfg;
        cfgLegacy.fastAuditNVB = false;
        cfgLegacy.fastAuditPeriodicSideEdges = false;
        cfgLegacy.fastAuditOperatorMesh = false;

        [nodeLegacy,elemLegacy,legacyStats] = ...
            build_continuous_score_operator_mesh( ...
                predScore,queryX,queryT,cfgLegacy);

        if ~isequal(node,nodeLegacy) || ~isequal(elem,elemLegacy)
            error(['FAST operator compiler produced a different node/element ', ...
                   'array from the legacy compiler.']);
        end

        if ~isequal(stats.finalGeneration,legacyStats.finalGeneration)
            error('FAST operator generation differs from legacy generation.');
        end
    end
end

function [node,elem,stats] = build_continuous_score_operator_mesh( ...
        predScore,queryX,queryT,cfg)
%BUILD_CONTINUOUS_SCORE_OPERATOR_MESH
% Recursively realize a continuous local target-generation field.
%
% Every current leaf triangle is queried independently after every NVB pass.
% Therefore one original coarse triangle may retain children of several
% generations simultaneously.
%
% The raw score is first multiplied by operatorRefinementMultiplier.  The
% optional score-reference offset is then added only when
% operatorApplyReferenceGridOffset is true.
%
% For every active leaf K,
%
%   s_K = max_{q in Q_K} s(q),
%   desiredGeneration(K) = Q_threshold(
%       multiplier*s_K + generationOffset + operatorScoreSafetyBias),
%
% where Q_threshold clips to [0,12] and uses cfg.generationThreshold.
% K is refined exactly when
%
%   generation(K) < desiredGeneration(K).

    Fscore = make_periodic_score_interpolant( ...
        predScore,queryX,queryT,cfg);

    [node,elem,A0] = make_uniform_spacetime_mesh( ...
        cfg.operatorInitialGrid,cfg);
    elem = nvb_label_initial_mesh(node,elem);

    if cfg.operatorApplyReferenceGridOffset
        generationOffset = 2*log2( ...
            cfg.scoreReferenceGrid/cfg.operatorInitialGrid);
    else
        generationOffset = 0;
    end

    Q0 = mesh_angle_quality(node,elem);
    warn_bad_mesh_quality_local(Q0,cfg,'operator initial mesh');

    stats.referenceGrid = cfg.scoreReferenceGrid;
    stats.initialGrid = cfg.operatorInitialGrid;
    stats.generationOffset = generationOffset;
    stats.refinementMultiplier = cfg.operatorRefinementMultiplier;
    stats.queryPattern = cfg.operatorQueryPattern;
    stats.scoreSafetyBias = cfg.operatorScoreSafetyBias;
    stats.generationThreshold = cfg.generationThreshold;
    stats.pass = 0;
    stats.nodesByPass = size(node,1);
    stats.elemsByPass = size(elem,1);
    stats.freeDOFByPass = count_free_dofs(node,cfg);
    stats.activeByPass = 0;
    stats.markedByPass = 0;
    stats.totalBisectedParentsByPass = 0;
    stats.completionBisectedParentsByPass = 0;
    stats.minScoreByPass = NaN;
    stats.meanScoreByPass = NaN;
    stats.maxScoreByPass = NaN;
    stats.minAngleByPass = Q0.minAngleDeg;
    stats.p05AngleByPass = Q0.p05AngleDeg;

    for pass = 1:cfg.operatorMaxRefineCalls
        generation = element_generation_from_area_local(node,elem,A0);
        active = find(generation<cfg.operatorMaxLevel);

        if isempty(active)
            break;
        end

        scoreActive = evaluate_score_on_elements_vectorized( ...
            Fscore,node,elem,active,cfg);

        effectiveScore = cfg.operatorRefinementMultiplier*scoreActive + ...
            generationOffset + cfg.operatorScoreSafetyBias;
        effectiveScore = min(max(effectiveScore,0), ...
            cfg.operatorMaxLevel);
        desiredGeneration = score_to_generation_threshold03( ...
            effectiveScore,cfg.operatorMaxLevel, ...
            cfg.generationThreshold);

        localGeneration = generation(active);
        markedLocal = localGeneration < desiredGeneration;
        markedElem = active(markedLocal);

        stats.pass(end+1,1) = pass; %#ok<AGROW>
        stats.activeByPass(end+1,1) = numel(active); %#ok<AGROW>
        stats.markedByPass(end+1,1) = numel(markedElem); %#ok<AGROW>
        stats.minScoreByPass(end+1,1) = min(scoreActive); %#ok<AGROW>
        stats.meanScoreByPass(end+1,1) = mean(scoreActive); %#ok<AGROW>
        stats.maxScoreByPass(end+1,1) = max(scoreActive); %#ok<AGROW>

        if isempty(markedElem)
            stats.nodesByPass(end+1,1) = size(node,1); %#ok<AGROW>
            stats.elemsByPass(end+1,1) = size(elem,1); %#ok<AGROW>
            stats.freeDOFByPass(end+1,1) = ...
                count_free_dofs(node,cfg); %#ok<AGROW>
            stats.totalBisectedParentsByPass(end+1,1) = 0; %#ok<AGROW>
            stats.completionBisectedParentsByPass(end+1,1) = 0; %#ok<AGROW>
            stats.minAngleByPass(end+1,1) = ...
                stats.minAngleByPass(end); %#ok<AGROW>
            stats.p05AngleByPass(end+1,1) = ...
                stats.p05AngleByPass(end); %#ok<AGROW>
            break;
        end

        [node,elem,rstats] = nvb_refine_conforming_periodic( ...
            node,elem,markedElem,cfg);

        if size(elem,1)>cfg.maxElementsCase
            error(['Continuous-score operator mesh exceeded ', ...
                   'cfg.maxElementsCase=%d.'],cfg.maxElementsCase);
        end

        check_mesh_valid(node,elem, ...
            sprintf('continuous-score operator mesh pass %d',pass));
        Q = mesh_angle_quality(node,elem);
        warn_bad_mesh_quality_local(Q,cfg, ...
            sprintf('operator refine pass %d',pass));

        completion = max(0, ...
            rstats.totalBisectedParents-numel(markedElem));

        stats.nodesByPass(end+1,1) = size(node,1); %#ok<AGROW>
        stats.elemsByPass(end+1,1) = size(elem,1); %#ok<AGROW>
        stats.freeDOFByPass(end+1,1) = ...
            count_free_dofs(node,cfg); %#ok<AGROW>
        stats.totalBisectedParentsByPass(end+1,1) = ...
            rstats.totalBisectedParents; %#ok<AGROW>
        stats.completionBisectedParentsByPass(end+1,1) = ...
            completion; %#ok<AGROW>
        stats.minAngleByPass(end+1,1) = Q.minAngleDeg; %#ok<AGROW>
        stats.p05AngleByPass(end+1,1) = Q.p05AngleDeg; %#ok<AGROW>

        fprintf(['  Operator pass %02d: active=%d, marked=%d, ' ...
                 'DOF=%d, elems=%d, completion=%d, ' ...
                 'score=[%.2f,%.2f,%.2f]\n'], ...
            pass,numel(active),numel(markedElem), ...
            stats.freeDOFByPass(end),size(elem,1),completion, ...
            min(scoreActive),mean(scoreActive),max(scoreActive));
    end

    generation = element_generation_from_area_local(node,elem,A0);
    active = find(generation<cfg.operatorMaxLevel);
    remaining = 0;
    finalEffectiveScore = zeros(size(elem,1),1);

    if ~isempty(active)
        scoreActive = evaluate_score_on_elements_vectorized( ...
            Fscore,node,elem,active,cfg);
        effectiveScore = min(max( ...
            cfg.operatorRefinementMultiplier*scoreActive + ...
            generationOffset + cfg.operatorScoreSafetyBias,0), ...
            cfg.operatorMaxLevel);
        desiredGeneration = score_to_generation_threshold03( ...
            effectiveScore,cfg.operatorMaxLevel, ...
            cfg.generationThreshold);
        finalEffectiveScore(active) = effectiveScore;
        remaining = nnz(generation(active) < desiredGeneration);
    end

    if remaining>0
        error(['Operator mesh stopped with %d triangles still below the ', ...
               'continuous target. Increase operatorMaxRefineCalls.'], ...
            remaining);
    end

    Qfinal = mesh_angle_quality(node,elem);
    stats.finalNodes = size(node,1);
    stats.finalElems = size(elem,1);
    stats.finalFreeDOF = count_free_dofs(node,cfg);
    stats.finalGeneration = uint8(min(generation,255));
    stats.finalEffectiveScore = single(finalEffectiveScore);
    stats.remainingBelowTarget = remaining;
    stats.finalQuality = Qfinal;
end


function scoreElem = evaluate_score_on_elements_vectorized( ...
        Fscore,node,elem,elementIds,cfg)
%EVALUATE_SCORE_ON_ELEMENTS_VECTORIZED
% Query the score in chunks and take the maximum over each triangle.
%
% Pattern 'four':
%   centroid + three edge midpoints.
%
% Pattern 'seven':
%   centroid + three edge midpoints + three vertices.

    elementIds = elementIds(:);
    n = numel(elementIds);
    scoreElem = zeros(n,1);

    chunkSize = max(1,round(cfg.operatorQueryChunkElements));
    period = cfg.xmax-cfg.xmin;

    for first = 1:chunkSize:n
        last = min(first+chunkSize-1,n);
        ids = elementIds(first:last);

        p1 = node(elem(ids,1),:);
        p2 = node(elem(ids,2),:);
        p3 = node(elem(ids,3),:);

        centroid = (p1+p2+p3)/3;
        m12 = 0.5*(p1+p2);
        m23 = 0.5*(p2+p3);
        m31 = 0.5*(p3+p1);

        switch lower(cfg.operatorQueryPattern)
            case 'four'
                qx = [centroid(:,1),m12(:,1),m23(:,1),m31(:,1)];
                qt = [centroid(:,2),m12(:,2),m23(:,2),m31(:,2)];
            case 'seven'
                qx = [centroid(:,1),m12(:,1),m23(:,1),m31(:,1), ...
                      p1(:,1),p2(:,1),p3(:,1)];
                qt = [centroid(:,2),m12(:,2),m23(:,2),m31(:,2), ...
                      p1(:,2),p2(:,2),p3(:,2)];
            otherwise
                error('Unknown cfg.operatorQueryPattern="%s".', ...
                    cfg.operatorQueryPattern);
        end

        qx = cfg.xmin + mod(qx-cfg.xmin,period);
        qt = min(max(qt,cfg.tmin),cfg.tmax);

        values = Fscore(qx(:),qt(:));
        values = reshape(values,size(qx));
        values(~isfinite(values)) = 0;

        scoreElem(first:last) = max(values,[],2);
    end
end


function plot_fno_target_score_comparison( ...
        predScore,targetScore,queryX,queryT,u0vec,xInput, ...
        operatorNode,operatorElem,operatorStats, ...
        targetNode,targetElem,targetStats,caseTag,cfg)
%PLOT_FNO_TARGET_SCORE_COMPARISON
% Compare the FNO score, generated target score, realized generations and
% resulting periodic-NVB meshes.

    switch lower(cfg.operatorName)
        case 'fno'
            operatorLabel = 'FNO';
        case 'deeponet'
            operatorLabel = 'DeepONet';
        case 'pod_deeponet'
            operatorLabel = 'POD-DeepONet';
        case 'transolver'
            operatorLabel = 'Transolver';
        case 'cno'
            operatorLabel = 'CNO';
        otherwise
            operatorLabel = upper(cfg.operatorName);
    end

    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[20,30,1900,930]);
    tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

    nexttile;
    plot(xInput,u0vec,'LineWidth',1.5);
    grid on;
    box on;
    xlabel('x');
    ylabel('u_0(x)');
    title('Initial condition');

    nexttile;
    imagesc(queryX,queryT,predScore.');
    axis xy tight;
    xlabel('x');
    ylabel('t');
    caxis([0,cfg.exportedScoreMaxLevel]);
    cb = colorbar;
    cb.Label.String = 'continuous score';
    title(sprintf('%s predicted score',operatorLabel));

    nexttile;
    imagesc(queryX,queryT,targetScore.');
    axis xy tight;
    xlabel('x');
    ylabel('t');
    caxis([0,cfg.exportedScoreMaxLevel]);
    cb = colorbar;
    cb.Label.String = 'continuous score';
    title('Generated target score');

    nexttile;
    imagesc(queryX,queryT,abs(predScore-targetScore).');
    axis xy tight;
    xlabel('x');
    ylabel('t');
    caxis auto;
    cb = colorbar;
    cb.Label.String = sprintf('|s_{%s}-s_{target}|',operatorLabel);
    title('Absolute score error');

    nexttile;
    generation = double(operatorStats.finalGeneration(:));
    [elemPlot,generationPlot] = decimate_face_values_for_plot( ...
        operatorElem,generation,cfg.plotMaxElements);
    patch('Faces',elemPlot,'Vertices',operatorNode, ...
        'FaceVertexCData',generationPlot, ...
        'FaceColor','flat','EdgeColor','none');
    view(2);
    axis equal tight;
    box on;
    xlabel('x');
    ylabel('t');
    caxis([0,cfg.operatorMaxLevel]);
    cb = colorbar;
    cb.Label.String = 'realized generation';
    title(sprintf('%s seed generations: DOF=%d', ...
        operatorLabel,operatorStats.finalFreeDOF));

    nexttile;
    generation = double(targetStats.finalGeneration(:));
    [elemPlot,generationPlot] = decimate_face_values_for_plot( ...
        targetElem,generation,cfg.plotMaxElements);
    patch('Faces',elemPlot,'Vertices',targetNode, ...
        'FaceVertexCData',generationPlot, ...
        'FaceColor','flat','EdgeColor','none');
    view(2);
    axis equal tight;
    box on;
    xlabel('x');
    ylabel('t');
    caxis([0,cfg.operatorMaxLevel]);
    cb = colorbar;
    cb.Label.String = 'realized generation';
    title(sprintf('Target seed generations: DOF=%d', ...
        targetStats.finalFreeDOF));

    nexttile;
    draw_mesh_for_plot(operatorNode,operatorElem,cfg);
    title(sprintf('%s seed mesh: %d elements', ...
        operatorLabel,size(operatorElem,1)));

    nexttile;
    draw_mesh_for_plot(targetNode,targetElem,cfg);
    title(sprintf('Target seed mesh: %d elements',size(targetElem,1)));

    colormap(parula);
    sgtitle(strrep(caseTag,'_','\_'));

    exportgraphics(fig,fullfile(cfg.figureDir, ...
        [caseTag '_fno_vs_target_score_and_mesh.png']), ...
        'Resolution',cfg.figureResolution);
    close(fig);
end

function [elemPlot,valuePlot] = decimate_face_values_for_plot( ...
        elem,faceValues,maxElements)
    if size(elem,1)>maxElements
        stride = ceil(size(elem,1)/maxElements);
        ids = 1:stride:size(elem,1);
        elemPlot = elem(ids,:);
        valuePlot = faceValues(ids);
    else
        elemPlot = elem;
        valuePlot = faceValues;
    end
end


function [final,history] = run_fixed_afem_cycles_fast_exact( ...
        node,elem,u,initialTime,initialErr,initialNewton, ...
        u0fun,ref,cfg,nCycles,methodTag,Fref)
%RUN_FIXED_AFEM_CYCLES_FAST_EXACT
% Same numerical AFEM correction cycle as the legacy routine. Only the
% periodic-NVB implementation and optional intermediate quality diagnostics
% are accelerated. Estimator, Dörfler marked set, interpolation warm start,
% Newton equations/tolerances and nonconvergence policy are unchanged.

    validateattributes(nCycles,{'numeric'}, ...
        {'scalar','real','finite','integer','>=',0});
    if nargin<11 || isempty(methodTag)
        methodTag = 'fixed AFEM correction';
    end
    if nargin<12 || isempty(Fref)
        Fref = make_ref_interpolant(ref);
    end

    cumulativeTime = double(initialTime);
    dof = count_free_dofs(node,cfg);
    Q = mesh_angle_quality(node,elem);
    warn_bad_mesh_quality_local(Q,cfg,[methodTag ' seed mesh']);

    history.stage = 0;
    history.dof = dof;
    history.nodes = size(node,1);
    history.elements = size(elem,1);
    history.spaceTimeRelL2 = initialErr.spaceTimeL2Rel;
    history.finalTimeRelL2 = initialErr.finalTimeL2Rel;
    history.cumulativeTimeSec = cumulativeTime;
    history.incrementalCycleTimeSec = 0;
    history.estimateMarkTimeSec = 0;
    history.refineSolveTimeSec = 0;
    history.refineTimeSec = 0;
    history.assemblyTimeSec = 0;
    history.solveTimeSec = 0;
    history.cumulativeRefineTimeSec = 0;
    history.cumulativeAssemblyTimeSec = 0;
    history.cumulativeSolveTimeSec = 0;
    history.estimator = NaN;
    history.markedElements = 0;
    history.totalBisectedParents = 0;
    history.completionBisectedParents = 0;
    history.minAngleDeg = Q.minAngleDeg;
    history.p05AngleDeg = Q.p05AngleDeg;
    [nc0,ni0,nr0,nt0,ne0] = unpack_newton_summary(initialNewton);
    history.newtonConverged = nc0;
    history.newtonIterations = ni0;
    history.newtonFinalResidual = nr0;
    history.newtonTargetResidual = nt0;
    history.newtonExitReason = {ne0};

    currentErr = initialErr;
    currentNewton = initialNewton;
    cumRefineTime = 0.0;
    cumAssemblyTime = 0.0;
    cumSolveTime = 0.0;

    for cycle = 1:nCycles
        estimateTimer = tic;
        eta2 = residual_indicator_burgers_st(node,elem,u,u0fun,cfg);
        eta2(~isfinite(eta2) | eta2<0) = 0;
        etaTotal = sqrt(sum(eta2));
        markedElem = dorfler_ranked_marking(eta2,cfg.theta);
        estimateMarkTime = toc(estimateTimer);
        if isempty(markedElem)
            [~,id] = max(eta2);
            markedElem = id;
        end

        oldNode = node;
        oldElem = elem;
        oldU = u;
        state = struct();
        state.elementGeneration = [];
        state.elementNewLeaf = false(size(elem,1),1);

        refineSolveTimer = tic;
        refineTimer = tic;
        [node,elem,rstats] = nvb_refine_conforming_periodic_fast_exact( ...
            oldNode,oldElem,markedElem,cfg,state);
        cycleRefineTime = toc(refineTimer);
        if size(elem,1)>cfg.maxElementsCase
            error(['%s exceeded cfg.maxElementsCase=%d during fixed ', ...
                   'correction cycle %d.'], ...
                methodTag,cfg.maxElementsCase,cycle);
        end
        uGuess = interpolate_to_new_nodes(oldNode,oldElem,oldU,node);
        [u,currentNewton] = solve_burgers_newton_st( ...
            node,elem,u0fun,cfg,uGuess);
        currentNewton.initializationMode = ...
            'previous_afem_solution_interpolation';
        refineSolveTime = toc(refineSolveTimer);
        cycleAssemblyTime = newton_bucket_time( ...
            currentNewton,'assemblyTimeSec');
        cycleSolveTime = newton_bucket_time( ...
            currentNewton,'solveTimeSec');
        cumRefineTime = cumRefineTime + cycleRefineTime;
        cumAssemblyTime = cumAssemblyTime + cycleAssemblyTime;
        cumSolveTime = cumSolveTime + cycleSolveTime;

        incrementalTime = estimateMarkTime+refineSolveTime;
        cumulativeTime = cumulativeTime+incrementalTime;

        currentErr = compute_fem_vs_reference_error( ...
            node,elem,u,ref,cfg,Fref);
        dof = count_free_dofs(node,cfg);
        if isfield(cfg,'fastSkipIntermediateMeshQuality') && ...
                cfg.fastSkipIntermediateMeshQuality
            Q.minAngleDeg = NaN;
            Q.p05AngleDeg = NaN;
        else
            Q = mesh_angle_quality(node,elem);
            warn_bad_mesh_quality_local(Q,cfg, ...
                sprintf('%s fixed AFEM cycle %d',methodTag,cycle));
        end
        completion=max(0,rstats.totalBisectedParents-numel(markedElem));

        history.stage(end+1,1)=cycle; %#ok<AGROW>
        history.dof(end+1,1)=dof; %#ok<AGROW>
        history.nodes(end+1,1)=size(node,1); %#ok<AGROW>
        history.elements(end+1,1)=size(elem,1); %#ok<AGROW>
        history.spaceTimeRelL2(end+1,1)=currentErr.spaceTimeL2Rel; %#ok<AGROW>
        history.finalTimeRelL2(end+1,1)=currentErr.finalTimeL2Rel; %#ok<AGROW>
        history.cumulativeTimeSec(end+1,1)=cumulativeTime; %#ok<AGROW>
        history.incrementalCycleTimeSec(end+1,1)=incrementalTime; %#ok<AGROW>
        history.estimateMarkTimeSec(end+1,1)=estimateMarkTime; %#ok<AGROW>
        history.refineSolveTimeSec(end+1,1)=refineSolveTime; %#ok<AGROW>
        history.refineTimeSec(end+1,1)=cycleRefineTime; %#ok<AGROW>
        history.assemblyTimeSec(end+1,1)=cycleAssemblyTime; %#ok<AGROW>
        history.solveTimeSec(end+1,1)=cycleSolveTime; %#ok<AGROW>
        history.cumulativeRefineTimeSec(end+1,1)=cumRefineTime; %#ok<AGROW>
        history.cumulativeAssemblyTimeSec(end+1,1)=cumAssemblyTime; %#ok<AGROW>
        history.cumulativeSolveTimeSec(end+1,1)=cumSolveTime; %#ok<AGROW>
        history.estimator(end+1,1)=etaTotal; %#ok<AGROW>
        history.markedElements(end+1,1)=numel(markedElem); %#ok<AGROW>
        history.totalBisectedParents(end+1,1)=rstats.totalBisectedParents; %#ok<AGROW>
        history.completionBisectedParents(end+1,1)=completion; %#ok<AGROW>
        history.minAngleDeg(end+1,1)=Q.minAngleDeg; %#ok<AGROW>
        history.p05AngleDeg(end+1,1)=Q.p05AngleDeg; %#ok<AGROW>
        [nc,ni,nr,nt,ne]=unpack_newton_summary(currentNewton);
        history.newtonConverged(end+1,1)=nc; %#ok<AGROW>
        history.newtonIterations(end+1,1)=ni; %#ok<AGROW>
        history.newtonFinalResidual(end+1,1)=nr; %#ok<AGROW>
        history.newtonTargetResidual(end+1,1)=nt; %#ok<AGROW>
        history.newtonExitReason{end+1,1}=ne; %#ok<AGROW>

        fprintf(['  %s correction %d/%d: DOF=%d, elems=%d, ', ...
                 'marked=%d, completion=%d, eta=%.3e, ', ...
                 'ST rel=%.3e, cycle time=%.3f s, total=%.3f s\n'], ...
            methodTag,cycle,nCycles,dof,size(elem,1),numel(markedElem), ...
            completion,etaTotal,currentErr.spaceTimeL2Rel, ...
            incrementalTime,cumulativeTime);
    end

    final = make_afem_snapshot( ...
        nCycles,node,elem,u,dof,cumulativeTime,currentErr,currentNewton);
    final.seedTimeSec = double(initialTime);
    final.correctionTimeSec = cumulativeTime-double(initialTime);
    final.fixedCorrectionCycles = nCycles;
    final.methodTag = methodTag;
    final.refineTimeSec = cumRefineTime;
    final.assemblyTimeSec = cumAssemblyTime;
    final.solveTimeSec = cumSolveTime;

    fn=fieldnames(history);
    for k=1:numel(fn)
        history.(fn{k})=history.(fn{k})(:);
    end
end


function [final,history] = run_fixed_afem_cycles( ...
        node,elem,u,initialTime,initialErr,initialNewton, ...
        u0fun,ref,cfg,nCycles,methodTag,Fref)
%RUN_FIXED_AFEM_CYCLES
% Starting from an already solved score-generated mesh, perform exactly
% nCycles complete residual-based AFEM cycles:
%
%       ESTIMATE -> DORFLER MARK -> PERIODIC NVB REFINE -> NEWTON SOLVE.
%
% The cumulative method time includes all four online operations in every
% correction cycle. Error evaluation and mesh-quality diagnostics are
% deliberately excluded from method timing, matching the rest of the file.

    validateattributes(nCycles,{'numeric'}, ...
        {'scalar','real','finite','integer','>=',0});
    if nargin<11 || isempty(methodTag)
        methodTag = 'fixed AFEM correction';
    end
    if nargin<12 || isempty(Fref)
        Fref = make_ref_interpolant(ref);
    end

    cumulativeTime = double(initialTime);
    dof = count_free_dofs(node,cfg);
    Q = mesh_angle_quality(node,elem);
    warn_bad_mesh_quality_local(Q,cfg,[methodTag ' seed mesh']);

    history.stage = 0;
    history.dof = dof;
    history.nodes = size(node,1);
    history.elements = size(elem,1);
    history.spaceTimeRelL2 = initialErr.spaceTimeL2Rel;
    history.finalTimeRelL2 = initialErr.finalTimeL2Rel;
    history.cumulativeTimeSec = cumulativeTime;
    history.incrementalCycleTimeSec = 0;
    history.estimateMarkTimeSec = 0;
    history.refineSolveTimeSec = 0;
    history.estimator = NaN;
    history.markedElements = 0;
    history.totalBisectedParents = 0;
    history.completionBisectedParents = 0;
    history.minAngleDeg = Q.minAngleDeg;
    history.p05AngleDeg = Q.p05AngleDeg;
    [nc0,ni0,nr0,nt0,ne0] = unpack_newton_summary(initialNewton);
    history.newtonConverged = nc0;
    history.newtonIterations = ni0;
    history.newtonFinalResidual = nr0;
    history.newtonTargetResidual = nt0;
    history.newtonExitReason = {ne0};

    currentErr = initialErr;
    currentNewton = initialNewton;

    for cycle = 1:nCycles
        estimateTimer = tic;
        eta2 = residual_indicator_burgers_st(node,elem,u,u0fun,cfg);
        eta2(~isfinite(eta2) | eta2<0) = 0;
        etaTotal = sqrt(sum(eta2));
        markedElem = dorfler_ranked_marking(eta2,cfg.theta);
        estimateMarkTime = toc(estimateTimer);

        if isempty(markedElem)
            [~,id] = max(eta2);
            markedElem = id;
        end

        oldNode = node;
        oldElem = elem;
        oldU = u;

        refineSolveTimer = tic;
        [node,elem,rstats] = nvb_refine_conforming_periodic( ...
            oldNode,oldElem,markedElem,cfg);

        if size(elem,1)>cfg.maxElementsCase
            error(['%s exceeded cfg.maxElementsCase=%d during fixed ', ...
                   'correction cycle %d.'], ...
                methodTag,cfg.maxElementsCase,cycle);
        end

        check_mesh_valid(node,elem, ...
            sprintf('%s after fixed AFEM cycle %d',methodTag,cycle));

        uGuess = interpolate_to_new_nodes( ...
            oldNode,oldElem,oldU,node);
        [u,currentNewton] = solve_burgers_newton_st( ...
            node,elem,u0fun,cfg,uGuess);
        currentNewton.initializationMode = ...
            'previous_afem_solution_interpolation';
        refineSolveTime = toc(refineSolveTimer);

        incrementalTime = estimateMarkTime + refineSolveTime;
        cumulativeTime = cumulativeTime + incrementalTime;

        % Diagnostics below are excluded from method time.
        currentErr = compute_fem_vs_reference_error( ...
            node,elem,u,ref,cfg,Fref);
        dof = count_free_dofs(node,cfg);
        Q = mesh_angle_quality(node,elem);
        warn_bad_mesh_quality_local(Q,cfg, ...
            sprintf('%s fixed AFEM cycle %d',methodTag,cycle));

        completion = max(0, ...
            rstats.totalBisectedParents-numel(markedElem));

        history.stage(end+1,1) = cycle; %#ok<AGROW>
        history.dof(end+1,1) = dof; %#ok<AGROW>
        history.nodes(end+1,1) = size(node,1); %#ok<AGROW>
        history.elements(end+1,1) = size(elem,1); %#ok<AGROW>
        history.spaceTimeRelL2(end+1,1) = ...
            currentErr.spaceTimeL2Rel; %#ok<AGROW>
        history.finalTimeRelL2(end+1,1) = ...
            currentErr.finalTimeL2Rel; %#ok<AGROW>
        history.cumulativeTimeSec(end+1,1) = ...
            cumulativeTime; %#ok<AGROW>
        history.incrementalCycleTimeSec(end+1,1) = ...
            incrementalTime; %#ok<AGROW>
        history.estimateMarkTimeSec(end+1,1) = ...
            estimateMarkTime; %#ok<AGROW>
        history.refineSolveTimeSec(end+1,1) = ...
            refineSolveTime; %#ok<AGROW>
        history.estimator(end+1,1) = etaTotal; %#ok<AGROW>
        history.markedElements(end+1,1) = ...
            numel(markedElem); %#ok<AGROW>
        history.totalBisectedParents(end+1,1) = ...
            rstats.totalBisectedParents; %#ok<AGROW>
        history.completionBisectedParents(end+1,1) = ...
            completion; %#ok<AGROW>
        history.minAngleDeg(end+1,1) = Q.minAngleDeg; %#ok<AGROW>
        history.p05AngleDeg(end+1,1) = Q.p05AngleDeg; %#ok<AGROW>
        [nc,ni,nr,nt,ne] = unpack_newton_summary(currentNewton);
        history.newtonConverged(end+1,1) = nc; %#ok<AGROW>
        history.newtonIterations(end+1,1) = ni; %#ok<AGROW>
        history.newtonFinalResidual(end+1,1) = nr; %#ok<AGROW>
        history.newtonTargetResidual(end+1,1) = nt; %#ok<AGROW>
        history.newtonExitReason{end+1,1} = ne; %#ok<AGROW>

        fprintf(['  %s correction %d/%d: DOF=%d, elems=%d, ' ...
                 'marked=%d, completion=%d, eta=%.3e, ' ...
                 'ST rel=%.3e, cycle time=%.3f s, total=%.3f s\n'], ...
            methodTag,cycle,nCycles,dof,size(elem,1), ...
            numel(markedElem),completion,etaTotal, ...
            currentErr.spaceTimeL2Rel,incrementalTime,cumulativeTime);
    end

    final = make_afem_snapshot( ...
        nCycles,node,elem,u,dof,cumulativeTime,currentErr,currentNewton);
    final.seedTimeSec = double(initialTime);
    final.correctionTimeSec = cumulativeTime-double(initialTime);
    final.fixedCorrectionCycles = nCycles;
    final.methodTag = methodTag;

    fn = fieldnames(history);
    for k = 1:numel(fn)
        history.(fn{k}) = history.(fn{k})(:);
    end
end


function [selected,history] = ...
    run_afem_for_two_targets_and_fixed_fast_exact( ...
        node,elem,u,baseTime,baseErr,baseNewtonInfo, ...
        targetTime,targetError,u0fun,ref,cfg,Fref)
%RUN_AFEM_FOR_TWO_TARGETS_AND_FIXED_FAST_EXACT
% Run one ordinary standard-AFEM trajectory and retain only the enabled
% outputs. The independent row switches are:
%   cfg.runTimeMatchedAFEM
%   cfg.runAccuracyMatchedAFEM
%   cfg.runFixedRefinementAFEM
%
% Disabled criteria do not appear in the output and do not force the shared
% trajectory to continue. There is no standard-AFEM DOF-matched selection.

    wantTime = logical(cfg.runTimeMatchedAFEM);
    wantAccuracy = logical(cfg.runAccuracyMatchedAFEM);
    wantFixed = logical(cfg.runFixedRefinementAFEM);
    if ~(wantTime || wantAccuracy || wantFixed)
        error('Standard AFEM was called with no enabled selection criterion.');
    end
    if wantTime
        validateattributes(targetTime,{'numeric'}, ...
            {'scalar','real','finite','>=',0});
    end
    if wantAccuracy
        validateattributes(targetError,{'numeric'}, ...
            {'scalar','real','finite','>',0});
    end

    cumulativeTime = baseTime;
    dof = count_free_dofs(node,cfg);
    Q0 = mesh_angle_quality(node,elem);
    warn_bad_mesh_quality_local(Q0,cfg,'AFEM initial mesh');

    current = make_afem_snapshot( ...
        0,node,elem,u,dof,cumulativeTime,baseErr,baseNewtonInfo);

    history.stage = 0;
    history.dof = dof;
    history.nodes = size(node,1);
    history.elements = size(elem,1);
    history.spaceTimeRelL2 = baseErr.spaceTimeL2Rel;
    history.finalTimeRelL2 = baseErr.finalTimeL2Rel;
    history.cumulativeTimeSec = cumulativeTime;
    history.estimator = NaN;
    history.markedElements = 0;
    history.totalBisectedParents = 0;
    history.completionBisectedParents = 0;
    history.refineTimeSec = 0;
    history.assemblyTimeSec = 0;
    history.solveTimeSec = 0;
    history.cumulativeRefineTimeSec = 0;
    history.cumulativeAssemblyTimeSec = 0;
    history.cumulativeSolveTimeSec = 0;
    history.minAngleDeg = Q0.minAngleDeg;
    history.p05AngleDeg = Q0.p05AngleDeg;

    selected.time = [];
    selected.accuracy = [];
    selected.fixed = [];

    if wantFixed && cfg.fixedAFEMRefineCycles==0
        selected.fixed = annotate_afem_selection( ...
            current,'fixed_refine_cycles',0,0,true);
    end

    bestTime = [];
    bestAccuracy = [];
    bestTimeGap = Inf;
    bestAccuracyGap = Inf;

    if wantTime
        bestTime = current;
        bestTimeGap = relative_gap(current.timeSec,targetTime);
        if current.timeSec>=targetTime
            selected.time = annotate_afem_selection( ...
                current,'time',targetTime,current.timeSec,true);
        end
    end
    if wantAccuracy
        bestAccuracy = current;
        bestAccuracyGap = relative_gap( ...
            current.error.spaceTimeL2Rel,targetError);
        if current.error.spaceTimeL2Rel<=targetError
            selected.accuracy = annotate_afem_selection( ...
                current,'accuracy',targetError, ...
                current.error.spaceTimeL2Rel,true);
        end
    end

    previous = current;
    cumRefineTime = 0.0;
    cumAssemblyTime = 0.0;
    cumSolveTime = 0.0;

    for cycle = 1:cfg.afemMaxCycles
        doneTime = ~wantTime || ~isempty(selected.time);
        doneAccuracy = ~wantAccuracy || ~isempty(selected.accuracy);
        doneFixed = ~wantFixed || ~isempty(selected.fixed);
        if doneTime && doneAccuracy && doneFixed
            break;
        end

        estimateTimer = tic;
        eta2 = residual_indicator_burgers_st(node,elem,u,u0fun,cfg);
        eta2(~isfinite(eta2) | eta2<0) = 0;
        etaTotal = sqrt(sum(eta2));
        rankedMarked = dorfler_ranked_marking(eta2,cfg.theta);
        estimateMarkTime = toc(estimateTimer);

        if isempty(rankedMarked)
            [~,id] = max(eta2);
            rankedMarked = id;
        end

        fullStepTimer = tic;
        oldNode = node;
        oldElem = elem;
        oldU = u;

        state = struct();
        state.elementGeneration = [];
        state.elementNewLeaf = false(size(elem,1),1);

        refineTimer = tic;
        [nodeNew,elemNew,rstats] = ...
            nvb_refine_conforming_periodic_fast_exact( ...
                node,elem,rankedMarked,cfg,state);
        cycleRefineTime = toc(refineTimer);

        if size(elemNew,1)>cfg.maxElementsCase
            if wantFixed && isempty(selected.fixed)
                error(['Standard AFEM exceeded cfg.maxElementsCase=%d ', ...
                       'before the requested fixed-%d-refinement stage ', ...
                       'was completed.'], ...
                    cfg.maxElementsCase,cfg.fixedAFEMRefineCycles);
            end

            warning('NOEM:AFEMElementCapReached', ...
                ['Standard AFEM would exceed cfg.maxElementsCase=%d. ', ...
                 'Stopping at the last completed stage %d and using the ', ...
                 'closest available enabled matching stages.'], ...
                cfg.maxElementsCase,previous.stage);
            break;
        end

        uGuess = interpolate_to_new_nodes( ...
            oldNode,oldElem,oldU,nodeNew);
        [uNew,newtonInfo] = solve_burgers_newton_st( ...
            nodeNew,elemNew,u0fun,cfg,uGuess);
        newtonInfo.initializationMode = ...
            'previous_afem_solution_interpolation';

        refineSolveTime = toc(fullStepTimer);
        cycleAssemblyTime = newton_bucket_time( ...
            newtonInfo,'assemblyTimeSec');
        cycleSolveTime = newton_bucket_time( ...
            newtonInfo,'solveTimeSec');
        cumRefineTime = cumRefineTime + cycleRefineTime;
        cumAssemblyTime = cumAssemblyTime + cycleAssemblyTime;
        cumSolveTime = cumSolveTime + cycleSolveTime;
        cumulativeTime = cumulativeTime + ...
            estimateMarkTime + refineSolveTime;

        node = nodeNew;
        elem = elemNew;
        u = uNew;
        dof = count_free_dofs(node,cfg);

        err = compute_fem_vs_reference_error( ...
            node,elem,u,ref,cfg,Fref);

        if isfield(cfg,'fastSkipIntermediateMeshQuality') && ...
                cfg.fastSkipIntermediateMeshQuality
            Q.minAngleDeg = NaN;
            Q.p05AngleDeg = NaN;
        else
            Q = mesh_angle_quality(node,elem);
            warn_bad_mesh_quality_local(Q,cfg, ...
                sprintf('AFEM cycle %d',cycle));
        end

        completion = max(0, ...
            rstats.totalBisectedParents-numel(rankedMarked));

        history.stage(end+1,1) = cycle; %#ok<AGROW>
        history.dof(end+1,1) = dof; %#ok<AGROW>
        history.nodes(end+1,1) = size(node,1); %#ok<AGROW>
        history.elements(end+1,1) = size(elem,1); %#ok<AGROW>
        history.spaceTimeRelL2(end+1,1) = ...
            err.spaceTimeL2Rel; %#ok<AGROW>
        history.finalTimeRelL2(end+1,1) = ...
            err.finalTimeL2Rel; %#ok<AGROW>
        history.cumulativeTimeSec(end+1,1) = ...
            cumulativeTime; %#ok<AGROW>
        history.estimator(end+1,1) = etaTotal; %#ok<AGROW>
        history.markedElements(end+1,1) = ...
            numel(rankedMarked); %#ok<AGROW>
        history.totalBisectedParents(end+1,1) = ...
            rstats.totalBisectedParents; %#ok<AGROW>
        history.completionBisectedParents(end+1,1) = ...
            completion; %#ok<AGROW>
        history.refineTimeSec(end+1,1) = cycleRefineTime; %#ok<AGROW>
        history.assemblyTimeSec(end+1,1) = cycleAssemblyTime; %#ok<AGROW>
        history.solveTimeSec(end+1,1) = cycleSolveTime; %#ok<AGROW>
        history.cumulativeRefineTimeSec(end+1,1) = ...
            cumRefineTime; %#ok<AGROW>
        history.cumulativeAssemblyTimeSec(end+1,1) = ...
            cumAssemblyTime; %#ok<AGROW>
        history.cumulativeSolveTimeSec(end+1,1) = ...
            cumSolveTime; %#ok<AGROW>
        history.minAngleDeg(end+1,1) = Q.minAngleDeg; %#ok<AGROW>
        history.p05AngleDeg(end+1,1) = Q.p05AngleDeg; %#ok<AGROW>

        current = make_afem_snapshot( ...
            cycle,node,elem,u,dof,cumulativeTime,err,newtonInfo);

        if wantFixed && isempty(selected.fixed) && ...
                cycle==cfg.fixedAFEMRefineCycles
            selected.fixed = annotate_afem_selection( ...
                current,'fixed_refine_cycles', ...
                cfg.fixedAFEMRefineCycles,cycle,true);
        end

        if wantTime
            g = relative_gap(current.timeSec,targetTime);
            if g<bestTimeGap
                bestTimeGap = g;
                bestTime = current;
            end
            if isempty(selected.time) && current.timeSec>=targetTime
                chosen = closer_snapshot( ...
                    previous,current,previous.timeSec,current.timeSec, ...
                    targetTime);
                selected.time = annotate_afem_selection( ...
                    chosen,'time',targetTime,chosen.timeSec,true);
            end
        end

        if wantAccuracy
            g = relative_gap(current.error.spaceTimeL2Rel,targetError);
            if g<bestAccuracyGap
                bestAccuracyGap = g;
                bestAccuracy = current;
            end
            if isempty(selected.accuracy) && ...
                    current.error.spaceTimeL2Rel<=targetError
                chosen = closer_snapshot( ...
                    previous,current, ...
                    previous.error.spaceTimeL2Rel, ...
                    current.error.spaceTimeL2Rel,targetError);
                selected.accuracy = annotate_afem_selection( ...
                    chosen,'accuracy',targetError, ...
                    chosen.error.spaceTimeL2Rel,true);
            end
        end

        fprintf(['  AFEM cycle %02d: DOF=%d, elems=%d, ', ...
                 'marked=%d, completion=%d, eta=%.3e, ', ...
                 'ST rel=%.3e, cumulative time=%.3f s\n'], ...
            cycle,dof,size(elem,1),numel(rankedMarked), ...
            completion,etaTotal,err.spaceTimeL2Rel,cumulativeTime);

        previous = current;
    end

    if wantTime && isempty(selected.time)
        selected.time = annotate_afem_selection( ...
            bestTime,'time',targetTime,bestTime.timeSec,false);
        warning(['AFEM did not reach hybrid time %.3f s; ', ...
                 'using closest available stage %d at %.3f s.'], ...
            targetTime,bestTime.stage,bestTime.timeSec);
    end

    if wantAccuracy && isempty(selected.accuracy)
        selected.accuracy = annotate_afem_selection( ...
            bestAccuracy,'accuracy',targetError, ...
            bestAccuracy.error.spaceTimeL2Rel,false);
        warning(['AFEM did not reach hybrid error %.3e; ', ...
                 'using closest available stage %d at %.3e.'], ...
            targetError,bestAccuracy.stage, ...
            bestAccuracy.error.spaceTimeL2Rel);
    end

    if wantFixed && isempty(selected.fixed)
        error(['AFEM did not reach the requested fixed stage %d within ', ...
               'cfg.afemMaxCycles=%d.'], ...
            cfg.fixedAFEMRefineCycles,cfg.afemMaxCycles);
    end

    fn = fieldnames(history);
    for k = 1:numel(fn)
        history.(fn{k}) = history.(fn{k})(:);
    end
end


function rankedMarked = dorfler_ranked_marking(eta2,theta)
%DORFLER_RANKED_MARKING Return the descending eta_K^2 prefix.

    eta2 = double(eta2(:));
    eta2(~isfinite(eta2) | eta2<0) = 0;

    [values,order] = sort(eta2,'descend');
    total = sum(values);

    if total<=0
        rankedMarked = order(1);
        return;
    end

    m = find(cumsum(values)>=theta*total,1,'first');
    rankedMarked = order(1:m);
end


function chosen = closer_snapshot(S1,S2,v1,v2,target)
    if relative_gap(v1,target)<=relative_gap(v2,target)
        chosen = S1;
    else
        chosen = S2;
    end
end


function g = relative_gap(value,target)
    g = abs(double(value)-double(target))/ ...
        max(abs(double(target)),eps);
end


function S = annotate_afem_selection( ...
        S,criterion,target,achieved,reached)
    S.matchCriterion = criterion;
    S.matchTarget = target;
    S.matchAchieved = achieved;
    S.matchRelativeGap = relative_gap(achieved,target);
    S.targetWasBracketed = logical(reached);
end


function T = strip_afem_mesh_fields(S)
    T = S;
    for f = {'node','elem','u'}
        if isfield(T,f{1})
            T = rmfield(T,f{1});
        end
    end
end


function plot_six_meshes_generic( ...
        nodes,elems,dofs,qualities,names,caseTag,cfg)
%PLOT_SIX_MESHES_GENERIC Dynamic enabled-method mesh panel.
% The historical function name is retained for compatibility.

    nMethods = numel(nodes);
    if nMethods<1 || numel(elems)~=nMethods || ...
            numel(dofs)~=nMethods || numel(qualities)~=nMethods || ...
            numel(names)~=nMethods
        error('Enabled-method mesh plotting inputs are inconsistent.');
    end

    nCols = min(3,nMethods);
    nRows = ceil(nMethods/nCols);
    figHeight = max(480,430*nRows);
    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[30,40,1850,figHeight]);
    tiledlayout(nRows,nCols,'Padding','compact','TileSpacing','compact');

    for m=1:nMethods
        nexttile;
        draw_mesh_for_plot(nodes{m},elems{m},cfg);
        title({names{m},sprintf( ...
            'DOF=%d, elems=%d, min angle=%.1f^\\circ', ...
            dofs(m),size(elems{m},1),qualities{m}.minAngleDeg)}, ...
            'Interpreter','tex');
    end

    sgtitle(strrep(caseTag,'_','\\_'));
    exportgraphics(fig,fullfile(cfg.figureDir, ...
        [caseTag '_enabled_method_meshes.png']), ...
        'Resolution',cfg.figureResolution);
    close(fig);
end

function data = plot_six_pointwise_absolute_error( ...
        nodes,elems,solutions,ref,Fref,names,errors,times,dofs,caseTag,cfg)
%PLOT_SIX_POINTWISE_ABSOLUTE_ERROR Dynamic enabled-method error panel.
% The historical function name is retained for compatibility.

    nMethods = numel(nodes);
    if nMethods<1 || numel(elems)~=nMethods || ...
            numel(solutions)~=nMethods || numel(names)~=nMethods || ...
            numel(errors)~=nMethods || numel(times)~=nMethods || ...
            numel(dofs)~=nMethods
        error('Enabled-method pointwise plotting inputs are inconsistent.');
    end

    nCols = min(3,nMethods);
    nRows = ceil(nMethods/nCols);
    figHeight = max(480,430*nRows);
    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[30,40,1850,figHeight]);
    tiledlayout(nRows,nCols,'Padding','compact','TileSpacing','compact');

    data.quantity = '|u_h-u_ref|';
    data.methodTimesSec = double(times(:)).';
    data.methods = repmat(struct( ...
        'name','','node',[],'error',[],'colorLimit',[]),nMethods,1);

    for m=1:nMethods
        refValues = Fref(nodes{m}(:,2),nodes{m}(:,1));
        refValues(~isfinite(refValues)) = 0;
        nodalError = abs(solutions{m}(:)-refValues(:));

        limits = plot_one_fem_absolute_error_six( ...
            nodes{m},elems{m},nodalError,names{m}, ...
            errors{m}.spaceTimeL2Rel,times(m),dofs(m),cfg);

        data.methods(m).name = names{m};
        data.methods(m).node = single(nodes{m});
        data.methods(m).error = single(nodalError);
        data.methods(m).colorLimit = limits;
    end

    colormap(parula);
    sgtitle({strrep(caseTag,'_','\\_'), ...
        sprintf('Pointwise absolute FEM error for %d enabled methods', ...
            nMethods)});

    exportgraphics(fig,fullfile(cfg.figureDir, ...
        [caseTag '_enabled_method_pointwise_absolute_error.png']), ...
        'Resolution',cfg.figureResolution);
    close(fig);
end

function limits = plot_one_fem_absolute_error_six( ...
        node,elem,nodalError,methodName,stRel,methodTime,dof,cfg)
%PLOT_ONE_FEM_ABSOLUTE_ERROR_SIX
% Display-only decimation protects very large 64x64 comparison figures.

    nexttile;

    if size(elem,1)>cfg.plotMaxElements
        stride = ceil(size(elem,1)/cfg.plotMaxElements);
        elemPlot = elem(1:stride:end,:);
    else
        elemPlot = elem;
    end

    if strcmpi(cfg.pointwiseErrorFaceColor,'flat')
        faceError = mean(nodalError(elemPlot),2);
        h = patch('Faces',elemPlot,'Vertices',node, ...
            'FaceVertexCData',faceError,'FaceColor','flat');
    else
        h = patch('Faces',elemPlot,'Vertices',node, ...
            'FaceVertexCData',nodalError,'FaceColor','interp');
    end

    if cfg.pointwiseShowMeshEdges
        h.EdgeColor = [0.25,0.25,0.25];
        h.LineWidth = 0.08;
    else
        h.EdgeColor = 'none';
    end

    view(2);
    axis tight;
    box on;
    xlabel('x');
    ylabel('t');
    caxis auto;
    limits = caxis;
    cb = colorbar;
    cb.Label.String = '|u_h-u_{ref}|';
    title({sprintf('%s   |u_h-u_{ref}|',methodName), ...
        sprintf('ST rel. L^2=%.3e, time=%.3f s, DOF=%d', ...
        stRel,methodTime,dof)},'Interpreter','tex');
end


function plot_afem_selected_stage_convergence( ...
        history,targetDOF,targetTime,targetError,selected,caseTag,cfg)
%PLOT_AFEM_SELECTED_STAGE_CONVERGENCE
% Plot the shared standard-AFEM trajectory and only the enabled selections.

    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[80,80,1450,570]);
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    h = loglog(history.dof,history.spaceTimeRelL2, ...
        '-o','LineWidth',1.4,'MarkerSize',5);
    handles = h;
    labels = {'AFEM history'};
    hold on;
    if isfinite(targetDOF)
        xline(targetDOF,'--','hybrid DOF reference', ...
            'HandleVisibility','off');
    end
    if isfinite(targetError)
        yline(targetError,'--','hybrid error', ...
            'HandleVisibility','off');
    end
    if isfield(selected,'time') && ~isempty(selected.time)
        h = loglog(selected.time.dof, ...
            selected.time.error.spaceTimeL2Rel,'p', ...
            'MarkerSize',12,'LineWidth',1.5);
        handles(end+1) = h; %#ok<AGROW>
        labels{end+1} = 'time match'; %#ok<AGROW>
    end
    if isfield(selected,'accuracy') && ~isempty(selected.accuracy)
        h = loglog(selected.accuracy.dof, ...
            selected.accuracy.error.spaceTimeL2Rel,'s', ...
            'MarkerSize',9,'LineWidth',1.5);
        handles(end+1) = h; %#ok<AGROW>
        labels{end+1} = 'accuracy match'; %#ok<AGROW>
    end
    if isfield(selected,'fixed') && ~isempty(selected.fixed)
        h = loglog(selected.fixed.dof, ...
            selected.fixed.error.spaceTimeL2Rel,'d', ...
            'MarkerSize',9,'LineWidth',1.5);
        handles(end+1) = h; %#ok<AGROW>
        labels{end+1} = 'fixed-cycle stage'; %#ok<AGROW>
    end
    grid on; box on;
    xlabel('free nodal DOFs');
    ylabel('space-time relative L^2 error');
    title('AFEM trajectory versus DOF');
    legend(handles,labels,'Location','southwest');

    nexttile;
    h = loglog(history.cumulativeTimeSec, ...
        history.spaceTimeRelL2,'-o', ...
        'LineWidth',1.4,'MarkerSize',5);
    handles = h;
    labels = {'AFEM history'};
    hold on;
    if isfinite(targetTime)
        xline(targetTime,'--','hybrid time', ...
            'HandleVisibility','off');
    end
    if isfinite(targetError)
        yline(targetError,'--','hybrid error', ...
            'HandleVisibility','off');
    end
    if isfield(selected,'time') && ~isempty(selected.time)
        h = loglog(selected.time.timeSec, ...
            selected.time.error.spaceTimeL2Rel,'p', ...
            'MarkerSize',12,'LineWidth',1.5);
        handles(end+1) = h; %#ok<AGROW>
        labels{end+1} = 'time match'; %#ok<AGROW>
    end
    if isfield(selected,'accuracy') && ~isempty(selected.accuracy)
        h = loglog(selected.accuracy.timeSec, ...
            selected.accuracy.error.spaceTimeL2Rel,'s', ...
            'MarkerSize',9,'LineWidth',1.5);
        handles(end+1) = h; %#ok<AGROW>
        labels{end+1} = 'accuracy match'; %#ok<AGROW>
    end
    if isfield(selected,'fixed') && ~isempty(selected.fixed)
        h = loglog(selected.fixed.timeSec, ...
            selected.fixed.error.spaceTimeL2Rel,'d', ...
            'MarkerSize',9,'LineWidth',1.5);
        handles(end+1) = h; %#ok<AGROW>
        labels{end+1} = 'fixed-cycle stage'; %#ok<AGROW>
    end
    grid on; box on;
    xlabel('cumulative AFEM method time (s)');
    ylabel('space-time relative L^2 error');
    title('AFEM trajectory versus time');
    legend(handles,labels,'Location','southwest');

    sgtitle(strrep(caseTag,'_','\_'));
    exportgraphics(fig,fullfile(cfg.figureDir, ...
        [caseTag '_afem_selected_stages.png']), ...
        'Resolution',cfg.figureResolution);
    close(fig);
end


function S = make_lightweight_six_method_result(fullS)
    S = fullS;
    methodFields = { ...
        'operatorSeed','operatorHybrid','targetSeed','targetHybrid', ...
        'afemTime','afemAccuracy','afemFixed','uniform','fixedUniform'};
    for k=1:numel(methodFields)
        f = methodFields{k};
        if isfield(S,f)
            if isfield(S.(f),'node')
                S.(f) = rmfield(S.(f),'node');
            end
            if isfield(S.(f),'elem')
                S.(f) = rmfield(S.(f),'elem');
            end
            if isfield(S.(f),'u')
                S.(f) = rmfield(S.(f),'u');
            end
        end
    end
    if isfield(S,'pointwiseError')
        S = rmfield(S,'pointwiseError');
    end
end

function row = make_extended_summary_row( ...
        caseNo,testId,sourceId,method,matchType,targetValue, ...
        achievedValue,selectedStage,dof,nodes,elements, ...
        err,timeSec,quality,referenceTime,newtonInfo)

    if isfinite(targetValue) && isfinite(achievedValue)
        matchGap = relative_gap(achievedValue,targetValue);
    else
        matchGap = NaN;
    end

    [newtonConverged,newtonIterations,newtonFinalResidual, ...
        newtonTargetResidual,newtonExitReason] = ...
        unpack_newton_summary(newtonInfo);

    row = { ...
        caseNo,testId,sourceId,method,matchType, ...
        targetValue,achievedValue,matchGap,selectedStage, ...
        dof,nodes,elements,err.spaceTimeL2Rel, ...
        err.finalTimeL2Rel,timeSec,quality.minAngleDeg, ...
        quality.p05AngleDeg,referenceTime, ...
        newtonConverged,newtonIterations,newtonFinalResidual, ...
        newtonTargetResidual,newtonExitReason};
end

function [converged,iterations,finalResidual,targetResidual,exitReason] = ...
        unpack_newton_summary(ninfo)
%UNPACK_NEWTON_SUMMARY Convert a Newton-info struct to table-safe scalars.

    converged = NaN;
    iterations = NaN;
    finalResidual = NaN;
    targetResidual = NaN;
    exitReason = '';

    if isempty(ninfo) || ~isstruct(ninfo)
        return;
    end
    if isfield(ninfo,'converged')
        converged = double(logical(ninfo.converged));
    end
    if isfield(ninfo,'iter')
        iterations = double(ninfo.iter);
    end
    if isfield(ninfo,'finalResidual')
        finalResidual = double(ninfo.finalResidual);
    end
    if isfield(ninfo,'targetResidual')
        targetResidual = double(ninfo.targetResidual);
    end
    if isfield(ninfo,'exitReason')
        exitReason = char(string(ninfo.exitReason));
    end
end

%% ========================================================================
%                       Prediction-data utilities
% ========================================================================
function sourceId = get_source_index(D,testId,nTest,nDatasetSamples)
%GET_SOURCE_INDEX Return the one-based dataset slot represented by testId.
%
% The canonical exporter stores original_indices as MATLAB one-based dataset
% indices.  When that metadata is absent, the fixed test split is assumed to
% be the final nTest samples of the original dataset.

    if isfield(D,'original_indices')
        ids = double(D.original_indices(:));
        if testId>numel(ids)
            error('testId=%d exceeds original_indices length %d.', ...
                testId,numel(ids));
        end
        sourceId = round(ids(testId));
    else
        if ~isfinite(nDatasetSamples)
            error(['Prediction MAT has no original_indices and the original ', ...
                   'dataset size is unavailable.']);
        end
        sourceId = round(nDatasetSamples-nTest+testId);
    end

    if isfinite(nDatasetSamples) && ...
            (sourceId<1 || sourceId>nDatasetSamples)
        error('Resolved source ID %d lies outside 1,...,%d.', ...
            sourceId,nDatasetSamples);
    end
end

function L = get_label_sample(A, k, nTest)
%GET_LABEL_SAMPLE Return one prediction as G x G x 2.
%
% The triangular Python exporter writes MATLAB arrays as
%       pred_labels : G x G x 2 x Ntest.
% This reader also accepts several permuted layouts to make the comparison
% script robust to files saved by other tools.

    sz = size(A);
    if ndims(A) ~= 4
        error(['pred_labels must be four-dimensional for triangular ', ...
               'labels. Got size %s.'], mat2str(sz));
    end

    candidateDims = find(sz == nTest);
    if isempty(candidateDims)
        error(['Cannot find the test-sample dimension in pred_labels ', ...
               'with size %s and nTest=%d.'], mat2str(sz), nTest);
    end

    % Prefer the last dimension because this is the format exported by the
    % supplied Python training code: G x G x 2 x Ntest.
    if sz(4) == nTest
        L = squeeze(A(:,:,:,k));
    elseif sz(1) == nTest
        L = squeeze(A(k,:,:,:));
        L = permute_triangular_label_to_GGT(L);
    elseif sz(3) == nTest
        L = squeeze(A(:,:,k,:));
        L = permute_triangular_label_to_GGT(L);
    elseif sz(2) == nTest
        L = squeeze(A(:,k,:,:));
        L = permute_triangular_label_to_GGT(L);
    else
        error('Unsupported pred_labels layout: %s.', mat2str(sz));
    end

    L = permute_triangular_label_to_GGT(L);
end

function L = permute_triangular_label_to_GGT(L)
%PERMUTE_TRIANGULAR_LABEL_TO_GGT Canonicalize a 3-D sample to G x G x 2.
    if ndims(L) ~= 3
        error('A triangular label sample must be 3-D, got %s.', ...
            mat2str(size(L)));
    end

    sz = size(L);
    triDim = find(sz == 2, 1, 'last');
    if isempty(triDim)
        error('Cannot identify the triangle dimension in size %s.', ...
            mat2str(sz));
    end

    gridDims = setdiff(1:3, triDim, 'stable');
    if sz(gridDims(1)) ~= sz(gridDims(2))
        error('The two macro-grid dimensions must be equal, got %s.', ...
            mat2str(sz));
    end

    L = permute(L, [gridDims, triDim]);
end


function validate_exact_burgers_dataset(B,nSamples)
%VALIDATE_EXACT_BURGERS_DATASET Check coefficient-array dimensions.

    required = { ...
        'sine_amplitude','sine_shift', ...
        'grf_xi_cos','grf_xi_sin','grf_modes','grf_spectral_std', ...
        'forcing_xi_cos','forcing_xi_sin', ...
        'forcing_modes','forcing_spectral_std'};

    for k=1:numel(required)
        if ~isfield(B,required{k})
            error('Exact Burgers dataset is missing field "%s".',required{k});
        end
    end

    if numel(B.sine_shift)~=nSamples
        error('sine_shift length differs from sine_amplitude length.');
    end
    if size(B.grf_xi_cos,2)~=nSamples || ...
            size(B.grf_xi_sin,2)~=nSamples
        error('GRF coefficient arrays do not have one column per sample.');
    end
    if size(B.forcing_xi_cos,2)~=nSamples || ...
            size(B.forcing_xi_sin,2)~=nSamples
        error('Forcing coefficient arrays do not have one column per sample.');
    end
    if size(B.grf_xi_cos,1)~=numel(B.grf_modes) || ...
            size(B.grf_xi_sin,1)~=numel(B.grf_modes) || ...
            numel(B.grf_spectral_std)~=numel(B.grf_modes)
        error('GRF modes, spectral standard deviations and coefficients disagree.');
    end
    if size(B.forcing_xi_cos,1)~=numel(B.forcing_modes) || ...
            size(B.forcing_xi_sin,1)~=numel(B.forcing_modes) || ...
            numel(B.forcing_spectral_std)~=numel(B.forcing_modes)
        error(['Forcing modes, spectral standard deviations and ', ...
               'coefficients disagree.']);
    end

    numericFields = required;
    for k=1:numel(numericFields)
        A = double(B.(numericFields{k}));
        if isempty(A) || any(~isfinite(A(:)))
            error('Exact Burgers dataset field "%s" contains NaN/Inf.', ...
                numericFields{k});
        end
    end
end


function [u0fun,ffun,meta] = ...
    make_exact_burgers_fourier_functions(B,sourceId,cfg)
%MAKE_EXACT_BURGERS_FOURIER_FUNCTIONS
% Reconstruct exactly the same truncated Fourier initial condition and
% forcing used by the MATLAB data generator:
%
%   u0(x) = -a*sin(pi*(x+b))
%           + sum_k sigma_k [xi_k^c cos(k*pi*x)+xi_k^s sin(k*pi*x)],
%
%   f(x)  = sum_k tau_k [zeta_k^c cos(k*pi*x)+zeta_k^s sin(k*pi*x)].
%
% The returned handles preserve the shape of the query array.

    sourceId = round(double(sourceId));
    nSamples = numel(B.sine_amplitude);
    if sourceId<1 || sourceId>nSamples
        error('sourceId=%d is outside the exact dataset range 1,...,%d.', ...
            sourceId,nSamples);
    end

    a = double(B.sine_amplitude(sourceId));
    b = double(B.sine_shift(sourceId));

    grfModes = double(B.grf_modes(:));
    grfSigma = double(B.grf_spectral_std(:));
    grfCos = double(B.grf_xi_cos(:,sourceId));
    grfSin = double(B.grf_xi_sin(:,sourceId));

    forcingModes = double(B.forcing_modes(:));
    forcingSigma = double(B.forcing_spectral_std(:));
    forcingCos = double(B.forcing_xi_cos(:,sourceId));
    forcingSin = double(B.forcing_xi_sin(:,sourceId));

    u0fun = @(x)evaluate_exact_burgers_initial_condition( ...
        x,a,b,grfModes,grfSigma,grfCos,grfSin);
    ffun = @(x)evaluate_exact_periodic_fourier_field( ...
        x,forcingModes,forcingSigma,forcingCos,forcingSin);

    % The formulas use k*pi*x and are periodic on [-1,1].
    testX = linspace(cfg.xmin,cfg.xmax,17).';
    if any(~isfinite(u0fun(testX))) || any(~isfinite(ffun(testX)))
        error('Exact Burgers Fourier reconstruction produced NaN/Inf.');
    end

    meta = struct();
    meta.sourceId = sourceId;
    meta.representation = 'exact-saved-Fourier-coefficients';
    meta.sineAmplitude = a;
    meta.sineShift = b;
    meta.grfModes = numel(grfModes);
    meta.forcingModes = numel(forcingModes);
end


function values = evaluate_exact_burgers_initial_condition( ...
        x,a,b,modes,sigma,xiCos,xiSin)

    originalSize = size(x);
    xv = double(x(:));
    values = -a*sin(pi*(xv+b)) + ...
        cos(pi*xv*modes.')*(sigma.*xiCos) + ...
        sin(pi*xv*modes.')*(sigma.*xiSin);
    values = reshape(values,originalSize);
end


function values = evaluate_exact_periodic_fourier_field( ...
        x,modes,sigma,xiCos,xiSin)

    originalSize = size(x);
    xv = double(x(:));
    values = cos(pi*xv*modes.')*(sigma.*xiCos) + ...
        sin(pi*xv*modes.')*(sigma.*xiSin);
    values = reshape(values,originalSize);
end


function u0fun = make_discrete_u0_function(xInput,u0vec,cfg)
%MAKE_DISCRETE_U0_FUNCTION
% Build a periodic interpolant from distinct samples on [xmin,xmax).
%
% The 101x101 axis-fix FNO exporter saves x_input without a duplicated
% right endpoint.  Therefore the last sensor must NOT be averaged with the
% first sensor.  The periodic endpoint is appended internally instead.

    xInput = double(xInput(:));
    u0vec = double(u0vec(:));

    if numel(xInput)~=numel(u0vec)
        error('x_input and u0 must have the same length.');
    end
    if any(~isfinite(xInput)) || any(~isfinite(u0vec))
        error('x_input or U0_test contains NaN/Inf.');
    end

    [xInput,order] = sort(xInput);
    u0vec = u0vec(order);

    tol = 100*eps(max(1,max(abs([cfg.xmin,cfg.xmax]))));
    if abs(xInput(1)-cfg.xmin)>100*tol
        error('The first x_input point must coincide with cfg.xmin.');
    end

    % Accept either distinct periodic sensors [xmin,xmax) or an older
    % representation that already includes xmax.
    if abs(xInput(end)-cfg.xmax)<=100*tol
        xPeriodic = xInput;
        uPeriodic = u0vec;
        uPeriodic(end) = uPeriodic(1);
    else
        if xInput(end)>=cfg.xmax
            error('x_input extends beyond cfg.xmax.');
        end
        xPeriodic = [xInput;cfg.xmax];
        uPeriodic = [u0vec;u0vec(1)];
    end

    try
        F = griddedInterpolant(xPeriodic,uPeriodic,'pchip','nearest');
    catch
        F = griddedInterpolant(xPeriodic,uPeriodic,'linear','nearest');
    end

    u0fun = @(x) evaluate_discrete_u0(F,x,cfg.xmin,cfg.xmax);
end

function y = evaluate_discrete_u0(F,x,xmin,xmax)
%EVALUATE_DISCRETE_U0 Evaluate the initial condition periodically.

    x = double(x);
    period = xmax-xmin;
    if period<=0
        error('Invalid periodic interval.');
    end

    xWrapped = xmin+mod(x-xmin,period);
    tol = 100*eps(max(1,max(abs([xmin,xmax]))));
    xWrapped(abs(x-xmax)<tol) = xmin;
    y = F(xWrapped);
end

function [node,elem,transitionLabel,stats] = ...
        build_neural_policy_mesh(predLabel,cfg)
%BUILD_NEURAL_POLICY_MESH
% Realize a GxGx2 macro-triangle target map with conforming periodic NVB.
%
% Triangle convention in macro rectangle (i,j):
%   triangle 1: [n00,n10,n11], local tau <= local xi;
%   triangle 2: [n00,n11,n01], local tau >  local xi.
%
% Each macro triangle has its own predicted target generation. Optional
% macro-level grading may be disabled; periodic NVB completion then provides
% only the conformity propagation required by the actual mesh.

    rawLabel=round(double(predLabel));
    expected=[cfg.neuralMacroGrid,cfg.neuralMacroGrid, ...
              cfg.trianglesPerRectangle];
    if ~isequal(size(rawLabel),expected)
        error('Triangular prediction has size %s, expected %s.', ...
            mat2str(size(rawLabel)),mat2str(expected));
    end
    if any(rawLabel(:)<1) || any(rawLabel(:)>cfg.maxLabel)
        error('Triangular prediction contains labels outside 1,...,%d.', ...
            cfg.maxLabel);
    end

    [edgeAdjacency,cornerAdjacency]= ...
        build_macro_triangle_adjacency(cfg.neuralMacroGrid,cfg);

    if cfg.removeSingletonHighLabels
        processedLabel=remove_singleton_high_labels_tri( ...
            rawLabel,cfg.maxLabel,edgeAdjacency);
    else
        processedLabel=rawLabel;
    end

    rawLevel=processedLabel-1+cfg.neuralSafetyLevels;
    rawLevel=min(max(rawLevel,0),cfg.neuralMaxLevel);

    doMacroGrading=isfield(cfg,'applyMacroLevelGrading') && ...
        logical(cfg.applyMacroLevelGrading);
    if doMacroGrading
        transitionLevel=grade_triangle_level_map_controlled_upward( ...
            rawLevel,edgeAdjacency,cornerAdjacency, ...
            cfg.edgeLevelDifference,cfg.cornerLevelDifference, ...
            cfg.neuralMaxLevel,cfg.maxTransitionIterations);
    else
        % Preserve the direct FNO labels.  Necessary conformity propagation
        % is produced by the actual periodic NVB completion below.
        transitionLevel=rawLevel;
    end
    transitionLabel=transitionLevel+1;

    [rawEdgeJump,rawCornerJump]=triangle_level_jump_diagnostics( ...
        rawLevel,edgeAdjacency,cornerAdjacency);
    [newEdgeJump,newCornerJump]=triangle_level_jump_diagnostics( ...
        transitionLevel,edgeAdjacency,cornerAdjacency);

    if doMacroGrading
        if newEdgeJump>cfg.edgeLevelDifference
            error('Transition map violates the full-edge level constraint.');
        end
        if newCornerJump>cfg.cornerLevelDifference
            error('Transition map violates the vertex-only level constraint.');
        end
        if any(transitionLevel(:)<rawLevel(:))
            error('Transition processing lowered a predicted triangle level.');
        end
    end

    [node,elem,A0]=make_uniform_spacetime_mesh( ...
        cfg.neuralMacroGrid,cfg);

    % The geometric macro-triangle convention is defined before this
    % one-time NVB vertex reordering.  Later target lookup is centroid based,
    % so the reordering does not change triangle ownership.
    elem=nvb_label_initial_mesh(node,elem);

    Qinitial=mesh_angle_quality(node,elem);

    stats.nodesByPass=size(node,1);
    stats.elemsByPass=size(elem,1);
    stats.targetMarkedByPass=[];
    stats.totalBisectedParentsByPass=[];
    stats.completionBisectedParentsByPass=[];
    stats.minAngleByPass=Qinitial.minAngleDeg;
    stats.p05AngleByPass=Qinitial.p05AngleDeg;
    stats.rawLabelCounts=histcounts( ...
        rawLabel(:),0.5:1:(cfg.maxLabel+0.5));
    stats.transitionLabelCounts=histcounts( ...
        transitionLabel(:),0.5:1:(cfg.maxLabel+0.5));
    stats.modifiedMacroTriangles=nnz(transitionLevel~=rawLevel);
    stats.modifiedMacroCells=stats.modifiedMacroTriangles; % compatibility
    stats.totalMacroTriangles=numel(rawLevel);
    stats.totalAddedLevels=round(sum(transitionLevel(:)-rawLevel(:)));
    stats.maxEdgeJumpRaw=rawEdgeJump;
    stats.maxCornerJumpRaw=rawCornerJump;
    stats.maxEdgeJumpTransition=newEdgeJump;
    stats.maxCornerJumpTransition=newCornerJump;

    refineCall=0;
    while true
        generation=element_generation_from_area_local(node,elem,A0);
        targetForElem=macro_triangle_target_for_elements( ...
            node,elem,transitionLevel,cfg);
        markedElem=find(generation<targetForElem);

        if isempty(markedElem)
            break;
        end

        refineCall=refineCall+1;
        if refineCall>cfg.neuralMaxRefineCalls
            error(['Neural NVB did not reach all triangular target levels ', ...
                   'after %d refinement calls.'], ...
                cfg.neuralMaxRefineCalls);
        end

        [node,elem,rstats]=nvb_refine_conforming_periodic( ...
            node,elem,markedElem,cfg);

        if size(elem,1)>cfg.maxElementsCase
            error(['Neural conforming mesh exceeded ', ...
                   'cfg.maxElementsCase=%d.'],cfg.maxElementsCase);
        end

        check_mesh_valid(node,elem, ...
            sprintf('neural triangular local-NVB mesh after call %d', ...
                refineCall));
        Q=mesh_angle_quality(node,elem);
        warn_bad_mesh_quality_local(Q,cfg, ...
            sprintf('neural triangular refine call %d',refineCall));

        stats.nodesByPass(end+1,1)=size(node,1); %#ok<AGROW>
        stats.elemsByPass(end+1,1)=size(elem,1); %#ok<AGROW>
        stats.targetMarkedByPass(end+1,1)=numel(markedElem); %#ok<AGROW>
        stats.totalBisectedParentsByPass(end+1,1)= ...
            rstats.totalBisectedParents; %#ok<AGROW>
        stats.completionBisectedParentsByPass(end+1,1)= ...
            max(0,rstats.totalBisectedParents-numel(markedElem)); %#ok<AGROW>
        stats.minAngleByPass(end+1,1)=Q.minAngleDeg; %#ok<AGROW>
        stats.p05AngleByPass(end+1,1)=Q.p05AngleDeg; %#ok<AGROW>
    end

    generation=element_generation_from_area_local(node,elem,A0);
    targetForElem=macro_triangle_target_for_elements( ...
        node,elem,transitionLevel,cfg);
    remaining=nnz(generation<targetForElem);

    Qfinal=mesh_angle_quality(node,elem);
    warn_bad_mesh_quality_local(Qfinal,cfg,'final neural triangular mesh');

    stats.finalNodes=size(node,1);
    stats.finalElems=size(elem,1);
    stats.finalFreeDOF=count_free_dofs(node,cfg);
    stats.remainingBelowTarget=remaining;
    stats.rawLevel=uint8(rawLevel);
    stats.transitionLevel=uint8(transitionLevel);
    stats.actualElementGeneration=uint8(min(generation,255));
    stats.finalQuality=Qfinal;
end

function target = macro_triangle_target_for_elements( ...
        node,elem,levelMap,cfg)
%MACRO_TRIANGLE_TARGET_FOR_ELEMENTS
% Assign every current descendant element to one of the 2048 original
% macro triangles by its centroid.

    centers=(node(elem(:,1),:)+node(elem(:,2),:)+ ...
             node(elem(:,3),:))/3;

    G=size(levelMap,1);
    dx=(cfg.xmax-cfg.xmin)/G;
    dt=(cfg.tmax-cfg.tmin)/G;

    ix=floor((centers(:,1)-cfg.xmin)/dx)+1;
    it=floor((centers(:,2)-cfg.tmin)/dt)+1;
    ix=min(max(ix,1),G);
    it=min(max(it,1),G);

    xLeft=cfg.xmin+(ix-1)*dx;
    tBottom=cfg.tmin+(it-1)*dt;
    xi=(centers(:,1)-xLeft)/dx;
    tau=(centers(:,2)-tBottom)/dt;

    triId=ones(size(ix));
    triId(tau>xi)=2;

    linearId=sub2ind([G,G,2],ix,it,triId);
    target=double(levelMap(linearId));
end

function [edgeAdjacency,cornerAdjacency] = ...
        build_macro_triangle_adjacency(n,cfg)
%BUILD_MACRO_TRIANGLE_ADJACENCY
% Build adjacency on the periodic-x quotient mesh.
%
% edgeAdjacency   : macro triangles sharing a complete physical edge,
%                   including pairs across x=xmin/xmax;
% cornerAdjacency : macro triangles sharing a physical vertex but no edge,
%                   also including periodic seam neighbours.

    [node0,elem0]=make_uniform_spacetime_mesh(n,cfg);
    NT=size(elem0,1);
    nLabels=2*n*n;

    % Map element-row order to MATLAB L(i,j,k) linear indexing.
    rowToLabel=zeros(NT,1);
    row=0;
    for j=1:n
        for i=1:n
            row=row+1;
            rowToLabel(row)=sub2ind([n,n,2],i,j,1);
            row=row+1;
            rowToLabel(row)=sub2ind([n,n,2],i,j,2);
        end
    end

    % Canonical vertex IDs identify x=xmax with x=xmin at the same t.
    ix=round((node0(:,1)-cfg.xmin)/(cfg.xmax-cfg.xmin)*n);
    it=round((node0(:,2)-cfg.tmin)/(cfg.tmax-cfg.tmin)*n);
    ix=mod(ix,n);
    canonicalVertex=it*n+ix+1;
    nCanonical=n*(n+1);

    % Shared-edge adjacency on the quotient mesh.
    edgeAdjacency=cell(nLabels,1);
    e12=sort([canonicalVertex(elem0(:,1)), ...
              canonicalVertex(elem0(:,2))],2);
    e23=sort([canonicalVertex(elem0(:,2)), ...
              canonicalVertex(elem0(:,3))],2);
    e31=sort([canonicalVertex(elem0(:,3)), ...
              canonicalVertex(elem0(:,1))],2);
    allEdges=[e12;e23;e31];
    owners=[(1:NT)';(1:NT)';(1:NT)'];

    [~,~,ic]=unique(allEdges,'rows');
    edgeOwners=accumarray(ic,owners,[],@(z){unique(z)});

    for e=1:numel(edgeOwners)
        own=edgeOwners{e};
        if numel(own)~=2
            continue;
        end
        a=rowToLabel(own(1));
        b=rowToLabel(own(2));
        if a~=b
            edgeAdjacency{a}(end+1)=b; %#ok<AGROW>
            edgeAdjacency{b}(end+1)=a; %#ok<AGROW>
        end
    end

    % Vertex adjacency on the same quotient mesh.
    canonicalElem=[canonicalVertex(elem0(:,1)), ...
                   canonicalVertex(elem0(:,2)), ...
                   canonicalVertex(elem0(:,3))];
    vertexOwners=accumarray( ...
        canonicalElem(:),repmat((1:NT)',3,1), ...
        [nCanonical,1],@(z){unique(z)});

    vertexAdjacency=cell(nLabels,1);
    for v=1:numel(vertexOwners)
        own=vertexOwners{v};
        if numel(own)<2
            continue;
        end
        labelsAtVertex=rowToLabel(own(:));
        for aPos=1:numel(labelsAtVertex)
            a=labelsAtVertex(aPos);
            others=labelsAtVertex;
            others(aPos)=[];
            vertexAdjacency{a}=[vertexAdjacency{a},others(:).']; %#ok<AGROW>
        end
    end

    cornerAdjacency=cell(nLabels,1);
    for a=1:nLabels
        edgeAdjacency{a}=unique(edgeAdjacency{a});
        vertexAdjacency{a}=unique(vertexAdjacency{a});
        cornerAdjacency{a}=setdiff( ...
            vertexAdjacency{a},[a,edgeAdjacency{a}]);
    end
end

function L = grade_triangle_level_map_controlled_upward( ...
        L,edgeAdjacency,cornerAdjacency,edgeDiff,cornerDiff, ...
        maxLevel,maxIterations)
%GRADE_TRIANGLE_LEVEL_MAP_CONTROLLED_UPWARD
% Enforce graph-based level differences using upward changes only.

    originalSize=size(L);
    v=min(max(round(double(L(:))),0),maxLevel);

    for iter=1:maxIterations
        vnew=v;

        for a=1:numel(v)
            required=v(a);

            edgeNeighbours=edgeAdjacency{a};
            if ~isempty(edgeNeighbours)
                required=max(required, ...
                    max(v(edgeNeighbours)-edgeDiff));
            end

            cornerNeighbours=cornerAdjacency{a};
            if ~isempty(cornerNeighbours)
                required=max(required, ...
                    max(v(cornerNeighbours)-cornerDiff));
            end

            vnew(a)=min(max(vnew(a),required),maxLevel);
        end

        if isequal(vnew,v)
            L=reshape(v,originalSize);
            return;
        end
        v=vnew;
    end

    error(['Triangular controlled transition map did not converge in ', ...
           '%d iterations.'],maxIterations);
end

function [edgeJump,cornerJump] = triangle_level_jump_diagnostics( ...
        L,edgeAdjacency,cornerAdjacency)
    v=double(L(:));
    edgeJump=0;
    cornerJump=0;

    for a=1:numel(v)
        e=edgeAdjacency{a};
        if ~isempty(e)
            edgeJump=max(edgeJump,max(abs(v(a)-v(e))));
        end
        c=cornerAdjacency{a};
        if ~isempty(c)
            cornerJump=max(cornerJump,max(abs(v(a)-v(c))));
        end
    end
end

function Lout = remove_singleton_high_labels_tri( ...
        L,maxLabel,edgeAdjacency)
%REMOVE_SINGLETON_HIGH_LABELS_TRI
% Lower only one-triangle connected components at each threshold.

    originalSize=size(L);
    v=min(max(round(double(L(:))),1),maxLabel);
    n=numel(v);

    for lev=maxLabel:-1:2
        active=v>=lev;
        visited=false(n,1);

        for seed=1:n
            if ~active(seed) || visited(seed)
                continue;
            end

            queue=zeros(n,1);
            component=zeros(n,1);
            head=1;
            tail=1;
            count=0;
            queue(1)=seed;
            visited(seed)=true;

            while head<=tail
                current=queue(head);
                head=head+1;
                count=count+1;
                component(count)=current;

                neighbours=edgeAdjacency{current};
                for q=1:numel(neighbours)
                    nb=neighbours(q);
                    if active(nb) && ~visited(nb)
                        tail=tail+1;
                        queue(tail)=nb;
                        visited(nb)=true;
                    end
                end
            end

            if count==1
                id=component(1);
                v(id)=max(v(id)-1,1);
            end
        end
    end

    Lout=reshape(v,originalSize);
end

%% ========================================================================
%                           Uniform mesh matching
% ========================================================================
function n = uniform_grid_from_fixed_free_dof(freeDOF)
%UNIFORM_GRID_FROM_FIXED_FREE_DOF
% For the periodic-x, initial-time-Dirichlet space-time grid used here,
% freeDOF=n^2. A fixed free-DOF baseline must therefore use a perfect square.

    validateattributes(freeDOF,{'numeric'}, ...
        {'scalar','real','finite','integer','>=',4});
    n = round(sqrt(double(freeDOF)));
    if n^2~=double(freeDOF)
        error(['cfg.fixedUniformFreeDOF=%d is not a perfect square. ', ...
               'Choose n^2, for example 16384 (128^2), ', ...
               '40000 (200^2), or 65536 (256^2).'],freeDOF);
    end
    n = max(n,2);
end


function n = choose_uniform_grid_strictly_above_dof(targetDOF,cfg)
%CHOOSE_UNIFORM_GRID_STRICTLY_ABOVE_DOF
%
% For make_uniform_spacetime_mesh(n,cfg):
%   total nodes              = (n+1)^2,
%   periodic x identification removes n+1 right-side copies,
%   the initial line fixes n reduced unknowns,
% hence
%       freeDOF_uniform = n^2.
%
% Therefore the smallest completed uniform grid with
%       freeDOF_uniform > targetDOF
% is obtained analytically.

    targetDOF = double(targetDOF);
    n = floor(sqrt(max(targetDOF,0)))+1;
    n = max(n,2);

    if ~(n^2>targetDOF && (n-1)^2<=targetDOF)
        error('Internal uniform-grid DOF formula failed.');
    end

    if isfield(cfg,'fastAuditDOFCount') && cfg.fastAuditDOFCount
        [nodeNow,~] = make_uniform_spacetime_mesh(n,cfg);
        dofNow = count_free_dofs(nodeNow,cfg);
        if dofNow<=targetDOF
            error('Analytic uniform grid is not strictly above target DOF.');
        end

        if n>2
            [nodePrev,~] = make_uniform_spacetime_mesh(n-1,cfg);
            dofPrev = count_free_dofs(nodePrev,cfg);
            if dofPrev>targetDOF
                error('Analytic uniform grid is not minimal.');
            end
        end
    end
end

function S = make_afem_snapshot( ...
        stage,node,elem,u,dof,timeSec,err,newtonInfo)
    S.stage = stage;
    S.node = node;
    S.elem = elem;
    S.u = u;
    S.dof = dof;
    S.timeSec = timeSec;
    S.error = err;
    S.newtonInfo = newtonInfo;
end

%% ========================================================================
%                          DOF and quality measures
% ========================================================================
function dof = count_free_dofs(node,cfg)
%COUNT_FREE_DOFS Count independent free unknowns after periodic reduction.
%
% FAST-EXACT formula:
%   nReduced = N - (# right-side nodes),
% because every right-side node is identified with exactly one left-side
% node.  Initial-time reduced DOFs are fixed; right-bottom is identified
% with left-bottom, hence
%
%   nFixed = (# t=tmin nodes) - (# nodes simultaneously at x=xmax,t=tmin).
%
% This is exactly the same quotient used by build_periodic_reduction.

    useFast = isfield(cfg,'fastUseDirectDOFCount') && ...
        logical(cfg.fastUseDirectDOFCount);

    if ~useFast
        map = build_periodic_reduction(node,cfg);
        dof = numel(map.freeReduced);
        return;
    end

    tol = periodic_tolerance(cfg);
    N = size(node,1);
    x = node(:,1);
    t = node(:,2);

    right = abs(x-cfg.xmax)<tol;
    initial = abs(t-cfg.tmin)<tol;
    rightInitial = right & initial;

    nReduced = N - nnz(right);
    nFixed = nnz(initial) - nnz(rightInitial);
    dof = nReduced - nFixed;

    if dof<0
        error('FAST periodic DOF formula produced a negative count.');
    end

    if isfield(cfg,'fastAuditDOFCount') && cfg.fastAuditDOFCount
        map = build_periodic_reduction(node,cfg);
        dofLegacy = numel(map.freeReduced);
        if dof~=dofLegacy
            error(['FAST periodic DOF count %d differs from legacy ', ...
                   'reduction count %d.'],dof,dofLegacy);
        end
    end
end

function counts = periodic_dof_counts(node,cfg)
    useFast = isfield(cfg,'fastUseDirectDOFCount') && ...
        logical(cfg.fastUseDirectDOFCount);

    if useFast
        tol = periodic_tolerance(cfg);
        N = size(node,1);
        x = node(:,1);
        t = node(:,2);

        right = abs(x-cfg.xmax)<tol;
        initial = abs(t-cfg.tmin)<tol;
        rightInitial = right & initial;

        counts.totalNodes = N;
        counts.reducedDofs = N-nnz(right);
        counts.fixedDofs = nnz(initial)-nnz(rightInitial);
        counts.freeDofs = counts.reducedDofs-counts.fixedDofs;

        if isfield(cfg,'fastAuditDOFCount') && cfg.fastAuditDOFCount
            legacy = build_periodic_reduction(node,cfg);
            if counts.reducedDofs~=legacy.nReduced || ...
                    counts.fixedDofs~=numel(legacy.fixedReduced) || ...
                    counts.freeDofs~=numel(legacy.freeReduced)
                error('FAST periodic_dof_counts differs from legacy map.');
            end
        end
    else
        map = build_periodic_reduction(node,cfg);
        counts.totalNodes = size(node,1);
        counts.reducedDofs = map.nReduced;
        counts.fixedDofs = numel(map.fixedReduced);
        counts.freeDofs = numel(map.freeReduced);
    end
end

function map = build_periodic_reduction(node,cfg)
%BUILD_PERIODIC_REDUCTION Build P such that u_full=P*u_reduced.
% Right-boundary nodes are identified with left-boundary nodes having the
% same time coordinate.  The initial-time reduced DOFs are fixed.

    N=size(node,1);
    tol=periodic_tolerance(cfg);
    x=node(:,1);
    t=node(:,2);

    left=find(abs(x-cfg.xmin)<tol);
    right=find(abs(x-cfg.xmax)<tol);

    [tLeft,oLeft]=sort(t(left));
    [tRight,oRight]=sort(t(right));
    left=left(oLeft);
    right=right(oRight);

    if numel(left)~=numel(right) || ...
            (~isempty(left) && max(abs(tLeft-tRight))>20*tol)
        error(['Periodic boundary node sets do not match. ', ...
               'Run periodic NVB completion before solving.']);
    end

    master=(1:N).';
    master(right)=left;
    [~,~,rid]=unique(master);
    nReduced=max(rid);

    P=sparse((1:N).',rid,1,N,nReduced);
    representative=accumarray(rid,(1:N).',[],@min);

    initFull=find(abs(t-cfg.tmin)<tol);
    fixedReduced=unique(rid(initFull));
    allReduced=(1:nReduced).';
    freeReduced=setdiff(allReduced,fixedReduced);

    map.P=P;
    map.rid=rid;
    map.representative=representative;
    map.nReduced=nReduced;
    map.fixedReduced=fixedReduced;
    map.freeReduced=freeReduced;
    map.leftNodes=left;
    map.rightNodes=right;
end

function tol = periodic_tolerance(cfg)
    tol=1e-10*max([1,abs(cfg.xmin),abs(cfg.xmax), ...
        abs(cfg.tmin),abs(cfg.tmax)]);
end

function Q = mesh_angle_quality(node,elem)
    p1 = node(elem(:,1),:);
    p2 = node(elem(:,2),:);
    p3 = node(elem(:,3),:);

    a = sqrt(sum((p2-p3).^2,2));
    b = sqrt(sum((p1-p3).^2,2));
    c = sqrt(sum((p1-p2).^2,2));

    cosA = clamp_cos((b.^2+c.^2-a.^2)./(2*b.*c));
    cosB = clamp_cos((a.^2+c.^2-b.^2)./(2*a.*c));
    cosC = clamp_cos((a.^2+b.^2-c.^2)./(2*a.*b));

    angles = [acosd(cosA),acosd(cosB),acosd(cosC)];
    triMin = min(angles,[],2);
    triMin = triMin(isfinite(triMin));

    Q.minAngleDeg = min(triMin);
    Q.p05AngleDeg = simple_percentile(triMin,5);
    Q.medianMinAngleDeg = simple_percentile(triMin,50);
end

function y = clamp_cos(y)
    y = min(max(y,-1),1);
end

function q = simple_percentile(x,p)
    x = sort(x(:));
    if isempty(x)
        q = NaN;
        return;
    end
    pos = 1 + (numel(x)-1)*p/100;
    lo = floor(pos);
    hi = ceil(pos);
    if lo==hi
        q = x(lo);
    else
        q = x(lo) + (pos-lo)*(x(hi)-x(lo));
    end
end

%% ========================================================================
%                                Plotting
% ========================================================================
function plot_triangular_label_maps( ...
        rawLabel,gradedLabel,u0vec,xInput,caseTag,cfg)
%PLOT_TRIANGULAR_LABEL_MAPS Show the two triangle labels without merging.

    [macroNode,macroElem]=make_uniform_spacetime_mesh( ...
        cfg.neuralMacroGrid,cfg);
    rawFace=triangle_label_array_to_element_order(rawLabel);
    gradedFace=triangle_label_array_to_element_order(gradedLabel);
    addedFace=gradedFace-rawFace;

    fig=figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[40,80,1850,470]);
    tiledlayout(1,4,'Padding','compact','TileSpacing','compact');

    nexttile;
    plot(xInput,u0vec,'LineWidth',1.5);
    grid on;
    box on;
    xlabel('x');
    ylabel('u_0(x)');
    title('Initial condition');

    nexttile;
    draw_one_triangular_label_map( ...
        macroNode,macroElem,rawFace,[1,cfg.maxLabel]);
    title('Raw predicted triangle labels');
    cb=colorbar;
    cb.Ticks=1:cfg.maxLabel;
    cb.Label.String='label';

    nexttile;
    draw_one_triangular_label_map( ...
        macroNode,macroElem,gradedFace,[1,cfg.maxLabel]);
    title('Triangle targets used for NVB realization');
    cb=colorbar;
    cb.Ticks=1:cfg.maxLabel;
    cb.Label.String='label';

    nexttile;
    draw_one_triangular_label_map( ...
        macroNode,macroElem,addedFace, ...
        [0,max(1,max(addedFace))]);
    title('Added levels before NVB realization');
    cb=colorbar;
    cb.Label.String='added level';

    colormap(parula);
    sgtitle(strrep(caseTag,'_','\_'));
    exportgraphics(fig, ...
        fullfile(cfg.figureDir,[caseTag '_triangular_labels.png']), ...
        'Resolution',cfg.figureResolution);
    close(fig);
end

function draw_one_triangular_label_map(node,elem,faceValues,limits)
    patch('Faces',elem,'Vertices',node, ...
        'FaceVertexCData',faceValues(:), ...
        'FaceColor','flat', ...
        'EdgeColor',[0.30,0.30,0.30], ...
        'LineWidth',0.08);
    view(2);
    axis equal tight;
    box on;
    xlabel('x');
    ylabel('t');
    caxis(limits);
end

function faceValues = triangle_label_array_to_element_order(L)
%TRIANGLE_LABEL_ARRAY_TO_ELEMENT_ORDER
% Convert L(i,j,k) to make_uniform_spacetime_mesh element-row order.
    n=size(L,1);
    if ~isequal(size(L),[n,n,2])
        error('Expected triangular label array n x n x 2, got %s.', ...
            mat2str(size(L)));
    end

    faceValues=zeros(2*n*n,1);
    row=0;
    for j=1:n
        for i=1:n
            row=row+1;
            faceValues(row)=double(L(i,j,1));
            row=row+1;
            faceValues(row)=double(L(i,j,2));
        end
    end
end

function plot_three_meshes( ...
        n1,e1,d1,q1,n2,e2,d2,q2,n3,e3,d3,q3,caseTag,cfg)

    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[80,80,1700,500]);
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

    nexttile;
    draw_mesh_for_plot(n1,e1,cfg);
    title(sprintf(['Neural conforming CG mesh\nDOF=%d, elems=%d, ', ...
        'min angle=%.1f^\\circ'],d1,size(e1,1),q1.minAngleDeg));

    nexttile;
    draw_mesh_for_plot(n2,e2,cfg);
    title(sprintf(['Smallest strictly-above-DOF uniform CG mesh\nDOF=%d, elems=%d, ', ...
        'min angle=%.1f^\\circ'],d2,size(e2,1),q2.minAngleDeg));

    nexttile;
    draw_mesh_for_plot(n3,e3,cfg);
    title(sprintf(['16x16-start AFEM CG mesh\nDOF=%d, elems=%d, ', ...
        'min angle=%.1f^\\circ'],d3,size(e3,1),q3.minAngleDeg));

    sgtitle(strrep(caseTag,'_','\_'));
    outFile = fullfile(cfg.figureDir,[caseTag '_meshes.png']);
    exportgraphics(fig,outFile,'Resolution',cfg.figureResolution);
    close(fig);
end

function draw_mesh_for_plot(node,elem,cfg)
    if size(elem,1) > cfg.plotMaxElements
        stride = ceil(size(elem,1)/cfg.plotMaxElements);
        elemPlot = elem(1:stride:end,:);
    else
        elemPlot = elem;
    end

    patch('Faces',elemPlot,'Vertices',node, ...
        'FaceColor','none','EdgeColor',[0.15 0.15 0.15], ...
        'LineWidth',0.15);
    axis equal;
    axis tight;
    box on;
    xlabel('x');
    ylabel('t');
    xlim([cfg.xmin cfg.xmax]);
    ylim([cfg.tmin cfg.tmax]);
end

function data = plot_fem_pointwise_absolute_error( ...
        neuralNode,neuralElem,neuralU, ...
        uniformNode,uniformElem,uniformU, ...
        afemNode,afemElem,afemU, ...
        ref,names,stErr,methodTimes,caseTag,cfg)
%PLOT_FEM_POINTWISE_ABSOLUTE_ERROR
% Draw |u_h-u_ref| directly on each FEM triangulation.
%
% Each subplot uses its own independent linear color scale.
% The only reported scalar metric is the global space-time relative L2 error.

    Fref = make_ref_interpolant(ref);

    refNeural = Fref(neuralNode(:,2),neuralNode(:,1));
    refUniform = Fref(uniformNode(:,2),uniformNode(:,1));
    refAfem = Fref(afemNode(:,2),afemNode(:,1));

    refNeural(~isfinite(refNeural)) = 0;
    refUniform(~isfinite(refUniform)) = 0;
    refAfem(~isfinite(refAfem)) = 0;

    errNeural = abs(neuralU(:)-refNeural(:));
    errUniform = abs(uniformU(:)-refUniform(:));
    errAfem = abs(afemU(:)-refAfem(:));

    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[40,70,1850,570]);
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

    lim1 = plot_one_fem_absolute_error( ...
        neuralNode,neuralElem,errNeural,names{1}, ...
        stErr(1),methodTimes(1),cfg);

    lim2 = plot_one_fem_absolute_error( ...
        uniformNode,uniformElem,errUniform,names{2}, ...
        stErr(2),methodTimes(2),cfg);

    lim3 = plot_one_fem_absolute_error( ...
        afemNode,afemElem,errAfem,names{3}, ...
        stErr(3),methodTimes(3),cfg);

    colormap(parula);
    sgtitle({strrep(caseTag,'_','\_'), ...
        'Absolute pointwise FEM error  |u_h-u_{ref}|  (fully automatic color scales)'});

    outFile = fullfile( ...
        cfg.figureDir,[caseTag '_pointwise_absolute_error.png']);
    exportgraphics(fig,outFile,'Resolution',cfg.figureResolution);
    close(fig);

    % Store nodal errors and each subplot's independent color limits.
    data.quantity = '|u_h-u_ref|';
    data.methodTimesSec = double(methodTimes(:)).';
    data.neural.node = single(neuralNode);
    data.neural.error = single(errNeural);
    data.neural.colorLimit = lim1;

    data.uniform.node = single(uniformNode);
    data.uniform.error = single(errUniform);
    data.uniform.colorLimit = lim2;

    data.afem.node = single(afemNode);
    data.afem.error = single(errAfem);
    data.afem.colorLimit = lim3;
end

function limits = plot_one_fem_absolute_error( ...
        node,elem,nodalError,methodName,stRel,methodTime,cfg)

    nexttile;

    if strcmpi(cfg.pointwiseErrorFaceColor,'flat')
        faceError = mean(nodalError(elem),2);
        h = patch('Faces',elem,'Vertices',node, ...
            'FaceVertexCData',faceError, ...
            'FaceColor','flat');
    else
        h = patch('Faces',elem,'Vertices',node, ...
            'FaceVertexCData',nodalError, ...
            'FaceColor','interp');
    end

    if cfg.pointwiseShowMeshEdges
        h.EdgeColor = [0.25,0.25,0.25];
        h.LineWidth = 0.08;
    else
        h.EdgeColor = 'none';
    end

    view(2);
    axis tight;
    box on;
    xlabel('x');
    ylabel('t');

    % Do not impose any manual color limits.
    % MATLAB determines the limits independently from the actual CData
    % of the current subplot.
    caxis auto;
    limits = caxis;
    cb = colorbar;
    cb.Label.String = '|u_h-u_{ref}|';

    title({sprintf('%s   |u_h-u_{ref}|',methodName), ...
        sprintf(['Global space-time rel. L^2 = %.3e, ', ...
                 'time = %.3f s'],stRel,methodTime)}, ...
        'Interpreter','tex');
end


function plot_afem_convergence(history,targetDOF,selectedStage,caseTag,cfg)
    fig = figure('Color','w','Visible',cfg.figureVisible, ...
        'Position',[120,120,780,560]);

    h1 = loglog(history.dof,history.spaceTimeRelL2,'-o', ...
        'LineWidth',1.5,'MarkerSize',6);
    hold on;
    h2 = loglog(history.dof,history.finalTimeRelL2,'-s', ...
        'LineWidth',1.5,'MarkerSize',6);

    xline(targetDOF,'--','Neural-mesh DOF', ...
        'LabelVerticalAlignment','middle', ...
        'HandleVisibility','off');

    idx = find(history.stage==selectedStage,1);
    if ~isempty(idx)
        h3 = loglog(history.dof(idx),history.spaceTimeRelL2(idx), ...
            'p','MarkerSize',13,'LineWidth',1.5);
    else
        h3 = [];
    end

    grid on;
    box on;
    xlabel('free nodal DOFs');
    ylabel('relative L^2 error');
    if isempty(h3)
        legend([h1,h2],{'space-time','final time'}, ...
            'Location','southwest');
    else
        legend([h1,h2,h3], ...
            {'space-time','final time','selected AFEM stage'}, ...
            'Location','southwest');
    end
    title({strrep(caseTag,'_','\_'), ...
        'AFEM error decay versus degrees of freedom'});

    outFile = fullfile(cfg.figureDir,[caseTag '_afem_convergence.png']);
    exportgraphics(fig,outFile,'Resolution',cfg.figureResolution);
    close(fig);
end

%% ========================================================================
%                             Result packaging
% ========================================================================
function S = pack_method_result( ...
        node,elem,u,dof,timeSec,err,quality,newtonInfo)
    S.node = node;
    S.elem = elem;
    S.u = u;
    S.dof = dof;
    S.nodes = size(node,1);
    S.elements = size(elem,1);
    S.timeSec = timeSec;
    S.error = err;
    S.quality = quality;
    S.newtonInfo = newtonInfo;
end


function S = make_lightweight_sample_result(fullS)
% Keep the global comparison_result.mat compact.  Full meshes and
% solutions are already stored in each sample-specific MAT file.
    S = fullS;
    methodFields = {'neural','uniform','afem'};
    for k = 1:numel(methodFields)
        f = methodFields{k};
        if isfield(S,f)
            if isfield(S.(f),'node'), S.(f) = rmfield(S.(f),'node'); end
            if isfield(S.(f),'elem'), S.(f) = rmfield(S.(f),'elem'); end
            if isfield(S.(f),'u'),    S.(f) = rmfield(S.(f),'u'); end
        end
    end
    if isfield(S,'pointwiseError')
        S = rmfield(S,'pointwiseError');
    end
end

function row = make_summary_row( ...
        caseNo,testId,sourceId,method,dof,nodes,elements, ...
        err,timeSec,quality,referenceTime)
    row = {caseNo,testId,sourceId,method,dof,nodes,elements, ...
        err.spaceTimeL2Rel,err.finalTimeL2Rel,timeSec, ...
        quality.minAngleDeg,quality.p05AngleDeg,referenceTime};
end


function [node, elem, A0] = make_uniform_spacetime_mesh(n, cfg)
    xg = linspace(cfg.xmin, cfg.xmax, n+1);
    tg = linspace(cfg.tmin, cfg.tmax, n+1);
    [XX, TT] = meshgrid(xg, tg);
    node = [XX(:), TT(:)];

    id = @(i,j) j*(n+1) + i + 1; % i,j start from 0
    elem = zeros(2*n*n,3);
    cnt = 0;
    for j = 0:n-1
        for i = 0:n-1
            n00 = id(i,   j);
            n10 = id(i+1, j);
            n01 = id(i,   j+1);
            n11 = id(i+1, j+1);
            cnt = cnt + 1; elem(cnt,:) = [n00,n10,n11];
            cnt = cnt + 1; elem(cnt,:) = [n00,n11,n01];
        end
    end
    elem = elem(1:cnt,:);
    A0 = ((cfg.xmax-cfg.xmin)/n) * ((cfg.tmax-cfg.tmin)/n) / 2;
end

%% ========================================================================
%                   Nonlinear space-time FEM solver
% ========================================================================
function [uFull,ninfo] = solve_burgers_newton_st( ...
        node,elem,u0fun,cfg,u_init)
%SOLVE_BURGERS_NEWTON_ST
% Damped Newton solve with algebraic periodic identification in x.

    % --- Wall-clock breakdown: matrix assembly vs. linear solve ---
    tAsmTotal = 0.0;
    tSolveTotal = 0.0;
    tAsmSetup = tic;
    map=build_periodic_reduction(node,cfg);
    geom=precompute_fem_assembly_geometry(node,elem);
    geom.fq=cfg.forcingFun(geom.xq);
    tAsmTotal = toc(tAsmSetup);
    P=map.P;
    rep=map.representative;
    free=map.freeReduced;
    fixed=map.fixedReduced;

    N=size(node,1);
    x=node(:,1);
    t=node(:,2);
    tol=periodic_tolerance(cfg);

    gFull=zeros(N,1);
    initMask=abs(t-cfg.tmin)<tol;
    gFull(initMask)=u0fun(x(initMask));

    gRed=zeros(map.nReduced,1);
    gRed(fixed)=gFull(rep(fixed));

    if nargin<5 || isempty(u_init)
        % Backward-compatible default: extend u0(x) through the time slab.
        initializationMode = 'initial_condition';
        initialFull = u0fun(x);
    elseif ischar(u_init) || (isstring(u_init) && isscalar(u_init))
        initializationMode = normalize_newton_initialization_mode(u_init);
        switch initializationMode
            case 'zero'
                initialFull = zeros(N,1);
            case 'initial_condition'
                initialFull = u0fun(x);
            case 'coarse_interpolation'
                error(['The solver cannot construct a coarse interpolation ', ...
                       'without an explicit numeric vector.']);
            otherwise
                error('Unsupported Newton initialization mode: %s', ...
                    initializationMode);
        end
    elseif isnumeric(u_init) && isvector(u_init) && numel(u_init)==N
        initialFull = double(u_init(:));
        initializationMode = 'provided_interpolated_solution';
    else
        error(['u_init must be empty, a supported mode string, or a ', ...
               'numeric vector with one value per mesh node.']);
    end

    if any(~isfinite(initialFull))
        error('Newton initial guess contains NaN/Inf.');
    end

    uRed=initialFull(rep);
    uRed(fixed)=gRed(fixed);
    uFull=P*uRed;

    tAsm = tic;
    [Rfull,Jfull]=assemble_burgers_res_jac_precomputed( ...
        geom,uFull,cfg.nu);
    tAsmTotal = tAsmTotal + toc(tAsm);
    Rred=P.'*Rfull;
    Jred=P.'*Jfull*P;

    initialResidual=norm(Rred(free))/sqrt(max(numel(free),1));
    target=max(cfg.newtonTol,cfg.newtonRelTol*initialResidual);

    ninfo.initialResidual=initialResidual;
    ninfo.targetResidual=target;
    ninfo.initializationMode=initializationMode;
    ninfo.acceptedSteps=[];
    ninfo.regularization=[];
    ninfo.converged=false;
    ninfo.usedLastIterate=false;
    ninfo.exitReason='running';
    ninfo.assemblyTimeSec = tAsmTotal;
    ninfo.solveTimeSec = tSolveTotal;

    for it=1:cfg.newtonMaxIt
        residualVector=Rred(free);
        resNorm=norm(residualVector)/sqrt(max(numel(free),1));

        if cfg.newtonVerbose
            fprintf('      Newton %02d residual %.3e\n',it-1,resNorm);
        end

        if resNorm<=target
            ninfo.iter=it-1;
            ninfo.finalResidual=resNorm;
            ninfo.converged=true;
            ninfo.usedLastIterate=false;
            ninfo.exitReason='converged';
            uFull=P*uRed;
            return;
        end

        A=Jred(free,free);
        rhs=-residualVector;
        accepted=false;
        baseNorm=norm(residualVector);
        bestNorm=baseNorm;
        bestRed=uRed;
        bestStep=0;
        bestMu=NaN;

        matrixScale=max(1,norm(A,1));
        baseMu=max(cfg.newtonRegularizationBase*matrixScale, ...
            eps(matrixScale));

        for regTry=0:cfg.newtonRegularizationTrials
            if regTry==0
                mu=0;
            else
                mu=baseMu*10^(regTry-1);
            end

            Atry=A;
            if mu>0
                Atry=Atry+mu*speye(size(Atry));
            end

            tSol = tic;
            try
                duFree=Atry\rhs;
            catch
                continue;
            end
            ninfo.solveTimeSec = ninfo.solveTimeSec + toc(tSol);
            if any(~isfinite(duFree))
                continue;
            end

            maxAllowed=cfg.newtonMaxUpdateFactor* ...
                max(1,max(abs(uRed(free))));
            duInf=norm(duFree,inf);
            if duInf>maxAllowed
                duFree=duFree*(maxAllowed/duInf);
            end

            if cfg.newtonLineSearch
                maxLS=cfg.newtonLineSearchMax;
            else
                maxLS=0;
            end

            for ls=0:maxLS
                step=2^(-ls);
                trialRed=uRed;
                trialRed(free)=uRed(free)+step*duFree;
                trialRed(fixed)=gRed(fixed);
                trialFull=P*trialRed;

                tAsm = tic;
                RtrialFull=assemble_burgers_res_only_precomputed( ...
                    geom,trialFull,cfg.nu);
                ninfo.assemblyTimeSec = ninfo.assemblyTimeSec + toc(tAsm);
                RtrialRed=P.'*RtrialFull;
                trialNorm=norm(RtrialRed(free));

                if isfinite(trialNorm) && trialNorm<bestNorm
                    bestNorm=trialNorm;
                    bestRed=trialRed;
                    bestStep=step;
                    bestMu=mu;
                end

                if isfinite(trialNorm) && ...
                        trialNorm<=(1-1e-4*step)*baseNorm
                    uRed=trialRed;
                    Rred=RtrialRed;
                    accepted=true;
                    ninfo.acceptedSteps(end+1,1)=step; %#ok<AGROW>
                    ninfo.regularization(end+1,1)=mu; %#ok<AGROW>
                    break;
                end
            end

            if accepted
                break;
            end
        end

        if ~accepted && bestNorm<baseNorm*(1-1e-12)
            uRed=bestRed;
            uFull=P*uRed;
            tAsm = tic;
            Rfull=assemble_burgers_res_only_precomputed( ...
                geom,uFull,cfg.nu);
            ninfo.assemblyTimeSec = ninfo.assemblyTimeSec + toc(tAsm);
            Rred=P.'*Rfull;
            accepted=true;
            ninfo.acceptedSteps(end+1,1)=bestStep; %#ok<AGROW>
            ninfo.regularization(end+1,1)=bestMu; %#ok<AGROW>
        end

        if ~accepted
            % Keep the last finite accepted iterate.  Newton nonconvergence
            % is recorded, but the test sample is not discarded.
            ninfo.iter=it-1;
            ninfo.finalResidual=resNorm;
            ninfo.converged=false;
            ninfo.usedLastIterate=true;
            ninfo.exitReason='stalled_return_last_iterate';
            uFull=P*uRed;

            warning(['Periodic Newton stalled at iteration %d: ', ...
                     'RMS residual %.3e > target %.3e. ', ...
                     'Keeping the last finite accepted iterate and ', ...
                     'continuing this test case.'], ...
                it-1,resNorm,target);
            return;
        end

        uFull=P*uRed;
        tAsm = tic;
        [Rfull,Jfull]=assemble_burgers_res_jac_precomputed( ...
            geom,uFull,cfg.nu);
        ninfo.assemblyTimeSec = ninfo.assemblyTimeSec + toc(tAsm);
        Rred=P.'*Rfull;
        Jred=P.'*Jfull*P;
    end

    finalResidual=norm(Rred(free))/sqrt(max(numel(free),1));
    ninfo.iter=cfg.newtonMaxIt;
    ninfo.finalResidual=finalResidual;
    ninfo.converged=finalResidual<=target;
    uFull=P*uRed;

    if ninfo.converged
        ninfo.usedLastIterate=false;
        ninfo.exitReason='converged_at_max_iterations';
    else
        ninfo.usedLastIterate=true;
        ninfo.exitReason='max_iterations_return_last_iterate';
        warning(['Periodic Newton reached maxIt=%d with RMS residual ', ...
                 '%.3e > target %.3e. Keeping the last finite iterate ', ...
                 'and continuing this test case.'], ...
            cfg.newtonMaxIt,finalResidual,target);
    end
end

function geom = precompute_fem_assembly_geometry(node,elem)
%PRECOMPUTE_FEM_ASSEMBLY_GEOMETRY Cache mesh-only quantities once per solve.
% Quadrature, sparse-entry ordering and algebra are unchanged.
    geom.N = size(node,1);
    geom.elem = elem;
    p1 = node(elem(:,1),:);
    p2 = node(elem(:,2),:);
    p3 = node(elem(:,3),:);
    detJ = (p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
           (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2));
    geom.area = 0.5*abs(detJ);
    if any(geom.area<=1e-15)
        error('FEM assembly geometry contains degenerate triangles.');
    end
    NT = size(elem,1);
    geom.dphidx = zeros(NT,3);
    geom.dphidt = zeros(NT,3);
    geom.dphidx(:,1) = (p2(:,2)-p3(:,2))./detJ;
    geom.dphidx(:,2) = (p3(:,2)-p1(:,2))./detJ;
    geom.dphidx(:,3) = (p1(:,2)-p2(:,2))./detJ;
    geom.dphidt(:,1) = (p3(:,1)-p2(:,1))./detJ;
    geom.dphidt(:,2) = (p1(:,1)-p3(:,1))./detJ;
    geom.dphidt(:,3) = (p2(:,1)-p1(:,1))./detJ;
    [geom.lambda,geom.w] = tri_quad_3();
    geom.xq=zeros(NT,3);
    for q=1:3
        phi=geom.lambda(q,:);
        geom.xq(:,q)=phi(1)*p1(:,1)+phi(2)*p2(:,1)+phi(3)*p3(:,1);
    end
    geom.I = zeros(NT,9);
    geom.JJ = zeros(NT,9);
    col = 0;
    for a = 1:3
        for b = 1:3
            col = col+1;
            geom.I(:,col) = elem(:,a);
            geom.JJ(:,col) = elem(:,b);
        end
    end
end

function [R,J] = assemble_burgers_res_jac_precomputed(geom,u,nu)
    elem = geom.elem;
    Ue = u(elem);
    ux = sum(Ue.*geom.dphidx,2);
    ut = sum(Ue.*geom.dphidt,2);
    Uq = Ue*geom.lambda.';
    NT = size(elem,1);
    Re = zeros(NT,3);
    Je = zeros(NT,9);
    for q = 1:3
        phi = geom.lambda(q,:);
        uq = Uq(:,q);
        strong = ut+uq.*ux-geom.fq(:,q);
        weight = geom.area*geom.w(q);
        for a = 1:3
            Re(:,a) = Re(:,a)+weight.*( ...
                strong*phi(a)+nu*ux.*geom.dphidx(:,a));
            for b = 1:3
                col = (a-1)*3+b;
                dstrong = geom.dphidt(:,b)+phi(b)*ux+ ...
                    uq.*geom.dphidx(:,b);
                Je(:,col) = Je(:,col)+weight.*( ...
                    dstrong*phi(a)+ ...
                    nu*geom.dphidx(:,b).*geom.dphidx(:,a));
            end
        end
    end
    R = accumarray(elem(:),Re(:),[geom.N,1],@sum,0);
    J = sparse(geom.I(:),geom.JJ(:),Je(:),geom.N,geom.N);
end

function R = assemble_burgers_res_only_precomputed(geom,u,nu)
    elem = geom.elem;
    Ue = u(elem);
    ux = sum(Ue.*geom.dphidx,2);
    ut = sum(Ue.*geom.dphidt,2);
    Uq = Ue*geom.lambda.';
    NT = size(elem,1);
    Re = zeros(NT,3);
    for q = 1:3
        phi = geom.lambda(q,:);
        uq = Uq(:,q);
        strong = ut+uq.*ux-geom.fq(:,q);
        weight = geom.area*geom.w(q);
        for a = 1:3
            Re(:,a) = Re(:,a)+weight.*( ...
                strong*phi(a)+nu*ux.*geom.dphidx(:,a));
        end
    end
    R = accumarray(elem(:),Re(:),[geom.N,1],@sum,0);
end

function [R,J] = assemble_burgers_res_jac(node,elem,u,nu)
% Vectorized P1 space-time residual and Jacobian assembly.

    N=size(node,1);
    NT=size(elem,1);

    p1=node(elem(:,1),:);
    p2=node(elem(:,2),:);
    p3=node(elem(:,3),:);

    detJ=(p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
         (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2));
    area=0.5*abs(detJ);

    valid=area>1e-15;
    if ~all(valid)
        elem=elem(valid,:);
        p1=p1(valid,:);
        p2=p2(valid,:);
        p3=p3(valid,:);
        detJ=detJ(valid);
        area=area(valid);
        NT=size(elem,1);
    end

    dphidx=zeros(NT,3);
    dphidt=zeros(NT,3);

    dphidx(:,1)=(p2(:,2)-p3(:,2))./detJ;
    dphidx(:,2)=(p3(:,2)-p1(:,2))./detJ;
    dphidx(:,3)=(p1(:,2)-p2(:,2))./detJ;

    dphidt(:,1)=(p3(:,1)-p2(:,1))./detJ;
    dphidt(:,2)=(p1(:,1)-p3(:,1))./detJ;
    dphidt(:,3)=(p2(:,1)-p1(:,1))./detJ;

    Ue=u(elem);
    ux=sum(Ue.*dphidx,2);
    ut=sum(Ue.*dphidt,2);

    [lambda,w]=tri_quad_3();
    Uq=Ue*lambda.';

    Re=zeros(NT,3);
    Je=zeros(NT,9);

    for q=1:3
        phi=lambda(q,:);
        uq=Uq(:,q);
        strong=ut+uq.*ux;
        weight=area*w(q);

        for a=1:3
            Re(:,a)=Re(:,a)+weight.*( ...
                strong*phi(a)+nu*ux.*dphidx(:,a));

            for b=1:3
                col=(a-1)*3+b;
                dstrong=dphidt(:,b)+phi(b)*ux+uq.*dphidx(:,b);
                Je(:,col)=Je(:,col)+weight.*( ...
                    dstrong*phi(a)+nu*dphidx(:,b).*dphidx(:,a));
            end
        end
    end

    R=accumarray(elem(:),Re(:),[N,1],@sum,0);

    I=zeros(NT,9);
    JJ=zeros(NT,9);
    col=0;
    for a=1:3
        for b=1:3
            col=col+1;
            I(:,col)=elem(:,a);
            JJ(:,col)=elem(:,b);
        end
    end
    J=sparse(I(:),JJ(:),Je(:),N,N);
end

function R = assemble_burgers_res_only(node,elem,u,nu,ffun)
% Vectorized residual-only assembly used by Newton line search.

    N=size(node,1);
    p1=node(elem(:,1),:);
    p2=node(elem(:,2),:);
    p3=node(elem(:,3),:);

    detJ=(p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
         (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2));
    area=0.5*abs(detJ);

    valid=area>1e-15;
    if ~all(valid)
        elem=elem(valid,:);
        p1=p1(valid,:);
        p2=p2(valid,:);
        p3=p3(valid,:);
        detJ=detJ(valid);
        area=area(valid);
    end
    NT=size(elem,1);

    dphidx=zeros(NT,3);
    dphidt=zeros(NT,3);
    dphidx(:,1)=(p2(:,2)-p3(:,2))./detJ;
    dphidx(:,2)=(p3(:,2)-p1(:,2))./detJ;
    dphidx(:,3)=(p1(:,2)-p2(:,2))./detJ;
    dphidt(:,1)=(p3(:,1)-p2(:,1))./detJ;
    dphidt(:,2)=(p1(:,1)-p3(:,1))./detJ;
    dphidt(:,3)=(p2(:,1)-p1(:,1))./detJ;

    Ue=u(elem);
    ux=sum(Ue.*dphidx,2);
    ut=sum(Ue.*dphidt,2);

    [lambda,w]=tri_quad_3();
    Uq=Ue*lambda.';
    Re=zeros(NT,3);

    for q=1:3
        phi=lambda(q,:);
        uq=Uq(:,q);
        xq=phi(1)*p1(:,1)+phi(2)*p2(:,1)+phi(3)*p3(:,1);
        strong=ut+uq.*ux-ffun(xq);
        weight=area*w(q);

        for a=1:3
            Re(:,a)=Re(:,a)+weight.*( ...
                strong*phi(a)+nu*ux.*dphidx(:,a));
        end
    end

    R=accumarray(elem(:),Re(:),[N,1],@sum,0);
end


%% ========================================================================
%                       Residual indicator and marking
% ========================================================================


function eta2 = residual_indicator_burgers_st(node,elem,u,u0fun,cfg)
%RESIDUAL_INDICATOR_BURGERS_ST
% Vectorized space-time residual indicator including interior diffusion-flux
% jumps and the periodic x-seam jump.

    NT = size(elem,1);

    p1 = node(elem(:,1),:);
    p2 = node(elem(:,2),:);
    p3 = node(elem(:,3),:);

    detJ = (p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
           (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2));
    area = 0.5*abs(detJ);

    if any(area<=1e-15)
        error('Residual estimator encountered degenerate triangles.');
    end

    dphidx = zeros(NT,3);
    dphidt = zeros(NT,3);

    dphidx(:,1) = (p2(:,2)-p3(:,2))./detJ;
    dphidx(:,2) = (p3(:,2)-p1(:,2))./detJ;
    dphidx(:,3) = (p1(:,2)-p2(:,2))./detJ;

    dphidt(:,1) = (p3(:,1)-p2(:,1))./detJ;
    dphidt(:,2) = (p1(:,1)-p3(:,1))./detJ;
    dphidt(:,3) = (p2(:,1)-p1(:,1))./detJ;

    Ue = u(elem);
    uxElem = sum(Ue.*dphidx,2);
    utElem = sum(Ue.*dphidt,2);

    [lambda,w] = tri_quad_3();
    Uq = Ue*lambda.';
    residualQ=zeros(NT,size(lambda,1));
    for q=1:size(lambda,1)
        xq=lambda(q,1)*p1(:,1)+lambda(q,2)*p2(:,1)+lambda(q,3)*p3(:,1);
        residualQ(:,q)=utElem+Uq(:,q).*uxElem-cfg.forcingFun(xq);
    end

    l12 = sqrt(sum((p1-p2).^2,2));
    l23 = sqrt(sum((p2-p3).^2,2));
    l31 = sqrt(sum((p3-p1).^2,2));
    hK = max([l12,l23,l31],[],2);

    volume = area.*sum(residualQ.^2.*w.',2);
    eta2 = hK.^2.*volume;

    % Interior edge jumps.
    allEdges = [elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])];
    owners = [(1:NT)';(1:NT)';(1:NT)'];
    sortedEdges = sort(allEdges,2);

    [uniqueEdges,~,ic] = unique(sortedEdges,'rows');
    multiplicity = accumarray(ic,1);
    ownerMin = accumarray(ic,owners,[],@min);
    ownerMax = accumarray(ic,owners,[],@max);

    interior = multiplicity==2;
    if any(interior)
        edgeNodes = uniqueEdges(interior,:);
        K1 = ownerMin(interior);
        K2 = ownerMax(interior);

        edgeVec = node(edgeNodes(:,2),:) - node(edgeNodes(:,1),:);
        hE = sqrt(sum(edgeVec.^2,2));
        normalX = edgeVec(:,2)./max(hE,realmin);

        jump = cfg.nu*(uxElem(K1)-uxElem(K2)).*normalX;
        contribution = 0.5*hE.^2.*jump.^2;

        eta2 = eta2 + accumarray( ...
            [K1;K2],[contribution;contribution],[NT,1],@sum,0);
    end

    % Periodic seam jump.
    [~,leftOwner,leftKey] = periodic_side_edges( ...
        node,elem,cfg,'left');
    [~,rightOwner,rightKey] = periodic_side_edges( ...
        node,elem,cfg,'right');

    if size(leftKey,1)~=size(rightKey,1) || ...
            (~isempty(leftKey) && ...
             any(abs(leftKey(:)-rightKey(:))> ...
                 50*periodic_tolerance(cfg)))
        error('Periodic boundary edge partitions do not match in estimator.');
    end

    if ~isempty(leftOwner)
        hE = leftKey(:,2)-leftKey(:,1);
        jump = cfg.nu*(uxElem(leftOwner)-uxElem(rightOwner));
        contribution = 0.5*hE.^2.*jump.^2;

        eta2 = eta2 + accumarray( ...
            [leftOwner;rightOwner], ...
            [contribution;contribution],[NT,1],@sum,0);
    end

    if any(~isfinite(eta2)); error('NOEM:NonfiniteEstimator','Burgers estimator contains NaN/Inf.'); end
    eta2(eta2<0) = 0;

    % Same initial-trace data approximation term used during generation.
    initialEdges=[elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])];
    initialOwners=repmat((1:NT)',3,1);
    tolInitial=50*periodic_tolerance(cfg);
    onInitial=abs(node(initialEdges(:,1),2)-cfg.tmin)<tolInitial & abs(node(initialEdges(:,2),2)-cfg.tmin)<tolInitial;
    initialOwners=initialOwners(onInitial); initialEdges=initialEdges(onInitial,:);
    if ~isempty(initialEdges)
        x1=node(initialEdges(:,1),1); x2=node(initialEdges(:,2),1); hE=abs(x2-x1);
        xi=[-sqrt(3/5),0,sqrt(3/5)]; w0=[5/9,8/9,5/9]; u1=u0fun(x1); u2=u0fun(x2);
        edgeContribution=zeros(size(initialOwners));
        for q=1:3
            lambda=0.5*(xi(q)+1); xq=(1-lambda).*x1+lambda.*x2;
            edgeContribution=edgeContribution+0.5*hE*w0(q).*(u0fun(xq)-((1-lambda).*u1+lambda.*u2)).^2;
        end
        if any(~isfinite(edgeContribution)); error('NOEM:NonfiniteEstimator','Initial-data estimator contains NaN/Inf.'); end
        eta2=eta2+accumarray(initialOwners,edgeContribution,[NT,1],@sum,0);
    end
end



function marked = dorfler_marking(eta2, theta)
    total = sum(eta2);
    marked = false(size(eta2));
    if total <= 0 || ~isfinite(total)
        [~,id] = max(eta2);
        marked(id) = true;
        return;
    end
    [vals, ids] = sort(eta2, 'descend');
    cs = cumsum(vals);
    m = find(cs >= theta*total, 1, 'first');
    marked(ids(1:m)) = true;
end

%% ========================================================================
%             Self-contained conforming newest-vertex bisection
% ========================================================================
function elem = nvb_label_initial_mesh(node,elem)
%NVB_LABEL_INITIAL_MESH
% Order every positively oriented initial triangle as [z0,z1,z2], where
% edge [z1,z2] is its longest/reference edge and z0 is the newest vertex.
%
% For the structured rectangular meshes used here, the diagonal is the
% longest edge for both triangles in every rectangle, giving a compatible
% initial reference-edge assignment.

    elem=double(elem);
    NT=size(elem,1);

    for K=1:NT
        v=elem(K,:);
        p=node(v,:);

        area2=det([p(2,:)-p(1,:);p(3,:)-p(1,:)]);
        if area2<0
            v=v([1,3,2]);
            p=node(v,:);
        end

        % Candidate reference edges, opposite vertices 1,2,3.
        l23=sum((p(2,:)-p(3,:)).^2);
        l31=sum((p(3,:)-p(1,:)).^2);
        l12=sum((p(1,:)-p(2,:)).^2);
        [~,id]=max([l23,l31,l12]);

        if id==1
            vv=[v(1),v(2),v(3)];
        elseif id==2
            vv=[v(2),v(3),v(1)];
        else
            vv=[v(3),v(1),v(2)];
        end

        % Cyclic permutations preserve positive orientation.
        elem(K,:)=vv;
    end
end

function key = edge_key_lex_pair_fast_exact(a,b)
%EDGE_KEY_LEX_PAIR_FAST_EXACT
% Lexicographic undirected-edge key preserving sortrows([lo,hi]) order.

    lo = uint64(min(a,b));
    hi = uint64(max(a,b));
    if any(hi>=2^32)
        error('FAST edge key requires node IDs < 2^32.');
    end
    key = bitshift(lo,32)+hi;
end

function key = edge_key_lex_sorted_fast_exact(E)
    if isempty(E)
        key = zeros(0,1,'uint64');
        return;
    end
    if any(E(:,1)>E(:,2))
        error('edge_key_lex_sorted_fast_exact received an unsorted edge.');
    end
    a = uint64(E(:,1));
    b = uint64(E(:,2));
    if any(b>=2^32)
        error('FAST edge key requires node IDs < 2^32.');
    end
    key = bitshift(a,32)+b;
end

function [node,elem,stats,state] = ...
        nvb_refine_conforming_periodic_fast_exact( ...
            node,elem,markedElem,cfg,state)
%NVB_REFINE_CONFORMING_PERIODIC_FAST_EXACT
% Same periodic-NVB closure as the legacy routine, with packed edge
% membership tests and optional element-generation payload propagation.

    if nargin<5 || isempty(state)
        state = struct();
    end
    if ~isfield(state,'elementGeneration')
        state.elementGeneration = [];
    end
    if ~isfield(state,'elementNewLeaf') || ...
            numel(state.elementNewLeaf)~=size(elem,1)
        state.elementNewLeaf = false(size(elem,1),1);
    end

    doAudit = isfield(cfg,'fastAuditNVB') && cfg.fastAuditNVB;
    if doAudit
        nodeInput = node;
        elemInput = elem;
        markedInput = markedElem;
        cfgLegacy = cfg;
        cfgLegacy.fastAuditNVB = false;
        cfgLegacy.fastAuditPeriodicSideEdges = false;
        [nodeLegacy,elemLegacy,statsLegacy] = ...
            nvb_refine_conforming_periodic( ...
                nodeInput,elemInput,markedInput,cfgLegacy);
    end

    oldElems = size(elem,1);

    [node,elem,localStats,state] = ...
        nvb_refine_conforming_local_fast_exact( ...
            node,elem,markedElem,cfg,state);

    afterLocal = size(elem,1);

    [node,elem,periodicStats,state] = ...
        enforce_periodic_boundary_partition_fast_exact( ...
            node,elem,cfg,state);

    stats.requestedMarkedElements = numel(markedElem);
    stats.totalBisectedParents = localStats.totalBisectedParents + ...
        periodicStats.totalBisectedParents;
    stats.completionSubsteps = localStats.completionSubsteps + ...
        periodicStats.completionSubsteps;
    stats.createdNodes = localStats.createdNodes + ...
        periodicStats.createdNodes;
    stats.localAddedElems = afterLocal-oldElems;
    stats.periodicAddedElems = size(elem,1)-afterLocal;
    stats.totalAddedElems = size(elem,1)-oldElems;
    stats.localStats = localStats;
    stats.periodicStats = periodicStats;

    doCallCheck = isfield(cfg,'fastCheckEveryNVBCall') && ...
        logical(cfg.fastCheckEveryNVBCall);
    if doCallCheck
        check_mesh_valid(node,elem,'FAST conforming periodic NVB mesh');
        assert_periodic_boundary_match_fast_exact( ...
            node,elem,cfg,'FAST periodic NVB');
    end

    if doAudit
        if ~isequal(node,nodeLegacy) || ~isequal(elem,elemLegacy)
            error(['FAST periodic NVB produced different node/element ', ...
                   'arrays from the legacy periodic NVB.']);
        end
        if stats.totalBisectedParents~=statsLegacy.totalBisectedParents || ...
                stats.createdNodes~=statsLegacy.createdNodes
            error('FAST periodic NVB statistics differ from legacy NVB.');
        end
    end
end

function [node,elem,stats,state] = ...
        nvb_refine_conforming_local_fast_exact( ...
            node,elem,markedElem,cfg,state)
%NVB_REFINE_CONFORMING_LOCAL_FAST_EXACT
% Same reference-edge completion and SAME element ordering as legacy code.
% Only edge-set membership is changed from N-by-2 row tables to uint64 keys.

    markedElem = unique(round(markedElem(:)));
    markedElem = markedElem( ...
        markedElem>=1 & markedElem<=size(elem,1));

    trackGeneration = ...
        isfield(state,'elementGeneration') && ...
        ~isempty(state.elementGeneration);

    if trackGeneration && ...
            numel(state.elementGeneration)~=size(elem,1)
        error('FAST tracked generation size mismatch.');
    end

    if ~isfield(state,'elementNewLeaf') || ...
            numel(state.elementNewLeaf)~=size(elem,1)
        state.elementNewLeaf = false(size(elem,1),1);
    end

    stats.requestedMarkedElements = numel(markedElem);
    stats.totalBisectedParents = 0;
    stats.completionSubsteps = 0;
    stats.createdNodes = 0;

    if isempty(markedElem)
        return;
    end

    pending = unique(sort(elem(markedElem,[2,3]),2),'rows');

    midpointEdges = zeros(0,2);
    midpointKeys = zeros(0,1,'uint64');
    midpointIds = zeros(0,1);

    substep = 0;

    while ~isempty(pending)
        substep = substep+1;
        if substep>cfg.nvbMaxCompletionSteps
            error(['FAST local NVB completion exceeded %d substeps. ', ...
                   'Check initial reference-edge compatibility.'], ...
                cfg.nvbMaxCompletionSteps);
        end

        % Loop invariants for this substep: the mesh is not modified until
        % the split below, so reference edges and packed keys are computed
        % once and reused by both the dependency scan and the split step.
        NT = size(elem,1);

        % Reference edges are still kept explicitly because dependencies
        % must be appended as exact node pairs.
        refEdges = sort(elem(:,[2,3]),2);
        refKeys = edge_key_lex_sorted_fast_exact(refEdges);

        allKeys = [ ...
            edge_key_lex_pair_fast_exact(elem(:,1),elem(:,2)); ...
            refKeys; ...
            edge_key_lex_pair_fast_exact(elem(:,3),elem(:,1))];

        changed = true;
        while changed
            pendingKeys = edge_key_lex_sorted_fast_exact(pending);
            hit = ismember(allKeys,pendingKeys);
            hitPos = find(hit);
            incident = unique(mod(hitPos-1,NT)+1);

            if isempty(incident)
                changed = false;
                break;
            end

            requiredRef = unique(refEdges(incident,:),'rows');
            newPending = unique([pending;requiredRef],'rows');
            changed = size(newPending,1)>size(pending,1);
            pending = newPending;
        end

        pendingKeys = edge_key_lex_sorted_fast_exact(pending);

        splitMask = ismember(refKeys,pendingKeys);
        splitIds = find(splitMask);

        if isempty(splitIds)
            error(['FAST NVB completion stalled: pending edges remain ', ...
                   'but no reference edge can be split.']);
        end

        splitEdges = unique(refEdges(splitIds,:),'rows');
        splitKeys = edge_key_lex_sorted_fast_exact(splitEdges);

        [known,locKnown] = ismember(splitKeys,midpointKeys);
        splitMidIds = zeros(size(splitEdges,1),1);

        if any(known)
            splitMidIds(known) = midpointIds(locKnown(known));
        end

        newEdgeMask = ~known;
        if any(newEdgeMask)
            newEdges = splitEdges(newEdgeMask,:);
            newPts = 0.5*(node(newEdges(:,1),:)+node(newEdges(:,2),:));

            firstId = size(node,1)+1;
            newIds = (firstId:firstId+size(newPts,1)-1).';
            node = [node;newPts]; %#ok<AGROW>

            midpointEdges = [midpointEdges;newEdges]; %#ok<AGROW>
            midpointKeys = [midpointKeys; ...
                splitKeys(newEdgeMask)]; %#ok<AGROW>
            midpointIds = [midpointIds;newIds]; %#ok<AGROW>
            splitMidIds(newEdgeMask) = newIds;
            stats.createdNodes = stats.createdNodes+size(newPts,1);
        end

        [tf,loc] = ismember(refKeys(splitIds),splitKeys);
        if ~all(tf)
            error('FAST internal NVB midpoint lookup failed.');
        end
        mids = splitMidIds(loc);

        unsplitIds = find(~splitMask);
        newElem = zeros(numel(unsplitIds)+2*numel(splitIds),3);
        ptr = 0;

        if ~isempty(unsplitIds)
            nr = numel(unsplitIds);
            newElem(1:nr,:) = elem(unsplitIds,:);
            ptr = nr;
        end

        z0 = elem(splitIds,1);
        z1 = elem(splitIds,2);
        z2 = elem(splitIds,3);

        child1 = [mids,z0,z1];
        child2 = [mids,z2,z0];

        nr = numel(splitIds);
        newElem(ptr+(1:nr),:) = child1;
        newElem(ptr+nr+(1:nr),:) = child2;

        if trackGeneration
            parentGeneration = uint8(state.elementGeneration(splitIds));
            childGeneration = parentGeneration+uint8(1);
            state.elementGeneration = [ ...
                uint8(state.elementGeneration(unsplitIds)); ...
                childGeneration; ...
                childGeneration];
        end

        state.elementNewLeaf = [ ...
            state.elementNewLeaf(unsplitIds); ...
            true(nr,1); ...
            true(nr,1)];

        elem = newElem;

        stats.totalBisectedParents = ...
            stats.totalBisectedParents+numel(splitIds);
        stats.completionSubsteps = substep;

        currentKeys = unique([ ...
            edge_key_lex_pair_fast_exact(elem(:,1),elem(:,2)); ...
            edge_key_lex_pair_fast_exact(elem(:,2),elem(:,3)); ...
            edge_key_lex_pair_fast_exact(elem(:,3),elem(:,1))]);

        pendingKeys = edge_key_lex_sorted_fast_exact(pending);
        stillPresent = ismember(pendingKeys,currentKeys);
        pending = pending(stillPresent,:);
    end

    if cfg.fastCheckNVBSubcalls
        check_mesh_valid(node,elem,'FAST local conforming NVB mesh');
    end
end

function [node,elem,stats,state] = ...
        enforce_periodic_boundary_partition_fast_exact( ...
            node,elem,cfg,state)
%ENFORCE_PERIODIC_BOUNDARY_PARTITION_FAST_EXACT
% Same dyadic boundary synchronization. All missing targets on one side are
% located from one side-edge scan instead of rescanning the whole mesh once
% per missing time coordinate.

    stats.iterations = 0;
    stats.requestedBoundaryEdges = 0;
    stats.totalBisectedParents = 0;
    stats.completionSubsteps = 0;
    stats.createdNodes = 0;

    for it = 1:cfg.periodicCompletionMax
        stats.iterations = it;

        [leftT,rightT] = periodic_boundary_time_sets(node,cfg);
        tol = periodic_tolerance(cfg);

        if numel(leftT)==numel(rightT) && ...
                (isempty(leftT) || max(abs(leftT-rightT))<=20*tol)
            assert_periodic_boundary_match_fast_exact( ...
                node,elem,cfg,'FAST periodic completion');
            return;
        end

        missingLeft = setdiff_tol(rightT,leftT,20*tol);
        missingRight = setdiff_tol(leftT,rightT,20*tol);

        requested = zeros(0,2);

        if ~isempty(missingLeft)
            if cfg.fastBatchPeriodicRequests
                requested = [requested; ...
                    find_boundary_edges_to_create_times_fast_exact( ...
                        node,elem,cfg,'left',missingLeft)]; %#ok<AGROW>
            else
                for q = 1:numel(missingLeft)
                    requested = [requested; ...
                        find_boundary_edge_to_create_time( ...
                            node,elem,cfg,'left',missingLeft(q))]; %#ok<AGROW>
                end
            end
        end

        if ~isempty(missingRight)
            if cfg.fastBatchPeriodicRequests
                requested = [requested; ...
                    find_boundary_edges_to_create_times_fast_exact( ...
                        node,elem,cfg,'right',missingRight)]; %#ok<AGROW>
            else
                for q = 1:numel(missingRight)
                    requested = [requested; ...
                        find_boundary_edge_to_create_time( ...
                            node,elem,cfg,'right',missingRight(q))]; %#ok<AGROW>
                end
            end
        end

        requested = unique(sort(requested,2),'rows');
        if isempty(requested)
            error('FAST periodic boundary completion stalled.');
        end

        stats.requestedBoundaryEdges = ...
            stats.requestedBoundaryEdges+size(requested,1);

        [node,elem,edgeStats,state] = ...
            nvb_refine_conforming_edges_local_fast_exact( ...
                node,elem,requested,cfg,state);

        stats.totalBisectedParents = stats.totalBisectedParents + ...
            edgeStats.totalBisectedParents;
        stats.completionSubsteps = stats.completionSubsteps + ...
            edgeStats.completionSubsteps;
        stats.createdNodes = stats.createdNodes+edgeStats.createdNodes;
    end

    error('FAST periodic boundary completion exceeded %d iterations.', ...
        cfg.periodicCompletionMax);
end

function [node,elem,stats,state] = ...
        nvb_refine_conforming_edges_local_fast_exact( ...
            node,elem,pending,cfg,state)
%NVB_REFINE_CONFORMING_EDGES_LOCAL_FAST_EXACT
% Explicit-edge version of the same packed-key NVB kernel.

    pending = unique(sort(round(pending),2),'rows');

    trackGeneration = ...
        isfield(state,'elementGeneration') && ...
        ~isempty(state.elementGeneration);

    if trackGeneration && ...
            numel(state.elementGeneration)~=size(elem,1)
        error('FAST tracked generation size mismatch in edge completion.');
    end

    if ~isfield(state,'elementNewLeaf') || ...
            numel(state.elementNewLeaf)~=size(elem,1)
        state.elementNewLeaf = false(size(elem,1),1);
    end

    stats.totalBisectedParents = 0;
    stats.completionSubsteps = 0;
    stats.createdNodes = 0;

    if isempty(pending)
        return;
    end

    midpointEdges = zeros(0,2);
    midpointKeys = zeros(0,1,'uint64');
    midpointIds = zeros(0,1);
    substep = 0;

    while ~isempty(pending)
        substep = substep+1;

        if substep>cfg.nvbMaxCompletionSteps
            error('FAST NVB edge completion exceeded %d substeps.', ...
                cfg.nvbMaxCompletionSteps);
        end

        changed = true;
        while changed
            NT = size(elem,1);
            e23 = sort(elem(:,[2,3]),2);

            allKeys = [ ...
                edge_key_lex_pair_fast_exact(elem(:,1),elem(:,2)); ...
                edge_key_lex_sorted_fast_exact(e23); ...
                edge_key_lex_pair_fast_exact(elem(:,3),elem(:,1))];

            pendingKeys = edge_key_lex_sorted_fast_exact(pending);
            hit = ismember(allKeys,pendingKeys);
            hitPos = find(hit);
            incident = unique(mod(hitPos-1,NT)+1);

            if isempty(incident)
                changed = false;
                break;
            end

            requiredRef = unique(e23(incident,:),'rows');
            newPending = unique([pending;requiredRef],'rows');
            changed = size(newPending,1)>size(pending,1);
            pending = newPending;
        end

        refEdges = sort(elem(:,[2,3]),2);
        refKeys = edge_key_lex_sorted_fast_exact(refEdges);
        pendingKeys = edge_key_lex_sorted_fast_exact(pending);

        splitMask = ismember(refKeys,pendingKeys);
        splitIds = find(splitMask);

        if isempty(splitIds)
            error('FAST NVB edge completion stalled with pending edges.');
        end

        splitEdges = unique(refEdges(splitIds,:),'rows');
        splitKeys = edge_key_lex_sorted_fast_exact(splitEdges);

        [known,locKnown] = ismember(splitKeys,midpointKeys);
        splitMidIds = zeros(size(splitEdges,1),1);

        if any(known)
            splitMidIds(known) = midpointIds(locKnown(known));
        end

        newMask = ~known;
        if any(newMask)
            newEdges = splitEdges(newMask,:);
            newPts = 0.5*(node(newEdges(:,1),:)+ ...
                node(newEdges(:,2),:));

            firstId = size(node,1)+1;
            newIds = (firstId:firstId+size(newPts,1)-1).';
            node = [node;newPts]; %#ok<AGROW>

            midpointEdges = [midpointEdges;newEdges]; %#ok<AGROW>
            midpointKeys = [midpointKeys;splitKeys(newMask)]; %#ok<AGROW>
            midpointIds = [midpointIds;newIds]; %#ok<AGROW>
            splitMidIds(newMask) = newIds;
            stats.createdNodes = stats.createdNodes+size(newPts,1);
        end

        [tf,loc] = ismember(refKeys(splitIds),splitKeys);
        if ~all(tf)
            error('FAST internal NVB midpoint lookup failed.');
        end

        mids = splitMidIds(loc);
        unsplitIds = find(~splitMask);

        newElem = zeros(numel(unsplitIds)+2*numel(splitIds),3);
        ptr = 0;

        if ~isempty(unsplitIds)
            nr = numel(unsplitIds);
            newElem(1:nr,:) = elem(unsplitIds,:);
            ptr = nr;
        end

        z0 = elem(splitIds,1);
        z1 = elem(splitIds,2);
        z2 = elem(splitIds,3);

        child1 = [mids,z0,z1];
        child2 = [mids,z2,z0];

        nr = numel(splitIds);
        newElem(ptr+(1:nr),:) = child1;
        newElem(ptr+nr+(1:nr),:) = child2;

        if trackGeneration
            parentGeneration = uint8(state.elementGeneration(splitIds));
            childGeneration = parentGeneration+uint8(1);
            state.elementGeneration = [ ...
                uint8(state.elementGeneration(unsplitIds)); ...
                childGeneration; ...
                childGeneration];
        end

        state.elementNewLeaf = [ ...
            state.elementNewLeaf(unsplitIds); ...
            true(nr,1); ...
            true(nr,1)];

        elem = newElem;

        stats.totalBisectedParents = ...
            stats.totalBisectedParents+numel(splitIds);
        stats.completionSubsteps = substep;

        currentKeys = unique([ ...
            edge_key_lex_pair_fast_exact(elem(:,1),elem(:,2)); ...
            edge_key_lex_pair_fast_exact(elem(:,2),elem(:,3)); ...
            edge_key_lex_pair_fast_exact(elem(:,3),elem(:,1))]);

        pendingKeys = edge_key_lex_sorted_fast_exact(pending);
        stillPresent = ismember(pendingKeys,currentKeys);
        pending = pending(stillPresent,:);
    end

    if cfg.fastCheckNVBSubcalls
        check_mesh_valid(node,elem,'FAST explicit-edge NVB mesh');
    end
end

function requested = find_boundary_edges_to_create_times_fast_exact( ...
        node,elem,cfg,side,targetT)
%FIND_BOUNDARY_EDGES_TO_CREATE_TIMES_FAST_EXACT
% Same edge choice as repeated find_boundary_edge_to_create_time calls, but
% periodic_side_edges is evaluated once for the whole target vector.

    targetT = targetT(:);
    requested = zeros(numel(targetT),2);

    [edges,~,keys] = periodic_side_edges_fast_exact( ...
        node,elem,cfg,side);

    tol = 50*periodic_tolerance(cfg);

    for q = 1:numel(targetT)
        tq = targetT(q);
        mid = 0.5*(keys(:,1)+keys(:,2));

        candidates = find(abs(mid-tq)<=tol);
        if ~isempty(candidates)
            requested(q,:) = edges(candidates(1),:);
            continue;
        end

        candidates = find(tq>keys(:,1)+tol & tq<keys(:,2)-tol);
        if ~isempty(candidates)
            [~,loc] = max(keys(candidates,2)-keys(candidates,1));
            requested(q,:) = edges(candidates(loc),:);
            continue;
        end

        error('Cannot locate a %s boundary edge containing t=%.16g.', ...
            side,tq);
    end
end

function [edges,owner,key] = periodic_side_edges_fast_exact( ...
        node,elem,cfg,side)
%PERIODIC_SIDE_EDGES_FAST_EXACT
% Return exactly the side-edge partition without materializing/uniquing all
% 3*NT mesh edges. Only edge occurrences whose two endpoints lie on the
% requested physical side are collected.

    tol = periodic_tolerance(cfg);

    if strcmpi(side,'left')
        xSide = cfg.xmin;
    elseif strcmpi(side,'right')
        xSide = cfg.xmax;
    else
        error('side must be left or right.');
    end

    NT = size(elem,1);
    edgePairs = [1 2;2 3;3 1];

    sideEdges = zeros(0,2);
    sideOwner = zeros(0,1);

    for b = 1:3
        a = elem(:,edgePairs(b,1));
        c = elem(:,edgePairs(b,2));

        mask = abs(node(a,1)-xSide)<tol & ...
               abs(node(c,1)-xSide)<tol;

        if any(mask)
            ee = sort([a(mask),c(mask)],2);
            sideEdges = [sideEdges;ee]; %#ok<AGROW>
            sideOwner = [sideOwner;find(mask)]; %#ok<AGROW>
        end
    end

    if isempty(sideEdges)
        edges = zeros(0,2);
        owner = zeros(0,1);
        key = zeros(0,2);
    else
        [uniqueEdges,~,ic] = unique(sideEdges,'rows');
        firstOwner = accumarray(ic,sideOwner,[],@min);

        edges = uniqueEdges;
        owner = firstOwner;

        t1 = node(edges(:,1),2);
        t2 = node(edges(:,2),2);
        key = [min(t1,t2),max(t1,t2)];

        [key,order] = sortrows(key,[1,2]);
        edges = edges(order,:);
        owner = owner(order);
    end

    if isfield(cfg,'fastAuditPeriodicSideEdges') && ...
            cfg.fastAuditPeriodicSideEdges
        [eLegacy,oLegacy,kLegacy] = periodic_side_edges( ...
            node,elem,cfg,side);

        if ~isequal(edges,eLegacy) || ...
                ~isequal(owner,oLegacy) || ...
                ~isequal(key,kLegacy)
            error(['FAST periodic_side_edges differs from the legacy ', ...
                   '%s-side partition.'],side);
        end
    end
end

function assert_periodic_boundary_match_fast_exact(node,elem,cfg,tag)
    [leftT,rightT] = periodic_boundary_time_sets(node,cfg);
    tol = periodic_tolerance(cfg);

    if numel(leftT)~=numel(rightT) || ...
            (~isempty(leftT) && max(abs(leftT-rightT))>20*tol)
        error('%s has unmatched periodic boundary nodes.',tag);
    end

    [~,~,leftKey] = periodic_side_edges_fast_exact( ...
        node,elem,cfg,'left');
    [~,~,rightKey] = periodic_side_edges_fast_exact( ...
        node,elem,cfg,'right');

    if size(leftKey,1)~=size(rightKey,1) || ...
            (~isempty(leftKey) && ...
             any(abs(leftKey(:)-rightKey(:))>50*tol))
        error('%s has unmatched periodic boundary edges.',tag);
    end
end

function [node,elem,stats] = nvb_refine_conforming_local( ...
        node,elem,markedElem,cfg)
%NVB_REFINE_CONFORMING_LOCAL
% Conforming NVB with iterative completion.
%
% Reference edge of triangle [z0,z1,z2] is [z1,z2].  Marked triangles
% request splitting of their reference edges.  If a pending edge is not the
% reference edge of an incident neighbour, that neighbour's reference edge
% is added first.  This process is repeated until all pending edges have
% disappeared from the mesh.
%
% Shared midpoint IDs are reused across completion substeps.

    markedElem=unique(round(markedElem(:)));
    markedElem=markedElem( ...
        markedElem>=1 & markedElem<=size(elem,1));

    stats.requestedMarkedElements=numel(markedElem);
    stats.totalBisectedParents=0;
    stats.completionSubsteps=0;
    stats.createdNodes=0;

    if isempty(markedElem)
        return;
    end

    pending=unique(sort(elem(markedElem,[2,3]),2),'rows');

    % Midpoint registry persists across substeps so both sides of a formerly
    % nonmatching edge use exactly the same midpoint node.
    midpointEdges=zeros(0,2);
    midpointIds=zeros(0,1);

    substep=0;
    while ~isempty(pending)
        substep=substep+1;
        if substep>cfg.nvbMaxCompletionSteps
            error(['Local NVB completion exceeded %d substeps. ', ...
                   'Check the initial reference-edge compatibility.'], ...
                cfg.nvbMaxCompletionSteps);
        end

        % -------------------------------------------------------------
        % Completion closure on the CURRENT mesh.
        % If any pending edge touches a triangle, its reference edge must
        % also become pending before that non-reference edge can be split.
        % -------------------------------------------------------------
        changed=true;
        while changed
            NT=size(elem,1);
            e12=sort(elem(:,[1,2]),2);
            e23=sort(elem(:,[2,3]),2); % reference edges
            e31=sort(elem(:,[3,1]),2);

            allEdges=[e12;e23;e31];
            owner=[(1:NT)';(1:NT)';(1:NT)'];

            hit=ismember(allEdges,pending,'rows');
            incident=unique(owner(hit));

            if isempty(incident)
                changed=false;
                break;
            end

            requiredRef=unique(e23(incident,:),'rows');
            newPending=unique([pending;requiredRef],'rows');
            changed=size(newPending,1)>size(pending,1);
            pending=newPending;
        end

        % Every triangle whose reference edge is pending can now be bisected.
        refEdges=sort(elem(:,[2,3]),2);
        splitMask=ismember(refEdges,pending,'rows');
        splitIds=find(splitMask);

        if isempty(splitIds)
            error('NVB completion stalled: pending edges remain but no reference edge can be split.');
        end

        splitEdges=unique(refEdges(splitIds,:),'rows');

        % Reuse old midpoint IDs; create only genuinely new midpoint nodes.
        [known,locKnown]=ismember(splitEdges,midpointEdges,'rows');
        splitMidIds=zeros(size(splitEdges,1),1);

        if any(known)
            splitMidIds(known)=midpointIds(locKnown(known));
        end

        newEdgeMask=~known;
        if any(newEdgeMask)
            newEdges=splitEdges(newEdgeMask,:);
            newPts=0.5*(node(newEdges(:,1),:)+node(newEdges(:,2),:));
            firstId=size(node,1)+1;
            newIds=(firstId:firstId+size(newPts,1)-1).';
            node=[node;newPts];

            midpointEdges=[midpointEdges;newEdges]; %#ok<AGROW>
            midpointIds=[midpointIds;newIds]; %#ok<AGROW>
            splitMidIds(newEdgeMask)=newIds;
            stats.createdNodes=stats.createdNodes+size(newPts,1);
        end

        % Map each split parent reference edge to its midpoint.
        [tf,loc]=ismember(refEdges(splitIds,:),splitEdges,'rows');
        if ~all(tf)
            error('Internal NVB midpoint lookup failed.');
        end
        mids=splitMidIds(loc);

        unsplitIds=find(~splitMask);
        newElem=zeros(numel(unsplitIds)+2*numel(splitIds),3);
        ptr=0;

        if ~isempty(unsplitIds)
            nr=numel(unsplitIds);
            newElem(1:nr,:)=elem(unsplitIds,:);
            ptr=nr;
        end

        % Standard NVB children for parent [z0,z1,z2]:
        %   child 1 = [m,z0,z1], reference edge [z0,z1]
        %   child 2 = [m,z2,z0], reference edge [z2,z0]
        z0=elem(splitIds,1);
        z1=elem(splitIds,2);
        z2=elem(splitIds,3);

        child1=[mids,z0,z1];
        child2=[mids,z2,z0];

        nr=numel(splitIds);
        newElem(ptr+(1:nr),:)=child1;
        newElem(ptr+nr+(1:nr),:)=child2;

        elem=newElem;
        stats.totalBisectedParents= ...
            stats.totalBisectedParents+numel(splitIds);
        stats.completionSubsteps=substep;

        % A pending edge is finished only when the full unsplit edge no
        % longer exists anywhere in the current mesh.  This is what forces
        % the opposite side of a temporarily hanging edge to be completed.
        NT=size(elem,1);
        currentEdges=unique(sort([ ...
            elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])],2),'rows');
        pending=intersect(pending,currentEdges,'rows','stable');
    end

    check_mesh_valid(node,elem,'self-contained conforming NVB mesh');
end

function [node,elem,stats] = nvb_refine_conforming_periodic( ...
        node,elem,markedElem,cfg)
%NVB_REFINE_CONFORMING_PERIODIC
% Perform local conforming NVB, then synchronize the dyadic partitions of
% the left and right periodic boundaries.

    oldElems=size(elem,1);
    [node,elem,localStats]=nvb_refine_conforming_local( ...
        node,elem,markedElem,cfg);
    afterLocal=size(elem,1);

    [node,elem,periodicStats]=enforce_periodic_boundary_partition( ...
        node,elem,cfg);

    stats.requestedMarkedElements=numel(markedElem);
    stats.totalBisectedParents=localStats.totalBisectedParents+ ...
        periodicStats.totalBisectedParents;
    stats.completionSubsteps=localStats.completionSubsteps+ ...
        periodicStats.completionSubsteps;
    stats.createdNodes=localStats.createdNodes+ ...
        periodicStats.createdNodes;
    stats.localAddedElems=afterLocal-oldElems;
    stats.periodicAddedElems=size(elem,1)-afterLocal;
    stats.totalAddedElems=size(elem,1)-oldElems;
    stats.localStats=localStats;
    stats.periodicStats=periodicStats;
end

function [node,elem,stats] = enforce_periodic_boundary_partition( ...
        node,elem,cfg)
%ENFORCE_PERIODIC_BOUNDARY_PARTITION
% If a dyadic time coordinate occurs on only one vertical boundary, refine
% the containing edge on the opposite boundary until both partitions match.

    stats.iterations=0;
    stats.requestedBoundaryEdges=0;
    stats.totalBisectedParents=0;
    stats.completionSubsteps=0;
    stats.createdNodes=0;

    for it=1:cfg.periodicCompletionMax
        stats.iterations=it;
        [leftT,rightT]=periodic_boundary_time_sets(node,cfg);
        tol=periodic_tolerance(cfg);

        if numel(leftT)==numel(rightT) && ...
                (isempty(leftT) || max(abs(leftT-rightT))<=20*tol)
            assert_periodic_boundary_match( ...
                node,elem,cfg,'periodic completion');
            return;
        end

        missingLeft=setdiff_tol(rightT,leftT,20*tol);
        missingRight=setdiff_tol(leftT,rightT,20*tol);
        requested=zeros(0,2);

        for q=1:numel(missingLeft)
            requested=[requested; ...
                find_boundary_edge_to_create_time( ...
                    node,elem,cfg,'left',missingLeft(q))]; %#ok<AGROW>
        end
        for q=1:numel(missingRight)
            requested=[requested; ...
                find_boundary_edge_to_create_time( ...
                    node,elem,cfg,'right',missingRight(q))]; %#ok<AGROW>
        end

        requested=unique(sort(requested,2),'rows');
        if isempty(requested)
            error('Periodic boundary completion stalled.');
        end

        stats.requestedBoundaryEdges= ...
            stats.requestedBoundaryEdges+size(requested,1);

        [node,elem,edgeStats]=nvb_refine_conforming_edges_local( ...
            node,elem,requested,cfg);
        stats.totalBisectedParents=stats.totalBisectedParents+ ...
            edgeStats.totalBisectedParents;
        stats.completionSubsteps=stats.completionSubsteps+ ...
            edgeStats.completionSubsteps;
        stats.createdNodes=stats.createdNodes+edgeStats.createdNodes;
    end

    error('Periodic boundary completion exceeded %d iterations.', ...
        cfg.periodicCompletionMax);
end

function [node,elem,stats] = nvb_refine_conforming_edges_local( ...
        node,elem,pending,cfg)
%NVB_REFINE_CONFORMING_EDGES_LOCAL
% Same NVB kernel as nvb_refine_conforming_local, but starts from explicitly
% requested full mesh edges.

    pending=unique(sort(round(pending),2),'rows');
    stats.totalBisectedParents=0;
    stats.completionSubsteps=0;
    stats.createdNodes=0;

    if isempty(pending)
        return;
    end

    midpointEdges=zeros(0,2);
    midpointIds=zeros(0,1);
    substep=0;

    while ~isempty(pending)
        substep=substep+1;
        if substep>cfg.nvbMaxCompletionSteps
            error('NVB completion exceeded %d substeps.', ...
                cfg.nvbMaxCompletionSteps);
        end

        changed=true;
        while changed
            NT=size(elem,1);
            e12=sort(elem(:,[1,2]),2);
            e23=sort(elem(:,[2,3]),2);
            e31=sort(elem(:,[3,1]),2);
            allEdges=[e12;e23;e31];
            owner=[(1:NT)';(1:NT)';(1:NT)'];

            hit=ismember(allEdges,pending,'rows');
            incident=unique(owner(hit));
            if isempty(incident)
                changed=false;
                break;
            end

            requiredRef=unique(e23(incident,:),'rows');
            newPending=unique([pending;requiredRef],'rows');
            changed=size(newPending,1)>size(pending,1);
            pending=newPending;
        end

        refEdges=sort(elem(:,[2,3]),2);
        splitMask=ismember(refEdges,pending,'rows');
        splitIds=find(splitMask);
        if isempty(splitIds)
            error('NVB completion stalled with pending edges.');
        end

        splitEdges=unique(refEdges(splitIds,:),'rows');
        [known,locKnown]=ismember( ...
            splitEdges,midpointEdges,'rows');
        splitMidIds=zeros(size(splitEdges,1),1);

        if any(known)
            splitMidIds(known)=midpointIds(locKnown(known));
        end

        newMask=~known;
        if any(newMask)
            newEdges=splitEdges(newMask,:);
            newPts=0.5*(node(newEdges(:,1),:)+ ...
                        node(newEdges(:,2),:));
            firstId=size(node,1)+1;
            newIds=(firstId:firstId+size(newPts,1)-1).';
            node=[node;newPts]; %#ok<AGROW>
            midpointEdges=[midpointEdges;newEdges]; %#ok<AGROW>
            midpointIds=[midpointIds;newIds]; %#ok<AGROW>
            splitMidIds(newMask)=newIds;
            stats.createdNodes=stats.createdNodes+size(newPts,1);
        end

        [tf,loc]=ismember( ...
            refEdges(splitIds,:),splitEdges,'rows');
        if ~all(tf)
            error('Internal NVB midpoint lookup failed.');
        end
        mids=splitMidIds(loc);

        unsplitIds=find(~splitMask);
        newElem=zeros(numel(unsplitIds)+2*numel(splitIds),3);
        ptr=0;

        if ~isempty(unsplitIds)
            nr=numel(unsplitIds);
            newElem(1:nr,:)=elem(unsplitIds,:);
            ptr=nr;
        end

        z0=elem(splitIds,1);
        z1=elem(splitIds,2);
        z2=elem(splitIds,3);
        child1=[mids,z0,z1];
        child2=[mids,z2,z0];

        nr=numel(splitIds);
        newElem(ptr+(1:nr),:)=child1;
        newElem(ptr+nr+(1:nr),:)=child2;
        elem=newElem;

        stats.totalBisectedParents= ...
            stats.totalBisectedParents+numel(splitIds);
        stats.completionSubsteps=substep;

        currentEdges=unique(sort([ ...
            elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])],2), ...
            'rows');
        pending=intersect(pending,currentEdges,'rows','stable');
    end

    check_mesh_valid(node,elem,'conforming NVB mesh');
end

function values = setdiff_tol(a,b,tol)
    values=zeros(0,1);
    for i=1:numel(a)
        if isempty(b) || min(abs(b-a(i)))>tol
            values(end+1,1)=a(i); %#ok<AGROW>
        end
    end
end

function edge = find_boundary_edge_to_create_time( ...
        node,elem,cfg,side,targetT)

    [edges,~,keys]=periodic_side_edges( ...
        node,elem,cfg,side);
    tol=50*periodic_tolerance(cfg);
    edge=zeros(0,2);

    mid=0.5*(keys(:,1)+keys(:,2));
    candidates=find(abs(mid-targetT)<=tol);
    if ~isempty(candidates)
        edge=edges(candidates(1),:);
        return;
    end

    candidates=find(targetT>keys(:,1)+tol & ...
                    targetT<keys(:,2)-tol);
    if ~isempty(candidates)
        [~,loc]=max(keys(candidates,2)-keys(candidates,1));
        edge=edges(candidates(loc),:);
        return;
    end

    error('Cannot locate a %s boundary edge containing t=%.16g.', ...
        side,targetT);
end

function [edges,owner,key] = periodic_side_edges( ...
        node,elem,cfg,side)

    tol=periodic_tolerance(cfg);
    if strcmpi(side,'left')
        xSide=cfg.xmin;
    elseif strcmpi(side,'right')
        xSide=cfg.xmax;
    else
        error('side must be left or right.');
    end

    NT=size(elem,1);
    allEdges=[elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])];
    allOwner=[(1:NT)';(1:NT)';(1:NT)'];
    sortedEdges=sort(allEdges,2);
    [uniqueEdges,~,ic]=unique(sortedEdges,'rows');
    multiplicity=accumarray(ic,1);
    firstOwner=accumarray(ic,allOwner,[],@min);

    boundary=multiplicity==1;
    uniqueEdges=uniqueEdges(boundary,:);
    firstOwner=firstOwner(boundary);

    isSide=abs(node(uniqueEdges(:,1),1)-xSide)<tol & ...
           abs(node(uniqueEdges(:,2),1)-xSide)<tol;
    edges=uniqueEdges(isSide,:);
    owner=firstOwner(isSide);

    t1=node(edges(:,1),2);
    t2=node(edges(:,2),2);
    key=[min(t1,t2),max(t1,t2)];
    [key,order]=sortrows(key,[1,2]);
    edges=edges(order,:);
    owner=owner(order);
end

function [leftT,rightT] = periodic_boundary_time_sets(node,cfg)
    tol=periodic_tolerance(cfg);
    leftT=sort(node(abs(node(:,1)-cfg.xmin)<tol,2));
    rightT=sort(node(abs(node(:,1)-cfg.xmax)<tol,2));
end

function assert_periodic_boundary_match(node,elem,cfg,tag)
    [leftT,rightT]=periodic_boundary_time_sets(node,cfg);
    tol=periodic_tolerance(cfg);

    if numel(leftT)~=numel(rightT) || ...
            (~isempty(leftT) && max(abs(leftT-rightT))>20*tol)
        error('%s has unmatched periodic boundary nodes.',tag);
    end

    [~,~,leftKey]=periodic_side_edges(node,elem,cfg,'left');
    [~,~,rightKey]=periodic_side_edges(node,elem,cfg,'right');
    if size(leftKey,1)~=size(rightKey,1) || ...
            (~isempty(leftKey) && ...
             any(abs(leftKey(:)-rightKey(:))>50*tol))
        error('%s has unmatched periodic boundary edges.',tag);
    end
end

function generation = element_generation_from_area_local(node,elem,A0)
%ELEMENT_GENERATION_FROM_AREA_LOCAL
% Every NVB bisection halves the parent area.

    p1=node(elem(:,1),:);
    p2=node(elem(:,2),:);
    p3=node(elem(:,3),:);
    area=0.5*abs( ...
        (p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
        (p2(:,2)-p1(:,2)).*(p3(:,1)-p1(:,1)));

    generation=max(0,round(log2(A0./max(area,realmin))));
end

function warn_bad_mesh_quality_local(Q,cfg,tag)
    if Q.minAngleDeg<cfg.meshQualityWarningAngle
        warning(['Mesh quality warning at %s: min angle %.3f deg is ', ...
                 'below %.3f deg.'], ...
            tag,Q.minAngleDeg,cfg.meshQualityWarningAngle);
    end
end

%% ========================================================================
%                           Label projection
% ========================================================================
function L32 = label32_from_explicit_levels(node, elem, elemLevel, nMacro, cfg)
%LABEL32_FROM_EXPLICIT_LEVELS Project exact element refinement levels to
%the 32x32 macro grid. Each macro cell receives the maximum label among
%final triangles whose centroids lie in that macro cell.

    centers = (node(elem(:,1),:) + node(elem(:,2),:) + node(elem(:,3),:))/3;
    ii = floor((centers(:,1)-cfg.xmin)/(cfg.xmax-cfg.xmin)*nMacro) + 1;
    jj = floor((centers(:,2)-cfg.tmin)/(cfg.tmax-cfg.tmin)*nMacro) + 1;
    ii = min(max(ii,1),nMacro);
    jj = min(max(jj,1),nMacro);

    classLabel = min(double(elemLevel(:))+1, cfg.maxLabel);
    linearId = sub2ind([nMacro,nMacro], ii, jj);
    maxPerCell = accumarray(linearId, classLabel, [nMacro*nMacro,1], @max, 1);
    L32 = uint8(reshape(maxPerCell, [nMacro,nMacro]));
end

function Lcoarse = aggregate_label_map_max(Lfine, labelGrid)
    [nf1, nf2] = size(Lfine);
    if nf1 ~= nf2
        error('Fine label map must be square.');
    end
    if mod(nf1, labelGrid) ~= 0
        error('Fine label size %d is not divisible by labelGrid %d.', nf1, labelGrid);
    end
    block = nf1 / labelGrid;
    Lcoarse = ones(labelGrid, labelGrid, 'uint8');
    for i = 1:labelGrid
        for j = 1:labelGrid
            sub = Lfine((i-1)*block+1:i*block, (j-1)*block+1:j*block);
            Lcoarse(i,j) = uint8(max(sub(:)));
        end
    end
end


function Lout = remove_singleton_high_label_components(L, maxLabel)
%REMOVE_SINGLETON_HIGH_LABEL_COMPONENTS
% Remove only one-cell islands at each label threshold. A singleton peak is
% lowered one level at a time, while every component containing at least two
% coarse cells is retained. No Image Processing Toolbox is required.

    Lout = uint8(min(max(round(double(L)),1),maxLabel));

    for lev = maxLabel:-1:2
        mask = Lout >= lev;
        comps = connected_components_4(mask);
        for k = 1:numel(comps)
            pix = comps{k};
            if numel(pix) == 1
                Lout(pix) = uint8(max(double(Lout(pix))-1,1));
            end
        end
    end

    Lout = uint8(min(max(round(double(Lout)),1),maxLabel));
end

function comps = connected_components_4(mask)
% 4-neighbor connected components for a small logical matrix.
    mask = logical(mask);
    [nx, ny] = size(mask);
    visited = false(nx, ny);
    comps = {};

    for i = 1:nx
        for j = 1:ny
            if ~mask(i,j) || visited(i,j)
                continue;
            end

            queue = zeros(nx*ny, 2);
            head = 1;
            tail = 1;
            queue(tail,:) = [i,j];
            visited(i,j) = true;
            pix = zeros(nx*ny, 1);
            pcnt = 0;

            while head <= tail
                ci = queue(head,1);
                cj = queue(head,2);
                head = head + 1;
                pcnt = pcnt + 1;
                pix(pcnt) = sub2ind([nx,ny], ci, cj);

                neigh = [ci-1,cj; ci+1,cj; ci,cj-1; ci,cj+1];
                for q = 1:4
                    ni = neigh(q,1);
                    nj = neigh(q,2);
                    if ni < 1 || ni > nx || nj < 1 || nj > ny
                        continue;
                    end
                    if mask(ni,nj) && ~visited(ni,nj)
                        tail = tail + 1;
                        queue(tail,:) = [ni,nj];
                        visited(ni,nj) = true;
                    end
                end
            end

            comps{end+1} = pix(1:pcnt); %#ok<AGROW>
        end
    end
end

function Lb = balance_label_map(L, maxLabel)
    Lb = double(L);
    changed = true;
    it = 0;
    while changed && it < 20
        changed = false;
        it = it + 1;
        [nx, ny] = size(Lb);
        for i = 1:nx
            for j = 1:ny
                neigh = [];
                if i > 1,  neigh(end+1) = Lb(i-1,j); end %#ok<AGROW>
                if i < nx, neigh(end+1) = Lb(i+1,j); end %#ok<AGROW>
                if j > 1,  neigh(end+1) = Lb(i,j-1); end %#ok<AGROW>
                if j < ny, neigh(end+1) = Lb(i,j+1); end %#ok<AGROW>
                if isempty(neigh), continue; end
                mmax = max(neigh);
                if mmax - Lb(i,j) > 1
                    Lb(i,j) = mmax - 1;
                    changed = true;
                end
            end
        end
    end
    Lb = uint8(min(max(round(Lb),1),maxLabel));
end

%% ========================================================================
%                           Geometry utilities
% ========================================================================
function [area, dphidx, dphidt] = tri_geom_xt(xe, te)
    areaSigned = 0.5 * det([xe(2)-xe(1), te(2)-te(1); xe(3)-xe(1), te(3)-te(1)]);
    area = abs(areaSigned);
    if area <= 1e-15
        dphidx = zeros(3,1);
        dphidt = zeros(3,1);
        return;
    end
    dphidx = [te(2)-te(3); te(3)-te(1); te(1)-te(2)] / (2*areaSigned);
    dphidt = [xe(3)-xe(2); xe(1)-xe(3); xe(2)-xe(1)] / (2*areaSigned);
end

function area = triangle_area(x, t)
    area = 0.5 * abs(det([x(2)-x(1), t(2)-t(1); x(3)-x(1), t(3)-t(1)]));
end

function h = max_triangle_edge(node, vid)
    h = max([norm(node(vid(1),:) - node(vid(2),:)), ...
             norm(node(vid(2),:) - node(vid(3),:)), ...
             norm(node(vid(3),:) - node(vid(1),:))]);
end

function [lambda, w] = tri_quad_3()
    lambda = [1/6, 1/6, 2/3;
              1/6, 2/3, 1/6;
              2/3, 1/6, 1/6];
    w = [1/3; 1/3; 1/3];
end

%% ========================================================================
%                         Diagnostics and checks
% ========================================================================
function diag = compute_solution_diagnostics( ...
        node, elem, u, u0fun, cfg, newtonRes, eta2)
%COMPUTE_SOLUTION_DIAGNOSTICS
% Optional newtonRes and eta2 inputs allow the main AFEM loop to reuse
% quantities it has already computed, substantially reducing runtime.

    x = node(:,1);
    t = node(:,2);
    tolBd = 1e-12;
    isBd = abs(x-cfg.xmin)<tolBd | abs(x-cfg.xmax)<tolBd | abs(t-cfg.tmin)<tolBd;
    free = find(~isBd);

    if nargin < 6 || isempty(newtonRes)
        R = assemble_burgers_res_only(node, elem, u, cfg.nu,cfg.forcingFun);
        if isempty(free)
            newtonRes = norm(R) / sqrt(max(numel(R),1));
        else
            newtonRes = norm(R(free)) / sqrt(max(numel(free),1));
        end
    end

    if isempty(free)
        uRms = norm(u) / sqrt(max(numel(u),1));
    else
        uRms = norm(u(free)) / sqrt(max(numel(free),1));
    end

    if nargin < 7 || isempty(eta2)
        eta2 = residual_indicator_burgers_st(node, elem, u, u0fun, cfg);
    end
    eta2(~isfinite(eta2) | eta2 < 0) = 0;

    diag.newtonResidual = newtonRes;
    diag.relResidual = newtonRes / (uRms + 1e-14);
    diag.etaTotal = sqrt(sum(eta2));
    diag.uMax = max(abs(u));
    diag.uL2 = spacetime_l2_norm_fast(node, elem, u);

    xb = abs(x-cfg.xmin)<tolBd | abs(x-cfg.xmax)<tolBd;
    tb = abs(t-cfg.tmin)<tolBd;
    if any(xb)
        diag.sideBoundaryMax = max(abs(u(xb)));
    else
        diag.sideBoundaryMax = NaN;
    end
    if any(tb)
        diag.initialMax = max(abs(u(tb) - u0fun(x(tb))));
    else
        diag.initialMax = NaN;
    end
end

function nrm = spacetime_l2_norm_fast(node, elem, u)
% Exact vectorized P1 mass-matrix formula on every triangle.
    p1 = node(elem(:,1),:);
    p2 = node(elem(:,2),:);
    p3 = node(elem(:,3),:);
    area = 0.5*abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
                   (p2(:,2)-p1(:,2)).*(p3(:,1)-p1(:,1)));

    u1 = u(elem(:,1));
    u2 = u(elem(:,2));
    u3 = u(elem(:,3));
    local = area/6 .* (u1.^2 + u2.^2 + u3.^2 + u1.*u2 + u2.*u3 + u3.*u1);
    nrm = sqrt(max(sum(local),0));
end

function check_mesh_valid(node, elem, tag)
    if nargin < 3, tag = 'mesh'; end
    if isempty(node) || isempty(elem)
        error('%s is empty.', tag);
    end
    if size(node,2) ~= 2 || size(elem,2) ~= 3
        error('%s has wrong array size.', tag);
    end
    if any(~isfinite(node(:))) || any(~isfinite(elem(:)))
        error('%s contains NaN or Inf.', tag);
    end

    elem = round(elem);
    n = size(node,1);
    if any(elem(:) < 1) || any(elem(:) > n)
        error('%s has invalid element indices.', tag);
    end

    % Vectorized degeneracy check; same tolerance as the original code.
    p1 = node(elem(:,1),:);
    p2 = node(elem(:,2),:);
    p3 = node(elem(:,3),:);
    area = 0.5*abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
                   (p2(:,2)-p1(:,2)).*(p3(:,1)-p1(:,1)));
    if any(area <= 1e-14)
        error('%s contains degenerate triangles.', tag);
    end
end

function check_vector_valid(u, n, tag)
    if nargin < 3, tag = 'vector'; end
    if numel(u) ~= n
        error('%s has wrong length: got %d, expected %d.', tag, numel(u), n);
    end
    if any(~isfinite(u(:)))
        error('%s contains NaN or Inf.', tag);
    end
end

function check_label_valid(L, cfg)
    if any(~isfinite(double(L(:))))
        error('Label map contains NaN or Inf.');
    end
    if any(L(:) < 1) || any(L(:) > cfg.maxLabel)
        error('Label map contains values outside 1..%d.', cfg.maxLabel);
    end
    if ~isequal(size(L), [cfg.labelGrid, cfg.labelGrid])
        error('Label map size is wrong. Expected %dx%d.', cfg.labelGrid, cfg.labelGrid);
    end
end

function check_solution_sane(diagNow, ninfo, etaNow, etaPrev, cfg, tag)
    if ~cfg.rejectUnstableSamples
        return;
    end
    if ~isfinite(diagNow.uMax) || ~isfinite(diagNow.uL2) || ~isfinite(diagNow.etaTotal)
        error('Unstable sample at %s: diagnostic contains NaN/Inf.', tag);
    end
    if ninfo.finalResidual > cfg.maxNewtonResidual
        error('Unstable sample at %s: Newton residual %.3e > %.3e.', tag, ninfo.finalResidual, cfg.maxNewtonResidual);
    end
    if diagNow.uMax > cfg.maxAbsUAllowed
        error('Unstable sample at %s: max|u| %.3e > %.3e.', tag, diagNow.uMax, cfg.maxAbsUAllowed);
    end
    if ~isnan(etaPrev) && etaPrev > 0
        etaRatio = etaNow / etaPrev;
        if isfinite(etaRatio) && etaRatio > cfg.maxEtaGrowth
            error('Unstable sample at %s: EtaGrowth %.3e > %.3e.', tag, etaRatio, cfg.maxEtaGrowth);
        end
    end
end

function check_final_solution_sane(finalDiag, ninfoFinal, cfg)
    if ~cfg.rejectUnstableSamples
        return;
    end
    if ~isfinite(finalDiag.uMax) || ~isfinite(finalDiag.uL2) || ~isfinite(finalDiag.etaTotal)
        error('Unstable final sample: diagnostic contains NaN/Inf.');
    end
    if ninfoFinal.finalResidual > cfg.maxNewtonResidual
        error('Unstable final sample: Newton residual %.3e > %.3e.', ninfoFinal.finalResidual, cfg.maxNewtonResidual);
    end
    if finalDiag.relResidual > cfg.maxFinalRelResidual
        error('Unstable final sample: final RelRes %.3e > %.3e.', finalDiag.relResidual, cfg.maxFinalRelResidual);
    end
    if finalDiag.uMax > cfg.maxAbsUAllowed
        error('Unstable final sample: max|u| %.3e > %.3e.', finalDiag.uMax, cfg.maxAbsUAllowed);
    end
end

%% ========================================================================
%              Spectral reference solution and error computation
% ========================================================================
function ref = solve_burgers_fourier_reference(u0fun,cfg)
%SOLVE_BURGERS_FOURIER_REFERENCE
% Periodic Fourier pseudo-spectral discretization in x with ETDRK4 in time.
% The nonlinear term is evaluated in conservative form and filtered by the
% two-thirds de-aliasing rule.

    N=cfg.NxRefSpectral;
    if mod(N,2)~=0
        error('cfg.NxRefSpectral must be even.');
    end

    Lx=cfg.xmax-cfg.xmin;
    x=cfg.xmin+Lx*(0:N-1)'/N;
    u0=u0fun(x);
    forcingHat=fft(cfg.forcingFun(x));

    modes=[0:N/2,-N/2+1:-1].';
    kLap=(2*pi/Lx)*modes;
    kDer=kLap;
    kDer(N/2+1)=0;

    cutoff=floor(cfg.dealiasFraction*(N/2));
    dealiasMask=abs(modes)<=cutoff;

    Lop=-cfg.nu*(kLap.^2);
    dt=cfg.refDt;
    totalStepsReal=(cfg.tmax-cfg.tmin)/dt;
    totalSteps=round(totalStepsReal);
    if abs(totalSteps-totalStepsReal)>1e-10
        error('cfg.refDt must divide the total time interval exactly.');
    end

    nSaveIntervals=cfg.NtRefSpectral-1;
    saveEveryReal=totalSteps/nSaveIntervals;
    saveEvery=round(saveEveryReal);
    if abs(saveEvery-saveEveryReal)>1e-10
        error(['The ETDRK4 step count must be divisible by ', ...
               'cfg.NtRefSpectral-1.']);
    end

    E=exp(dt*Lop);
    E2=exp(dt*Lop/2);

    M=32;
    r=exp(1i*pi*((1:M)-0.5)/M);
    LR=dt*Lop(:,ones(M,1))+r(ones(N,1),:);
    Q=dt*real(mean((exp(LR/2)-1)./LR,2));
    f1=dt*real(mean( ...
        (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3,2));
    f2=dt*real(mean( ...
        (2+LR+exp(LR).*(-2+LR))./LR.^3,2));
    f3=dt*real(mean( ...
        (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3,2));

    v=fft(u0);
    U=zeros(cfg.NtRefSpectral,N+1);
    tSave=linspace(cfg.tmin,cfg.tmax,cfg.NtRefSpectral).';
    U(1,:)=[u0(:).',u0(1)];
    saveId=2;

    for step=1:totalSteps
        Nv=burgers_fourier_nonlinear(v,kDer,dealiasMask)+forcingHat;
        a=E2.*v+Q.*Nv;
        Na=burgers_fourier_nonlinear(a,kDer,dealiasMask)+forcingHat;
        b=E2.*v+Q.*Na;
        Nb=burgers_fourier_nonlinear(b,kDer,dealiasMask)+forcingHat;
        c=E2.*a+Q.*(2*Nb-Nv);
        Nc=burgers_fourier_nonlinear(c,kDer,dealiasMask)+forcingHat;
        v=E.*v+f1.*Nv+2*f2.*(Na+Nb)+f3.*Nc;

        if mod(step,saveEvery)==0
            uNow=real(ifft(v));
            U(saveId,:)=[uNow(:).',uNow(1)];
            saveId=saveId+1;
        end
    end

    if saveId~=cfg.NtRefSpectral+1
        error('Internal spectral snapshot count mismatch.');
    end

    ref.x=[x;cfg.xmax];
    ref.t=tSave;
    ref.U=U;
    ref.N=N;
    ref.dt=dt;
    ref.method= ...
        'Periodic Fourier pseudo-spectral ETDRK4 with 2/3 dealiasing';
    ref.note= ...
        'High-resolution numerical reference, not a closed-form solution';
end

function Nv = burgers_fourier_nonlinear(v,kDer,dealiasMask)
    u=real(ifft(v));
    Nv=-0.5i*kDer.*fft(u.^2);
    Nv(~dealiasMask)=0;
end



function err = compute_fem_vs_reference_error(node,elem,u,ref,cfg,Fref)
%COMPUTE_FEM_VS_REFERENCE_ERROR
% Chunked vectorized three-point triangle quadrature for the global
% space-time error. Reference work is outside all method timings.

    if nargin<6 || isempty(Fref)
        Fref = make_ref_interpolant(ref);
    end
    [lambda,w] = tri_quad_3();

    NT = size(elem,1);
    chunkSize = cfg.errorChunkElements;
    e2 = 0.0;
    r2 = 0.0;

    for first = 1:chunkSize:NT
        last = min(first+chunkSize-1,NT);
        ids = first:last;
        E = elem(ids,:);

        p1 = node(E(:,1),:);
        p2 = node(E(:,2),:);
        p3 = node(E(:,3),:);

        area = 0.5*abs( ...
            (p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
            (p2(:,2)-p1(:,2)).*(p3(:,1)-p1(:,1)));

        X = [p1(:,1),p2(:,1),p3(:,1)];
        T = [p1(:,2),p2(:,2),p3(:,2)];
        Ue = u(E);

        Xq = X*lambda.';
        Tq = T*lambda.';
        Uhq = Ue*lambda.';

        Urq = Fref(Tq(:),Xq(:));
        Urq = reshape(Urq,size(Xq));
        Urq(~isfinite(Urq)) = 0;

        localE = area.*sum((Uhq-Urq).^2.*w.',2);
        localR = area.*sum(Urq.^2.*w.',2);

        e2 = e2 + sum(localE);
        r2 = r2 + sum(localR);
    end

    uRefNode = Fref(node(:,2),node(:,1));
    uRefNode(~isfinite(uRefNode)) = 0;
    nodalErr = u(:)-uRefNode(:);

    xLine = linspace(cfg.xmin,cfg.xmax, ...
        max(1001,cfg.compareGridX)).';
    tLine = (cfg.tmax-1e-12)*ones(size(xLine));
    uFemLine = eval_p1_at_points(node,elem,u,[xLine,tLine]);
    uRefLine = Fref(tLine,xLine);

    bad = ~isfinite(uFemLine) | ~isfinite(uRefLine);
    uFemLine(bad) = 0;
    uRefLine(bad) = 0;

    eFinal2 = trapz(xLine,(uFemLine-uRefLine).^2);
    rFinal2 = trapz(xLine,uRefLine.^2);

    err.spaceTimeL2Abs = sqrt(max(e2,0));
    err.spaceTimeL2Rel = sqrt(max(e2,0)/max(r2,1e-30));
    err.referenceSpaceTimeL2 = sqrt(max(r2,0));
    err.nodalLinf = max(abs(nodalErr));
    err.nodalRMS = sqrt(mean(nodalErr.^2));
    err.finalTimeL2Abs = sqrt(max(eFinal2,0));
    err.finalTimeL2Rel = sqrt(max(eFinal2,0)/max(rFinal2,1e-30));
end



function Fref = make_ref_interpolant(ref)
    try
        Fref = griddedInterpolant({ref.t(:), ref.x(:)}, ref.U, 'spline', 'nearest');
    catch
        Fref = griddedInterpolant({ref.t(:), ref.x(:)}, ref.U, 'linear', 'nearest');
    end
end

function vals = eval_p1_at_points(node, elem, u, pts)
    vals = NaN(size(pts,1),1);
    try
        TR = triangulation(elem, node);
        ti = pointLocation(TR, pts);
        inside = ~isnan(ti);
        if any(inside)
            bc = cartesianToBarycentric(TR, ti(inside), pts(inside,:));
            tri = elem(ti(inside),:);
            vals(inside) = sum(bc .* u(tri), 2);
        end
        if any(~inside)
            F = scatteredInterpolant(node(:,1), node(:,2), u(:), 'linear', 'nearest');
            vals(~inside) = F(pts(~inside,1), pts(~inside,2));
        end
    catch
        F = scatteredInterpolant(node(:,1), node(:,2), u(:), 'linear', 'nearest');
        vals = F(pts(:,1), pts(:,2));
    end
end

%% ========================================================================
%                         Interpolation and output
% ========================================================================
function uNew = interpolate_to_new_nodes(oldNode,oldElem,oldU,newNode)
%INTERPOLATE_TO_NEW_NODES For NVB meshes, old nodes remain the prefix of the
% new node array. Copy those values exactly and locate only appended nodes.
    nOld = size(oldNode,1);
    nNew = size(newNode,1);
    prefixMatch = nNew>=nOld && isequal(newNode(1:nOld,:),oldNode);
    uNew = zeros(nNew,1);
    if prefixMatch
        uNew(1:nOld) = oldU(:);
        queryIds = (nOld+1:nNew).';
        if isempty(queryIds)
            return;
        end
    else
        queryIds = (1:nNew).';
    end
    try
        TR = triangulation(oldElem,oldNode);
        queryPoints = newNode(queryIds,:);
        ti = pointLocation(TR,queryPoints);
        inside = ~isnan(ti);
        if any(inside)
            bc = cartesianToBarycentric(TR,ti(inside),queryPoints(inside,:));
            tri = oldElem(ti(inside),:);
            uNew(queryIds(inside)) = sum(bc.*oldU(tri),2);
        end
        if any(~inside)
            F = scatteredInterpolant(oldNode(:,1),oldNode(:,2), ...
                oldU,'linear','nearest');
            uNew(queryIds(~inside)) = ...
                F(queryPoints(~inside,1),queryPoints(~inside,2));
        end
    catch
        F = scatteredInterpolant(oldNode(:,1),oldNode(:,2), ...
            oldU,'linear','nearest');
        uNew(queryIds) = F(newNode(queryIds,1),newNode(queryIds,2));
    end
end

function save_dataset(fileName, U0_input, x_input, labels, labels32, params, final_nodes, final_elems, final_u, info, refErrors, refCounter, cfg, completedSamples)
    fileName = char(fileName);
    ns = completedSamples;
    if ns < 1
        warning('No completed samples to save yet.');
        return;
    end

    U0_input = U0_input(:, 1:ns);
    labels   = labels(:, :, 1:ns);
    params   = params(1:ns, :);
    info     = info(1:ns);
    if refCounter > 0
        refErrors = refErrors(1:refCounter);
    else
        refErrors = refErrors([]);
    end

    outFolder = fileparts(fileName);
    if ~isempty(outFolder) && ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end

    completedSamples = ns; %#ok<NASGU>

    if isfield(cfg, 'saveLabels32') && cfg.saveLabels32
        labels32 = labels32(:, :, 1:ns); %#ok<NASGU>
    else
        labels32 = []; %#ok<NASGU>
    end

    if isfield(cfg, 'saveFinalMeshes') && cfg.saveFinalMeshes
        final_nodes = final_nodes(1:ns); %#ok<NASGU>
        final_elems = final_elems(1:ns); %#ok<NASGU>
        final_u     = final_u(1:ns); %#ok<NASGU>
        save(fileName, 'U0_input', 'x_input', 'labels', 'labels32', 'params', ...
            'final_nodes', 'final_elems', 'final_u', 'info', 'refErrors', 'cfg', 'completedSamples', '-v7.3');
    else
        save(fileName, 'U0_input', 'x_input', 'labels', 'labels32', 'params', ...
            'info', 'refErrors', 'cfg', 'completedSamples', '-v7.3');
    end
end

function save_preview_figure(node, elem, u, L32, x_input, u0vec, s, cfg)
    fig = figure('Color','w', 'Visible', cfg.figureVisible, 'Position',[100,100,1400,420]);

    subplot(1,3,1);
    plot(x_input, u0vec, 'LineWidth', 1.5);
    grid on; xlabel('x'); ylabel('u_0(x)');
    title(sprintf('Sample %d initial condition', s));

    subplot(1,3,2);
    triplot(elem, node(:,1), node(:,2), 'k-');
    axis tight; box on; xlabel('x'); ylabel('t');
    title(sprintf('Final adaptive mesh, nodes=%d, elems=%d', size(node,1), size(elem,1)));

    subplot(1,3,3);
    imagesc(L32.'); axis image xy; colorbar; caxis([1 cfg.maxLabel]);
    xlabel('label x index'); ylabel('label t index');
    title('Saved 32x32 label map');

    outPng = fullfile(cfg.figureDir, sprintf('preview_sample_%04d.png', s));
    try
        exportgraphics(fig, outPng, 'Resolution', 180);
    catch
        saveas(fig, outPng);
    end
    close(fig);
end
