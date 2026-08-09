function result = reaction_diffusion_accu_fem_comparison(cfg)
%REACTION_DIFFUSION_ACCU_FEM_COMPARISON Five-method accuracy comparison.
% VERSION: REACTION_DIFFUSION_CONFIGURABLE_INIT_AND_VIS_SAVE_20260805_V2
%
% Methods
% -------
% 1. FNO-seeded AFEM:
%      predicted score -> conforming mesh -> solve -> AFEM until tolerance.
% 2. Target-seeded AFEM (diagnostic):
%      target score -> conforming mesh -> solve -> AFEM until tolerance.
% 3. Standard AFEM:
%      common 16^3 initial mesh -> solve -> AFEM until tolerance.
% 4. Accuracy-matched uniform FEM (final solve only):
%      search the configured uniform-grid candidates offline, select the
%      first tested grid reaching the tolerance, then rebuild and solve that
%      final grid once. Only this final mesh+solve is MethodTimeSec. The full
%      search wall time is saved separately as SearchWallTimeSec.
% 5. Fixed-size uniform FEM:
%      every test case uses the same preconfigured uniform grid size.
%
% Timing excludes reference construction, relative-error evaluation,
% plotting, offline operator training and offline target-label generation.
% When cfg.newtonInitializationMode='coarse_interpolation', the common
% coarse-stage solve and target-mesh interpolation are included in FNO,
% target-score and uniform MethodTimeSec values.
% Per-sample cumulative buckets MeshTotalTimeSec / AssemblyTotalTimeSec /
% SolveTotalTimeSec are additionally reported for every method row:
%   mesh      = score-mesh build (or uniform mesh generation) + every AFEM
%               NVB refinement cycle;
%   assembly  = geometry+matrix assembly of the coarse (if included), seed
%               and every AFEM-cycle solve;
%   solve     = PCG/direct linear solves of the coarse (if included), seed
%               and every AFEM-cycle solve.
% Their means per method appear in method_statistics_*.csv; the residual
% MethodTimeSec-(mesh+assembly+solve) is reported as MeanOtherTimeSec
% (estimator/marking, interpolation, score inference, uniform search).
%
% This file reuses the same PDE, boundary reconstruction, P1 assembly,
% complete residual/jump/Dirichlet estimator, Dörfler marking and
% Maubach-NVB rules as the reaction-diffusion accuracy data generator.
%
% Saving policy:
%   - all samples: lightweight statistics, histories and diagnostics;
%   - first three selected samples by default: boundary, score tensors,
%     initial/final adaptive meshes, nodal solutions and compact uniform
%     grid solutions in a separate visualization MAT file.

if nargin < 1 || isempty(cfg)
    error('A configuration struct is required.');
end

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

% -------------------------------------------------------------------------
% USER-SELECTABLE FIRST-SOLVE INITIALIZATION
%
% The same selector controls the independent first linear solve on:
%   1. the FNO score-generated mesh;
%   2. the target-score-generated mesh;
%   3. every accuracy-search uniform mesh and its final selected mesh;
%   4. the fixed-size uniform mesh.
%
% Modes:
%   zero
%       Free-node PCG initial vector is zero.
%
%   initial_condition
%       For this elliptic problem the name is retained only to match the
%       Burgers interface. It means the boundary-data extension
%           u^(0)(x,y,z) = (1-z) g_D(x,y),
%       followed by exact enforcement of all Dirichlet boundary values.
%
%   coarse_interpolation
%       Solve once on the common cfg.initialCells^3 mesh from the above
%       boundary-data extension, then interpolate that coarse FEM solution
%       to the FNO, target-score and uniform meshes as the PCG initial vector.
%       The coarse-stage cost and interpolation cost are included in each
%       corresponding method's MethodTimeSec.
%
% Standard AFEM stage 0 is deliberately left unchanged (zero free-node
% initialization); all later AFEM stages retain the existing P1 warm start.

cfg = fill_accu_fem_defaults(cfg);
cfg.newtonInitializationMode = normalize_linear_initialization_mode( ...
    cfg.newtonInitializationMode);
validate_accu_fem_config(cfg);

clc;
close all;

if ~exist(cfg.outDir,'dir'), mkdir(cfg.outDir); end
if ~exist(cfg.figureDir,'dir'), mkdir(cfg.figureDir); end

if exist(cfg.predictionMat,'file') ~= 2
    error('Prediction MAT not found:\n%s',cfg.predictionMat);
end
if exist(cfg.datasetMat,'file') ~= 2
    error('Accuracy data set not found:\n%s',cfg.datasetMat);
end

%% ------------------------------------------------------------------------
% Load prediction export and exact original random coefficients.
% -------------------------------------------------------------------------
D = load(cfg.predictionMat);
requiredPredictionFields = { ...
    'pred_score','target_score_test', ...
    'query_x','query_y','query_z','original_indices'};
for k = 1:numel(requiredPredictionFields)
    if ~isfield(D,requiredPredictionFields{k})
        error('Prediction MAT is missing field "%s".', ...
            requiredPredictionFields{k});
    end
end

queryX = double(D.query_x(:));
queryY = double(D.query_y(:));
queryZ = double(D.query_z(:));
if isfield(D,'query_s')
    queryS = double(D.query_s(:));
    if numel(queryS)==numel(queryZ) && ...
            max(abs(queryZ-queryS.^2)) > 5e-7
        error('Prediction coordinates are not the expected z=s^2 grid.');
    end
else
    queryS = sqrt(max(queryZ,0));
end

nx = numel(queryX);
ny = numel(queryY);
nz = numel(queryZ);
if ~isequal([nx,ny,nz],[65,65,65])
    error('This comparison expects a fixed 65x65x65 score field.');
end

if cfg.requireExportedThresholdMatch && ...
        isfield(D,'generation_threshold')
    exportedThreshold = double(D.generation_threshold(1));
    if abs(exportedThreshold-cfg.generationThreshold)>1e-12
        error(['Exported generation threshold %.17g differs from ', ...
               'configured threshold %.17g.'], ...
            exportedThreshold,cfg.generationThreshold);
    end
end
if isfield(D,'max_score')
    exportedMaxScore = round(double(D.max_score(1)));
    if exportedMaxScore ~= cfg.exportedScoreMaxLevel
        error('Exported max_score=%d differs from configured maximum=%d.', ...
            exportedMaxScore,cfg.exportedScoreMaxLevel);
    end
end

nTest = infer_score_test_count(D.pred_score,nx,ny,nz);
if cfg.numSamples > nTest
    error('Prediction MAT has %d tests, but cfg.numSamples=%d.', ...
        nTest,cfg.numSamples);
end

originalIndices = round(double(D.original_indices(:)));
if numel(originalIndices) ~= nTest
    error('original_indices length %d differs from nTest=%d.', ...
        numel(originalIndices),nTest);
end

C = load(cfg.datasetMat, ...
    'grf_xi_cos','grf_xi_sin','grf_modes','grf_spectral_std', ...
    'completedSamples');
if ~isfield(C,'grf_xi_cos') || ~isfield(C,'grf_xi_sin')
    error('Accuracy data set does not contain exact GRF coefficients.');
end
if min(originalIndices)<1 || ...
        max(originalIndices)>size(C.grf_xi_cos,2)
    error(['Prediction original_indices=[%d,%d] are incompatible with ', ...
           '%d stored samples.'], ...
        min(originalIndices),max(originalIndices),size(C.grf_xi_cos,2));
end

datasetModes = double(C.grf_modes);
datasetSpectralStd = double(C.grf_spectral_std(:));
lambdaCheck = cfg.grfKappa^2 + 4*pi^2*sum(datasetModes.^2,2);
spectralStdFormula = sqrt(cfg.grfGamma).* ...
    lambdaCheck.^(-cfg.grfBeta/2);
if max(abs(datasetSpectralStd-spectralStdFormula)) > ...
        1e-12*max(1,max(abs(datasetSpectralStd)))
    error('Data-set GRF metadata differs from the configured law.');
end

if cfg.runFNOSeededAFEM
    if ~isfinite(cfg.operatorInferenceTimePerSample)
        cfg.operatorInferenceTimePerSample = ...
            read_fno_inference_time_per_sample(cfg.metricsJson);
    end
    if cfg.operatorInferenceTimePerSample < 0
        error('Invalid operator inference time per sample.');
    end
else
    cfg.operatorInferenceTimePerSample = 0;
end

if isempty(cfg.sampleIds)
    rng(cfg.sampleSelectionSeed,'twister');
    order = randperm(nTest);
    sampleIds = order(1:cfg.numSamples);
else
    sampleIds = unique(round(cfg.sampleIds(:).'),'stable');
    if numel(sampleIds)<cfg.numSamples
        error('cfg.sampleIds contains fewer than cfg.numSamples entries.');
    end
    sampleIds = sampleIds(1:cfg.numSamples);
    if any(sampleIds<1) || any(sampleIds>nTest)
        error('cfg.sampleIds contains an invalid test ID.');
    end
end

fprintf('\n');
fprintf('====================================================================\n');
fprintf('Reaction-diffusion accuracy-to-tolerance FEM comparison\n');
fprintf('====================================================================\n');
fprintf('Prediction file       : %s\n',cfg.predictionMat);
fprintf('Accuracy data set     : %s\n',cfg.datasetMat);
fprintf('Selected test IDs     : %s\n',mat2str(sampleIds));
fprintf('Target relative L2    : %.6e\n',cfg.targetRelativeL2);
fprintf('epsilon               : %.6e\n',cfg.epsilon);
fprintf('initial cubes         : %d^3\n',cfg.initialCells);
fprintf('Dorfler theta         : %.3f\n',cfg.theta);
fprintf('score maximum         : %d\n',cfg.operatorMaxLevel);
fprintf('score threshold       : %.2f\n',cfg.generationThreshold);
fprintf('accuracy-uniform      : %d\n',cfg.runAccuracyMatchedUniform);
fprintf('fixed uniform cells   : %d (free DOF=%d)\n', ...
    resolve_fixed_uniform_cells(cfg), ...
    (resolve_fixed_uniform_cells(cfg)-1)^3);
fprintf('linear init mode      : %s\n',cfg.newtonInitializationMode);
fprintf(['initial_condition     : boundary extension ', ...
         'u^(0)=(1-z)g_D(x,y)\n']);
fprintf('standard AFEM stage 0 : unchanged zero free-node initialization\n');
fprintf('====================================================================\n\n');

%% ------------------------------------------------------------------------
% Fixed common initial mesh.
% -------------------------------------------------------------------------
[P0,T0,nvb0] = make_structured_tetra_mesh(cfg.initialCells);
V0 = 1/(6*cfg.initialCells^3);
boundary0 = find_boundary_nodes(P0);

methodRowNames = { ...
    'Case','TestID','DatasetIndex','AttemptID','Method','SeedType', ...
    'TargetRelL2','ReachedTarget','StopReason','CorrectionCycles', ...
    'FreeDOF','Elements','RelL2','MethodTimeSec','SearchWallTimeSec', ...
    'MeshTimeSec','SolveTimeSec','AssemblyTimeSec', ...
    'LinearSolveTimeSec','PcgRelRes','PcgIter', ...
    'MeshTotalTimeSec','AssemblyTotalTimeSec','SolveTotalTimeSec'};
scoreRowNames = { ...
    'Case','TestID','DatasetIndex','AttemptID', ...
    'ScoreMAE','ScoreRMSE','UnderGenerationRate', ...
    'ExactGenerationRate','OverGenerationRate', ...
    'FNOSeedRelL2','TargetSeedRelL2'};

methodRows = cell(0,numel(methodRowNames));
scoreRows = cell(0,numel(scoreRowNames));
sampleResults = cell(cfg.numSamples,1);

numVisualizationSamples = min( ...
    cfg.numSamples,round(cfg.numSavedVisualizationSamples));
if cfg.saveVisualizationData
    visualizationSamples = cell(numVisualizationSamples,1);
else
    visualizationSamples = cell(0,1);
end

for caseNo = 1:cfg.numSamples
    testId = sampleIds(caseNo);
    datasetIndex = originalIndices(testId);
    if isfield(D,'source_attempt_id_test')
        attemptIds = round(double(D.source_attempt_id_test(:)));
        attemptId = attemptIds(testId);
    else
        attemptId = datasetIndex;
    end

    fprintf('\n');
    fprintf('--------------------------------------------------------------------\n');
    fprintf('Case %d/%d | test=%d | dataset=%d | attempt=%d\n', ...
        caseNo,cfg.numSamples,testId,datasetIndex,attemptId);
    fprintf('--------------------------------------------------------------------\n');

    predScore = get_score_sample_3d( ...
        D.pred_score,testId,nTest,nx,ny,nz);
    targetScore = get_score_sample_3d( ...
        D.target_score_test,testId,nTest,nx,ny,nz);
    predScore = min(max(double(predScore),0),cfg.exportedScoreMaxLevel);
    targetScore = min(max(double(targetScore),0),cfg.exportedScoreMaxLevel);

    scoreDiff = predScore-targetScore;
    scoreMAE = mean(abs(scoreDiff(:)));
    scoreRMSE = sqrt(mean(scoreDiff(:).^2));

    predScoreUsed = min(max( ...
        cfg.operatorRefinementMultiplier*predScore,0),cfg.operatorMaxLevel);
    targetScoreUsed = min(max( ...
        cfg.operatorRefinementMultiplier*targetScore,0),cfg.operatorMaxLevel);
    predGeneration = score_to_generation_threshold03_3d(predScoreUsed,cfg);
    targetGeneration = score_to_generation_threshold03_3d(targetScoreUsed,cfg);
    underRate = mean(predGeneration(:)<targetGeneration(:));
    exactRate = mean(predGeneration(:)==targetGeneration(:));
    overRate = mean(predGeneration(:)>targetGeneration(:));

    sample = struct();
    sample.grf.modes = datasetModes;
    sample.grf.spectralStd = datasetSpectralStd;
    sample.grf.xiCos = double(C.grf_xi_cos(:,datasetIndex));
    sample.grf.xiSin = double(C.grf_xi_sin(:,datasetIndex));
    sample.grf.K = cfg.grfK;
    sample.grf.beta = cfg.grfBeta;
    sample.grf.kappa = cfg.grfKappa;
    sample.grf.gamma = cfg.grfGamma;
    sample.grf.pointVariance = sum(datasetSpectralStd.^2);
    sample.boundaryMultiplier = cfg.boundaryMultiplier;

    [GbcFun,boundaryInfo] = build_boundary_grf_interpolant( ...
        sample,cfg.boundaryGridN,cfg.boundaryMultiplier);

    boundaryMismatch = NaN;
    if isfield(D,'boundary_test')
        exportedBoundary = get_boundary_sample_2d( ...
            D.boundary_test,testId,nTest,nx,ny);
        [BX,BY] = ndgrid(queryX,queryY);
        reconstructedBoundary = GbcFun(BX,BY);
        boundaryMismatch = max(abs( ...
            reconstructedBoundary(:)-double(exportedBoundary(:))));
        boundaryScale = max(1,max(abs(reconstructedBoundary(:))));
        if boundaryMismatch > 5e-6*boundaryScale
            error(['Exact boundary reconstruction mismatch %.3e is too ', ...
                   'large for test ID %d.'],boundaryMismatch,testId);
        end
    end

    tReference = tic;
    reference = solve_boundary_reference_sine_spectral( ...
        GbcFun,cfg.epsilon,cfg.referenceCellsXY, ...
        cfg.referenceZPoints,cfg.referenceZPower);
    referenceTime = toc(tReference);

    S = struct();
    S.caseNo = caseNo;
    S.testId = testId;
    S.datasetIndex = datasetIndex;
    S.attemptId = attemptId;
    S.boundaryMismatch = boundaryMismatch;
    S.boundaryInfo = boundaryInfo;
    S.referenceTimeSec = referenceTime;
    S.scoreMAE = scoreMAE;
    S.scoreRMSE = scoreRMSE;
    S.underGenerationRate = underRate;
    S.exactGenerationRate = exactRate;
    S.overGenerationRate = overRate;

    saveVisualizationThisSample = ...
        cfg.saveVisualizationData && ...
        caseNo <= numVisualizationSamples;

    if saveVisualizationThisSample
        V = struct();
        V.meta = struct( ...
            'caseNo',caseNo, ...
            'testId',testId, ...
            'datasetIndex',datasetIndex, ...
            'attemptId',attemptId, ...
            'epsilon',cfg.epsilon, ...
            'initialCells',cfg.initialCells, ...
            'targetRelativeL2',cfg.targetRelativeL2, ...
            'initializationMode',cfg.newtonInitializationMode);

        % Save the exact random coefficients as well as a ready-to-plot
        % boundary grid. This makes later plotting independent of the
        % original prediction export for these selected samples.
        [BXvis,BYvis] = ndgrid(queryX,queryY);
        V.input = struct();
        V.input.grf = sample.grf;
        V.input.boundary = struct( ...
            'x',queryX, ...
            'y',queryY, ...
            'value',single(GbcFun(BXvis,BYvis)), ...
            'reconstructionMismatch',boundaryMismatch, ...
            'boundaryMultiplier',cfg.boundaryMultiplier);

        V.score = struct( ...
            'queryX',queryX, ...
            'queryY',queryY, ...
            'queryZ',queryZ, ...
            'queryS',queryS, ...
            'predicted',single(predScore), ...
            'target',single(targetScore), ...
            'predictedUsed',single(predScoreUsed), ...
            'targetUsed',single(targetScoreUsed), ...
            'predictedGeneration',uint8(predGeneration), ...
            'targetGeneration',uint8(targetGeneration));
    end

    % ------------------------------------------------------------------
    % Initialization context shared by FNO, target-score and uniform FEM.
    % A common coarse solve is performed only in coarse_interpolation mode.
    % Its cost is added separately to every corresponding standalone method
    % time, even though this benchmark computes the coarse solution once.
    % ------------------------------------------------------------------
    initializationContext = struct();
    initializationContext.mode = cfg.newtonInitializationMode;
    initializationContext.coarseP = [];
    initializationContext.coarseT = [];
    initializationContext.coarseU = [];
    initializationContext.coarseBoundary = [];
    initializationContext.coarseSolveInfo = [];
    initializationContext.coarseGeom = [];
    initializationContext.coarseStageTimeSec = 0.0;

    needsConfigurableInitialization = any([ ...
        cfg.runFNOSeededAFEM, ...
        cfg.runTargetSeededAFEM, ...
        cfg.runAccuracyMatchedUniform, ...
        cfg.runFixedUniform]);

    if strcmp(cfg.newtonInitializationMode,'coarse_interpolation') && ...
            needsConfigurableInitialization

        coarseStageTimer = tic;
        coarseInitialGuess = ...
            make_reaction_diffusion_boundary_extension(P0,GbcFun);
        [uCoarseInitialization,coarseInitializationSolveInfo, ...
                coarseInitializationGeom] = ...
            solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
                P0,T0,GbcFun,cfg.epsilon,cfg, ...
                coarseInitialGuess,boundary0);
        coarseStageTimeSec = toc(coarseStageTimer);

        initializationContext.coarseP = P0;
        initializationContext.coarseT = T0;
        initializationContext.coarseU = uCoarseInitialization;
        initializationContext.coarseBoundary = boundary0;
        initializationContext.coarseSolveInfo = ...
            coarseInitializationSolveInfo;
        initializationContext.coarseGeom = coarseInitializationGeom;
        initializationContext.coarseStageTimeSec = coarseStageTimeSec;

        fprintf(['  Common coarse initialization %d^3: DOF=%d, ', ...
                 'elems=%d, PCG iter=%d, relres=%.3e, time=%.3f s\n'], ...
            cfg.initialCells,nnz(~boundary0),size(T0,1), ...
            coarseInitializationSolveInfo.iter, ...
            coarseInitializationSolveInfo.relres, ...
            coarseStageTimeSec);
    end

    % Timing-breakdown buckets of the optional common coarse solve.  In
    % coarse_interpolation mode its cost is included in the FNO, target and
    % uniform MethodTimeSec values, so it is also added to the per-sample
    % assembly/solve totals of those methods.
    coarseAssemblyTime = 0.0;
    coarseSolveTime = 0.0;
    if ~isempty(initializationContext.coarseSolveInfo)
        coarseAssemblyTime = safe_info_field( ...
            initializationContext.coarseSolveInfo,'assemblyTime');
        coarseSolveTime = safe_info_field( ...
            initializationContext.coarseSolveInfo,'linearSolveTime');
    end

    S.linearInitializationPolicy = struct( ...
        'selectedMode',cfg.newtonInitializationMode, ...
        'initialConditionMeaning','boundary_extension_(1-z)g_D', ...
        'fnoSeed',cfg.newtonInitializationMode, ...
        'targetSeed',cfg.newtonInitializationMode, ...
        'accuracyUniform',cfg.newtonInitializationMode, ...
        'fixedUniform',cfg.newtonInitializationMode, ...
        'standardAFEMStage0','zero', ...
        'laterAFEM','previous_P1_solution_prolongation', ...
        'coarseStageTimeSec', ...
            initializationContext.coarseStageTimeSec);

    if saveVisualizationThisSample
        V.initializationPolicy = S.linearInitializationPolicy;

        switch cfg.newtonInitializationMode
            case 'zero'
                commonInitialArgument = [];
            case 'initial_condition'
                commonInitialArgument = ...
                    make_reaction_diffusion_boundary_extension(P0,GbcFun);
            case 'coarse_interpolation'
                commonInitialArgument = coarseInitialGuess;
            otherwise
                error('Unexpected initialization mode.');
        end

        V.commonInitialMesh = make_visualization_adaptive_state( ...
            P0,T0,[],nvb0,boundary0, ...
            materialize_reaction_diffusion_initial_guess( ...
                P0,GbcFun,commonInitialArgument), ...
            struct(),NaN);

        if strcmp(cfg.newtonInitializationMode,'coarse_interpolation')
            V.commonCoarseSolution = make_visualization_adaptive_state( ...
                P0,T0,uCoarseInitialization,nvb0,boundary0, ...
                materialize_reaction_diffusion_initial_guess( ...
                    P0,GbcFun,coarseInitialGuess), ...
                coarseInitializationSolveInfo,NaN);
            V.commonCoarseSolution.stageTimeSec = ...
                initializationContext.coarseStageTimeSec;
        end
    end

    fnoSeedRelL2 = NaN;
    targetSeedRelL2 = NaN;

    % ------------------------------------------------------------------
    % 1. FNO-seeded AFEM to the common target tolerance.
    % ------------------------------------------------------------------
    if cfg.runFNOSeededAFEM
        tMesh = tic;
        [PFNO,TFNO,tagFNO,fnoMeshStats,fnoBoundary] = ...
            build_score_driven_mesh_3d_fast_exact( ...
                predScoreUsed,queryX,queryY,queryZ, ...
                P0,T0,nvb0,boundary0,V0,cfg, ...
                cfg.fnoScoreInterpolation);
        fnoMeshTime = toc(tMesh);

        [fnoInitialGuess,fnoInitializationInfo] = ...
            make_reaction_diffusion_linear_initial_guess( ...
                PFNO,GbcFun,initializationContext);

        tSolve = tic;
        [uFNO,fnoSolveInfo,geomFNO] = ...
            solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
                PFNO,TFNO,GbcFun,cfg.epsilon,cfg, ...
                fnoInitialGuess,fnoBoundary);
        fnoSolveTime = toc(tSolve);
        fnoSolveInfo.initializationMode = ...
            fnoInitializationInfo.label;
        fnoSeedRelL2 = relative_l2_error_against_reference( ...
            PFNO,TFNO,uFNO,reference,geomFNO);
        fnoInitialTime = ...
            initializationContext.coarseStageTimeSec + ...
            cfg.operatorInferenceTimePerSample + ...
            fnoMeshTime + ...
            fnoInitializationInfo.timeSec + ...
            fnoSolveTime;

        [fnoFinal,fnoHistory] = run_afem_to_tolerance_3d_fast_exact( ...
            PFNO,TFNO,tagFNO,fnoBoundary,uFNO,geomFNO,fnoSolveInfo, ...
            fnoInitialTime,fnoSeedRelL2,GbcFun,reference,cfg, ...
            sprintf('%s-seeded AFEM',operatorLabel));

        [fnoMeshTotal,fnoAsmTotal,fnoSolveTotal] = ...
            seeded_afem_timing_totals( ...
                fnoMeshTime,fnoSolveInfo,fnoFinal, ...
                initializationContext.coarseSolveInfo);
        methodRows(end+1,:) = make_accu_method_row( ...
            caseNo,testId,datasetIndex,attemptId, ...
            sprintf('%s-seeded AFEM',operatorLabel), ...
            'predicted_score',cfg.targetRelativeL2, ...
            fnoFinal,fnoMeshTime,fnoSolveTime,0, ...
            fnoMeshTotal,fnoAsmTotal,fnoSolveTotal); %#ok<AGROW>

        S.fnoSeed = struct( ...
            'dof',nnz(~fnoBoundary),'elements',size(TFNO,1), ...
            'relL2',fnoSeedRelL2,'timeSec',fnoInitialTime, ...
            'inferenceTimeSec',cfg.operatorInferenceTimePerSample, ...
            'meshTimeSec',fnoMeshTime, ...
            'initializationMode',fnoInitializationInfo.label, ...
            'initializationTimeSec',fnoInitializationInfo.timeSec, ...
            'coarseStageTimeSec', ...
                initializationContext.coarseStageTimeSec, ...
            'solveTimeSec',fnoSolveTime, ...
            'meshStats',fnoMeshStats);
        S.fnoSeededAFEM = compact_method_result(fnoFinal);
        S.fnoHistory = fnoHistory;

        if saveVisualizationThisSample
            V.fno = struct();
            V.fno.seed = make_visualization_adaptive_state( ...
                PFNO,TFNO,uFNO,tagFNO,fnoBoundary, ...
                materialize_reaction_diffusion_initial_guess( ...
                    PFNO,GbcFun,fnoInitialGuess), ...
                fnoSolveInfo,fnoSeedRelL2);
            V.fno.seed.meshStats = fnoMeshStats;
            V.fno.seed.methodTimeSec = fnoInitialTime;

            V.fno.final = make_visualization_adaptive_state( ...
                fnoFinal.P,fnoFinal.T,fnoFinal.u, ...
                fnoFinal.nvbTag,fnoFinal.boundaryMask, ...
                [],fnoFinal.solveInfo,fnoFinal.relL2);
            V.fno.final.methodTimeSec = fnoFinal.timeSec;
            V.fno.final.correctionCycles = ...
                fnoFinal.correctionCycles;
            V.fno.history = fnoHistory;
        end

        if cfg.saveSelectedMeshes && ~cfg.saveVisualizationData
            S.fnoFinalMesh = struct( ...
                'P',fnoFinal.P,'T',fnoFinal.T, ...
                'tag',fnoFinal.nvbTag,'u',fnoFinal.u);
        end
    end

    % ------------------------------------------------------------------
    % 2. Target-score-seeded AFEM to the common target tolerance.
    % ------------------------------------------------------------------
    if cfg.runTargetSeededAFEM
        tMesh = tic;
        [PTarget,TTarget,tagTarget,targetMeshStats,targetBoundary] = ...
            build_score_driven_mesh_3d_fast_exact( ...
                targetScoreUsed,queryX,queryY,queryZ, ...
                P0,T0,nvb0,boundary0,V0,cfg, ...
                cfg.targetScoreInterpolation);
        targetMeshTime = toc(tMesh);

        [targetInitialGuess,targetInitializationInfo] = ...
            make_reaction_diffusion_linear_initial_guess( ...
                PTarget,GbcFun,initializationContext);

        tSolve = tic;
        [uTarget,targetSolveInfo,geomTarget] = ...
            solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
                PTarget,TTarget,GbcFun,cfg.epsilon,cfg, ...
                targetInitialGuess,targetBoundary);
        targetSolveTime = toc(tSolve);
        targetSolveInfo.initializationMode = ...
            targetInitializationInfo.label;
        targetSeedRelL2 = relative_l2_error_against_reference( ...
            PTarget,TTarget,uTarget,reference,geomTarget);
        targetInitialTime = ...
            initializationContext.coarseStageTimeSec + ...
            targetMeshTime + ...
            targetInitializationInfo.timeSec + ...
            targetSolveTime;

        [targetFinal,targetHistory] = run_afem_to_tolerance_3d_fast_exact( ...
            PTarget,TTarget,tagTarget,targetBoundary, ...
            uTarget,geomTarget,targetSolveInfo, ...
            targetInitialTime,targetSeedRelL2,GbcFun,reference,cfg, ...
            'Target-score-seeded AFEM');

        [targetMeshTotal,targetAsmTotal,targetSolveTotal] = ...
            seeded_afem_timing_totals( ...
                targetMeshTime,targetSolveInfo,targetFinal, ...
                initializationContext.coarseSolveInfo);
        methodRows(end+1,:) = make_accu_method_row( ...
            caseNo,testId,datasetIndex,attemptId, ...
            'Target-score-seeded AFEM','target_score', ...
            cfg.targetRelativeL2,targetFinal, ...
            targetMeshTime,targetSolveTime,0, ...
            targetMeshTotal,targetAsmTotal,targetSolveTotal); %#ok<AGROW>

        S.targetSeed = struct( ...
            'dof',nnz(~targetBoundary),'elements',size(TTarget,1), ...
            'relL2',targetSeedRelL2,'timeSec',targetInitialTime, ...
            'meshTimeSec',targetMeshTime, ...
            'initializationMode',targetInitializationInfo.label, ...
            'initializationTimeSec',targetInitializationInfo.timeSec, ...
            'coarseStageTimeSec', ...
                initializationContext.coarseStageTimeSec, ...
            'solveTimeSec',targetSolveTime, ...
            'meshStats',targetMeshStats);
        S.targetSeededAFEM = compact_method_result(targetFinal);
        S.targetHistory = targetHistory;

        if saveVisualizationThisSample
            V.target = struct();
            V.target.seed = make_visualization_adaptive_state( ...
                PTarget,TTarget,uTarget,tagTarget,targetBoundary, ...
                materialize_reaction_diffusion_initial_guess( ...
                    PTarget,GbcFun,targetInitialGuess), ...
                targetSolveInfo,targetSeedRelL2);
            V.target.seed.meshStats = targetMeshStats;
            V.target.seed.methodTimeSec = targetInitialTime;

            V.target.final = make_visualization_adaptive_state( ...
                targetFinal.P,targetFinal.T,targetFinal.u, ...
                targetFinal.nvbTag,targetFinal.boundaryMask, ...
                [],targetFinal.solveInfo,targetFinal.relL2);
            V.target.final.methodTimeSec = targetFinal.timeSec;
            V.target.final.correctionCycles = ...
                targetFinal.correctionCycles;
            V.target.history = targetHistory;
        end

        if cfg.saveSelectedMeshes && ~cfg.saveVisualizationData
            S.targetFinalMesh = struct( ...
                'P',targetFinal.P,'T',targetFinal.T, ...
                'tag',targetFinal.nvbTag,'u',targetFinal.u);
        end
    end

    % ------------------------------------------------------------------
    % 3. Standard AFEM from the common initial mesh.
    % ------------------------------------------------------------------
    if cfg.runStandardAFEM
        tSolve = tic;
        [uStandard,standardSolveInfo,geomStandard] = ...
            solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
                P0,T0,GbcFun,cfg.epsilon,cfg,[],boundary0);
        standardSolveTime = toc(tSolve);
        standardInitialRelL2 = relative_l2_error_against_reference( ...
            P0,T0,uStandard,reference,geomStandard);

        [standardFinal,standardHistory] = ...
            run_afem_to_tolerance_3d_fast_exact( ...
                P0,T0,nvb0,boundary0,uStandard,geomStandard, ...
                standardSolveInfo,standardSolveTime,standardInitialRelL2, ...
                GbcFun,reference,cfg,'Standard AFEM');

        [standardMeshTotal,standardAsmTotal,standardSolveTotal] = ...
            seeded_afem_timing_totals( ...
                0,standardSolveInfo,standardFinal,[]);
        methodRows(end+1,:) = make_accu_method_row( ...
            caseNo,testId,datasetIndex,attemptId, ...
            'Standard AFEM','common_coarse_mesh', ...
            cfg.targetRelativeL2,standardFinal, ...
            0,standardSolveTime,0, ...
            standardMeshTotal,standardAsmTotal,standardSolveTotal); %#ok<AGROW>

        S.standardAFEM = compact_method_result(standardFinal);
        S.standardHistory = standardHistory;

        if saveVisualizationThisSample
            V.standard = struct();
            V.standard.seed = make_visualization_adaptive_state( ...
                P0,T0,uStandard,nvb0,boundary0, ...
                materialize_reaction_diffusion_initial_guess( ...
                    P0,GbcFun,[]), ...
                standardSolveInfo,standardInitialRelL2);
            V.standard.seed.methodTimeSec = standardSolveTime;

            V.standard.final = make_visualization_adaptive_state( ...
                standardFinal.P,standardFinal.T,standardFinal.u, ...
                standardFinal.nvbTag,standardFinal.boundaryMask, ...
                [],standardFinal.solveInfo,standardFinal.relL2);
            V.standard.final.methodTimeSec = standardFinal.timeSec;
            V.standard.final.correctionCycles = ...
                standardFinal.correctionCycles;
            V.standard.history = standardHistory;
        end

        if cfg.saveSelectedMeshes && ~cfg.saveVisualizationData
            S.standardFinalMesh = struct( ...
                'P',standardFinal.P,'T',standardFinal.T, ...
                'tag',standardFinal.nvbTag,'u',standardFinal.u);
        end
    end

    % ------------------------------------------------------------------
    % 4. Accuracy-matched uniform FEM; search excluded from reported time.
    % ------------------------------------------------------------------
    if cfg.runAccuracyMatchedUniform
        accuracyUniform = run_accuracy_matched_uniform_final_only( ...
            GbcFun,reference,cfg,initializationContext);
        [accMeshTotal,accAsmTotal,accSolveTotal] = ...
            uniform_timing_totals(accuracyUniform, ...
                initializationContext.coarseSolveInfo);
        methodRows(end+1,:) = make_uniform_method_row( ...
            caseNo,testId,datasetIndex,attemptId, ...
            'Accuracy-matched uniform FEM (final solve only)', ...
            'offline_uniform_search',cfg.targetRelativeL2, ...
            accuracyUniform, ...
            accMeshTotal,accAsmTotal,accSolveTotal); %#ok<AGROW>
        S.accuracyMatchedUniform = compact_method_result(accuracyUniform);

        if saveVisualizationThisSample
            V.accuracyMatchedUniform = ...
                make_visualization_uniform_state(accuracyUniform);
        end

        if cfg.saveSelectedMeshes && ~cfg.saveVisualizationData
            S.accuracyUniformMesh = struct( ...
                'P',accuracyUniform.P,'T',accuracyUniform.T, ...
                'u',accuracyUniform.u);
        end
    end

    % ------------------------------------------------------------------
    % 5. Fixed-size uniform FEM; identical grid size for every sample.
    % ------------------------------------------------------------------
    if cfg.runFixedUniform
        fixedUniform = run_fixed_uniform_once( ...
            GbcFun,reference,cfg,initializationContext);
        [fixedMeshTotal,fixedAsmTotal,fixedSolveTotal] = ...
            uniform_timing_totals(fixedUniform, ...
                initializationContext.coarseSolveInfo);
        methodRows(end+1,:) = make_uniform_method_row( ...
            caseNo,testId,datasetIndex,attemptId, ...
            'Fixed-size uniform FEM','fixed_uniform_grid', ...
            cfg.targetRelativeL2,fixedUniform, ...
            fixedMeshTotal,fixedAsmTotal,fixedSolveTotal); %#ok<AGROW>
        S.fixedUniform = compact_method_result(fixedUniform);

        if saveVisualizationThisSample
            V.fixedUniform = ...
                make_visualization_uniform_state(fixedUniform);
        end

        if cfg.saveSelectedMeshes && ~cfg.saveVisualizationData
            S.fixedUniformMesh = struct( ...
                'P',fixedUniform.P,'T',fixedUniform.T, ...
                'u',fixedUniform.u);
        end
    end

    scoreRows(end+1,:) = { ...
        caseNo,testId,datasetIndex,attemptId, ...
        scoreMAE,scoreRMSE,underRate,exactRate,overRate, ...
        fnoSeedRelL2,targetSeedRelL2}; %#ok<AGROW>

    sampleResults{caseNo} = S;

    if saveVisualizationThisSample
        visualizationSamples{caseNo} = V;
        fprintf(['  Saved visualization payload for selected case %d ', ...
                 '(test ID %d).\n'],caseNo,testId);
    end
end

%% ------------------------------------------------------------------------
% Save tables and aggregate statistics.
% -------------------------------------------------------------------------
summaryTable = cell2table(methodRows,'VariableNames',methodRowNames);
scoreDiagnosticsTable = cell2table(scoreRows,'VariableNames',scoreRowNames);
methodStatisticsTable = build_accu_method_statistics(summaryTable);

summaryCsv = fullfile(cfg.outDir,sprintf( ...
    'comparison_summary_%d_samples.csv',cfg.numSamples));
scoreCsv = fullfile(cfg.outDir,sprintf( ...
    'score_diagnostics_%d_samples.csv',cfg.numSamples));
statsCsv = fullfile(cfg.outDir,sprintf( ...
    'method_statistics_%d_samples.csv',cfg.numSamples));
resultMat = fullfile(cfg.outDir,sprintf( ...
    'comparison_result_%d_samples.mat',cfg.numSamples));
visualizationMat = fullfile(cfg.outDir,sprintf( ...
    'visualization_data_first_%d_of_%d_samples.mat', ...
    numVisualizationSamples,cfg.numSamples));

writetable(summaryTable,summaryCsv);
writetable(scoreDiagnosticsTable,scoreCsv);
writetable(methodStatisticsTable,statsCsv);

% The main result remains lightweight for all samples.
save(resultMat, ...
    'sampleResults','summaryTable','scoreDiagnosticsTable', ...
    'methodStatisticsTable','cfg','sampleIds','queryX','queryY', ...
    'queryS','queryZ','-v7.3');

% Heavy P/T/u payloads are stored only for the first selected cases and in
% a separate file, so later plotting is easy without bloating the main MAT.
if cfg.saveVisualizationData && numVisualizationSamples>0
    visualizationSampleIds = sampleIds(1:numVisualizationSamples);
    save(visualizationMat, ...
        'visualizationSamples','visualizationSampleIds', ...
        'cfg','queryX','queryY','queryS','queryZ','-v7.3');
end

if cfg.makePlots
    plot_accu_method_statistics(methodStatisticsTable,cfg);
end

fprintf('\n');
fprintf('====================================================================\n');
fprintf('Finished five-method accuracy comparison.\n');
fprintf('Linear initialization mode: %s\n', ...
    cfg.newtonInitializationMode);
fprintf('====================================================================\n');
disp(methodStatisticsTable(:,{ ...
    'Method','Samples','SuccessRate','MeanMethodTimeSec', ...
    'MeanMeshTimeSec','MeanAssemblyTimeSec','MeanSolveTimeSec', ...
    'MeanOtherTimeSec', ...
    'MeanRelL2','MeanFreeDOF','MeanCorrectionCycles'}));
fprintf('Summary CSV : %s\n',summaryCsv);
fprintf('Score CSV   : %s\n',scoreCsv);
fprintf('Stats CSV   : %s\n',statsCsv);
fprintf('Result MAT  : %s\n',resultMat);
if cfg.saveVisualizationData && numVisualizationSamples>0
    fprintf('Visualization MAT: %s\n',visualizationMat);
    fprintf('Visualization cases: %s\n', ...
        mat2str(sampleIds(1:numVisualizationSamples)));
end
fprintf(['Accuracy-uniform SearchWallTimeSec is recorded but excluded ', ...
         'from MethodTimeSec.\n']);
fprintf('====================================================================\n');

result = struct();
result.sampleResults = sampleResults;
result.summaryTable = summaryTable;
result.scoreDiagnosticsTable = scoreDiagnosticsTable;
result.methodStatisticsTable = methodStatisticsTable;
result.cfg = cfg;
result.sampleIds = sampleIds;
result.resultMat = resultMat;
if cfg.saveVisualizationData && numVisualizationSamples>0
    result.visualizationMat = visualizationMat;
    result.visualizationSampleIds = ...
        sampleIds(1:numVisualizationSamples);
else
    result.visualizationMat = '';
    result.visualizationSampleIds = zeros(0,1);
end

end


function cfg = fill_accu_fem_defaults(cfg)
cfg = set_default(cfg,'targetRelativeL2',4.0e-3);
cfg = set_default(cfg,'epsilon',2.0e-2);
cfg = set_default(cfg,'initialCells',16);
cfg = set_default(cfg,'theta',0.80);
cfg = set_default(cfg,'exportedScoreMaxLevel',12);
cfg = set_default(cfg,'operatorRefinementMultiplier',1.0);
cfg = set_default(cfg,'operatorMaxLevel',12);
cfg = set_default(cfg,'generationThreshold',0.50);
cfg = set_default(cfg,'generationThresholdEps',1.0e-6);
cfg = set_default(cfg,'requireExportedThresholdMatch',true);

cfg = set_default(cfg,'grfK',16);
cfg = set_default(cfg,'grfBeta',4);
cfg = set_default(cfg,'grfKappa',5);
cfg = set_default(cfg,'grfGamma',625);
cfg = set_default(cfg,'boundaryMultiplier',30.0);
cfg = set_default(cfg,'boundaryGridN',256);

cfg = set_default(cfg,'linearSolver','auto');
cfg = set_default(cfg,'pcgTolerance',1.0e-10);
cfg = set_default(cfg,'pcgMaxIteration',3000);
cfg = set_default(cfg,'pcgDiagMaxIteration',200);
cfg = set_default(cfg,'icholDropTolerance',1.0e-3);
cfg = set_default(cfg,'icholDiagComp',1.0e-3);
cfg = set_default(cfg,'useWarmStart',true);

cfg = set_default(cfg,'nvbMaxCompletionSteps',500);
cfg = set_default(cfg,'nvbCheckConformity',false);
cfg = set_default(cfg,'maxFreeDOF',500000);
cfg = set_default(cfg,'maxElementsCase',3500000);
cfg = set_default(cfg,'afemMaxCycles',30);
cfg = set_default(cfg,'operatorMaxRefineCalls',30);

cfg = set_default(cfg,'includeDirichletDataTerm',true);
cfg = set_default(cfg,'estimatorVersion', ...
    'residual_jump_plus_dirichlet_data_v1');
cfg = set_default(cfg,'scoreQueryPattern','fifteen');
cfg = set_default(cfg,'scoreQueryChunkElements',100000);
cfg = set_default(cfg,'fnoScoreInterpolation','linear');
cfg = set_default(cfg,'targetScoreInterpolation','linear');

cfg = set_default(cfg,'fastUseWorkQueue',true);
cfg = set_default(cfg,'fastTrackGeneration',true);
cfg = set_default(cfg,'fastTrackBoundary',true);
cfg = set_default(cfg,'fastPackedNVBEdges',true);
cfg = set_default(cfg,'fastVerifyGenerationByVolume',false);
cfg = set_default(cfg,'fastVerifyFinalScoreMesh',false);
cfg = set_default(cfg,'fastVerifyBoundaryMask',false);
cfg = set_default(cfg,'fastVerifyPackedNVBEdges',false);

cfg = set_default(cfg,'referenceCellsXY',256);
cfg = set_default(cfg,'referenceZPoints',161);
cfg = set_default(cfg,'referenceZPower',2.0);
cfg = set_default(cfg,'sampleSelectionSeed',20260725);
cfg = set_default(cfg,'sampleIds',[]);
cfg = set_default(cfg,'operatorInferenceTimePerSample',NaN);

cfg = set_default(cfg,'runFNOSeededAFEM',true);
cfg = set_default(cfg,'runTargetSeededAFEM',true);
cfg = set_default(cfg,'runStandardAFEM',true);
cfg = set_default(cfg,'runAccuracyMatchedUniform',true);
cfg = set_default(cfg,'runFixedUniform',true);

cfg = set_default(cfg,'uniformSearchCells',16:4:80);
cfg = set_default(cfg,'fixedUniformTargetDOF',250000);
cfg = set_default(cfg,'fixedUniformCells',[]);

cfg = set_default(cfg,'makePlots',false);
cfg = set_default(cfg,'figureVisible','off');
cfg = set_default(cfg,'figureResolution',180);

% Lightweight statistics are saved for every sample. Heavy visualization
% data are saved only for the first numSavedVisualizationSamples selected
% cases. These defaults work without editing the external config file.
cfg = set_default(cfg,'saveVisualizationData',true);
cfg = set_default(cfg,'numSavedVisualizationSamples',3);

% Legacy all-sample heavy-mesh switch. Leave false when the selective
% visualization-data mechanism above is enabled.
cfg = set_default(cfg,'saveSelectedMeshes',false);
end


function cfg = set_default(cfg,name,value)
if ~isfield(cfg,name) || isempty(cfg.(name))
    cfg.(name) = value;
end
end


function validate_accu_fem_config(cfg)
required = {'predictionMat','metricsJson','datasetMat','outDir', ...
    'figureDir','numSamples'};
for k = 1:numel(required)
    if ~isfield(cfg,required{k}) || isempty(cfg.(required{k}))
        error('Configuration field cfg.%s is required.',required{k});
    end
end
validateattributes(cfg.numSamples,{'numeric'}, ...
    {'scalar','integer','>=',1,'finite'});
validateattributes(cfg.targetRelativeL2,{'numeric'}, ...
    {'scalar','positive','finite'});
validateattributes(cfg.afemMaxCycles,{'numeric'}, ...
    {'scalar','integer','>=',0,'finite'});
validateattributes(cfg.operatorMaxLevel,{'numeric'}, ...
    {'scalar','integer','>=',0,'finite'});
validateattributes(cfg.numSavedVisualizationSamples,{'numeric'}, ...
    {'scalar','integer','>=',0,'finite'});
if ~(islogical(cfg.saveVisualizationData) || ...
        (isnumeric(cfg.saveVisualizationData) && ...
         isscalar(cfg.saveVisualizationData)))
    error('cfg.saveVisualizationData must be a logical scalar.');
end
cfg.saveVisualizationData = logical(cfg.saveVisualizationData);
if cfg.operatorMaxLevel ~= cfg.exportedScoreMaxLevel
    error(['operatorMaxLevel and exportedScoreMaxLevel must agree for ', ...
           'this accuracy data set.']);
end
if ~logical(cfg.includeDirichletDataTerm) || ...
        ~strcmp(cfg.estimatorVersion, ...
            'residual_jump_plus_dirichlet_data_v1')
    error(['The accuracy comparison must use residual + jump + ', ...
           'Dirichlet-data estimator terms.']);
end
if ~any([cfg.runFNOSeededAFEM,cfg.runTargetSeededAFEM, ...
         cfg.runStandardAFEM,cfg.runAccuracyMatchedUniform, ...
         cfg.runFixedUniform])
    error('At least one comparison method must be enabled.');
end
if cfg.runAccuracyMatchedUniform
    cells = unique(round(double(cfg.uniformSearchCells(:).')),'stable');
    if isempty(cells) || any(cells<2) || any(diff(cells)<=0)
        error('uniformSearchCells must be a strictly increasing vector >=2.');
    end
end
resolve_fixed_uniform_cells(cfg);
end


function n = resolve_fixed_uniform_cells(cfg)
if isfield(cfg,'fixedUniformCells') && ~isempty(cfg.fixedUniformCells)
    n = round(double(cfg.fixedUniformCells));
else
    targetDOF = double(cfg.fixedUniformTargetDOF);
    validateattributes(targetDOF,{'numeric'}, ...
        {'scalar','positive','finite'});
    n = max(2,ceil(targetDOF^(1/3))+1);
    while (n-1)^3 < targetDOF
        n = n+1;
    end
end
validateattributes(n,{'numeric'}, ...
    {'scalar','integer','>=',2,'finite'});
if 6*n^3 > cfg.maxElementsCase
    error(['Fixed uniform grid n=%d has %d elements, exceeding ', ...
           'cfg.maxElementsCase=%d.'],n,6*n^3,cfg.maxElementsCase);
end
if (n-1)^3 > cfg.maxFreeDOF
    error(['Fixed uniform grid n=%d has %d free DOF, exceeding ', ...
           'cfg.maxFreeDOF=%d.'],n,(n-1)^3,cfg.maxFreeDOF);
end
end


%% ========================================================================
% Prediction readers and score-driven mesh compiler reused from the prior
% FAST-EXACT reaction-diffusion implementation.
% =========================================================================

function nTest = infer_score_test_count(A,nx,ny,nz)
sz = size(A);
if ndims(A)~=4
    error('Score array must be 4-D, got %s.',mat2str(sz));
end
spatial = [nx,ny,nz];
if all(sz(1:3)==spatial)
    nTest = sz(4);
elseif all(sz(2:4)==spatial)
    nTest = sz(1);
else
    error('Cannot identify test dimension in score array size %s.', ...
        mat2str(sz));
end
end


function score = get_score_sample_3d(A,k,nTest,nx,ny,nz)
sz = size(A);

if isequal(sz(1:3),[nx,ny,nz]) && sz(4)==nTest
    score = A(:,:,:,k);
elseif sz(1)==nTest && isequal(sz(2:4),[nx,ny,nz])
    score = squeeze(A(k,:,:,:));
else
    error('Unsupported 3-D score layout %s.',mat2str(sz));
end

score = squeeze(score);
if ~isequal(size(score),[nx,ny,nz])
    error('Extracted score has size %s, expected [%d %d %d].', ...
        mat2str(size(score)),nx,ny,nz);
end
end


function B = get_boundary_sample_2d(A,k,nTest,nx,ny)
sz = size(A);
if ndims(A)~=3
    error('boundary_test must be 3-D.');
end
if isequal(sz(1:2),[nx,ny]) && sz(3)==nTest
    B = A(:,:,k);
elseif sz(1)==nTest && isequal(sz(2:3),[nx,ny])
    B = squeeze(A(k,:,:));
else
    error('Unsupported boundary_test layout %s.',mat2str(sz));
end
B = squeeze(B);
end


function t = read_fno_inference_time_per_sample(jsonFile)
if exist(jsonFile,'file')~=2
    error(['final_metrics.json not found:\n%s\n', ...
           'The trained Python run must finish before the strict time ', ...
           'comparison can use the exported inference timing.'],jsonFile);
end

J = jsondecode(fileread(jsonFile));

if isfield(J,'timing') && ...
        isfield(J.timing,'mean_inference_time_sec_per_sample')
    t = double(J.timing.mean_inference_time_sec_per_sample);
else
    error('mean_inference_time_sec_per_sample is missing from %s.',jsonFile);
end
end


%% ========================================================================
% Score conversion and score-driven 3-D mesh
% =========================================================================

function generation = score_to_generation_threshold03_3d(score,cfg)
score = min(max(double(score),0),double(cfg.operatorMaxLevel));
generation = floor( ...
    score + 1.0 - cfg.generationThreshold + cfg.generationThresholdEps);
generation = min(max(generation,0),double(cfg.operatorMaxLevel));
end


function Fscore = make_score_interpolant_3d( ...
        score,queryX,queryY,queryZ,method)

score = double(score);
queryX = double(queryX(:));
queryY = double(queryY(:));
queryZ = double(queryZ(:));

if ~isequal(size(score), ...
        [numel(queryX),numel(queryY),numel(queryZ)])
    error('Score tensor and query grids are incompatible.');
end

if any(diff(queryX)<=0) || any(diff(queryY)<=0) || any(diff(queryZ)<=0)
    error('Score query coordinates must be strictly increasing.');
end

Fscore = griddedInterpolant( ...
    {queryX,queryY,queryZ},score,method,'nearest');
end


function [P,T,nvbTag,stats,boundaryMask] = ...
    build_score_driven_mesh_3d_fast_exact( ...
        score,queryX,queryY,queryZ, ...
        P0,T0,nvb0,boundary0,V0,cfg,interpMethod)
%BUILD_SCORE_DRIVEN_MESH_3D_FAST_EXACT
%
% Exact-algorithm fast compiler for the FNO / target score field.
%
% The legacy implementation re-evaluated every leaf tetrahedron after each
% refinement pass. That is unnecessary: a leaf that was unchanged and was
% already found to satisfy
%
%       generation(K) >= desiredGeneration(K)
%
% has the same geometry, same score field and same generation at the next
% pass, hence it cannot become newly marked. Therefore only newly created
% leaves have to be queried again.
%
% No score-query rule is changed: the exact same 15-point maximum and the
% exact same thresholding routine are used.

Fscore = make_score_interpolant_3d( ...
    score,queryX,queryY,queryZ,interpMethod);

P = P0;
T = T0;
nvbTag = nvb0;
boundaryMask = logical(boundary0(:));

if numel(boundaryMask) ~= size(P,1)
    error('Initial boundary mask size mismatch.');
end

% Exact binary-NVB generation. Initial Kuhn tetrahedra are generation zero.
generation = zeros(size(T,1),1,'uint8');

% Initially every leaf must be checked once.
workMask = true(size(T,1),1);

stats.pass = zeros(0,1);
stats.freeDOF = nnz(~boundaryMask);
stats.elements = size(T,1);
stats.marked = zeros(0,1);
stats.completionParents = zeros(0,1);
stats.maxDesired = zeros(0,1);
stats.queriedElements = zeros(0,1);
stats.fullScanEquivalentElements = zeros(0,1);
stats.savedScoreQueries = zeros(0,1);

for pass = 1:cfg.operatorMaxRefineCalls

    if cfg.fastUseWorkQueue
        active = find(workMask & generation<cfg.operatorMaxLevel);
    else
        active = find(generation<cfg.operatorMaxLevel);
    end

    if isempty(active)
        break;
    end

    scoreElem = evaluate_score_on_tetrahedra_3d( ...
        Fscore,P,T,active,cfg);

    desired = score_to_generation_threshold03_3d(scoreElem,cfg);
    marked = active(double(generation(active))<desired);

    % Optional expensive proof-by-comparison against the old full scan.
    % This is useful for one-sample validation, not production timing.
    if isfield(cfg,'fastVerifyFinalScoreMesh') && ...
            cfg.fastVerifyFinalScoreMesh && cfg.fastUseWorkQueue

        legacyActive = find(generation<cfg.operatorMaxLevel);
        legacyScore = evaluate_score_on_tetrahedra_3d( ...
            Fscore,P,T,legacyActive,cfg);
        legacyDesired = score_to_generation_threshold03_3d(legacyScore,cfg);
        legacyMarked = legacyActive( ...
            double(generation(legacyActive))<legacyDesired);

        if ~isequal(marked(:),legacyMarked(:))
            error(['FAST work queue and legacy full scan selected different ', ...
                   'marked tetrahedra at score pass %d.'],pass);
        end
    end

    stats.pass(end+1,1) = pass; %#ok<AGROW>
    stats.marked(end+1,1) = numel(marked); %#ok<AGROW>
    stats.maxDesired(end+1,1) = max(desired); %#ok<AGROW>
    stats.queriedElements(end+1,1) = numel(active); %#ok<AGROW>
    stats.fullScanEquivalentElements(end+1,1) = ...
        nnz(generation<cfg.operatorMaxLevel); %#ok<AGROW>
    stats.savedScoreQueries(end+1,1) = ...
        stats.fullScanEquivalentElements(end)-numel(active); %#ok<AGROW>

    if isempty(marked)
        stats.freeDOF(end+1,1) = nnz(~boundaryMask); %#ok<AGROW>
        stats.elements(end+1,1) = size(T,1); %#ok<AGROW>
        stats.completionParents(end+1,1) = 0; %#ok<AGROW>

        % Every current work item was checked and satisfied its target.
        workMask(:) = false;
        break;
    end

    % Payload is propagated through EXACTLY the same Maubach closure.
    state = struct();
    if cfg.fastTrackGeneration
        state.elementGeneration = generation;
    else
        state.elementGeneration = [];
    end
    state.elementNewLeaf = false(size(T,1),1);

    if cfg.fastTrackBoundary
        state.nodeBoundary = boundaryMask;
    else
        state.nodeBoundary = [];
    end

    [P,T,nvbTag,rinfo,state] = ...
        nvb_refine_conforming_3d_fast_exact( ...
            P,T,nvbTag,marked,cfg,state);

    if cfg.fastTrackGeneration
        generation = state.elementGeneration;
    else
        generation = uint8( ...
            element_generation_from_volume_comparison_3d(P,T,V0));
    end

    if cfg.fastTrackBoundary
        boundaryMask = state.nodeBoundary;
    else
        boundaryMask = find_boundary_nodes(P);
    end

    % Only final leaves born during this NVB call can require a new score
    % query. Unchanged leaves were already checked on this pass.
    workMask = state.elementNewLeaf;

    if size(T,1)>cfg.maxElementsCase
        error('Score-driven mesh exceeded cfg.maxElementsCase=%d.', ...
            cfg.maxElementsCase);
    end

    freeDOF = nnz(~boundaryMask);
    if freeDOF>cfg.maxFreeDOF
        error('Score-driven mesh free DOF %d exceeded cfg.maxFreeDOF=%d.', ...
            freeDOF,cfg.maxFreeDOF);
    end

    % Optional exact audits.
    if cfg.fastVerifyGenerationByVolume
        generationByVolume = uint8( ...
            element_generation_from_volume_comparison_3d(P,T,V0));
        if ~isequal(generation,generationByVolume)
            error('Tracked NVB generation differs from the volume definition.');
        end
    end

    if cfg.fastVerifyBoundaryMask
        legacyBoundary = find_boundary_nodes(P);
        if ~isequal(boundaryMask,legacyBoundary)
            error('Incremental boundary mask differs from full recomputation.');
        end
    end

    stats.freeDOF(end+1,1) = freeDOF; %#ok<AGROW>
    stats.elements(end+1,1) = size(T,1); %#ok<AGROW>
    stats.completionParents(end+1,1) = ...
        max(0,rinfo.totalBisectedParents-numel(marked)); %#ok<AGROW>
end

% If the pass cap was hit, only the remaining work queue can be unresolved.
remaining = 0;
active = find(workMask & generation<cfg.operatorMaxLevel);

if ~isempty(active)
    scoreElem = evaluate_score_on_tetrahedra_3d( ...
        Fscore,P,T,active,cfg);
    desired = score_to_generation_threshold03_3d(scoreElem,cfg);
    remaining = nnz(double(generation(active))<desired);
end

if remaining>0
    error(['Score-driven mesh stopped with %d tetrahedra still below ', ...
           'the requested generation. Increase operatorMaxRefineCalls.'], ...
        remaining);
end

% Optional expensive final audit of the whole leaf mesh.
if cfg.fastVerifyFinalScoreMesh
    allActive = find(generation<cfg.operatorMaxLevel);
    if ~isempty(allActive)
        scoreAll = evaluate_score_on_tetrahedra_3d( ...
            Fscore,P,T,allActive,cfg);
        desiredAll = score_to_generation_threshold03_3d(scoreAll,cfg);
        bad = nnz(double(generation(allActive))<desiredAll);
        if bad~=0
            error('Final FAST score mesh has %d under-refined leaves.',bad);
        end
    end
end

stats.finalFreeDOF = nnz(~boundaryMask);
stats.finalElements = size(T,1);
stats.finalGenerationMin = double(min(generation));
stats.finalGenerationMax = double(max(generation));
stats.remainingBelowTarget = remaining;
stats.totalQueriedElements = sum(stats.queriedElements);
stats.totalFullScanEquivalentElements = ...
    sum(stats.fullScanEquivalentElements);
stats.totalSavedScoreQueries = sum(stats.savedScoreQueries);

end


function scoreElem = evaluate_score_on_tetrahedra_3d( ...
    Fscore,P,T,elementIds,cfg)
%EVALUATE_SCORE_ON_TETRAHEDRA_3D
% Vectorized score evaluation.
%
% 'five':
%   4 vertices + tetrahedron centroid.
%
% 'fifteen':
%   4 vertices
% + 6 edge midpoints
% + 4 face centroids
% + 1 tetrahedron centroid.
%
% All query points in one chunk are passed to the interpolant in one call.

elementIds = elementIds(:);
n = numel(elementIds);

scoreElem = zeros(n,1);

chunkSize = max(1,round(cfg.scoreQueryChunkElements));

for first = 1:chunkSize:n

last = min(first+chunkSize-1,n);
ids = elementIds(first:last);
m = numel(ids);

p1 = P(T(ids,1),:);
p2 = P(T(ids,2),:);
p3 = P(T(ids,3),:);
p4 = P(T(ids,4),:);

centroid = 0.25*(p1+p2+p3+p4);

switch lower(cfg.scoreQueryPattern)

    case 'five'

        X = [ ...
            p1(:,1), ...
            p2(:,1), ...
            p3(:,1), ...
            p4(:,1), ...
            centroid(:,1)];

        Y = [ ...
            p1(:,2), ...
            p2(:,2), ...
            p3(:,2), ...
            p4(:,2), ...
            centroid(:,2)];

        Z = [ ...
            p1(:,3), ...
            p2(:,3), ...
            p3(:,3), ...
            p4(:,3), ...
            centroid(:,3)];

    case 'fifteen'

        m12 = 0.5*(p1+p2);
        m13 = 0.5*(p1+p3);
        m14 = 0.5*(p1+p4);
        m23 = 0.5*(p2+p3);
        m24 = 0.5*(p2+p4);
        m34 = 0.5*(p3+p4);

        f123 = (p1+p2+p3)/3;
        f124 = (p1+p2+p4)/3;
        f134 = (p1+p3+p4)/3;
        f234 = (p2+p3+p4)/3;

        X = [ ...
            p1(:,1),p2(:,1),p3(:,1),p4(:,1), ...
            m12(:,1),m13(:,1),m14(:,1), ...
            m23(:,1),m24(:,1),m34(:,1), ...
            f123(:,1),f124(:,1),f134(:,1),f234(:,1), ...
            centroid(:,1)];

        Y = [ ...
            p1(:,2),p2(:,2),p3(:,2),p4(:,2), ...
            m12(:,2),m13(:,2),m14(:,2), ...
            m23(:,2),m24(:,2),m34(:,2), ...
            f123(:,2),f124(:,2),f134(:,2),f234(:,2), ...
            centroid(:,2)];

        Z = [ ...
            p1(:,3),p2(:,3),p3(:,3),p4(:,3), ...
            m12(:,3),m13(:,3),m14(:,3), ...
            m23(:,3),m24(:,3),m34(:,3), ...
            f123(:,3),f124(:,3),f134(:,3),f234(:,3), ...
            centroid(:,3)];

    otherwise

        error('Unknown cfg.scoreQueryPattern="%s".', ...
            cfg.scoreQueryPattern);

end

% Column-major ordering:
% all elements at query point 1, then all elements at query point 2, ...
values = Fscore(X(:),Y(:),Z(:));
values(~isfinite(values)) = 0;

values = reshape(values,m,[]);

scoreElem(first:last) = max(values,[],2);

end

end


function generation = element_generation_from_volume_comparison_3d(P,T,V0)
p1 = P(T(:,1),:);
p2 = P(T(:,2),:);
p3 = P(T(:,3),:);
p4 = P(T(:,4),:);

sixV = abs(dot( ...
    p2-p1,cross(p3-p1,p4-p1,2),2));
volume = sixV/6;

if any(volume<=0) || any(~isfinite(volume))
    error('Invalid tetrahedral volume in score-driven mesh.');
end

generation = max(0,round(log2(V0./volume)));

reconstructed = V0./(2.^generation);
relativeMismatch = abs(volume-reconstructed)./max(volume,realmin);

if max(relativeMismatch)>1e-8
    error('Mesh volumes are inconsistent with binary NVB generation.');
end
end


%% ========================================================================
% Strict standard-AFEM first crossings
% =========================================================================


function [final,history] = run_afem_to_tolerance_3d_fast_exact( ...
        P,T,nvbTag,boundaryMask,u,geom,solveInfo, ...
        initialTime,initialRelL2,GbcFun,reference,cfg,methodTag)
%RUN_AFEM_TO_TOLERANCE_3D_FAST_EXACT Continue AFEM until target or a cap.

boundaryMask = logical(boundaryMask(:));
cumulativeTime = double(initialTime);
correctionTime = 0;
cumRefineTime = 0.0;
cumAssemblyTime = 0.0;
cumSolveTime = 0.0;
currentRelL2 = double(initialRelL2);
currentSolveInfo = solveInfo;
stopReason = 'target_reached';

history = repmat(struct( ...
    'stage',0,'dof',0,'elements',0,'timeSec',0,'relL2',NaN, ...
    'estimator',NaN,'marked',0,'completionParents',0, ...
    'pcgRelRes',NaN,'pcgIter',NaN, ...
    'estimateMarkTimeSec',0,'refineTimeSec',0,'solveTimeSec',0, ...
    'incrementalCycleTimeSec',0),0,1);

history(end+1,1) = make_accu_afem_history_row( ...
    0,nnz(~boundaryMask),size(T,1),cumulativeTime,currentRelL2, ...
    NaN,0,0,currentSolveInfo.relres,currentSolveInfo.iter,0,0,0); %#ok<AGROW>

if currentRelL2 > cfg.targetRelativeL2
    stopReason = 'maximum_cycles';
end

for cycle = 1:cfg.afemMaxCycles
    if currentRelL2 <= cfg.targetRelativeL2
        stopReason = 'target_reached';
        break;
    end

    tEstimate = tic;
    eta2 = residual_jump_estimator_homogeneous_source( ...
        P,T,u,cfg.epsilon,geom,GbcFun);
    eta2(~isfinite(eta2) | eta2<0) = 0;
    etaTotal = sqrt(sum(eta2));
    [marked,~] = dorfler_ranked_marking_3d(eta2,cfg.theta);
    if isempty(marked)
        [~,id] = max(eta2);
        marked = id;
    end
    estimateMarkTime = toc(tEstimate);

    if isempty(marked) || all(eta2==0)
        stopReason = 'zero_estimator';
        break;
    end

    state = struct();
    state.elementGeneration = [];
    state.elementNewLeaf = false(size(T,1),1);
    if cfg.fastTrackBoundary
        state.nodeBoundary = boundaryMask;
    else
        state.nodeBoundary = [];
    end

    oldU = u;
    tRefine = tic;
    [Pnew,Tnew,tagNew,rinfo,state] = ...
        nvb_refine_conforming_3d_fast_exact( ...
            P,T,nvbTag,marked,cfg,state);
    refineTime = toc(tRefine);

    if size(Tnew,1)>cfg.maxElementsCase
        stopReason = 'maximum_elements';
        break;
    end
    if cfg.fastTrackBoundary
        boundaryNew = logical(state.nodeBoundary(:));
    else
        boundaryNew = find_boundary_nodes(Pnew);
    end
    dofNew = nnz(~boundaryNew);
    if dofNew>cfg.maxFreeDOF
        stopReason = 'maximum_dof';
        break;
    end

    if cfg.useWarmStart
        uGuess = prolongate_p1_after_nvb(oldU,size(Pnew,1),rinfo);
    else
        uGuess = [];
    end

    tSolve = tic;
    [uNew,solveInfoNew,geomNew] = ...
        solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
            Pnew,Tnew,GbcFun,cfg.epsilon,cfg,uGuess,boundaryNew);
    solveTime = toc(tSolve);
    cycleAssemblyTime = safe_info_field( ...
        solveInfoNew,'assemblyTime');
    cycleLinearSolveTime = safe_info_field( ...
        solveInfoNew,'linearSolveTime');
    cumRefineTime = cumRefineTime + refineTime;
    cumAssemblyTime = cumAssemblyTime + cycleAssemblyTime;
    cumSolveTime = cumSolveTime + cycleLinearSolveTime;

    incrementalTime = estimateMarkTime + refineTime + solveTime;
    correctionTime = correctionTime + incrementalTime;
    cumulativeTime = double(initialTime) + correctionTime;

    relL2New = relative_l2_error_against_reference( ...
        Pnew,Tnew,uNew,reference,geomNew);
    completionParents = max( ...
        0,rinfo.totalBisectedParents-numel(marked));

    history(end+1,1) = make_accu_afem_history_row( ...
        cycle,dofNew,size(Tnew,1),cumulativeTime,relL2New, ...
        etaTotal,numel(marked),completionParents, ...
        solveInfoNew.relres,solveInfoNew.iter, ...
        estimateMarkTime,refineTime,solveTime); %#ok<AGROW>

    fprintf(['  %s stage %02d: DOF=%d, elems=%d, relL2=%.4e, ', ...
             'eta=%.3e, time=%.3f s\n'], ...
        methodTag,cycle,dofNew,size(Tnew,1),relL2New, ...
        etaTotal,cumulativeTime);

    P = Pnew;
    T = Tnew;
    nvbTag = tagNew;
    boundaryMask = boundaryNew;
    u = uNew;
    geom = geomNew;
    currentSolveInfo = solveInfoNew;
    currentRelL2 = relL2New;

    if currentRelL2 <= cfg.targetRelativeL2
        stopReason = 'target_reached';
        break;
    end
end

final = struct();
final.P = P;
final.T = T;
final.nvbTag = nvbTag;
final.boundaryMask = boundaryMask;
final.u = u;
final.geom = geom;
final.solveInfo = currentSolveInfo;
final.dof = nnz(~boundaryMask);
final.elements = size(T,1);
final.relL2 = currentRelL2;
final.timeSec = cumulativeTime;
final.seedTimeSec = double(initialTime);
final.correctionTimeSec = correctionTime;
final.correctionCycles = numel(history)-1;
final.refineTimeSec = cumRefineTime;
final.assemblyTimeSec = cumAssemblyTime;
final.solveTimeSec = cumSolveTime;
final.reachedTarget = currentRelL2 <= cfg.targetRelativeL2;
final.stopReason = stopReason;
final.methodTag = methodTag;

if ~final.reachedTarget
    fprintf('  %s stopped without reaching %.3e: %s\n', ...
        methodTag,cfg.targetRelativeL2,stopReason);
end
end


function H = make_accu_afem_history_row( ...
        stage,dof,elements,timeSec,relL2,estimator,marked, ...
        completionParents,pcgRelRes,pcgIter, ...
        estimateMarkTimeSec,refineTimeSec,solveTimeSec)
H = struct();
H.stage = stage;
H.dof = dof;
H.elements = elements;
H.timeSec = timeSec;
H.relL2 = relL2;
H.estimator = estimator;
H.marked = marked;
H.completionParents = completionParents;
H.pcgRelRes = pcgRelRes;
H.pcgIter = pcgIter;
H.estimateMarkTimeSec = estimateMarkTimeSec;
H.refineTimeSec = refineTimeSec;
H.solveTimeSec = solveTimeSec;
H.incrementalCycleTimeSec = ...
    estimateMarkTimeSec + refineTimeSec + solveTimeSec;
end


function mode = normalize_linear_initialization_mode(modeInput)
%NORMALIZE_LINEAR_INITIALIZATION_MODE Canonicalize the three public modes.

if ~(ischar(modeInput) || (isstring(modeInput) && isscalar(modeInput)))
    error(['cfg.newtonInitializationMode must be a character vector ', ...
           'or scalar string.']);
end

token = lower(strtrim(char(string(modeInput))));
token = strrep(token,'-','_');
token = strrep(token,' ','_');

switch token
    case {'zero','0','zero_initialization','zero_init'}
        mode = 'zero';

    case {'initial','initial_value','initial_condition', ...
          'initial_condition_extension','boundary_extension', ...
          'dirichlet_extension'}
        mode = 'initial_condition';

    case {'interpolation','interpolated','interpolate', ...
          'coarse_interpolation','coarse_solution_interpolation', ...
          'coarse_warm_start','warm_start'}
        mode = 'coarse_interpolation';

    otherwise
        error(['Unsupported cfg.newtonInitializationMode="%s". ', ...
               'Use ''zero'', ''initial_condition'', or ', ...
               '''coarse_interpolation''.'], ...
            char(string(modeInput)));
end
end


function uInitial = make_reaction_diffusion_boundary_extension(P,GbcFun)
%MAKE_REACTION_DIFFUSION_BOUNDARY_EXTENSION
% Elliptic analogue of the Burgers initial-condition extension:
%
%       u^(0)(x,y,z) = (1-z) g_D(x,y).
%
% It matches the random bottom-face Dirichlet data and vanishes on z=1.
% The prescribed field already vanishes on the four vertical edges because
% g_D contains the factor 16*x*(1-x)*y*(1-y). Boundary values are still
% overwritten exactly inside the FEM solver.

P = double(P);
uInitial = (1-P(:,3)).*GbcFun(P(:,1),P(:,2));
uInitial(~isfinite(uInitial)) = 0;

boundary = find_boundary_nodes(P);
tol = 100*eps;
bottom = boundary & abs(P(:,3))<tol;
otherBoundary = boundary & ~bottom;

uInitial(bottom) = GbcFun(P(bottom,1),P(bottom,2));
uInitial(otherBoundary) = 0;
end


function [uInitial,info] = ...
    make_reaction_diffusion_linear_initial_guess( ...
        P,GbcFun,initializationContext)
%MAKE_REACTION_DIFFUSION_LINEAR_INITIAL_GUESS
% Construct the requested free-node PCG initial vector on an arbitrary mesh.

mode = normalize_linear_initialization_mode(initializationContext.mode);

info = struct();
info.requestedMode = mode;
info.timeSec = 0.0;
info.label = '';

switch mode

    case 'zero'
        % Passing [] preserves the existing solver behavior: zero on free
        % nodes and exact nonhomogeneous Dirichlet data on boundary nodes.
        uInitial = [];
        info.label = 'zero';

    case 'initial_condition'
        tInit = tic;
        uInitial = make_reaction_diffusion_boundary_extension(P,GbcFun);
        info.timeSec = toc(tInit);
        info.label = 'initial_condition_boundary_extension';

    case 'coarse_interpolation'
        required = {'coarseP','coarseT','coarseU','coarseStageTimeSec'};
        for k = 1:numel(required)
            if ~isfield(initializationContext,required{k}) || ...
                    isempty(initializationContext.(required{k}))
                error(['coarse_interpolation requires initializationContext.%s.'], ...
                    required{k});
            end
        end

        tInit = tic;
        uInitial = evaluate_tet_solution_dirichlet( ...
            initializationContext.coarseP, ...
            initializationContext.coarseT, ...
            initializationContext.coarseU, ...
            P,GbcFun);
        info.timeSec = toc(tInit);
        info.label = 'coarse_solution_interpolation';

        if any(~isfinite(uInitial))
            error(['Coarse-grid interpolation produced NaN/Inf on the ', ...
                   'target mesh.']);
        end

    otherwise
        error('Unsupported linear initialization mode "%s".',mode);
end
end


function U = run_accuracy_matched_uniform_final_only( ...
        GbcFun,reference,cfg,initializationContext)
% Search is audited separately and excluded from reported MethodTimeSec.

cells = unique(round(double(cfg.uniformSearchCells(:).')),'stable');
searchHistory = repmat(struct( ...
    'cellsPerDirection',0,'freeDOF',0,'elements',0, ...
    'relL2',NaN,'wallTimeSec',0,'reachedTarget',false),0,1);

selectedCells = NaN;
searchTimer = tic;
lastValidCells = NaN;

for k = 1:numel(cells)
    n = cells(k);
    elementsExpected = 6*n^3;
    dofExpected = (n-1)^3;
    if elementsExpected>cfg.maxElementsCase || dofExpected>cfg.maxFreeDOF
        break;
    end

    candidateTimer = tic;
    [P,T] = make_structured_tetra_mesh(n);
    boundary = find_boundary_nodes(P);
    [uInitial,~] = make_reaction_diffusion_linear_initial_guess( ...
        P,GbcFun,initializationContext);
    [u,~,geom] = solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
        P,T,GbcFun,cfg.epsilon,cfg,uInitial,boundary);
    relL2 = relative_l2_error_against_reference( ...
        P,T,u,reference,geom);
    candidateWall = toc(candidateTimer);

    row = struct();
    row.cellsPerDirection = n;
    row.freeDOF = nnz(~boundary);
    row.elements = size(T,1);
    row.relL2 = relL2;
    row.wallTimeSec = candidateWall;
    row.reachedTarget = relL2<=cfg.targetRelativeL2;
    searchHistory(end+1,1) = row; %#ok<AGROW>
    lastValidCells = n;

    fprintf(['  Uniform search n=%d, DOF=%d, elems=%d, ', ...
             'relL2=%.4e (excluded %.3f s)\n'], ...
        n,row.freeDOF,row.elements,relL2,candidateWall);

    if row.reachedTarget
        selectedCells = n;
        break;
    end
end
searchWallTime = toc(searchTimer);

if ~isfinite(selectedCells)
    if ~isfinite(lastValidCells)
        error('No valid uniform search grid fits the configured safeguards.');
    end
    selectedCells = lastValidCells;
end

U = solve_uniform_once( ...
    selectedCells,GbcFun,reference,cfg,initializationContext,true);
U.searchWallTimeSec = searchWallTime;
U.searchHistory = searchHistory;
U.searchCandidatesTested = numel(searchHistory);
U.requestedInitializationMode = initializationContext.mode;
U.stopReason = ternary_string(U.reachedTarget, ...
    'target_reached','search_cap_without_target');
end


function U = run_fixed_uniform_once( ...
        GbcFun,reference,cfg,initializationContext)
n = resolve_fixed_uniform_cells(cfg);
U = solve_uniform_once( ...
    n,GbcFun,reference,cfg,initializationContext,true);
U.searchWallTimeSec = 0;
U.searchHistory = struct([]);
U.searchCandidatesTested = 0;
U.requestedInitializationMode = initializationContext.mode;
U.stopReason = ternary_string(U.reachedTarget, ...
    'target_reached','fixed_grid_above_target_error');
end


function U = solve_uniform_once( ...
        n,GbcFun,reference,cfg,initializationContext, ...
        includeSharedCoarseStageTime)
tMesh = tic;
[P,T] = make_structured_tetra_mesh(n);
meshTime = toc(tMesh);
boundary = find_boundary_nodes(P);

[uInitial,initializationInfo] = ...
    make_reaction_diffusion_linear_initial_guess( ...
        P,GbcFun,initializationContext);

tSolve = tic;
[u,solveInfo,geom] = solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
    P,T,GbcFun,cfg.epsilon,cfg,uInitial,boundary);
solveTime = toc(tSolve);
solveInfo.initializationMode = initializationInfo.label;

relL2 = relative_l2_error_against_reference(P,T,u,reference,geom);

sharedCoarseStageTime = 0.0;
if includeSharedCoarseStageTime && ...
        strcmp(initializationContext.mode,'coarse_interpolation')
    sharedCoarseStageTime = ...
        initializationContext.coarseStageTimeSec;
end

U = struct();
U.P = P;
U.T = T;
U.u = u;
U.geom = geom;
U.solveInfo = solveInfo;
U.cellsPerDirection = n;
U.dof = nnz(~boundary);
U.elements = size(T,1);
U.relL2 = relL2;
U.timeSec = sharedCoarseStageTime + ...
    meshTime + initializationInfo.timeSec + solveTime;
U.meshTimeSec = meshTime;
U.initializationMode = initializationInfo.label;
U.initializationTimeSec = initializationInfo.timeSec;
U.initialGuess = materialize_reaction_diffusion_initial_guess( ...
    P,GbcFun,uInitial);
U.coarseStageTimeSec = sharedCoarseStageTime;
U.solveTimeSec = solveTime;
U.reachedTarget = relL2<=cfg.targetRelativeL2;
U.correctionCycles = 0;
end


function s = ternary_string(condition,a,b)
if condition
    s = a;
else
    s = b;
end
end


function row = make_accu_method_row( ...
        caseNo,testId,datasetIndex,attemptId,method,seedType, ...
        targetRelL2,F,meshTime,solveTime,searchWallTime, ...
        meshTotalTime,assemblyTotalTime,solveTotalTime)
row = { ...
    caseNo,testId,datasetIndex,attemptId,method,seedType, ...
    targetRelL2,logical(F.reachedTarget),F.stopReason, ...
    F.correctionCycles,F.dof,F.elements,F.relL2,F.timeSec, ...
    searchWallTime,meshTime,solveTime, ...
    safe_info_field(F.solveInfo,'assemblyTime'), ...
    safe_info_field(F.solveInfo,'linearSolveTime'), ...
    safe_info_field(F.solveInfo,'relres'), ...
    safe_info_field(F.solveInfo,'iter'), ...
    meshTotalTime,assemblyTotalTime,solveTotalTime};
end


function [meshTotal,asmTotal,solveTotal] = seeded_afem_timing_totals( ...
        seedMeshTime,seedSolveInfo,afemFinal,coarseSolveInfo)
% Cumulative wall-clock buckets for score/AFEM methods:
% mesh = score-mesh build + AFEM NVB refinement; assembly and linear solve
% include the optional common coarse solve, the seed solve and every AFEM
% correction cycle.
    if isempty(coarseSolveInfo) || ~isstruct(coarseSolveInfo)
        coarseAsm = 0.0;
        coarseSol = 0.0;
    else
        coarseAsm = safe_info_field(coarseSolveInfo,'assemblyTime');
        coarseSol = safe_info_field(coarseSolveInfo,'linearSolveTime');
    end
    meshTotal = seedMeshTime + ...
        safe_info_field(afemFinal,'refineTimeSec');
    asmTotal = coarseAsm + ...
        safe_info_field(seedSolveInfo,'assemblyTime') + ...
        safe_info_field(afemFinal,'assemblyTimeSec');
    solveTotal = coarseSol + ...
        safe_info_field(seedSolveInfo,'linearSolveTime') + ...
        safe_info_field(afemFinal,'solveTimeSec');
end


function row = make_uniform_method_row( ...
        caseNo,testId,datasetIndex,attemptId,method,seedType, ...
        targetRelL2,U, ...
        meshTotalTime,assemblyTotalTime,solveTotalTime)
row = { ...
    caseNo,testId,datasetIndex,attemptId,method,seedType, ...
    targetRelL2,logical(U.reachedTarget),U.stopReason,0, ...
    U.dof,U.elements,U.relL2,U.timeSec,U.searchWallTimeSec, ...
    U.meshTimeSec,U.solveTimeSec, ...
    safe_info_field(U.solveInfo,'assemblyTime'), ...
    safe_info_field(U.solveInfo,'linearSolveTime'), ...
    safe_info_field(U.solveInfo,'relres'), ...
    safe_info_field(U.solveInfo,'iter'), ...
    meshTotalTime,assemblyTotalTime,solveTotalTime};
end


function [meshTotal,asmTotal,solveTotal] = uniform_timing_totals( ...
        U,coarseSolveInfo)
% Uniform rows have no AFEM corrections; the coarse solve is the only extra
% bucket when coarse_interpolation is enabled.
    if isempty(coarseSolveInfo) || ~isstruct(coarseSolveInfo)
        coarseAsm = 0.0;
        coarseSol = 0.0;
    else
        coarseAsm = safe_info_field(coarseSolveInfo,'assemblyTime');
        coarseSol = safe_info_field(coarseSolveInfo,'linearSolveTime');
    end
    meshTotal = U.meshTimeSec;
    asmTotal = coarseAsm + ...
        safe_info_field(U.solveInfo,'assemblyTime');
    solveTotal = coarseSol + ...
        safe_info_field(U.solveInfo,'linearSolveTime');
end


function uFull = materialize_reaction_diffusion_initial_guess( ...
        P,GbcFun,uInitial)
%MATERIALIZE_REACTION_DIFFUSION_INITIAL_GUESS
% Convert [] or a supplied free-node starting field into a full nodal vector
% with the exact Dirichlet data imposed. This is used only for saving and
% plotting; it does not alter the numerical solve.

nNode = size(P,1);
if isempty(uInitial) || numel(uInitial)~=nNode
    uFull = zeros(nNode,1);
else
    uFull = double(uInitial(:));
    uFull(~isfinite(uFull)) = 0;
end

boundary = find_boundary_nodes(P);
tol = 100*eps;
bottom = boundary & abs(P(:,3))<tol;
otherBoundary = boundary & ~bottom;

uFull(bottom) = GbcFun(P(bottom,1),P(bottom,2));
uFull(otherBoundary) = 0;
end


function R = make_visualization_adaptive_state( ...
        P,T,u,nvbTag,boundaryMask,initialGuess,solveInfo,relL2)
%MAKE_VISUALIZATION_ADAPTIVE_STATE Store only plotting-relevant arrays.

R = struct();
R.P = P;
R.T = T;
R.u = u;
R.nvbTag = nvbTag;
R.boundaryMask = logical(boundaryMask(:));
R.initialGuess = initialGuess;
R.solveInfo = solveInfo;
R.relL2 = relL2;
R.freeDOF = nnz(~R.boundaryMask);
R.elements = size(T,1);
end


function R = make_visualization_uniform_state(U)
%MAKE_VISUALIZATION_UNIFORM_STATE
% A structured uniform mesh is exactly determined by cellsPerDirection.
% Saving the enormous tetrahedron-connectivity array is unnecessary.
% Store nodal fields as (n+1)^3 arrays, from which slices and the structured
% grid can be reconstructed directly.

n = U.cellsPerDirection;
expectedNodes = (n+1)^3;
if numel(U.u)~=expectedNodes
    error('Uniform solution size is incompatible with cellsPerDirection.');
end

R = struct();
R.meshType = 'structured_Kuhn_tetrahedralization_unit_cube';
R.cellsPerDirection = n;
R.x = linspace(0,1,n+1);
R.uGrid = reshape(U.u,n+1,n+1,n+1);

if isfield(U,'initialGuess') && numel(U.initialGuess)==expectedNodes
    R.initialGuessGrid = ...
        reshape(U.initialGuess,n+1,n+1,n+1);
else
    R.initialGuessGrid = [];
end

R.solveInfo = U.solveInfo;
R.relL2 = U.relL2;
R.freeDOF = U.dof;
R.elements = U.elements;
R.methodTimeSec = U.timeSec;
R.meshTimeSec = U.meshTimeSec;
R.initializationMode = U.initializationMode;
R.initializationTimeSec = U.initializationTimeSec;
R.coarseStageTimeSec = U.coarseStageTimeSec;
R.solveTimeSec = U.solveTimeSec;
R.reachedTarget = U.reachedTarget;

if isfield(U,'searchWallTimeSec')
    R.searchWallTimeSec = U.searchWallTimeSec;
end
if isfield(U,'searchHistory')
    R.searchHistory = U.searchHistory;
end
end


function C = compact_method_result(S)
% Remove mesh-sized arrays unless cfg.saveSelectedMeshes requested them.
C = S;
heavy = {'P','T','u','geom','nvbTag','boundaryMask','initialGuess'};
for k = 1:numel(heavy)
    if isfield(C,heavy{k})
        C = rmfield(C,heavy{k});
    end
end
end


function value = safe_info_field(S,name)
if isstruct(S) && isfield(S,name)
    value = double(S.(name));
else
    value = NaN;
end
end


function T = build_accu_method_statistics(summaryTable)
methodNames = unique(summaryTable.Method,'stable');
n = numel(methodNames);

Method = cell(n,1);
Samples = zeros(n,1);
SuccessRate = zeros(n,1);
MeanMethodTimeSec = zeros(n,1);
StdMethodTimeSec = zeros(n,1);
MedianMethodTimeSec = zeros(n,1);
MeanSearchWallTimeSec = zeros(n,1);
MeanRelL2 = zeros(n,1);
MedianRelL2 = zeros(n,1);
BestRelL2 = zeros(n,1);
WorstRelL2 = zeros(n,1);
MeanFreeDOF = zeros(n,1);
StdFreeDOF = zeros(n,1);
MeanCorrectionCycles = zeros(n,1);
MeanMeshTimeSec = zeros(n,1);
MeanAssemblyTimeSec = zeros(n,1);
MeanSolveTimeSec = zeros(n,1);
MeanOtherTimeSec = NaN(n,1);

for i = 1:n
    Method{i} = methodNames{i};
    mask = strcmp(summaryTable.Method,methodNames{i});
    t = double(summaryTable.MethodTimeSec(mask));
    ts = double(summaryTable.SearchWallTimeSec(mask));
    e = double(summaryTable.RelL2(mask));
    d = double(summaryTable.FreeDOF(mask));
    c = double(summaryTable.CorrectionCycles(mask));
    tm = double(summaryTable.MeshTotalTimeSec(mask));
    ta = double(summaryTable.AssemblyTotalTimeSec(mask));
    ts = double(summaryTable.SolveTotalTimeSec(mask));
    ok = double(summaryTable.ReachedTarget(mask));

    Samples(i) = nnz(mask);
    SuccessRate(i) = mean(ok,'omitnan');
    MeanMethodTimeSec(i) = mean(t,'omitnan');
    StdMethodTimeSec(i) = std(t,0,'omitnan');
    MedianMethodTimeSec(i) = median(t,'omitnan');
    MeanSearchWallTimeSec(i) = mean(ts,'omitnan');
    MeanRelL2(i) = mean(e,'omitnan');
    MedianRelL2(i) = median(e,'omitnan');
    BestRelL2(i) = min(e,[],'omitnan');
    WorstRelL2(i) = max(e,[],'omitnan');
    MeanFreeDOF(i) = mean(d,'omitnan');
    StdFreeDOF(i) = std(d,0,'omitnan');
    MeanCorrectionCycles(i) = mean(c,'omitnan');
    MeanMeshTimeSec(i) = mean(tm,'omitnan');
    MeanAssemblyTimeSec(i) = mean(ta,'omitnan');
    MeanSolveTimeSec(i) = mean(ts,'omitnan');
end

for i = 1:n
    if isfinite(MeanMethodTimeSec(i)) && isfinite(MeanMeshTimeSec(i))
        MeanOtherTimeSec(i) = MeanMethodTimeSec(i) - ...
            MeanMeshTimeSec(i) - ...
            MeanAssemblyTimeSec(i) - ...
            MeanSolveTimeSec(i);
    end
end

T = table(Method,Samples,SuccessRate, ...
    MeanMethodTimeSec,StdMethodTimeSec,MedianMethodTimeSec, ...
    MeanSearchWallTimeSec,MeanRelL2,MedianRelL2, ...
    BestRelL2,WorstRelL2,MeanFreeDOF,StdFreeDOF, ...
    MeanCorrectionCycles,MeanMeshTimeSec,MeanAssemblyTimeSec, ...
    MeanSolveTimeSec,MeanOtherTimeSec);
end


function plot_accu_method_statistics(T,cfg)
fig = figure('Color','w','Visible',cfg.figureVisible, ...
    'Position',[80,80,1500,820]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
names = T.Method;
idx = 1:height(T);

nexttile;
bar(idx,T.MeanRelL2);
set(gca,'YScale','log','XTick',idx,'XTickLabel',names);
xtickangle(20); grid on; ylabel('mean relative L^2');
title('Accuracy');

nexttile;
bar(idx,T.MeanMethodTimeSec);
set(gca,'XTick',idx,'XTickLabel',names);
xtickangle(20); grid on; ylabel('reported online time (s)');
title('Method time');

nexttile;
bar(idx,T.MeanFreeDOF);
set(gca,'XTick',idx,'XTickLabel',names);
xtickangle(20); grid on; ylabel('mean free DOF');
title('Degrees of freedom');

nexttile;
bar(idx,T.SuccessRate);
set(gca,'XTick',idx,'XTickLabel',names,'YLim',[0,1]);
xtickangle(20); grid on; ylabel('success rate');
title(sprintf('Target relative L^2 = %.1e',cfg.targetRelativeL2));

save_figure(fig,fullfile(cfg.figureDir, ...
    'reaction_diffusion_accu_method_statistics.png'), ...
    cfg.figureResolution);
close(fig);
end




%% ========================================================================
% Common GRF, reference, P1 FEM, estimator and Maubach-NVB kernels.
% =========================================================================

function g = render_grf_on_periodic_grid_2d(N,grf)

if N <= 2*grf.K+1
    error('Need N > 2*K+1. N=%d, K=%d.',N,grf.K);
end

modes = grf.modes;

ip = mod(modes(:,1),N)+1;
jp = mod(modes(:,2),N)+1;

im = mod(-modes(:,1),N)+1;
jm = mod(-modes(:,2),N)+1;

idxPlus = sub2ind([N,N],ip,jp);
idxMinus = sub2ind([N,N],im,jm);

fftScale = N^2;

coeffPlus = ...
    0.5*fftScale .* ...
    grf.spectralStd .* ...
    (grf.xiCos - 1i*grf.xiSin);

Fhat = complex(zeros(N,N));

Fhat(idxPlus) = coeffPlus;
Fhat(idxMinus) = conj(coeffPlus);

g = real(ifft2(Fhat));

end


%% ========================================================================
% Build bottom-face boundary interpolant
% ========================================================================

function [GbcFun,info] = build_boundary_grf_interpolant( ...
    sample,N,boundaryMultiplier)

gPer = render_grf_on_periodic_grid_2d(N,sample.grf);

% Periodically extend from [0,1)^2 to [0,1]^2.
gExt = zeros(N+1,N+1);
gExt(1:N,1:N) = gPer;
gExt(N+1,1:N) = gPer(1,:);
gExt(1:N,N+1) = gPer(:,1);
gExt(N+1,N+1) = gPer(1,1);

x = (0:N)'/N;
[X,Y] = ndgrid(x,x);

W = 16*X.*(1-X).*Y.*(1-Y);

gBoundary = boundaryMultiplier * W .* gExt;

GbcFun = griddedInterpolant( ...
    {x,x}, ...
    gBoundary, ...
    'linear', ...
    'nearest');

GrawFun = griddedInterpolant( ...
    {x,x}, ...
    gExt, ...
    'linear', ...
    'nearest');

edgeMask = ...
    (X==0) | (X==1) | ...
    (Y==0) | (Y==1);

info = struct();
info.N = N;
info.rawMean = mean(gPer(:));
info.rawStd = std(gPer(:));
info.rawMaxAbs = max(abs(gPer(:)));
info.boundaryMean = mean(gBoundary(:));
info.boundaryStd = std(gBoundary(:));
info.boundaryMaxAbs = max(abs(gBoundary(:)));
info.edgeMaxAbs = max(abs(gBoundary(edgeMask)));
info.rawInterpolant = GrawFun;

end


function values = evaluate_periodic_grf_from_interpolant_parts(F,X,Y)

values = F(X,Y);

end


%% ========================================================================
% High-resolution sine-spectral reference
% ========================================================================

function ref = solve_boundary_reference_sine_spectral( ...
    GbcFun,epsilon,Nxy,Nz,zPower)

if Nxy < 8
    error('referenceCellsXY is too small.');
end

if Nz < 3
    error('referenceZPoints must be at least 3.');
end

n = Nxy-1;

xInt = (1:n)'/Nxy;
[X,Y] = ndgrid(xInt,xInt);

gInterior = GbcFun(X,Y);

% DST-I coefficients.
% If H = DST2(gInterior), then the sine-series coefficients are
%
%     b_mn ~= (2/Nxy)^2 * H_mn.
%
% Since DST-I is symmetric, reconstructing nodal values from b_mn is
% simply DST2(b_mn).
gHat = dst2_type1(gInterior);
bCoeff = (2/Nxy)^2 * gHat;

[m1,m2] = ndgrid(1:n,1:n);

mu = sqrt( ...
    epsilon^(-2) + ...
    pi^2*(m1.^2 + m2.^2));

% z grid strongly clustered near the random Dirichlet face z=0.
t = linspace(0,1,Nz)';
z = t.^zPower;

xFull = (0:Nxy)'/Nxy;

uFull = zeros(Nxy+1,Nxy+1,Nz);

maxAbsU = 0;

for iz = 1:Nz

    zz = z(iz);

    if iz == 1

        uInterior = gInterior;

    elseif iz == Nz

        uInterior = zeros(n,n);

    else

        % Stable equivalent of
        %
        %   sinh(mu*(1-z))/sinh(mu).
        %
        % Avoids overflow for epsilon << 1.
        zFactor = ...
            exp(-mu*zz) .* ...
            (1-exp(-2*mu*(1-zz))) ./ ...
            (1-exp(-2*mu));

        modal = bCoeff .* zFactor;

        uInterior = dst2_type1(modal);

    end

    uFull(2:Nxy,2:Nxy,iz) = uInterior;

    maxAbsU = max(maxAbsU,max(abs(uInterior(:))));

end

F = griddedInterpolant( ...
    {xFull,xFull,z}, ...
    uFull, ...
    'linear', ...
    'nearest');

ref = struct();
ref.Nxy = Nxy;
ref.Nz = Nz;
ref.x = xFull;
ref.z = z;
ref.F = F;
ref.maxAbsU = maxAbsU;

end


%% ========================================================================
% 2-D DST-I
% ========================================================================

function Y = dst2_type1(X)

Y = dst1_along_dimension(X,1);
Y = dst1_along_dimension(Y,2);

end


function Y = dst1_along_dimension(X,dim)

nd = ndims(X);
perm = [dim,1:dim-1,dim+1:nd];
Xp = permute(X,perm);
sz = size(Xp);
n = sz(1);
X2 = reshape(Xp,n,[]);
m = size(X2,2);

Z = zeros(2*(n+1),m,'like',X2);
Z(2:n+1,:) = X2;
Z(n+3:end,:) = -flipud(X2);

F = fft(Z,[],1);
Y2 = -imag(F(2:n+1,:))/2;

Yp = reshape(Y2,sz);
Y = ipermute(Yp,perm);

end


%% ========================================================================
% Structured cube -> 6-tetrahedron mesh
% ========================================================================

function [P,T,nvbTag] = make_structured_tetra_mesh(n)
%MAKE_STRUCTURED_TETRA_MESH
% Structured Whitney-Tucker/Kuhn triangulation of the unit cube.
%
% The six tetrahedra in each Cartesian cube are the six Kuhn simplices.
% Their vertices can be 4-colored by
%
%     c(i,j,k) = mod(i+j+k,4),
%
% where i,j,k are integer lattice coordinates.  Every tetrahedron contains
% exactly the four colors 0,1,2,3.  We sort each tetrahedron by this color
% and initialize the Maubach tag gamma=3.  Thus the initial bisection edge
% is [v0,v3], exactly as in the 3-D Maubach NVB rule.

x = linspace(0,1,n+1);
[X,Y,Z] = ndgrid(x,x,x);
P = [X(:),Y(:),Z(:)];

node = reshape(1:(n+1)^3,n+1,n+1,n+1);

T = zeros(6*n^3,4);
row = 0;

for i = 1:n
    for j = 1:n
        for k = 1:n

            v000 = node(i  ,j  ,k  );
            v100 = node(i+1,j  ,k  );
            v010 = node(i  ,j+1,k  );
            v110 = node(i+1,j+1,k  );

            v001 = node(i  ,j  ,k+1);
            v101 = node(i+1,j  ,k+1);
            v011 = node(i  ,j+1,k+1);
            v111 = node(i+1,j+1,k+1);

            localT = [ ...
                v000 v100 v110 v111;
                v000 v110 v010 v111;
                v000 v010 v011 v111;
                v000 v011 v001 v111;
                v000 v001 v101 v111;
                v000 v101 v100 v111];

            T(row+(1:6),:) = localT;
            row = row+6;

        end
    end
end

% Integer lattice coordinates of all vertices.
ijk = round(n*P);
vertexColor = mod(sum(ijk,2),4);

% Every Kuhn tetrahedron must contain all four colors exactly once.
tetColor = vertexColor(T);
sortedColor = sort(tetColor,2);

expected = repmat(0:3,size(T,1),1);
if any(sortedColor(:) ~= expected(:))
    error(['The structured tetrahedral mesh is not 4-colorable by ', ...
           'mod(i+j+k,4); Maubach-NVB initialization failed.']);
end

% Reorder vertices of each tetrahedron by ascending color 0,1,2,3.
[~,perm] = sort(tetColor,2);
rows = repmat((1:size(T,1))',1,4);
idx = sub2ind(size(T),rows,perm);
T = T(idx);

% Maubach tag gamma in {1,2,3}; initial tag is gamma=3.
nvbTag = uint8(3*ones(size(T,1),1));

end


%% ========================================================================
% Boundary nodes
% ========================================================================

function boundary = find_boundary_nodes(P)

tol = 100*eps;

boundary = ...
    abs(P(:,1)) < tol | ...
    abs(P(:,1)-1) < tol | ...
    abs(P(:,2)) < tol | ...
    abs(P(:,2)-1) < tol | ...
    abs(P(:,3)) < tol | ...
    abs(P(:,3)-1) < tol;

end


%% ========================================================================
% Tetrahedron geometry
% ========================================================================

function geom = tetra_geometry(P,T)

p1 = P(T(:,1),:);
p2 = P(T(:,2),:);
p3 = P(T(:,3),:);
p4 = P(T(:,4),:);

e1 = p2-p1;
e2 = p3-p1;
e3 = p4-p1;

c23 = cross(e2,e3,2);
sixVSigned = dot(e1,c23,2);
volume = abs(sixVSigned)/6;

if any(volume <= 1.0e-16)
    error('Degenerate tetrahedron detected.');
end

grad = zeros(size(T,1),4,3);

g2 = c23 ./ sixVSigned;
g3 = cross(e3,e1,2) ./ sixVSigned;
g4 = cross(e1,e2,2) ./ sixVSigned;
g1 = -(g2+g3+g4);

grad(:,1,:) = g1;
grad(:,2,:) = g2;
grad(:,3,:) = g3;
grad(:,4,:) = g4;

edgeLength = zeros(size(T,1),6);

edgeLength(:,1) = vecnorm(p2-p1,2,2);
edgeLength(:,2) = vecnorm(p3-p1,2,2);
edgeLength(:,3) = vecnorm(p4-p1,2,2);
edgeLength(:,4) = vecnorm(p3-p2,2,2);
edgeLength(:,5) = vecnorm(p4-p2,2,2);
edgeLength(:,6) = vecnorm(p4-p3,2,2);

hK = max(edgeLength,[],2);

geom = struct();
geom.volume = volume;
geom.grad = grad;
geom.hK = hK;
geom.p1 = p1;
geom.p2 = p2;
geom.p3 = p3;
geom.p4 = p4;

end


%% ========================================================================
% P1 FEM assembly and solve with nonhomogeneous Dirichlet data
% ========================================================================

function [u,info,geom] = solve_p1_reaction_diffusion_dirichlet_fast_exact( ...
    P,T,GbcFun,epsilon,cfg,uInitial,boundaryKnown)
%SOLVE_P1_REACTION_DIFFUSION_DIRICHLET
% Assemble and solve the SPD P1 system.
%
% Important performance point:
%   no inverse(A) is formed anywhere.
%
% The default 'auto' path first tries diagonal-preconditioned PCG.  For the
% present reaction-diffusion operator this is often enough and avoids the
% relatively expensive ICT-ICHOL construction.  The PCG stopping tolerance
% is exactly cfg.pcgTolerance, so this speed optimization does not loosen
% the requested algebraic accuracy.

if nargin < 6
    uInitial = [];
end
if nargin < 7
    boundaryKnown = [];
end

nNode = size(P,1);
nElem = size(T,1);

assemblyTimer = tic;

geom = tetra_geometry(P,T);
volume = geom.volume;
grad = geom.grad;

% -------------------------------------------------------------------------
% Matrix assembly: epsilon^2*K + M
% -------------------------------------------------------------------------
%
% P1 tetrahedral mass matrix:
%       M_K = |K|/20 * (ones(4) + eye(4)).
%
% We keep the assembly vectorized over elements.  Only the fixed 4x4 local
% index loops remain; there are no loops over tetrahedra.

nnzLocal = 16*nElem;
I = zeros(nnzLocal,1);
J = zeros(nnzLocal,1);
V = zeros(nnzLocal,1);

counter = 0;

for a = 1:4

    ga = squeeze(grad(:,a,:));

    for b = 1:4

        gb = squeeze(grad(:,b,:));

        stiffness = ...
            epsilon^2 .* volume .* sum(ga.*gb,2);

        if a == b
            mass = volume/10;
        else
            mass = volume/20;
        end

        ids = counter+(1:nElem);

        I(ids) = T(:,a);
        J(ids) = T(:,b);
        V(ids) = stiffness + mass;

        counter = counter+nElem;

    end
end

A = sparse(I,J,V,nNode,nNode);
clear I J V;

% -------------------------------------------------------------------------
% Dirichlet values
% -------------------------------------------------------------------------

tol = 100*eps;

if isempty(boundaryKnown)
    boundary = find_boundary_nodes(P);
else
    boundary = logical(boundaryKnown(:));
    if numel(boundary)~=nNode
        error('Provided boundary mask size mismatch.');
    end
end

free = find(~boundary);
bnd = find(boundary);

bottom = boundary & abs(P(:,3)) < tol;

u = zeros(nNode,1);

% Bottom face carries the random boundary field.
u(bottom) = GbcFun(P(bottom,1),P(bottom,2));

% f=0, hence only the Dirichlet lifting remains on the free RHS.
bf = -A(free,bnd)*u(bnd);
Aff = A(free,free);

assemblyTime = toc(assemblyTimer);

% -------------------------------------------------------------------------
% Warm start on free nodes
% -------------------------------------------------------------------------

if ~isempty(uInitial) && numel(uInitial)==nNode

    x0 = double(uInitial(free));
    x0(~isfinite(x0)) = 0;

else

    x0 = zeros(numel(free),1);

end

% -------------------------------------------------------------------------
% Linear solve
% -------------------------------------------------------------------------

[uf,linfo] = solve_spd_system_fast(Aff,bf,x0,cfg);

u(free) = uf;

info = linfo;
info.maxAbsDirichlet = max(abs(u(bnd)));
info.assemblyTime = assemblyTime;

end


function [x,info] = solve_spd_system_fast(A,b,x0,cfg)
%SOLVE_SPD_SYSTEM_FAST Fast robust solve for a sparse SPD matrix.
%
% Supported cfg.linearSolver:
%   auto        : cheap diagonal-PCG, fallback to ICT-ICHOL PCG
%   pcg-diag    : diagonal-PCG only
%   pcg-ichol   : ICT-ICHOL PCG
%   chol        : sparse Cholesky with SYMAMD ordering
%   backslash   : MATLAB sparse direct solve
%
% All iterative paths use cfg.pcgTolerance; no accuracy relaxation is made.

n = size(A,1);

info = struct();
info.flag = 0;
info.relres = 0;
info.iter = 0;
info.solverUsed = '';
info.preconditionerTime = 0;
info.linearSolveTime = 0;

if n == 0
    x = zeros(0,1);
    info.solverUsed = 'empty';
    return;
end

if nargin < 3 || isempty(x0) || numel(x0)~=n
    x0 = zeros(n,1);
else
    x0 = double(x0(:));
end

solver = lower(strtrim(cfg.linearSolver));

switch solver

    case 'auto'

        % -------------------------------------------------------------
        % First attempt: very cheap diagonal-preconditioned PCG.
        % -------------------------------------------------------------
        d = full(diag(A));
        d = max(d,realmin);
        Mdiag = spdiags(d,0,n,n);

        tSolve = tic;

        [x,flag,relres,iter] = pcg( ...
            A,b, ...
            cfg.pcgTolerance, ...
            min(cfg.pcgDiagMaxIteration,cfg.pcgMaxIteration), ...
            Mdiag,[],x0);

        firstTime = toc(tSolve);

        if flag == 0

            info.flag = flag;
            info.relres = relres;
            info.iter = iter;
            info.solverUsed = 'pcg-diag';
            info.linearSolveTime = firstTime;
            info.preconditionerTime = 0;
            return;

        end

        % -------------------------------------------------------------
        % Robust fallback: ICT-ICHOL PCG, still at the same tolerance.
        % -------------------------------------------------------------
        tPrec = tic;

        setup = struct();
        setup.type = 'ict';
        setup.droptol = cfg.icholDropTolerance;
        setup.diagcomp = cfg.icholDiagComp;

        try
            L = ichol(A,setup);
            precOK = true;
        catch
            precOK = false;
            L = [];
        end

        precTime = toc(tPrec);

        tSolve = tic;

        if precOK

            [x,flag,relres,iter2] = pcg( ...
                A,b, ...
                cfg.pcgTolerance, ...
                cfg.pcgMaxIteration, ...
                L,L',x);

            solverUsed = 'pcg-diag -> pcg-ichol';

        else

            [x,flag,relres,iter2] = pcg( ...
                A,b, ...
                cfg.pcgTolerance, ...
                cfg.pcgMaxIteration, ...
                [],[],x);

            solverUsed = 'pcg-diag -> pcg(no ichol)';

        end

        secondTime = toc(tSolve);

        info.flag = flag;
        info.relres = relres;
        info.iter = iter + iter2;
        info.solverUsed = solverUsed;
        info.preconditionerTime = precTime;
        info.linearSolveTime = firstTime + secondTime;

    case 'pcg-diag'

        d = full(diag(A));
        d = max(d,realmin);
        Mdiag = spdiags(d,0,n,n);

        tSolve = tic;

        [x,flag,relres,iter] = pcg( ...
            A,b, ...
            cfg.pcgTolerance, ...
            cfg.pcgMaxIteration, ...
            Mdiag,[],x0);

        info.linearSolveTime = toc(tSolve);
        info.flag = flag;
        info.relres = relres;
        info.iter = iter;
        info.solverUsed = 'pcg-diag';

    case 'pcg-ichol'

        tPrec = tic;

        setup = struct();
        setup.type = 'ict';
        setup.droptol = cfg.icholDropTolerance;
        setup.diagcomp = cfg.icholDiagComp;

        try
            L = ichol(A,setup);
            precOK = true;
        catch
            precOK = false;
            L = [];
        end

        info.preconditionerTime = toc(tPrec);

        tSolve = tic;

        if precOK

            [x,flag,relres,iter] = pcg( ...
                A,b, ...
                cfg.pcgTolerance, ...
                cfg.pcgMaxIteration, ...
                L,L',x0);

            info.solverUsed = 'pcg-ichol';

        else

            [x,flag,relres,iter] = pcg( ...
                A,b, ...
                cfg.pcgTolerance, ...
                cfg.pcgMaxIteration, ...
                [],[],x0);

            info.solverUsed = 'pcg(no ichol)';

        end

        info.linearSolveTime = toc(tSolve);
        info.flag = flag;
        info.relres = relres;
        info.iter = iter;

    case 'chol'

        tSolve = tic;

        % Symmetric approximate minimum degree ordering reduces fill-in.
        p = symamd(A);
        Ap = A(p,p);
        bp = b(p);

        try

            L = chol(Ap,'lower');
            yp = L\bp;
            xp = L'\yp;

            x = zeros(n,1);
            x(p) = xp;

            info.flag = 0;
            info.solverUsed = 'sparse-chol';

        catch

            % Safe direct fallback.
            x = A\b;
            info.flag = 0;
            info.solverUsed = 'backslash(chol fallback)';

        end

        info.linearSolveTime = toc(tSolve);
        info.relres = norm(A*x-b)/max(norm(b),eps);
        info.iter = 1;

    case 'backslash'

        tSolve = tic;
        x = A\b;
        info.linearSolveTime = toc(tSolve);

        info.flag = 0;
        info.relres = norm(A*x-b)/max(norm(b),eps);
        info.iter = 1;
        info.solverUsed = 'backslash';

    otherwise

        error('Unknown cfg.linearSolver="%s".',cfg.linearSolver);

end

if info.flag ~= 0
    warning( ...
        'Linear solve did not fully converge: flag=%d, relres=%.3e, iter=%d.', ...
        info.flag,info.relres,info.iter);
end

end


%% ========================================================================
% Complete robust estimator: residual + interior flux jump + Dirichlet data
% ========================================================================

function [eta2,parts] = residual_jump_estimator_homogeneous_source( ...
    P,T,u,epsilon,geom,GbcFun)

nElem = size(T,1);

volume = geom.volume;
gradBasis = geom.grad;
hK = geom.hK;

[Lq,wq] = tetra_quadrature_degree2();

uLocal = u(T);

% -------------------------------------------------------------------------
% Strong residual
%
% P1 -> Delta u_h = 0 elementwise, and f=0, so
%
%     r_K = -u_h.
% -------------------------------------------------------------------------

residualL2sq = zeros(nElem,1);

for q = 1:4

    uhq = uLocal*Lq(q,:).';
    rq = -uhq;

    residualL2sq = ...
        residualL2sq + ...
        volume*wq(q).*rq.^2;

end

alphaK = min(hK/epsilon,1);
eta2 = alphaK.^2 .* residualL2sq;

% -------------------------------------------------------------------------
% Element gradients
% -------------------------------------------------------------------------

gradU = zeros(nElem,3);

for a = 1:4

    ga = squeeze(gradBasis(:,a,:));
    gradU = gradU + uLocal(:,a).*ga;

end

% -------------------------------------------------------------------------
% Interior face jumps
% -------------------------------------------------------------------------

faces = [ ...
    T(:,[2 3 4]);
    T(:,[1 3 4]);
    T(:,[1 2 4]);
    T(:,[1 2 3])];

owner = repmat((1:nElem)',4,1);
facesSorted = sort(faces,2);

[uniqueFaces,~,ic] = unique(facesSorted,'rows');
counts = accumarray(ic,1);

[~,ord] = sort(ic);
ownerSorted = owner(ord);
starts = cumsum([1;counts(1:end-1)]);

interiorFaceID = find(counts==2);

if ~isempty(interiorFaceID)

    pos1 = starts(interiorFaceID);
    pos2 = pos1+1;

    elem1 = ownerSorted(pos1);
    elem2 = ownerSorted(pos2);

    faceNode = uniqueFaces(interiorFaceID,:);

    q1 = P(faceNode(:,1),:);
    q2 = P(faceNode(:,2),:);
    q3 = P(faceNode(:,3),:);

    normalRaw = cross(q2-q1,q3-q1,2);
    normalNorm = vecnorm(normalRaw,2,2);
    faceArea = 0.5*normalNorm;
    normalUnit = normalRaw ./ normalNorm;

    h12 = vecnorm(q2-q1,2,2);
    h13 = vecnorm(q3-q1,2,2);
    h23 = vecnorm(q3-q2,2,2);
    hF = max([h12,h13,h23],[],2);

    jumpFlux = ...
        epsilon^2 .* ...
        sum((gradU(elem1,:)-gradU(elem2,:)).*normalUnit,2);

    alphaF = min(hF/epsilon,1);

    faceContribution = ...
        (alphaF/epsilon) .* ...
        faceArea .* ...
        jumpFlux.^2;

    eta2 = eta2 + ...
        accumarray( ...
            [elem1;elem2], ...
            0.5*[faceContribution;faceContribution], ...
            [nElem,1], ...
            @sum, ...
            0);

end

% Preserve the classical element-residual plus interior-flux-jump part.
parts = struct();
parts.residualJump = eta2;

% Nonhomogeneous Dirichlet-data approximation on z=0.  The P1 trace uses
% I_h g_D, while GbcFun represents the prescribed continuous dataset input.
% The computable L2 form below is scaling-equivalent on shape-regular faces
% to epsilon^2*h_F*||grad_tau(g_D-I_h g_D)||^2.
[bottomFaces,bottomOwner] = extract_bottom_boundary_faces(P,T);
q1 = P(bottomFaces(:,1),1:2);
q2 = P(bottomFaces(:,2),1:2);
q3 = P(bottomFaces(:,3),1:2);

hF = max([ ...
    vecnorm(q2-q1,2,2), ...
    vecnorm(q3-q2,2,2), ...
    vecnorm(q1-q3,2,2)],[],2);
areaF = 0.5*abs( ...
    (q2(:,1)-q1(:,1)).*(q3(:,2)-q1(:,2)) - ...
    (q3(:,1)-q1(:,1)).*(q2(:,2)-q1(:,2)));

[baryF,wF] = triangle_quadrature_degree5();
g1 = GbcFun(q1(:,1),q1(:,2));
g2 = GbcFun(q2(:,1),q2(:,2));
g3 = GbcFun(q3(:,1),q3(:,2));
boundaryL2sq = zeros(size(bottomOwner));

for q = 1:size(baryF,1)
    L = baryF(q,:);
    xyq = L(1)*q1 + L(2)*q2 + L(3)*q3;
    gExact = GbcFun(xyq(:,1),xyq(:,2));
    gInterp = L(1)*g1 + L(2)*g2 + L(3)*g3;
    boundaryL2sq = boundaryL2sq + ...
        areaF*wF(q).*(gExact-gInterp).^2;
end

faceData = epsilon^2./max(hF,realmin).*boundaryL2sq;
parts.dirichletData = accumarray( ...
    bottomOwner,faceData,[nElem,1],@sum,0);
eta2 = eta2 + parts.dirichletData;

end


function [lambda,w] = triangle_quadrature_degree5()
% Seven-point Dunavant rule; weights sum to one on a physical triangle.
a1 = 1/3;
a2 = 0.059715871789770;
b2 = 0.470142064105115;
a3 = 0.797426985353087;
b3 = 0.101286507323456;
lambda = [ ...
    a1,a1,a1; ...
    a2,b2,b2; b2,a2,b2; b2,b2,a2; ...
    a3,b3,b3; b3,a3,b3; b3,b3,a3];
w = [ ...
    0.225000000000000; ...
    repmat(0.132394152788506,3,1); ...
    repmat(0.125939180544827,3,1)];
end


%% ========================================================================
% Degree-2 tetrahedron quadrature
% ========================================================================

function [Lq,wq] = tetra_quadrature_degree2()

a = 0.5854101966249685;
b = 0.1381966011250105;

Lq = [ ...
    a b b b;
    b a b b;
    b b a b;
    b b b a];

wq = 0.25*ones(4,1);

end


%% ========================================================================
% Relative L2 error against spectral reference
% ========================================================================

function relL2 = relative_l2_error_against_reference( ...
    P,T,u,reference,geom)

[Lq,wq] = tetra_quadrature_degree2();

uLocal = u(T);
err2 = 0;
ref2 = 0;

for q = 1:4

    Xq = ...
        Lq(q,1)*geom.p1 + ...
        Lq(q,2)*geom.p2 + ...
        Lq(q,3)*geom.p3 + ...
        Lq(q,4)*geom.p4;

    uref = reference.F(Xq(:,1),Xq(:,2),Xq(:,3));
    uh = uLocal*Lq(q,:).';

    err2 = err2 + ...
        sum(geom.volume*wq(q).*(uh-uref).^2);

    ref2 = ref2 + ...
        sum(geom.volume*wq(q).*uref.^2);

end

relL2 = sqrt(err2/max(ref2,eps));

end


%% ========================================================================
% Dörfler marking
% ========================================================================

function [marked,info] = dorfler_ranked_marking_3d(eta2,theta)

eta2 = double(eta2(:));
eta2(~isfinite(eta2) | eta2<0) = 0;

if ~(isfinite(theta) && theta>0 && theta<=1)
    error('Dorfler theta must satisfy 0 < theta <= 1.');
end

[values,order] = sort(eta2,'descend');
total = sum(values);

if total <= 0

    marked = order(1);
    info.total = total;
    info.captured = 0;
    info.capturedFraction = 0;
    info.largestFraction = 0;
    return;

end

m = find(cumsum(values) >= theta*total,1,'first');
marked = order(1:m);
captured = sum(values(1:m));

info.total = total;
info.captured = captured;
info.capturedFraction = captured/total;
info.largestFraction = values(1)/total;

end


%% ========================================================================
% Marked tetrahedra -> longest-edge bisection with edge closure
% ========================================================================

function [P,T,nvbTag,info,state] = ...
    nvb_refine_conforming_3d_fast_exact( ...
        P,T,nvbTag,marked,cfg,state)
%NVB_REFINE_CONFORMING_3D_FAST_EXACT
%
% SAME Maubach-NVB closure and SAME child ordering as the original kernel.
%
% Performance-only changes:
%   1. Build six packed uint64 edge-key vectors directly from node columns.
%      The old 6*N-by-2 double allEdges matrix is not materialized.
%   2. Owner IDs for matching edge occurrences are recovered from occurrence
%      positions instead of allocating repmat((1:N)',6,1).
%   3. Already-sorted bisection/pending edges are packed without sorting them
%      a second time.
%   4. Optional payload tracks element generation, new leaves and boundary
%      nodes with exactly the same split operations.
%
% NO in-place tetrahedron reordering is used. We intentionally keep
%
%       T = [T(keep,:); child1; child2]
%
% exactly as before, because changing element ordering could change tie
% ordering in later Dörfler sorts.

if nargin < 6 || isempty(state)
    state = struct();
end

marked = unique(round(marked(:)));
marked = marked(marked>=1 & marked<=size(T,1));

if numel(nvbTag) ~= size(T,1)
    error('nvbTag must have one entry per tetrahedron.');
end

if any(nvbTag<1 | nvbTag>3)
    error('Every 3-D Maubach NVB tag must lie in {1,2,3}.');
end

trackGeneration = ...
    isfield(state,'elementGeneration') && ...
    ~isempty(state.elementGeneration);

trackBoundary = ...
    isfield(state,'nodeBoundary') && ...
    ~isempty(state.nodeBoundary);

if trackGeneration && numel(state.elementGeneration)~=size(T,1)
    error('Tracked generation size mismatch.');
end

if trackBoundary && numel(state.nodeBoundary)~=size(P,1)
    error('Tracked boundary-mask size mismatch.');
end

% Always maintain a final-leaf birth mask. This is only bookkeeping.
state.elementNewLeaf = false(size(T,1),1);

info = struct();
info.requestedMarkedElements = numel(marked);
info.selectedEdges = 0;
info.actualBisectedEdges = 0;
info.totalBisectedParents = 0;
info.completionParents = 0;
info.completionSteps = 0;
info.newNodes = 0;
info.newElements = 0;
info.newNodeIds = zeros(0,1);
info.newNodeParents = zeros(0,2);

if isempty(marked)
    return;
end

nNodeBefore = size(P,1);
nElemBefore = size(T,1);

pending = unique( ...
    nvb_bisection_edges_3d(T(marked,:),nvbTag(marked)), ...
    'rows');

info.selectedEdges = size(pending,1);

substep = 0;

while ~isempty(pending)

    substep = substep+1;

    if substep > cfg.nvbMaxCompletionSteps
        error(['3-D Maubach-NVB completion exceeded %d steps. ', ...
               'The initial coloring/tagging or refinement kernel ', ...
               'has been modified inconsistently.'], ...
            cfg.nvbMaxCompletionSteps);
    end

    NT = size(T,1);

    % Current bisection edge of every tetrahedron. This routine already
    % returns each edge sorted.
    bseAll = nvb_bisection_edges_3d(T,nvbTag);
    bseKeyAll = nvb_edge_key_sorted_3d(bseAll);

    if cfg.fastPackedNVBEdges
        % EXACT same occurrence order as the old concatenated allEdges:
        % [12;13;14;23;24;34].
        allKeys = [ ...
            nvb_edge_key_pair_3d(T(:,1),T(:,2)); ...
            nvb_edge_key_pair_3d(T(:,1),T(:,3)); ...
            nvb_edge_key_pair_3d(T(:,1),T(:,4)); ...
            nvb_edge_key_pair_3d(T(:,2),T(:,3)); ...
            nvb_edge_key_pair_3d(T(:,2),T(:,4)); ...
            nvb_edge_key_pair_3d(T(:,3),T(:,4))];

        % Optional audit: reconstruct the old 6*N-by-2 edge table and prove
        % that the packed fast keys are identical, occurrence by occurrence.
        if isfield(cfg,'fastVerifyPackedNVBEdges') && ...
                cfg.fastVerifyPackedNVBEdges
            legacyEdges = [ ...
                T(:,[1 2]);
                T(:,[1 3]);
                T(:,[1 4]);
                T(:,[2 3]);
                T(:,[2 4]);
                T(:,[3 4])];
            legacyEdges = sort(legacyEdges,2);
            legacyKeys = nvb_edge_key_sorted_3d(legacyEdges);
            if ~isequal(allKeys,legacyKeys)
                error('Packed NVB edge keys differ from legacy edge keys.');
            end
        end
    else
        % Debug fallback reproducing the old materialized representation.
        allEdges = [ ...
            T(:,[1 2]);
            T(:,[1 3]);
            T(:,[1 4]);
            T(:,[2 3]);
            T(:,[2 4]);
            T(:,[3 4])];
        allEdges = sort(allEdges,2);
        allKeys = nvb_edge_key_sorted_3d(allEdges);
    end

    pendingKeys = nvb_edge_key_sorted_3d(pending);

    [hit,loc] = ismember(allKeys,pendingKeys);
    hitPos = find(hit);

    % allKeys is stacked in six blocks of length NT, exactly like the old
    % owners = repmat((1:NT)',6,1).
    ownerHit = mod(hitPos-1,NT)+1;
    locHit = loc(hit);

    nPending = size(pending,1);

    counts = accumarray( ...
        locHit,1,[nPending,1],@sum,0);

    exists = counts > 0;

    if any(~exists)
        pending = pending(exists,:);
        continue;
    end

    sameEdge = ...
        bseKeyAll(ownerHit) == pendingKeys(locHit);

    sameCounts = accumarray( ...
        locHit,double(sameEdge),[nPending,1],@sum,0);

    ready = (sameCounts == counts);

    if ~any(ready)
        badOwner = ownerHit(~sameEdge);
        dependencies = unique(bseAll(badOwner,:),'rows');
        pending = unique([pending;dependencies],'rows');
        continue;
    end

    readyEdges = pending(ready,:);
    readyKeys = nvb_edge_key_sorted_3d(readyEdges);

    [splitMask,edgeLocAll] = ismember(bseKeyAll,readyKeys);
    splitIds = find(splitMask);
    edgeLoc = edgeLocAll(splitMask);

    if isempty(splitIds)
        error('Internal NVB error: ready edges found but no parent tetrahedra.');
    end

    % One shared midpoint per ready edge -- unchanged.
    newPoints = ...
        0.5*(P(readyEdges(:,1),:) + P(readyEdges(:,2),:));

    firstId = size(P,1)+1;
    newIds = (firstId:firstId+size(newPoints,1)-1)';

    P = [P;newPoints]; %#ok<AGROW>

    if trackBoundary
        state.nodeBoundary = [ ...
            logical(state.nodeBoundary(:)); ...
            find_boundary_nodes(newPoints)]; %#ok<AGROW>
    end

    info.newNodeIds = [info.newNodeIds;newIds]; %#ok<AGROW>
    info.newNodeParents = [ ...
        info.newNodeParents;
        readyEdges]; %#ok<AGROW>

    mids = newIds(edgeLoc);

    parent = T(splitIds,:);
    parentTag = double(nvbTag(splitIds));

    child1 = zeros(numel(splitIds),4);
    child2 = zeros(numel(splitIds),4);

    ids = parentTag==3;
    if any(ids)
        child1(ids,:) = [ ...
            parent(ids,1),parent(ids,2),parent(ids,3),mids(ids)];
        child2(ids,:) = [ ...
            parent(ids,2),parent(ids,3),parent(ids,4),mids(ids)];
    end

    ids = parentTag==2;
    if any(ids)
        child1(ids,:) = [ ...
            parent(ids,1),parent(ids,2),mids(ids),parent(ids,4)];
        child2(ids,:) = [ ...
            parent(ids,2),parent(ids,3),mids(ids),parent(ids,4)];
    end

    ids = parentTag==1;
    if any(ids)
        child1(ids,:) = [ ...
            parent(ids,1),mids(ids),parent(ids,3),parent(ids,4)];
        child2(ids,:) = [ ...
            parent(ids,2),mids(ids),parent(ids,3),parent(ids,4)];
    end

    childTag = uint8(parentTag-1);
    childTag(parentTag==1) = uint8(3);

    keep = ~splitMask;

    % Preserve the original element ordering exactly.
    T = [ ...
        T(keep,:);
        child1;
        child2];

    nvbTag = [ ...
        nvbTag(keep);
        childTag;
        childTag];

    % Propagate optional exact element generation.
    if trackGeneration
        parentGeneration = uint8(state.elementGeneration(splitIds));
        childGeneration = parentGeneration + uint8(1);
        state.elementGeneration = [ ...
            uint8(state.elementGeneration(keep)); ...
            childGeneration; ...
            childGeneration];
    end

    % Final leaves created anywhere inside this NVB call are marked true.
    state.elementNewLeaf = [ ...
        state.elementNewLeaf(keep); ...
        true(numel(splitIds),1); ...
        true(numel(splitIds),1)];

    info.totalBisectedParents = ...
        info.totalBisectedParents + numel(splitIds);

    pending = pending(~ready,:);

end

info.completionSteps = substep;
info.actualBisectedEdges = size(info.newNodeIds,1);
info.newNodes = size(P,1)-nNodeBefore;
info.newElements = size(T,1)-nElemBefore;
info.completionParents = max( ...
    info.totalBisectedParents-info.requestedMarkedElements,0);

if cfg.nvbCheckConformity
    check_conforming_tetra_mesh_cube(P,T,nvbTag);
end

end


function E = nvb_bisection_edges_3d(T,nvbTag)
%NVB_BISECTION_EDGES_3D Return sorted [v0,v_gamma] for every tetrahedron.

NT = size(T,1);

if numel(nvbTag) ~= NT
    error('NVB tag size mismatch.');
end

row = (1:NT)';
col = double(nvbTag(:))+1;  % gamma=1,2,3 -> MATLAB columns 2,3,4

second = T(sub2ind([NT,4],row,col));

E = sort([T(:,1),second],2);

end


function key = nvb_edge_key_3d(E)
%NVB_EDGE_KEY_3D Compact uint64 key for an undirected edge.
%
% Node IDs in this code are far below 2^32.  Packing the two uint32-sized
% IDs into one uint64 makes ISMEMBER substantially cheaper than repeated
% two-column row comparisons on large tetrahedral meshes.

if isempty(E)
    key = zeros(0,1,'uint64');
    return;
end

E = sort(E,2);

a = uint64(E(:,1));
b = uint64(E(:,2));

if any(b >= 2^32)
    error('NVB edge key requires node IDs < 2^32.');
end

key = a + bitshift(b,32);

end


function key = nvb_edge_key_sorted_3d(E)
%NVB_EDGE_KEY_SORTED_3D Pack already-sorted undirected edges into uint64.

if isempty(E)
    key = zeros(0,1,'uint64');
    return;
end

if any(E(:,1)>E(:,2))
    error('nvb_edge_key_sorted_3d received an unsorted edge.');
end

a = uint64(E(:,1));
b = uint64(E(:,2));

if any(b >= 2^32)
    error('NVB edge key requires node IDs < 2^32.');
end

key = a + bitshift(b,32);
end


function key = nvb_edge_key_pair_3d(a,b)
%NVB_EDGE_KEY_PAIR_3D Pack node-pair columns without materializing [a,b].

lo = min(a,b);
hi = max(a,b);

lo = uint64(lo);
hi = uint64(hi);

if any(hi >= 2^32)
    error('NVB edge key requires node IDs < 2^32.');
end

key = lo + bitshift(hi,32);
end


function uNew = prolongate_p1_after_nvb(uOld,nNew,refineInfo)
%PROLONGATE_P1_AFTER_NVB Exact nodal P1 prolongation to the refined mesh.
%
% Existing vertices keep their values.  Every NVB vertex is an edge
% midpoint, so its P1 prolongated value is the average of its two parents.
% New nodes are stored in creation order; parents therefore already have
% values when a later midpoint depends on an earlier midpoint.

nOld = numel(uOld);

uNew = zeros(nNew,1);
uNew(1:nOld) = uOld(:);

newIds = refineInfo.newNodeIds;
parents = refineInfo.newNodeParents;

for q = 1:numel(newIds)

    id = newIds(q);
    a = parents(q,1);
    b = parents(q,2);

    uNew(id) = 0.5*(uNew(a)+uNew(b));

end

end


function check_conforming_tetra_mesh_cube(P,T,nvbTag)
%CHECK_CONFORMING_TETRA_MESH_CUBE Expensive debugging audit.
%
% In a conforming tetrahedral partition of the unit cube, every triangular
% face occurs twice in the interior and once on the physical boundary.
% A hanging-node interface creates unmatched interior faces and is caught.

if numel(nvbTag)~=size(T,1)
    error('NVB tag size mismatch during conformity audit.');
end

if any(nvbTag<1 | nvbTag>3)
    error('Invalid Maubach NVB tag during conformity audit.');
end

faces = [ ...
    T(:,[2 3 4]);
    T(:,[1 3 4]);
    T(:,[1 2 4]);
    T(:,[1 2 3])];

faces = sort(faces,2);

[uniqueFaces,~,ic] = unique(faces,'rows');
counts = accumarray(ic,1);

if any(counts>2)
    error('A triangular face occurs more than twice.');
end

singleFaces = uniqueFaces(counts==1,:);

if isempty(singleFaces)
    error('No physical boundary faces found.');
end

tol = 1.0e-11;

x = reshape(P(singleFaces(:),1),size(singleFaces));
y = reshape(P(singleFaces(:),2),size(singleFaces));
z = reshape(P(singleFaces(:),3),size(singleFaces));

onPhysicalBoundary = ...
    all(abs(x) < tol,2) | ...
    all(abs(x-1) < tol,2) | ...
    all(abs(y) < tol,2) | ...
    all(abs(y-1) < tol,2) | ...
    all(abs(z) < tol,2) | ...
    all(abs(z-1) < tol,2);

if any(~onPhysicalBoundary)
    error(['NVB conformity audit found %d unmatched triangular faces ', ...
           'inside the cube.'],nnz(~onPhysicalBoundary));
end

% Positive volume check, independent of local vertex orientation.
p1 = P(T(:,1),:);
p2 = P(T(:,2),:);
p3 = P(T(:,3),:);
p4 = P(T(:,4),:);

sixV = abs(dot(p2-p1,cross(p3-p1,p4-p1,2),2));

if any(sixV <= 1.0e-14)
    error('NVB conformity audit found a degenerate tetrahedron.');
end

end


%% ========================================================================
% Uniform grid with the smallest FREE DOF strictly above target
% ========================================================================

function n = choose_uniform_cells_just_above_dof(targetDOF)

% For an n x n x n structured cube grid with Dirichlet data on all faces,
% the number of free P1 nodal DOF is
%
%     (n-1)^3.
%
% We require the smallest n such that
%
%     (n-1)^3 > targetDOF.

n = max(2,floor(targetDOF^(1/3))+2);

while (n-1)^3 <= targetDOF
    n = n+1;
end

while n > 2 && (n-2)^3 > targetDOF
    n = n-1;
end

end


%% ========================================================================
% Evaluate P1 tetrahedral solution at arbitrary query points
% ========================================================================

function values = evaluate_tet_solution_dirichlet( ...
    P,T,u,query,GbcFun)

TR = triangulation(T,P);
tetID = pointLocation(TR,query);

values = NaN(size(query,1),1);
inside = ~isnan(tetID);

if any(inside)

    bary = cartesianToBarycentric( ...
        TR, ...
        tetID(inside), ...
        query(inside,:));

    conn = T(tetID(inside),:);
    localU = u(conn);
    values(inside) = sum(bary.*localU,2);

end

% Correct any numerical pointLocation misses on the cube boundary.
tol = 1.0e-12;

onBottom = ...
    isnan(values) & ...
    abs(query(:,3)) < tol;

if any(onBottom)
    values(onBottom) = GbcFun( ...
        query(onBottom,1), ...
        query(onBottom,2));
end

onOtherBoundary = ...
    isnan(values) & ( ...
    abs(query(:,1))<tol | abs(query(:,1)-1)<tol | ...
    abs(query(:,2))<tol | abs(query(:,2)-1)<tol | ...
    abs(query(:,3)-1)<tol);

values(onOtherBoundary) = 0;

if any(isnan(values))
    warning('%d query points were not located inside the mesh.', ...
        nnz(isnan(values)));
end

end


%% ========================================================================
% Extract triangular faces on the bottom boundary z=0 and owner elements
% ========================================================================

function [bottomFaces,bottomOwner] = extract_bottom_boundary_faces(P,T)
%EXTRACT_BOTTOM_BOUNDARY_FACES
% Return all triangular boundary faces lying on z=0 and the tetrahedron
% owning each such face.

nElem = size(T,1);

% All four faces of every tetrahedron.
faces = [ ...
    T(:,[2 3 4]); ...
    T(:,[1 3 4]); ...
    T(:,[1 2 4]); ...
    T(:,[1 2 3])];

% Owner tetrahedron of every face occurrence.
owner = repmat((1:nElem)',4,1);

% Sort node numbers inside every face so neighboring tetrahedra give the
% same face representation.
facesSorted = sort(faces,2);

% Count how many times every triangular face occurs.
[~,~,ic] = unique(facesSorted,'rows');
counts = accumarray(ic,1);

% Interior faces occur twice.
% Boundary faces occur exactly once.
isBoundaryOccurrence = (counts(ic) == 1);

boundaryFaces = facesSorted(isBoundaryOccurrence,:);
boundaryOwner = owner(isBoundaryOccurrence);

% z coordinates of the three vertices of every boundary face.
%
% IMPORTANT:
% explicitly reshape to nBoundaryFaces-by-3.
zFace = reshape( ...
    P(boundaryFaces(:),3), ...
    size(boundaryFaces));

tol = 1.0e-12;

% A bottom face has all three vertices at z=0.
isBottom = all(abs(zFace) < tol,2);

bottomFaces = boundaryFaces(isBottom,:);
bottomOwner = boundaryOwner(isBottom);

if isempty(bottomFaces)
    error('No bottom boundary faces were found.');
end

end


%% ========================================================================
% Figure saver
% ========================================================================

function save_figure(fig,fileName,resolution)

try

    exportgraphics(fig,fileName,'Resolution',resolution);

catch

    set(fig,'PaperPositionMode','auto');
    print(fig,fileName,'-dpng',sprintf('-r%d',resolution));

end

end
