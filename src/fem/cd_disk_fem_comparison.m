function result = cd_disk_fem_comparison(cfg)
%CD_DISK_FEM_COMPARISON Compare FNO/target score meshes with standard AFEM
% and uniform FEM rows for the CD-disk problem, mirroring burgers.
%
%   cfg.hybridAFEMCycles = 0 -> 'base' stage
%   cfg.hybridAFEMCycles = 2 -> 'hybrid' stage
%
% All AFEM machinery is the SHARED cd_* core used by generate_data, so the
% score-driven meshes, hybrid corrections, standard rows and data
% generation use exactly the same initial mesh, theta=0.8, NVB and SUPG
% solver.
%
% Summary rows carry the burgers-style timing breakdown:
%   meshSec (score realization / initial mesh / NVB refine),
%   assemblySec, solveSec, otherSec (estimator+marking), totalSec.

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(root,'src','fem','cd_disk'));

    % Fill any fields the config file did not set from the shared defaults.
    base = cd_config();
    bfn = fieldnames(base);
    for bi = 1:numel(bfn)
        if ~isfield(cfg,bfn{bi})
            cfg.(bfn{bi}) = base.(bfn{bi});
        end
    end

    D = load(cfg.datasetMat);
    P = cd_load_predictions(cfg.predictionMat);

    % Burgers-style exported inference time.  NaN -> read from the
    % operator's final_metrics.json; a finite cfg override wins.
    if ~isfield(cfg,'useExportedInferenceTime')
        cfg.useExportedInferenceTime = true;
    end
    if ~isfield(cfg,'operatorInferenceTimePerSample')
        cfg.operatorInferenceTimePerSample = NaN;
    end
    if cfg.useExportedInferenceTime && ...
            (~isfinite(cfg.operatorInferenceTimePerSample) || ...
             cfg.operatorInferenceTimePerSample<0)
        cfg.operatorInferenceTimePerSample = ...
            read_operator_inference_time(cfg,root);
    end
    if ~isfinite(cfg.operatorInferenceTimePerSample) || ...
            cfg.operatorInferenceTimePerSample<0
        cfg.operatorInferenceTimePerSample = 0.0;
    end
    fprintf('[cd_disk_fem] operator inference time per sample: %.6f s\n', ...
        cfg.operatorInferenceTimePerSample);

    rVec = double(D.rVec(:));
    thetaVec = double(D.thetaVec(:));
    node0 = double(D.initial_node);
    elem0 = double(D.initial_elem);
    gen0 = zeros(size(elem0,1),1);

    nTest = size(P.pred_score,3);
    sampleIds = cfg.sampleIds(:).';
    if isempty(sampleIds)
        rng(20260814,'twister');
        perm = randperm(nTest);
        sampleIds = perm(1:min(cfg.numSamples,nTest));
    end

    stageName = 'base';
    if cfg.hybridAFEMCycles > 0
        stageName = 'hybrid';
    end
    tag = sprintf('scale_%03d_threshold_%03d', ...
        round(100*cfg.scoreMultiplier),round(100*cfg.generationThreshold));
    outDir = fullfile(root,'result','fem','cd_disk', ...
        cfg.operatorName,cfg.operatorExperiment,stageName,tag);
    if exist(outDir,'dir') ~= 7, mkdir(outDir); end
    figDir = fullfile(root,'figures','fem','cd_disk', ...
        cfg.operatorName,cfg.operatorExperiment,stageName,tag);
    if exist(figDir,'dir') ~= 7, mkdir(figDir); end

    % Fixed-uniform mesh is geometry-only (same seed, sample-independent),
    % so build it ONCE before the sample loop and reuse it.  No NVB
    % labeling is needed because this row is never refined.
    fixedUniform = struct('node',[],'elem',[]);
    if cfg.runFixedUniform
        tFixedMesh = tic;
        [fixedUniform.node,fixedUniform.elem] = ...
            cd_uniform_disk_mesh(cfg,cfg.uniformHmax,'fixed');
        fprintf('[cd_disk_fem] fixed uniform mesh built once: %d nodes, %d elems, %.3f s\n', ...
            size(fixedUniform.node,1),size(fixedUniform.elem,1),toc(tFixedMesh));
    end

    summary = repmat(struct('testId',0,'sourceId',0,'method','', ...
        'dof',0,'elems',0,'relL2',NaN, ...
        'scoreMae',NaN,'scoreRmse',NaN, ...
        'meshSec',NaN,'assemblySec',NaN,'solveSec',NaN, ...
        'otherSec',NaN,'totalSec',NaN,'stage',NaN),0,1);

    sampleNo = 0;
    for k = sampleIds
        sampleNo = sampleNo + 1;
        sourceId = double(P.source_attempt_id_test(k));
        grf = cd_grf('fromcoeffs',cfg, ...
            D.beta_angle(sourceId),D.forcing_center(sourceId), ...
            D.forcing_width(sourceId), ...
            D.grf_xi_cos(:,sourceId),D.grf_xi_sin(:,sourceId));
        beta = cfg.betaMagnitude*[cos(grf.betaAngle),sin(grf.betaAngle)];

        [~,refFine,specDiff] = cd_reference('solve',grf,beta,cfg);

        scoreDiff = double(P.pred_score(:,:,k)) - ...
            double(P.target_score_test(:,:,k));
        scoreMae = mean(abs(scoreDiff(:)));
        scoreRmse = sqrt(mean(scoreDiff(:).^2));
        fprintf('  score error: MAE=%.4f RMSE=%.4f\n',scoreMae,scoreRmse);

        S = struct('testId',k,'sourceId',sourceId, ...
            'predScore',P.pred_score(:,:,k), ...
            'targetScore',P.target_score_test(:,:,k), ...
            'rVec',rVec,'thetaVec',thetaVec, ...
            'specDiff',specDiff,'scoreError', ...
            struct('mae',scoreMae,'rmse',scoreRmse),'cfg',cfg);

        fnoRelL2 = NaN; fnoTotal = NaN; fnoDof = NaN;

        % ---------------- FNO seeded ----------------
        if cfg.runFNOSeededAFEM
            tMesh = tic;
            [n,e,g] = cd_score_to_mesh(cfg,P.pred_score(:,:,k), ...
                rVec,thetaVec,node0,elem0,gen0);
            meshSec = toc(tMesh);
            if cfg.hybridAFEMCycles > 0
                [n,e,g,u,histAfem,~] = cd_run_afem( ...
                    cfg,grf,beta,n,e,g,cfg.hybridAFEMCycles);
                b = histAfem(end);
                asmSec = b.assemblyTimeSec; solSec = b.solveTimeSec;
                otherSec = b.otherTimeSec + cfg.operatorInferenceTimePerSample;
                meshSec = meshSec + b.refineTimeSec;
                totalSec = meshSec + asmSec + solSec + otherSec;
            else
                [u,sinfo] = cd_solve_p1_supg(n,e,cfg,grf,beta);
                asmSec = sinfo.assemblyTimeSec; solSec = sinfo.solveTimeSec;
                otherSec = cfg.operatorInferenceTimePerSample;
                totalSec = meshSec + asmSec + solSec + otherSec;
            end
            [relL2,~] = cd_reference('error',n,e,u,refFine);
            fnoRelL2 = relL2; fnoDof = dof_disk(n,e); fnoTotal = totalSec;
            S.operatorHybrid = struct('node',n,'elem',e,'u',u, ...
                'relL2',relL2,'dof',fnoDof, ...
                'meshSec',meshSec,'assemblySec',asmSec, ...
                'solveSec',solSec,'otherSec',otherSec, ...
                'totalSec',totalSec);
            summary(end+1,1) = make_row(k,sourceId,'FNO score', ...
                fnoDof,size(e,1),relL2,scoreMae,scoreRmse,meshSec,asmSec,solSec,otherSec,totalSec); %#ok<AGROW>
        end

        % ---------------- Target seeded ----------------
        if cfg.runTargetSeededAFEM
            tMesh = tic;
            [n,e,g] = cd_score_to_mesh(cfg,P.target_score_test(:,:,k), ...
                rVec,thetaVec,node0,elem0,gen0);
            meshSec = toc(tMesh);
            if cfg.hybridAFEMCycles > 0
                [n,e,g,u,histAfem,~] = cd_run_afem( ...
                    cfg,grf,beta,n,e,g,cfg.hybridAFEMCycles);
                b = histAfem(end);
                asmSec = b.assemblyTimeSec; solSec = b.solveTimeSec;
                otherSec = b.otherTimeSec;
                meshSec = meshSec + b.refineTimeSec;
                totalSec = meshSec + asmSec + solSec + otherSec;
            else
                [u,sinfo] = cd_solve_p1_supg(n,e,cfg,grf,beta);
                asmSec = sinfo.assemblyTimeSec; solSec = sinfo.solveTimeSec;
                otherSec = 0; totalSec = meshSec + asmSec + solSec;
            end
            [relL2,~] = cd_reference('error',n,e,u,refFine);
            S.targetHybrid = struct('node',n,'elem',e,'u',u, ...
                'relL2',relL2,'dof',dof_disk(n,e), ...
                'meshSec',meshSec,'assemblySec',asmSec, ...
                'solveSec',solSec,'otherSec',otherSec, ...
                'totalSec',totalSec);
            summary(end+1,1) = make_row(k,sourceId,'Target score', ...
                dof_disk(n,e),size(e,1),relL2,scoreMae,scoreRmse,meshSec,asmSec,solSec,otherSec,totalSec); %#ok<AGROW>
        end

        % ---------------- Standard AFEM trajectory ----------------
        if cfg.runStandardAFEM
            trajCycles = cfg.fixedAFEMRefineCycles;
            if cfg.runTimeMatchedAFEM
                trajCycles = max(trajCycles,cfg.afemMaxCycles);
            end
            [sn,se,sg,su,sHist,sSnap] = cd_run_afem( ...
                cfg,grf,beta,node0,elem0,gen0,trajCycles);

            % fixed-refinement row
            fidx = min(cfg.fixedAFEMRefineCycles+1,numel(sSnap));
            fn = sSnap{fidx}.node; fe = sSnap{fidx}.elem;
            fu = sSnap{fidx}.u; hb = sHist(fidx);
            [relL2,~] = cd_reference('error',fn,fe,fu,refFine);
            S.afemFixed = struct('node',fn,'elem',fe,'u',fu, ...
                'relL2',relL2,'dof',dof_disk(fn,fe), ...
                'stage',cfg.fixedAFEMRefineCycles, ...
                'meshSec',hb.refineTimeSec, ...
                'assemblySec',hb.assemblyTimeSec, ...
                'solveSec',hb.solveTimeSec, ...
                'otherSec',hb.otherTimeSec, ...
                'totalSec',hb.timeSec);
            summary(end+1,1) = make_row(k,sourceId, ...
                sprintf('Fixed-%d AFEM',cfg.fixedAFEMRefineCycles), ...
                dof_disk(fn,fe),size(fe,1),relL2,scoreMae,scoreRmse, ...
                hb.refineTimeSec,hb.assemblyTimeSec,hb.solveTimeSec, ...
                hb.otherTimeSec,hb.timeSec,cfg.fixedAFEMRefineCycles); %#ok<AGROW>

            % accuracy-matched row (burgers semantics): stop at the first
            % snapshot whose error is at or below the FNO error, then keep
            % the closer of the two bracketing snapshots.  If the trajectory
            % never reaches the FNO accuracy, use the closest snapshot.
            if cfg.runAccuracyMatchedAFEM && ~isnan(fnoRelL2)
                best = 1; bestGap = inf; chosen = [];
                rl2Prev = NaN;
                for r = 1:numel(sSnap)
                    [rl2,~] = cd_reference('error', ...
                        sSnap{r}.node,sSnap{r}.elem,sSnap{r}.u,refFine);
                    gap = abs(rl2-fnoRelL2);
                    if gap < bestGap, bestGap = gap; best = r; end
                    if r > 1 && isempty(chosen) && rl2 <= fnoRelL2
                        if abs(rl2Prev-fnoRelL2) <= gap
                            chosen = r-1;
                        else
                            chosen = r;
                        end
                        break;
                    end
                    rl2Prev = rl2;
                end
                if isempty(chosen)
                    chosen = best;
                    warning('cd_disk_fem:AccuracyNotReached', ...
                        'AFEM did not reach FNO accuracy; using closest stage %d.', ...
                        best-1);
                end
                bn = sSnap{chosen}.node; be = sSnap{chosen}.elem;
                bu = sSnap{chosen}.u; hb = sHist(chosen);
                [relL2,~] = cd_reference('error',bn,be,bu,refFine);
                S.afemAccuracy = struct('node',bn,'elem',be,'u',bu, ...
                    'relL2',relL2,'dof',dof_disk(bn,be), ...
                    'stage',chosen-1, ...
                    'meshSec',hb.refineTimeSec, ...
                    'assemblySec',hb.assemblyTimeSec, ...
                    'solveSec',hb.solveTimeSec, ...
                    'otherSec',hb.otherTimeSec, ...
                    'totalSec',hb.timeSec);
                summary(end+1,1) = make_row(k,sourceId,'Accuracy AFEM', ...
                    dof_disk(bn,be),size(be,1),relL2,scoreMae,scoreRmse, ...
                    hb.refineTimeSec,hb.assemblyTimeSec,hb.solveTimeSec, ...
                    hb.otherTimeSec,hb.timeSec,chosen-1); %#ok<AGROW>
            end

            % time-matched row
            if cfg.runTimeMatchedAFEM && ~isnan(fnoTotal)
                best = 1; bestGap = inf;
                for r = 1:numel(sHist)
                    gap = abs(sHist(r).timeSec-fnoTotal);
                    if gap < bestGap, bestGap = gap; best = r; end
                end
                bn = sSnap{best}.node; be = sSnap{best}.elem;
                bu = sSnap{best}.u; hb = sHist(best);
                [relL2,~] = cd_reference('error',bn,be,bu,refFine);
                S.afemTime = struct('node',bn,'elem',be,'u',bu, ...
                    'relL2',relL2,'dof',dof_disk(bn,be), ...
                    'stage',best-1, ...
                    'meshSec',hb.refineTimeSec, ...
                    'assemblySec',hb.assemblyTimeSec, ...
                    'solveSec',hb.solveTimeSec, ...
                    'otherSec',hb.otherTimeSec, ...
                    'totalSec',hb.timeSec);
                summary(end+1,1) = make_row(k,sourceId,'Time AFEM', ...
                    dof_disk(bn,be),size(be,1),relL2,scoreMae,scoreRmse, ...
                    hb.refineTimeSec,hb.assemblyTimeSec,hb.solveTimeSec, ...
                    hb.otherTimeSec,hb.timeSec,best-1); %#ok<AGROW>
            end

            % Per-stage history for convergence plots.
            histRel = nan(numel(sHist),1);
            if cfg.makeConvergencePlots && ismember(k,cfg.plotSampleIds)
                for r = 1:numel(sSnap)
                    [rl2,~] = cd_reference('error', ...
                        sSnap{r}.node,sSnap{r}.elem,sSnap{r}.u,refFine);
                    histRel(r) = rl2;
                end
            end
            S.afemHistory = struct('cycle',[sHist.cycle], ...
                'nodes',[sHist.nodes],'elems',[sHist.elems], ...
                'eta',[sHist.eta],'relL2',histRel);
        end

        % ---------------- Uniform rows ----------------
        if cfg.runDOFMatchedUniform && ~isnan(fnoDof)
            tMesh = tic;
            [un,ue] = cd_uniform_disk_mesh(cfg,fnoDof,'above');
            meshSec = toc(tMesh);
            [uu,sinfo] = cd_solve_p1_supg(un,ue,cfg,grf,beta);
            asmSec = sinfo.assemblyTimeSec; solSec = sinfo.solveTimeSec;
            [relL2,~] = cd_reference('error',un,ue,uu,refFine);
            S.uniform = struct('node',un,'elem',ue,'u',uu, ...
                'relL2',relL2,'dof',dof_disk(un,ue), ...
                'meshSec',meshSec,'assemblySec',asmSec, ...
                'solveSec',solSec,'otherSec',0, ...
                'totalSec',meshSec+asmSec+solSec);
            summary(end+1,1) = make_row(k,sourceId,'Uniform(DOF)', ...
                dof_disk(un,ue),size(ue,1),relL2,scoreMae,scoreRmse, ...
                meshSec,asmSec,solSec,0,meshSec+asmSec+solSec); %#ok<AGROW>
        end
        if cfg.runFixedUniform
            tMesh = tic;
            un = fixedUniform.node; ue = fixedUniform.elem;
            meshSec = toc(tMesh);
            [uu,sinfo] = cd_solve_p1_supg(un,ue,cfg,grf,beta);
            asmSec = sinfo.assemblyTimeSec; solSec = sinfo.solveTimeSec;
            [relL2,~] = cd_reference('error',un,ue,uu,refFine);
            S.fixedUniform = struct('node',un,'elem',ue,'u',uu, ...
                'relL2',relL2,'dof',dof_disk(un,ue), ...
                'meshSec',meshSec,'assemblySec',asmSec, ...
                'solveSec',solSec,'otherSec',0, ...
                'totalSec',meshSec+asmSec+solSec);
            summary(end+1,1) = make_row(k,sourceId,'Uniform(fixed)', ...
                dof_disk(un,ue),size(ue,1),relL2,scoreMae,scoreRmse, ...
                meshSec,asmSec,solSec,0,meshSec+asmSec+solSec); %#ok<AGROW>
        end

        if any([cfg.makeMeshPlots cfg.makeScorePlots cfg.makeConvergencePlots]) && ...
                ismember(k,cfg.plotSampleIds)
            cd_plot_sample(cfg,S,D,figDir,k,sourceId);
        end

        % Save the per-sample MAT only for the requested plot samples
        % (same rule as burgers: makePlotsForThisSample && saveSampleMeshes).
        if cfg.saveSampleMeshes && ismember(k,cfg.plotSampleIds)
            filePath = fullfile(outDir,sprintf( ...
                'sample_%03d_test_%04d_source_%05d_enabled_method_comparison.mat', ...
                find(sampleIds==k,1),k,sourceId));
            sampleResult = S;
            save(filePath,'-v7.3','sampleResult','cfg');
        end
        fprintf('[cd_disk_fem] sample %d/%d: test %d (source %d) done\n', ...
            sampleNo,numel(sampleIds),k,sourceId);
    end

    if cfg.makeSummaryPlots
        cd_plot_summary(cfg,summary,figDir);
    end

    summary = summary(:);
    summaryFile = fullfile(outDir,'summary.mat');
    save(summaryFile,'summary');
    csvFile = fullfile(outDir,'summary.csv');
    write_summary_csv(summary,csvFile);

    % Burgers-style statistics tables.
    write_method_statistics(summary,outDir);
    write_timing_statistics(summary,outDir);
    result = struct();
    result.summary = summary;
    result.summaryFile = summaryFile;
    fprintf('Saved %s\n',summaryFile);
end

function dof = dof_disk(node,elem)
    n = size(node,1);
    edges = sort([elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])],2);
    [uE,~,ic] = unique(edges,'rows');
    mult = accumarray(ic,1);
    bndEdges = uE(mult==1,:);
    bnd = unique(bndEdges(:));
    bnd = bnd(bnd>=1 & bnd<=n);
    dof = n - numel(bnd);
end

function row = make_row(testId,sourceId,method,dof,elems,relL2, ...
        scoreMae,scoreRmse,meshSec,assemblySec,solveSec,otherSec,totalSec,stage)
    if nargin < 14, stage = NaN; end
    row = struct('testId',testId,'sourceId',sourceId,'method',method, ...
        'dof',dof,'elems',elems,'relL2',relL2, ...
        'scoreMae',scoreMae,'scoreRmse',scoreRmse, ...
        'meshSec',meshSec,'assemblySec',assemblySec, ...
        'solveSec',solveSec,'otherSec',otherSec,'totalSec',totalSec, ...
        'stage',stage);
    fprintf('  %-20s dof=%8d elems=%8d relL2=%.6e total=%.3f s\n', ...
        method,dof,elems,relL2,totalSec);
end

function [node,elem] = cd_uniform_disk_mesh(cfg,target,mode)
    % Uniform disk mesh built with initmesh (same construction and fixed
    % seed as the shared initial mesh).  'fixed': target is cfg.uniformHmax.
    % 'above': halve Hmax until free DOF >= target (DOF-matched row).
    % No NVB labeling: the uniform row is never refined.
    if strcmp(mode,'above')
        hmax = 0.4;
        while true
            [node,elem] = cd_uniform_initmesh(cfg,hmax);
            if dof_disk(node,elem) >= target
                break;
            end
            hmax = hmax/2;
            if hmax < 1e-4
                error('cd_disk_fem:UniformTooBig','DOF target too large.');
            end
        end
    else
        [node,elem] = cd_uniform_initmesh(cfg,target);
    end
end

function [node,elem] = cd_uniform_initmesh(cfg,hmax)
    % Plain initmesh call only.  Keeps the same fixed seed as the shared
    % initial mesh so the uniform row is deterministic.
    if exist('initmesh','file') ~= 2
        error('cd_disk_fem:NoPDEToolbox', ...
            'initmesh requires the MATLAB PDE Toolbox.');
    end
    rngState = rng;
    rng(20240815,'twister');
    [pLegacy,~,tLegacy] = initmesh(@circleg, ...
        'Hmax',hmax,'Jiggle','minimum');
    rng(rngState);
    node = double(pLegacy.');
    elem = double(tLegacy(1:3,:).');
end

function write_summary_csv(summary,csvFile)
    fid = fopen(csvFile,'w');
    fprintf(fid,'testId,sourceId,method,dof,elems,relL2,scoreMae,scoreRmse,meshSec,assemblySec,solveSec,otherSec,totalSec,stage\n');
    for i = 1:numel(summary)
        fprintf(fid,'%d,%d,%s,%d,%d,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6g\n', ...
            summary(i).testId,summary(i).sourceId,summary(i).method, ...
            summary(i).dof,summary(i).elems,summary(i).relL2, ...
            summary(i).scoreMae,summary(i).scoreRmse, ...
            summary(i).meshSec,summary(i).assemblySec, ...
            summary(i).solveSec,summary(i).otherSec,summary(i).totalSec, ...
            summary(i).stage);
    end
    fclose(fid);
end

function cd_plot_sample(cfg, S, D, figDir, k, sourceId)
    if exist(figDir,'dir') ~= 7, mkdir(figDir); end
    base = sprintf('cd_disk_test%03d_id%05d',k,sourceId);

    if cfg.makeMeshPlots
        cd_mesh_figure(cfg, S, D, figDir, base);
    end
    if cfg.makeScorePlots && isfield(S,'predScore') && isfield(S,'targetScore')
        cd_score_figure(cfg, S, figDir, base);
    end
    if cfg.makeConvergencePlots && isfield(S,'afemHistory')
        cd_convergence_figure(S, figDir, base);
    end
end

function cd_mesh_figure(cfg, S, D, figDir, base)
    node0 = double(D.initial_node);
    elem0 = double(D.initial_elem);

    entries = struct('label',{},'node',{},'elem',{});
    entries(end+1,1) = struct('label','Initial mesh', ...
        'node',node0,'elem',elem0);
    if isfield(S,'operatorHybrid')
        entries(end+1,1) = struct('label','FNO score mesh', ...
            'node',S.operatorHybrid.node,'elem',S.operatorHybrid.elem);
    end
    if isfield(S,'targetHybrid')
        entries(end+1,1) = struct('label','Target score mesh', ...
            'node',S.targetHybrid.node,'elem',S.targetHybrid.elem);
    end
    if isfield(S,'afemFixed')
        entries(end+1,1) = struct('label','Fixed AFEM', ...
            'node',S.afemFixed.node,'elem',S.afemFixed.elem);
    end
    if isfield(S,'afemAccuracy')
        entries(end+1,1) = struct('label','Accuracy AFEM', ...
            'node',S.afemAccuracy.node,'elem',S.afemAccuracy.elem);
    end
    if isfield(S,'afemTime')
        entries(end+1,1) = struct('label','Time AFEM', ...
            'node',S.afemTime.node,'elem',S.afemTime.elem);
    end

    n = numel(entries);
    ncol = min(3,n);
    nrow = ceil(n/ncol);
    fig = figure('Color','w','Units','points','Position',[10 10 1500 420*nrow],'Visible','on');
    tl = tiledlayout(fig,nrow,ncol,'TileSpacing','compact','Padding','compact');
    for i = 1:n
        nexttile(tl);
        patch('Faces',entries(i).elem,'Vertices',entries(i).node, ...
            'FaceColor','w','EdgeColor',[0.4 0.4 0.4],'LineWidth',0.25);
        axis equal tight off;
        title(entries(i).label,'FontSize',12);
    end
    cd_export_fig(fig, figDir, [base '_meshes']);
end

function cd_score_figure(cfg, S, figDir, base)
    if isfield(S,'rVec') && ~isempty(S.rVec)
        rVec = double(S.rVec(:));
        thetaVec = double(S.thetaVec(:));
    else
        rVec = double(cfg.rVec(:));
        thetaVec = double(cfg.thetaVec(:));
    end
    pred = double(S.predScore);
    tgt = double(S.targetScore);
    [X,Y,Ft] = cd_polar_to_cart(rVec,thetaVec,tgt,260);
    [~,~,Fp] = cd_polar_to_cart(rVec,thetaVec,pred,260);
    lim = [min([Ft(:);Fp(:)]), max([Ft(:);Fp(:)])];
    lim = [0, max(lim(2),cfg.scoreMaximum)];

    fig = figure('Color','w','Units','points','Position',[10 10 1200 560],'Visible','on');
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    nexttile(tl);
    imagesc(X(1,:),Y(:,1),Ft); axis xy equal tight; caxis(lim); colormap(parula);
    title('Target score'); colorbar;
    nexttile(tl);
    imagesc(X(1,:),Y(:,1),Fp); axis xy equal tight; caxis(lim); colormap(parula);
    title('Predicted score'); colorbar;
    cd_export_fig(fig, figDir, [base '_scores']);
end

function cd_convergence_figure(S, figDir, base)
    h = S.afemHistory;
    dof = double(h.nodes);
    eta = double(h.eta);
    fig = figure('Color','w','Units','points','Position',[10 10 1200 460],'Visible','on');
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    nexttile(tl);
    semilogy(dof,eta,'-o','LineWidth',1.2);
    xlabel('nodes'); ylabel('estimator eta'); grid on;
    title('AFEM estimator convergence');
    nexttile(tl);
    if isfield(h,'relL2') && any(~isnan(h.relL2))
        semilogy(dof,h.relL2,'-s','LineWidth',1.2);
        xlabel('nodes'); ylabel('relative L2'); grid on;
        title('AFEM error convergence');
    else
        text(0.5,0.5,'relL2 unavailable','HorizontalAlignment','center');
        axis off;
    end
    cd_export_fig(fig, figDir, [base '_afem_convergence']);
end

function cd_plot_summary(cfg, summary, figDir)
    if isempty(summary), return; end
    if exist(figDir,'dir') ~= 7, mkdir(figDir); end

    methods = unique({summary.method},'stable');
    n = numel(methods);
    meanL2 = zeros(n,1); stdL2 = zeros(n,1);
    medDof = zeros(n,1); meanTime = zeros(n,1);
    for i = 1:n
        sel = strcmp({summary.method},methods{i});
        rel = [summary(sel).relL2];
        dof = [summary(sel).dof];
        tm = [summary(sel).totalSec];
        meanL2(i) = mean(rel); stdL2(i) = std(rel);
        medDof(i) = median(dof);
        meanTime(i) = mean(tm(~isnan(tm)));
    end

    fig = figure('Color','w','Units','points','Position',[10 10 1200 460],'Visible','on');
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    nexttile(tl);
    errorbar((1:n),meanL2,stdL2,'-o','LineWidth',1.2);
    set(gca,'XTick',1:n,'XTickLabel',methods,'XTickLabelRotation',25);
    ylabel('relative L2 (mean +/- std)'); grid on;
    title('Method errors');
    nexttile(tl);
    for i = 1:n
        sel = strcmp({summary.method},methods{i});
        dof = [summary(sel).dof];
        rel = [summary(sel).relL2];
        loglog(dof,rel,'o','MarkerSize',4); hold on;
    end
    legend(methods,'Location','best','Interpreter','none');
    xlabel('DOF'); ylabel('relative L2'); grid on;
    title('Error vs DOF');
    cd_export_fig(fig, figDir, 'cd_disk_summary');
end

function [X,Y,F] = cd_polar_to_cart(rVec,thetaVec,field,nGrid)
    [RR,TT] = ndgrid(double(rVec(:)),double(thetaVec(:)));
    xp = RR(:).*cos(TT(:));
    yp = RR(:).*sin(TT(:));
    Fs = scatteredInterpolant(xp,yp,double(field(:)),'linear','nearest');
    g = linspace(-1,1,nGrid);
    [X,Y] = meshgrid(g);
    inside = X.^2+Y.^2 <= 1;
    F = nan(size(X));
    F(inside) = Fs(X(inside),Y(inside));
end

function cd_export_fig(fig, figDir, base)
    axs = findall(fig,'Type','axes');
    for ai = 1:numel(axs)
        if isprop(axs(ai),'Toolbar') && ~isempty(axs(ai).Toolbar)
            axs(ai).Toolbar.Visible = 'off';
        end
    end
    set(fig,'PaperUnits','points','PaperSize',[1500 900],'PaperPosition',[0 0 1500 900]);
    exportgraphics(fig, fullfile(figDir,[base '.pdf']),'ContentType','image','Resolution',150);
    try
        print(fig, fullfile(figDir,[base '.png']),'-dpng','-r150');
    catch
        exportgraphics(fig, fullfile(figDir,[base '.png']),'Resolution',150);
    end
    close(fig);
    fprintf('  [plot] %s\n',fullfile(figDir,[base '.pdf']));
end
function P = cd_load_predictions(predictionFile)
%CD_LOAD_PREDICTIONS Load the operator predictions.mat written by Python
% (h5py, HDF5).  MATLAB load cannot read h5py-written files, so use h5read.
% h5read reverses the raw HDF5 dims, giving (nr, nth, N) directly; a sanity
% check against query_r/query_theta guards against version differences.
    if exist(predictionFile,'file') ~= 2
        error('cd_disk_fem:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end
    try
        P = load(predictionFile);       % MATLAB-written (tests / legacy)
        if isfield(P,'pred_score')
            return;
        end
    catch
    end
    P = struct();
    P.pred_score = h5read(predictionFile,'/pred_score');
    P.target_score_test = h5read(predictionFile,'/target_score_test');
    P.query_r = h5read(predictionFile,'/query_r');
    P.query_theta = h5read(predictionFile,'/query_theta');
    P.source_attempt_id_test = h5read(predictionFile,'/source_attempt_id_test');
    if h5exist(predictionFile,'/max_score')
        P.max_score = h5read(predictionFile,'/max_score');
    else
        P.max_score = 12;
    end
    nr = numel(P.query_r);
    nth = numel(P.query_theta);
    if size(P.pred_score,1) ~= nr || size(P.pred_score,2) ~= nth
        P.pred_score = permute(P.pred_score,ndims(P.pred_score):-1:1);
        P.target_score_test = permute(P.target_score_test, ...
            ndims(P.target_score_test):-1:1);
    end
end

function tf = h5exist(fileName, dsName)
    tf = false;
    try
        info = h5info(fileName, dsName);
        tf = ~isempty(info);
    catch
    end
end
function tInf = read_operator_inference_time(cfg,root)
    tInf = 0.0;
    metricsFile = fullfile(root,'result','operators','cd_disk', ...
        cfg.operatorName,cfg.operatorExperiment,'final_metrics.json');
    if exist(metricsFile,'file') ~= 2
        warning('cd_disk_fem:NoMetrics', ...
            'final_metrics.json not found (%s); inference time set to 0.', ...
            metricsFile);
        return;
    end
    try
        m = jsondecode(fileread(metricsFile));
        if isfield(m,'timing') && ...
                isfield(m.timing,'mean_inference_time_sec_per_sample')
            v = double(m.timing.mean_inference_time_sec_per_sample);
            if isfinite(v) && v>=0
                tInf = v;
                return;
            end
        end
    catch
    end
    warning('cd_disk_fem:NoInferenceTiming', ...
        'Inference timing unavailable in %s; set cfg.operatorInferenceTimePerSample manually.', ...
        metricsFile);
end

function write_method_statistics(summary,outDir)
    methods = unique({summary.method},'stable');
    csvFile = fullfile(outDir,'method_statistics.csv');
    fid = fopen(csvFile,'w');
    fprintf(fid,['Method,Samples,MeanTimeSec,StdTimeSec,MedianTimeSec,MinTimeSec,', ...
        'MaxTimeSec,MeanRelL2,MedianRelL2,StdRelL2,BestRelL2,BestTestID,BestSourceID,', ...
        'WorstRelL2,WorstTestID,WorstSourceID,MeanDOF,StdDOF,', ...
        'MeanScoreMae,MeanScoreRmse,MeanStage\n']);
    for i = 1:numel(methods)
        sel = strcmp({summary.method},methods{i});
        rel = [summary(sel).relL2];
        tm = [summary(sel).totalSec];
        dof = [summary(sel).dof];
        smae = [summary(sel).scoreMae];
        srmse = [summary(sel).scoreRmse];
        stg = [summary(sel).stage];
        [bestRel,bI] = min(rel);
        [worstRel,wI] = max(rel);
        idx = find(sel);
        fprintf(fid,'%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%.9g,%d,%d,%.9g,%.9g,%.9g,%.9g,%.6g\n', ...
            methods{i},numel(rel),mean(tm),std(tm),median(tm),min(tm),max(tm), ...
            mean(rel),median(rel),std(rel),bestRel, ...
            summary(idx(bI)).testId,summary(idx(bI)).sourceId, ...
            worstRel,summary(idx(wI)).testId,summary(idx(wI)).sourceId, ...
            mean(dof),std(dof),mean(smae),mean(srmse),mean(stg,'omitnan'));
    end
    fclose(fid);
    fprintf('Saved %s\n',csvFile);
end

function write_timing_statistics(summary,outDir)
    methods = unique({summary.method},'stable');
    csvFile = fullfile(outDir,'timing_statistics.csv');
    fid = fopen(csvFile,'w');
    fprintf(fid,'Method,Samples,MeanMeshTimeSec,MeanAssemblyTimeSec,MeanSolveTimeSec,MeanOtherTimeSec,MeanMethodTimeSec\n');
    for i = 1:numel(methods)
        sel = strcmp({summary.method},methods{i});
        meanMesh = mean([summary(sel).meshSec]);
        meanAsm = mean([summary(sel).assemblySec]);
        meanSol = mean([summary(sel).solveSec]);
        meanTotal = mean([summary(sel).totalSec]);
        meanOther = meanTotal - meanMesh - meanAsm - meanSol;
        fprintf(fid,'%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g\n', ...
            methods{i},nnz(sel), ...
            meanMesh,meanAsm,meanSol,meanOther,meanTotal);
    end
    fclose(fid);
    fprintf('Saved %s\n',csvFile);
end