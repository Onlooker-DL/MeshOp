function plotmeshb_sol_cd(operatorName, operatorExperiment, testIds, stage, errQuantile, errCap)
%PLOTMESHB_SOL_CD Draw the CD-disk FEM solution field u_h and its
% pointwise absolute error |u_h - u_ref| against the spectral reference.
%
% Same arguments and MAT lookup as plotmeshb_cd. For every enabled method
% (FNO, target, AFEM, uniform) two figures are written: the finite-element
% solution field and the pointwise absolute error.
%
% errCap: fixed upper limit of the err colorbar (default 0.02). Values
%   above it are drawn as saturated color.
% errQuantile: percentile mode, used only when errCap is NaN:
%   plotmeshb_sol_cd(...,'base',85,NaN) caps the colorbar at the 85th
%   percentile of each figure's errors.
%
% Usage:
%   plotmeshb_sol_cd('fno','cd_b3000_mse',[1 4 69])          % cap 0.02
%   plotmeshb_sol_cd('fno','cd_b3000_mse',4,'base',85,NaN)   % 85th pct
%
% Output (figures/process/CD/):
%   cd_<op>_<exp>_t<k>_i<id>_<tag>_sol.pdf/.png
%   cd_<op>_<exp>_t<k>_i<id>_<tag>_err.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'cd_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 5;
    end
    if nargin < 4 || isempty(stage)
        stage = 'base';
    end
    if nargin < 5 || isempty(errQuantile)
        errQuantile = 95;
    end
    if nargin < 6 || isempty(errCap)
        errCap = 0.02;
    end
    validateattributes(errQuantile,{'numeric'}, ...
        {'scalar','finite','>',0,'<=',100});

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(root,'src','fem','cd_disk'));
    femRoot = fullfile(root,'result','fem','cd_disk', ...
        operatorName,operatorExperiment);
    if exist(femRoot,'dir') ~= 7
        error('plotmeshb_sol_cd:MissingResults', ...
            'FEM result folder not found:\n%s',femRoot);
    end

    matFiles = dir(fullfile(femRoot,'**','*_enabled_method_comparison.mat'));
    if isempty(matFiles)
        error('plotmeshb_sol_cd:NoSavedSamples', ...
            'No per-sample MAT found under:\n%s',femRoot);
    end

    nFiles = numel(matFiles);
    savedIds = nan(1,nFiles);
    isStage = false(1,nFiles);
    for i = 1:nFiles
        tok = regexp(matFiles(i).name,'test_(\d+)','tokens','once');
        if ~isempty(tok)
            savedIds(i) = str2double(tok{1});
        end
        isStage(i) = contains(matFiles(i).folder,[filesep stage filesep]);
    end

    for k = testIds(:).'
        candidates = find(savedIds == k);
        if isempty(candidates)
            error('plotmeshb_sol_cd:MissingSample', ...
                'Test %d has no saved MAT under:\n%s',k,femRoot);
        end
        cands = candidates(isStage(candidates));
        if isempty(cands)
            cands = candidates;
            fprintf('[plotmeshb_sol_cd] test %d: using %s (no %s file)\n', ...
                k,matFiles(cands(1)).name,stage);
        end
        scaleCands = cands(contains({matFiles(cands).folder},'scale_'));
        if ~isempty(scaleCands)
            idx = scaleCands(1);
        else
            idx = cands(1);
        end

        filePath = fullfile(matFiles(idx).folder,matFiles(idx).name);
        S = load(filePath);
        cfg = S.cfg;
        sampleResult = S.sampleResult;
        operatorLabel = plotmeshb_sol_cd_label(cfg.operatorName);

        methods = plotmeshb_sol_cd_collect(sampleResult,cfg,operatorLabel);
        if isempty(methods)
            error('plotmeshb_sol_cd:NoMethods', ...
                'No enabled method has saved solution data for test %d.',k);
        end

        sourceId = double(sampleResult.sourceId(1));
        refFine = build_cd_reference(root,cfg,sourceId);
        figDir = fullfile(root,'figures','process','CD');
        if exist(figDir,'dir') ~= 7, mkdir(figDir); end

        for m = 1:numel(methods)
            M = methods(m).M;
            if ~isfield(M,'u') || isempty(M.u)
                continue;
            end
            u = double(M.u(:));
            if numel(u) ~= size(M.node,1)
                warning('plotmeshb_sol_cd:SizeMismatch', ...
                    'Solution size does not match nodes for %s.',methods(m).tag);
                continue;
            end
            base = sprintf('cd_%s_%s_t%03d_i%04d_%s', ...
                cfg.operatorName, ...
                strrep(operatorExperiment,'_mse',''), ...
                k,sourceId,methods(m).tag);
            uref = eval_cd_reference(refFine,M.node(:,1),M.node(:,2));
            err = abs(u - uref);
            plotmeshb_sol_cd_field(figDir,base,M.node,M.elem,u,'sol',errQuantile,errCap);
            plotmeshb_sol_cd_field(figDir,base,M.node,M.elem,err,'err',errQuantile,errCap);
        end
        fprintf('[plotmeshb_sol_cd] test %d drawn from %s\n',k,filePath);
    end
end

function plotmeshb_sol_cd_field(figDir,base,node,elem,u,kind,errQuantile,errCap)
    fig = figure('Color','w','Units','points', ...
        'Position',[1,1,514,409],'Visible','on');
    ax = axes(fig,'Position',[0.03,0.05,0.82,0.90]);
    patch(ax,'Faces',elem,'Vertices',node, ...
        'FaceVertexCData',u,'FaceColor','interp','EdgeColor','none');
    axis(ax,'equal');
    axis(ax,'tight');
    axis(ax,'off');
    colormap(ax,parula);
    if strcmp(kind,'err')
        if isfinite(errCap) && errCap > 0
            clim = errCap;
        else
            clim = prctile(u,errQuantile);
            if ~isfinite(clim) || clim <= 0
                clim = max(u);
            end
        end
        caxis(ax,[0,clim]);
    else
        caxis(ax,'auto');
    end
    cb = colorbar(ax,'Position',[0.88,0.05,0.03,0.90]);
    if strcmp(kind,'err')
        cb.Label.String = '|u_h-u_{ref}|';
    else
        cb.Label.String = 'u_h';
    end
    plotmeshb_sol_cd_export(fig, ...
        fullfile(figDir,[base '_' kind '.pdf']), ...
        fullfile(figDir,[base '_' kind '.png']));
end

function refFine = build_cd_reference(root,cfg,sourceId)
    datasetFile = fullfile(root,'data','cd_disk','cd_disk_3100.mat');
    if isfield(cfg,'datasetMat') && ~isempty(cfg.datasetMat) && ...
            exist(cfg.datasetMat,'file') == 2
        datasetFile = cfg.datasetMat;
    end
    if exist(datasetFile,'file') ~= 2
        error('plotmeshb_sol_cd:MissingDataset', ...
            'Dataset file not found:\n%s',datasetFile);
    end
    D = cd_plot_load_dataset(datasetFile, ...
        {'beta_angle','forcing_center','forcing_width', ...
         'grf_xi_cos','grf_xi_sin'});
    grf = cd_grf('fromcoeffs',cfg, ...
        double(D.beta_angle(sourceId)), ...
        double(D.forcing_center(sourceId)), ...
        double(D.forcing_width(sourceId)), ...
        double(D.grf_xi_cos(:,sourceId)), ...
        double(D.grf_xi_sin(:,sourceId)));
    beta = cfg.betaMagnitude*[cos(grf.betaAngle),sin(grf.betaAngle)];
    [~,refFine,~] = cd_reference('solve',grf,beta,cfg);
end

function ur = eval_cd_reference(refFine,x,y)
    x = x(:);
    y = y(:);
    r = sqrt(x.^2 + y.^2);
    th = mod(atan2(y,x),2*pi);
    r = min(max(r,0),1);
    ur = refFine.interpolant(r,th);
    ur = double(ur(:));
end

function D = cd_plot_load_dataset(fileName,fields)
    D = struct();
    info = h5info(fileName);
    names = {info.Datasets.Name};
    for i = 1:numel(fields)
        fld = fields{i};
        if ~any(strcmp(names,fld))
            error('plotmeshb_sol_cd:MissingField', ...
                'Dataset %s is missing %s.',fileName,fld);
        end
        D.(fld) = h5read(fileName,['/' fld]);
    end
end

function methods = plotmeshb_sol_cd_collect(sampleResult,cfg,operatorLabel)
    methods = struct('label',{},'tag',{},'M',{});
    if cfg.runFNOSeededAFEM
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            sprintf('%s score FEM + %d AFEM',operatorLabel,cfg.hybridAFEMCycles), ...
            sprintf('op%d',cfg.hybridAFEMCycles), ...
            {'operatorHybrid'});
    end
    if cfg.runTargetSeededAFEM
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            sprintf('Target score FEM + %d AFEM',cfg.hybridAFEMCycles), ...
            sprintf('tgt%d',cfg.hybridAFEMCycles), ...
            {'targetHybrid'});
    end
    if cfg.runTimeMatchedAFEM
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            'Time-matched AFEM','atime',{'afemTime'});
    end
    if cfg.runAccuracyMatchedAFEM
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            'Accuracy-matched AFEM','aacc',{'afemAccuracy'});
    end
    if cfg.runFixedRefinementAFEM
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            sprintf('Fixed-%d-refine AFEM',cfg.fixedAFEMRefineCycles), ...
            'afix',{'afemFixed'});
    end
    if cfg.runDOFMatchedUniform
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            'DOF-matched uniform FEM','udof',{'uniform'});
    end
    if cfg.runFixedUniform
        methods = plotmeshb_sol_cd_add(methods,sampleResult, ...
            'Fixed uniform FEM','ufix',{'fixedUniform'});
    end
end

function out = plotmeshb_sol_cd_add(methods,sampleResult,label,tag,fieldNames)
    out = methods;
    for k = 1:numel(fieldNames)
        fld = fieldNames{k};
        if isfield(sampleResult,fld) && ...
                isfield(sampleResult.(fld),'node') && ...
                isfield(sampleResult.(fld),'elem') && ...
                isfield(sampleResult.(fld),'u')
            out(end+1) = struct('label',label,'tag',tag, ...
                'M',sampleResult.(fld)); %#ok<AGROW>
            return;
        end
    end
end

function label = plotmeshb_sol_cd_label(operatorName)
    switch lower(operatorName)
        case 'fno', label = 'FNO';
        case 'cno', label = 'CNO';
        case 'deeponet', label = 'DeepONet';
        case 'pod_deeponet', label = 'POD-DeepONet';
        otherwise, label = upper(operatorName);
    end
end

function plotmeshb_sol_cd_export(fig,outPdf,outPng)
    set(fig,'PaperUnits','points', ...
        'PaperSize',[514,409], ...
        'PaperPosition',[0,0,514,409]);
    exportgraphics(fig,outPdf,'ContentType','image','Resolution',150);
    try
        print(fig,outPng,'-dpng','-r300');
    catch
        exportgraphics(fig,outPng,'Resolution',300);
    end
    close(fig);
end