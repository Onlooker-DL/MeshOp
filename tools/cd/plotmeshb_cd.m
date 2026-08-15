function plotmeshb_cd(operatorName, operatorExperiment, testIds, stage)
%PLOTMESHB_CD Draw saved CD-disk FEM meshes for selected test samples.
%
% Usage:
%   plotmeshb_cd('fno','cd_b3000_mse',[1 4 69])
%   plotmeshb_cd('fno','cd_b3000_mse',[1 4 69],'base')  % 'base' or 'hybrid'
%
% Loads the per-sample MAT files saved by run_cd_disk_fem
% (sample_*_test_%04d_source_*_enabled_method_comparison.mat) and draws the
% meshes of exactly the methods enabled in the saved configuration:
%   FNO score (operatorHybrid), Target score (targetHybrid),
%   Accuracy AFEM, Fixed-N AFEM, Time AFEM, Uniform(DOF), Uniform(fixed).
%
% Output (figures/pltfig/cd/femmesh/):
%   cd_<op>_<exp>_t<k>_i<id>_<tag>.pdf/.png

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

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    femRoot = fullfile(root,'result','fem','cd_disk', ...
        operatorName,operatorExperiment);
    if exist(femRoot,'dir') ~= 7
        error('plotmeshb_cd:MissingResults', ...
            'FEM result folder not found:\n%s\nRun run_cd_disk_fem first.', ...
            femRoot);
    end

    matFiles = dir(fullfile(femRoot,'**','*_enabled_method_comparison.mat'));
    if isempty(matFiles)
        error('plotmeshb_cd:NoSavedSamples', ...
            ['No saved per-sample MAT found under:\n%s\n', ...
             'Set cfg.saveSampleMeshes=true and cfg.plotSampleIds to the ', ...
             'desired tests, then rerun run_cd_disk_fem.'],femRoot);
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
            error('plotmeshb_cd:MissingSample', ...
                ['Test %d has no saved mesh MAT under:\n%s\n', ...
                 'Make sure it was listed in cfg.plotSampleIds with ', ...
                 'cfg.saveSampleMeshes=true, then rerun run_cd_disk_fem.'], ...
                k,femRoot);
        end
        cands = candidates(isStage(candidates));
        if isempty(cands)
            cands = candidates;
            fprintf('[plotmeshb_cd] test %d: using %s (no %s file found)\n', ...
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
        operatorLabel = plotmeshb_cd_label(cfg.operatorName);

        methods = plotmeshb_cd_collect(sampleResult,cfg,operatorLabel);
        if isempty(methods)
            error('plotmeshb_cd:NoEnabledMethods', ...
                'No enabled method has saved mesh data for test %d.',k);
        end

        sourceId = sampleResult.sourceId;
        figDir = fullfile(root,'figures','pltfig','cd','femmesh');
        if exist(figDir,'dir') ~= 7, mkdir(figDir); end

        for m = 1:numel(methods)
            M = methods(m).M;
            fig = figure('Color','w','Units','points', ...
                'Position',[1,1,514,409],'Visible','on');
            ax = axes(fig,'Position',[0.03,0.03,0.94,0.94]);
            patch(ax,'Faces',M.elem,'Vertices',M.node, ...
                'FaceColor','w','EdgeColor',[0.4 0.4 0.4], ...
                'LineWidth',0.25);
            axis(ax,'equal');
            axis(ax,'tight');
            axis(ax,'off');
            if isprop(ax,'Toolbar') && ~isempty(ax.Toolbar)
                ax.Toolbar.Visible = 'off';
            end
            outPdf = fullfile(figDir,sprintf( ...
                'cd_%s_%s_t%03d_i%04d_%s.pdf', ...
                cfg.operatorName, ...
                strrep(operatorExperiment,'_mse',''), ...
                k,sourceId,methods(m).tag));
            plotmeshb_cd_export(fig,outPdf);
            outPng = strrep(outPdf,'.pdf','.png');
            try
                print(fig,outPng,'-dpng','-r150');
            catch
                exportgraphics(fig,outPng,'Resolution',150);
            end
            close(fig);
        end
        fprintf('[plotmeshb_cd] test %d drawn from %s\n',k,filePath);
    end
end

function methods = plotmeshb_cd_collect(sampleResult,cfg,operatorLabel)
    methods = struct('label',{},'tag',{},'M',{});
    if cfg.runFNOSeededAFEM
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            sprintf('%s score FEM + %d AFEM',operatorLabel,cfg.hybridAFEMCycles), ...
            sprintf('op%d',cfg.hybridAFEMCycles), ...
            {'operatorHybrid'});
    end
    if cfg.runTargetSeededAFEM
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            sprintf('Target score FEM + %d AFEM',cfg.hybridAFEMCycles), ...
            sprintf('tgt%d',cfg.hybridAFEMCycles), ...
            {'targetHybrid'});
    end
    if cfg.runTimeMatchedAFEM
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            'Time-matched AFEM','atime',{'afemTime'});
    end
    if cfg.runAccuracyMatchedAFEM
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            'Accuracy-matched AFEM','aacc',{'afemAccuracy'});
    end
    if cfg.runFixedRefinementAFEM
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            sprintf('Fixed-%d-refine AFEM',cfg.fixedAFEMRefineCycles), ...
            'afix',{'afemFixed'});
    end
    if cfg.runDOFMatchedUniform
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            'DOF-matched uniform FEM','udof',{'uniform'});
    end
    if cfg.runFixedUniform
        methods = plotmeshb_cd_add(methods,sampleResult, ...
            'Fixed uniform FEM','ufix',{'fixedUniform'});
    end
end

function out = plotmeshb_cd_add(methods,sampleResult,label,tag,fieldNames)
    out = methods;
    for k = 1:numel(fieldNames)
        fld = fieldNames{k};
        if isfield(sampleResult,fld) && ...
                isfield(sampleResult.(fld),'node') && ...
                isfield(sampleResult.(fld),'elem')
            out(end+1) = struct('label',label,'tag',tag, ...
                'M',sampleResult.(fld)); %#ok<AGROW>
            return;
        end
    end
end

function label = plotmeshb_cd_label(operatorName)
    switch lower(operatorName)
        case 'fno', label = 'FNO';
        case 'cno', label = 'CNO';
        case 'deeponet', label = 'DeepONet';
        case 'pod_deeponet', label = 'POD-DeepONet';
        otherwise, label = upper(operatorName);
    end
end

function plotmeshb_cd_export(fig,outPdf)
    set(fig,'PaperUnits','points', ...
        'PaperSize',[514,409], ...
        'PaperPosition',[0,0,514,409]);
    exportgraphics(fig,outPdf,'ContentType','image','Resolution',150);
end