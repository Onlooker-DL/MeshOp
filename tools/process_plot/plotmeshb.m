function plotmeshb(operatorName, operatorExperiment, testIds, stage)
%PLOTMESHB Draw saved Burgers FEM meshes for selected test samples.
%
% Usage:
%   plotmeshb('fno', 'b3000_mse', [1 69])
%   plotmeshb('cno', 'b3000_mse', 1)
%   plotmeshb('fno', 'b3000_mse', [1 69], 'hybrid')   % 'base' or 'hybrid'
%
% The script loads the per-sample MAT files saved by run_burgers_fem
% (sample_*_test_%04d_source_*_enabled_method_comparison.mat) and draws the
% meshes of the two score-driven methods only:
%   FNO predicted-score FEM (operator) and target-score FEM (target).
% The standard-AFEM and uniform rows are deliberately not drawn here.
% An error is raised when a requested test id has no saved MAT.
%
% Output PDF: figures/process/Burgers/burgers_<op>_<exp>_t<k>_i<id>_<tag>.pdf

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds =69;
    end
    if nargin < 4 || isempty(stage)
        stage = 'hybrid';
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    femRoot = fullfile(root, 'result', 'fem', 'burgers', ...
        operatorName, operatorExperiment);
    if exist(femRoot, 'dir') ~= 7
        error('plotmeshb:MissingResults', ...
            'FEM result folder not found:\n%s\nRun run_burgers_fem first.', ...
            femRoot);
    end

    matFiles = dir(fullfile(femRoot, '**', ...
        '*_enabled_method_comparison.mat'));
    if isempty(matFiles)
        error('plotmeshb:NoSavedSamples', ...
            ['No saved per-sample MAT found under:\n%s\n', ...
             'Set cfg.saveSampleMeshes=true and cfg.plotSampleIds to the ', ...
             'desired tests, then rerun run_burgers_fem.'], femRoot);
    end

    nFiles = numel(matFiles);
    savedIds = nan(1, nFiles);
    isStage = false(1, nFiles);
    for i = 1:nFiles
        tok = regexp(matFiles(i).name, 'test_(\d+)', 'tokens', 'once');
        if ~isempty(tok)
            savedIds(i) = str2double(tok{1});
        end
        isStage(i) = contains(matFiles(i).folder, ...
            [filesep stage filesep]);
    end

    for k = testIds(:).'
        candidates = find(savedIds == k);
        if isempty(candidates)
            error('plotmeshb:MissingSample', ...
                ['Test %d has no saved mesh MAT under:\n%s\n', ...
                 'Make sure it was listed in cfg.plotSampleIds with ', ...
                 'cfg.saveSampleMeshes=true, then rerun run_burgers_fem.'], ...
                k, femRoot);
        end

        idx = candidates(find(isStage(candidates), 1));
        if isempty(idx)
            idx = candidates(1);
            fprintf('[plotmeshb] test %d: using %s (no %s file found)\n', ...
                k, matFiles(idx).name, stage);
        end

        filePath = fullfile(matFiles(idx).folder, matFiles(idx).name);
        S = load(filePath);
        cfg = S.cfg;
        sampleResult = S.sampleResult;
        operatorLabel = plotmeshb_label(cfg.operatorName);

        plotmeshb_one_sample( ...
            sampleResult, cfg, operatorLabel, k, root, ...
            operatorExperiment);
        fprintf('[plotmeshb] test %d drawn from %s\n', k, filePath);
    end
end


function plotmeshb_one_sample( ...
        sampleResult, cfg, operatorLabel, testId, root, ...
        operatorExperiment)
    methods = plotmeshb_collect(sampleResult, cfg, operatorLabel);
    if isempty(methods)
        error('plotmeshb:NoEnabledMethods', ...
            'No enabled method has saved mesh data for test %d.', testId);
    end

    sourceId = sampleResult.sourceId;
    figDir = fullfile(root, 'figures', 'process', 'Burgers');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    for k = 1:numel(methods)
    % Burgers space--time domain: (-1,1) x (0,1), physical aspect ratio 2:1.
    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');

    ax = axes(fig, 'Position', [0.07, 0.07, 0.86, 0.86]);

    M = methods(k).M;

    patch(ax, ...
        'Faces', M.elem, ...
        'Vertices', M.node, ...
        'FaceColor', 'w', ...
        'EdgeColor', [0.4 0.4 0.4], ...
        'LineWidth', 0.25);

    axis(ax, 'tight');
    axis(ax, 'off');
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end

    outPdf = fullfile(figDir, sprintf( ...
        'burgers_%s_%s_t%03d_i%04d_%s.pdf', ...
        cfg.operatorName, ...
        strrep(operatorExperiment, '_mse', ''), ...
        testId, sourceId, methods(k).tag));

    plotmeshb_export(fig, outPdf);

    outPng = strrep(outPdf, '.pdf', '.png');
    try
        print(fig, outPng, '-dpng', '-r150');
    catch
        exportgraphics(fig, outPng, 'Resolution', 150);
    end

    close(fig);
end
end


function methods = plotmeshb_collect(sampleResult, cfg, operatorLabel)
    methods = struct('label', {}, 'tag', {}, 'M', {});

    if cfg.runFNOSeededAFEM
        methods = plotmeshb_add(methods, sampleResult, ...
            sprintf('%s predicted-score FEM + %d AFEM', ...
                operatorLabel, cfg.hybridAFEMCycles), ...
            sprintf('op%d', cfg.hybridAFEMCycles), ...
            {'operatorHybrid', 'operatorSeed'});
    end
    if cfg.runTargetSeededAFEM
        methods = plotmeshb_add(methods, sampleResult, ...
            sprintf('Target score FEM + %d AFEM', ...
                cfg.hybridAFEMCycles), ...
            sprintf('tgt%d', cfg.hybridAFEMCycles), ...
            {'targetHybrid', 'targetSeed'});
    end
end


function out = plotmeshb_add(methods, sampleResult, label, tag, fieldNames)
    out = methods;
    for k = 1:numel(fieldNames)
        if isfield(sampleResult, fieldNames{k}) && ...
                isfield(sampleResult.(fieldNames{k}), 'node') && ...
                isfield(sampleResult.(fieldNames{k}), 'elem')
            out(end+1) = struct('label', label, 'tag', tag, ...
                'M', sampleResult.(fieldNames{k})); %#ok<AGROW>
            return;
        end
    end
end


function label = plotmeshb_label(operatorName)
    switch lower(operatorName)
        case 'fno'
            label = 'FNO';
        case 'cno'
            label = 'CNO';
        case 'deeponet'
            label = 'DeepONet';
        case 'pod_deeponet'
            label = 'POD-DeepONet';
        case 'transolver'
            label = 'Transolver';
        otherwise
            label = upper(operatorName);
    end
end


function plotmeshb_export(fig, outPdf)
    % Fixed 800 x 400 pt (2:1) page matching the Burgers domain.
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
end
