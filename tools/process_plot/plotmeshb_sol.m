function plotmeshb_sol(operatorName, operatorExperiment, testIds, stage)
%PLOTMESHB_SOL Draw the FEM solution and pointwise error for Burgers.
%
%   Same arguments and MAT lookup as plotmeshb. For every enabled method in
%   the saved configuration, two figures are written: the finite-element
%   solution field u_h and the pointwise error |u_h - u_ref| evaluated on
%   the reference grid stored in the sample MAT.
%
% Usage:
%   plotmeshb_sol('fno', 'b3000_mse', [1 69])
%   plotmeshb_sol('cno', 'b3000_mse', 4, 'base')
%
% Output (figures/process/Burgers/):
%   burgers_<op>_<exp>_t<k>_i<id>_<tag>_sol.pdf/.png
%   burgers_<op>_<exp>_t<k>_i<id>_<tag>_err.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 69;
    end
    if nargin < 4 || isempty(stage)
        stage = 'hybrid';
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    femRoot = fullfile(root, 'result', 'fem', 'burgers', ...
        operatorName, operatorExperiment);
    if exist(femRoot, 'dir') ~= 7
        error('plotmeshb_sol:MissingResults', ...
            'FEM result folder not found:\n%s', femRoot);
    end

    matFiles = dir(fullfile(femRoot, '**', ...
        '*_enabled_method_comparison.mat'));
    if isempty(matFiles)
        error('plotmeshb_sol:NoSavedSamples', ...
            'No per-sample MAT found under:\n%s', femRoot);
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
            error('plotmeshb_sol:MissingSample', ...
                'Test %d has no saved MAT under:\n%s', k, femRoot);
        end
        idx = candidates(find(isStage(candidates), 1));
        if isempty(idx)
            idx = candidates(1);
            fprintf('[plotmeshb_sol] test %d: using %s (no %s file)\n', ...
                k, matFiles(idx).name, stage);
        end

        filePath = fullfile(matFiles(idx).folder, matFiles(idx).name);
        S = load(filePath);
        cfg = S.cfg;
        sampleResult = S.sampleResult;
        operatorLabel = plotmeshb_sol_label(cfg.operatorName);

        methods = plotmeshb_sol_collect(sampleResult, cfg, operatorLabel);
        if isempty(methods)
            error('plotmeshb_sol:NoMethods', ...
                'No enabled method has saved solution data for test %d.', k);
        end

        ref = S.ref;
        if ~isfield(ref, 'U') || ~isfield(ref, 'x') || ~isfield(ref, 't')
            error('plotmeshb_sol:NoReference', ...
                'Sample MAT has no reference solution (ref.U/x/t).');
        end
        refU = double(ref.U);
        refX = double(ref.x(:));
        refT = double(ref.t(:));

        figDir = fullfile(root, 'figures', 'process', 'Burgers');
        if exist(figDir, 'dir') ~= 7
            mkdir(figDir);
        end

        for m = 1:numel(methods)
            M = methods(m).M;
            if ~isfield(M, 'u') || isempty(M.u)
                continue;
            end
            u = double(M.u(:));
            if numel(u) ~= size(M.node, 1)
                warning('plotmeshb_sol:SizeMismatch', ...
                    'Solution size does not match nodes for %s.', methods(m).tag);
                continue;
            end

            base = sprintf('burgers_%s_%s_t%03d_i%04d_%s', ...
                cfg.operatorName, ...
                strrep(operatorExperiment, '_mse', ''), ...
                k, sampleResult.sourceId, methods(m).tag);

            % Solution field.
            plotmeshb_sol_field(figDir, base, M.node, M.elem, u, ...
                sprintf('%s solution', methods(m).label));

            % Pointwise error vs reference (interpolate u_h onto ref grid).
            F = scatteredInterpolant( ...
                double(M.node(:, 1)), double(M.node(:, 2)), u, ...
                'linear', 'none');
            [XX, TT] = meshgrid(refX, refT);
            uhRef = F(XX, TT);
            err = abs(uhRef - refU);
            plotmeshb_sol_error(figDir, base, refX, refT, err, ...
                sprintf('%s pointwise error', methods(m).label));
        end
        fprintf('[plotmeshb_sol] test %d drawn from %s\n', k, filePath);
    end
end


function plotmeshb_sol_field(figDir, base, node, elem, u, labelText)
    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.07, 0.07, 0.86, 0.86]);
    patch(ax, 'Faces', elem, 'Vertices', node, ...
        'FaceVertexCData', u, 'FaceColor', 'interp', ...
        'EdgeColor', 'none');
    axis(ax, 'tight');
    ax.XTick = [];
    ax.YTick = [];
    colormap(ax, parula);
    plotmeshb_sol_export(fig, ...
        fullfile(figDir, [base '_sol.pdf']), ...
        fullfile(figDir, [base '_sol.png']));
end


function plotmeshb_sol_error(figDir, base, refX, refT, err, labelText)
    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.07, 0.07, 0.86, 0.86]);
    imagesc(ax, refX, refT, err);
    axis(ax, 'xy');
    axis(ax, 'tight');
    ax.XTick = [];
    ax.YTick = [];
    colormap(ax, parula);
    plotmeshb_sol_export(fig, ...
        fullfile(figDir, [base '_err.pdf']), ...
        fullfile(figDir, [base '_err.png']));
end


function methods = plotmeshb_sol_collect(sampleResult, cfg, operatorLabel)
    methods = struct('label', {}, 'tag', {}, 'M', {});

    if cfg.runFNOSeededAFEM
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            sprintf('%s predicted-score FEM + %d AFEM', ...
                operatorLabel, cfg.hybridAFEMCycles), ...
            sprintf('op%d', cfg.hybridAFEMCycles), ...
            {'operatorHybrid', 'operatorSeed'});
    end
    if cfg.runTargetSeededAFEM
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            sprintf('Target score FEM + %d AFEM', cfg.hybridAFEMCycles), ...
            sprintf('tgt%d', cfg.hybridAFEMCycles), ...
            {'targetHybrid', 'targetSeed'});
    end
    if cfg.runTimeMatchedAFEM
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            'Time-matched AFEM', 'atime', {'afemTime'});
    end
    if cfg.runAccuracyMatchedAFEM
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            'Accuracy-matched AFEM', 'aacc', {'afemAccuracy'});
    end
    if cfg.runFixedRefinementAFEM
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            sprintf('Fixed-%d-refine AFEM', cfg.fixedAFEMRefineCycles), ...
            'afix', {'afemFixed'});
    end
    if cfg.runDOFMatchedUniform
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            sprintf('Smallest strictly-above-%s-DOF uniform FEM', operatorLabel), ...
            'udof', {'uniform'});
    end
    if cfg.runFixedUniform
        methods = plotmeshb_sol_add(methods, sampleResult, ...
            sprintf('Fixed-%d-DOF uniform FEM', cfg.fixedUniformFreeDOF), ...
            'ufix', {'fixedUniform'});
    end
end


function out = plotmeshb_sol_add(methods, sampleResult, label, tag, fieldNames)
    out = methods;
    for k = 1:numel(fieldNames)
        if isfield(sampleResult, fieldNames{k}) && ...
                isfield(sampleResult.(fieldNames{k}), 'node') && ...
                isfield(sampleResult.(fieldNames{k}), 'elem') && ...
                isfield(sampleResult.(fieldNames{k}), 'u')
            out(end + 1) = struct('label', label, 'tag', tag, ...
                'M', sampleResult.(fieldNames{k})); %#ok<AGROW>
            return;
        end
    end
end


function label = plotmeshb_sol_label(operatorName)
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


function plotmeshb_sol_export(fig, outPdf, outPng)
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end
    close(fig);
end
