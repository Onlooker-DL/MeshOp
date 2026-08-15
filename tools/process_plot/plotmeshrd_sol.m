function plotmeshrd_sol(operatorName, operatorExperiment, testIds, stage)
%PLOTMESHRD_SOL Draw reaction-diffusion FEM solution and pointwise error slices.
%
%   Same arguments and MAT lookup as plotmeshrd. For every enabled method
%   with a saved final solution, two figures are written per slice:
%   the finite-element solution field u_h and the pointwise error
%   |u_h - u_target| evaluated against the target-score-seeded AFEM
%   solution (the oracle baseline). The high-resolution spectral reference
%   is not stored in the visualization payload, so the target-seeded
%   solution is used as the reference; set the FEM save flag to store the
%   spectral reference if a true |u_h - u_ref| panel is required.
%
% Usage:
%   plotmeshrd_sol('fno', 'rd_accu_b3000_mse', [1 65])
%   plotmeshrd_sol('fno', 'rd_accu_b3000_mse', 4, 'tol_4e-02')
%
% Output (figures/process/ReactionDiffusion/):
%   reaction_diffusion_<op>_<exp>_t<k>_i<id>_<z0|y05>_<tag>_sol.pdf/.png
%   reaction_diffusion_<op>_<exp>_t<k>_i<id>_<z0|y05>_<tag>_err.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'rd_accu_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 65;
    end
    if nargin < 4 || isempty(stage)
        stage = '';
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    femRoot = fullfile(root, 'result', 'fem', 'reaction_diffusion_accu', ...
        operatorName, operatorExperiment);
    if exist(femRoot, 'dir') ~= 7
        error('plotmeshrd_sol:MissingResults', ...
            'FEM result folder not found:\n%s', femRoot);
    end

    visFiles = dir(fullfile(femRoot, '**', 'visualization_data_*.mat'));
    if isempty(visFiles)
        error('plotmeshrd_sol:NoVisualization', ...
            ['No visualization_data_*.mat found under:\n%s\n', ...
             'Set cfg.saveVisualizationData=true and rerun the FEM entry.'], ...
            femRoot);
    end

    figDir = fullfile(root, 'figures', 'process', 'ReactionDiffusion');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    sliceSpecs = { ...
        struct('tag', 'z0', 'dim', 3, 'val', 0.0, 'xl', 'x', 'yl', 'y'), ...
        struct('tag', 'y05', 'dim', 2, 'val', 0.5, 'xl', 'x', 'yl', 'z')};

    plotted = false;

    for k = testIds(:).'
        found = false;

        for f = 1:numel(visFiles)
            filePath = fullfile(visFiles(f).folder, visFiles(f).name);

            if ~isempty(stage) && ...
                    ~contains(visFiles(f).folder, [filesep stage filesep])
                continue;
            end

            S = load(filePath);
            if ~isfield(S, 'visualizationSamples')
                continue;
            end

            for v = 1:numel(S.visualizationSamples)
                Gc = S.visualizationSamples{v};

                if double(Gc.meta.testId(1)) ~= k
                    continue;
                end

                found = true;
                plotted = true;

                datasetIndex = double(Gc.meta.datasetIndex(1));
                refF = plotmeshrd_sol_reference(Gc);
                methods = plotmeshrd_sol_collect(Gc, operatorName, refF);

                if isempty(methods)
                    error('plotmeshrd_sol:NoMethods', ...
                        'No enabled method has saved solution data for test %d.', k);
                end

                for m = 1:numel(methods)
                    M = methods(m);

                    for s = 1:numel(sliceSpecs)
                        sp = sliceSpecs{s};

                        [g1, g2, uSlice] = plotmeshrd_sol_slice(M.F, sp);
                        base = sprintf( ...
                            'reaction_diffusion_%s_%s_t%03d_i%04d_%s_%s_sol', ...
                            operatorName, ...
                            strrep(operatorExperiment, '_mse', ''), ...
                            k, datasetIndex, sp.tag, M.tag);

                        plotmeshrd_sol_panel(figDir, base, g1, g2, uSlice, ...
                            sp.xl, sp.yl);

                        if M.hasErr
                            [~, ~, refSlice] = plotmeshrd_sol_slice(refF, sp);
                            err = abs(uSlice - refSlice);
                            baseErr = strrep(base, '_sol', '_err');
                            plotmeshrd_sol_panel(figDir, baseErr, ...
                                g1, g2, err, sp.xl, sp.yl);
                        end
                    end
                end
                fprintf('[plotmeshrd_sol] test %d drawn from %s\n', k, filePath);
            end

            if found
                break;
            end
        end

        if ~found
            warning('plotmeshrd_sol:MissingSample', ...
                'Test %d not found in any visualization file.', k);
        end
    end

    if ~plotted
        error('plotmeshrd_sol:NothingPlotted', ...
            'No requested test matched the visualization samples.');
    end
end


function refF = plotmeshrd_sol_reference(Gc)
%PLOTMESHRD_SOL_REFERENCE Build the oracle reference from the target score
%method, or return [] when the target-seeded AFEM was not saved.

    if isfield(Gc, 'target')
        refF = plotmeshrd_sol_interpolant(Gc.target);
    else
        refF = [];
    end
end


function methods = plotmeshrd_sol_collect(Gc, operatorName, refF)
%PLOTMESHRD_SOL_COLLECT Collect enabled methods with a solution interpolant.

    methods = struct('label', {}, 'tag', {}, 'F', {}, 'hasErr', {});

    opLabel = plotmeshrd_sol_label(operatorName);

    fields = { ...
        'fno',             sprintf('%s predicted-score FEM', opLabel), 'op'; ...
        'target',          'Target-score FEM', 'tgt'; ...
        'standard',        'Standard AFEM', 'std'; ...
        'accuracyUniform', 'Accuracy-matched uniform', 'ua'; ...
        'fixedUniform',    'Fixed uniform', 'uf'};

    for i = 1:size(fields, 1)
        mf = fields{i, 1};

        if ~isfield(Gc, mf)
            continue;
        end

        F = plotmeshrd_sol_interpolant(Gc.(mf));
        if isempty(F)
            continue;
        end

        methods(end + 1) = struct( ... %#ok<AGROW>
            'label', fields{i, 2}, ...
            'tag', fields{i, 3}, ...
            'F', F, ...
            'hasErr', ~strcmp(mf, 'target') && ~isempty(refF));
    end
end


function F = plotmeshrd_sol_interpolant(Sf)
%PLOTMESHRD_SOL_INTERPOLANT Build a u(x,y,z) interpolant for one method.
%Adaptive states store P/T/u; uniform states store a structured uGrid.

    if isfield(Sf, 'final') && isfield(Sf.final, 'P') && ...
            isfield(Sf.final, 'u')
        P = double(squeeze(Sf.final.P));
        u = double(Sf.final.u(:));

        if size(P, 1) ~= 3 && size(P, 2) == 3
            P = P.';
        end

        if size(P, 1) ~= 3 || numel(u) ~= size(P, 2)
            F = [];
            return;
        end

        F = scatteredInterpolant( ...
            P(1, :).', P(2, :).', P(3, :).', u, ...
            'linear', 'none');

    elseif isfield(Sf, 'uGrid')
        n = double(Sf.cellsPerDirection(1));
        uGrid = double(Sf.uGrid);

        if numel(uGrid) ~= (n + 1)^3
            F = [];
            return;
        end

        g = linspace(0, 1, n + 1);
        F = griddedInterpolant( ...
            {g, g, g}, reshape(uGrid, n + 1, n + 1, n + 1), ...
            'linear', 'nearest');
    else
        F = [];
    end
end


function [g1, g2, uSlice] = plotmeshrd_sol_slice(F, sp)
%PLOTMESHRD_SOL_SLICE Evaluate u on a regular grid over one slice plane.

    nGrid = 201;
    v1 = linspace(0, 1, nGrid);
    v2 = linspace(0, 1, nGrid);
    [XX, YY] = ndgrid(v1, v2);

    if sp.dim == 3
        uSlice = F(XX, YY, zeros(size(XX)));
    else
        uSlice = F(XX, sp.val * ones(size(XX)), YY);
    end

    uSlice = double(uSlice).';
    g1 = v1;
    g2 = v2;
end


function plotmeshrd_sol_panel(figDir, base, g1, g2, field, xl, yl)
%PLOTMESHRD_SOL_PANEL Draw one slice field and export it.

    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.07, 0.07, 0.86, 0.86]);

    imagesc(ax, g1, g2, field);
    axis(ax, 'xy');
    axis(ax, 'tight');
    ax.XTick = [];
    ax.YTick = [];
    colormap(ax, parula);

    plotmeshrd_sol_export(fig, fullfile(figDir, [base '.pdf']));
end


function plotmeshrd_sol_export(fig, outPdf)
%PLOTMESHRD_SOL_EXPORT Export the 514x409 figure as PDF and PNG.

    set(fig, ...
        'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);

    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);

    outPng = strrep(outPdf, '.pdf', '.png');
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end

    close(fig);
end


function label = plotmeshrd_sol_label(operatorName)
%PLOTMESHRD_SOL_LABEL Convert operator identifier to display label.

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