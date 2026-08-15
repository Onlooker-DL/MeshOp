function plotmeshrd_sol3d(operatorName, operatorExperiment, testIds, stage)
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
% Output (figures/pltfig/reaction_diffusion/sol_err/):
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

    figDir = fullfile(root, 'figures', 'pltfig', ...
        'reaction_diffusion', 'sol_err3d');
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
                    base = sprintf( ...
                        'reaction_diffusion_%s_%s_t%03d_i%04d_%s_3d_sol', ...
                        operatorName, ...
                        strrep(operatorExperiment, '_mse', ''), ...
                        k, datasetIndex, M.tag);
                    plotmeshrd_sol3d_panel(figDir, base, M.F);
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


function plotmeshrd_sol3d_panel(figDir, base, F)
%PLOTMESHRD_SOL3D_PANEL Draw u_h on z=0 and y=0.5 as colored 3-D surfaces.

    nGrid = 129;
    v = linspace(0, 1, nGrid);
    [X1, Y1] = meshgrid(v, v);
    Z1 = zeros(size(X1));
    C1 = double(F(X1, Y1, Z1));
    [X2, Z2] = meshgrid(v, v);
    Y2 = 0.5 * ones(size(X2));
    C2 = double(F(X2, Y2, Z2));

    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [10 10 760 620], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.04 0.04 0.86 0.88]);
    hold(ax, 'on');
    surf(ax, X1, Y1, Z1, C1, 'EdgeColor', 'none', 'FaceAlpha', 0.92);
    surf(ax, X2, Y2, Z2, C2, 'EdgeColor', 'none', 'FaceAlpha', 0.92);
    plotmeshrd_sol3d_cube(ax);
    view(ax, 3);
    axis(ax, 'equal');
    axis(ax, 'off');

    colormap(ax, parula);
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    plotmeshrd_sol_export(fig, fullfile(figDir, [base '.pdf']));
end

function plotmeshrd_sol3d_cube(ax)
    v = [0 0 0;1 0 0;1 1 0;0 1 0;0 0 1;1 0 1;1 1 1;0 1 1];
    e = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
    line(ax, v(e(:,1),1), v(e(:,1),2), v(e(:,1),3), ...
        'Color', [0.2 0.2 0.2], 'LineWidth', 0.8);
    line(ax, v(e(:,2),1), v(e(:,2),2), v(e(:,2),3), ...
        'Color', [0.2 0.2 0.2], 'LineWidth', 0.8);
end


function plotmeshrd_sol_export(fig, outPdf)
%PLOTMESHRD_SOL_EXPORT Export the 760x620 figure as PDF and PNG.

    set(fig, ...
        'PaperUnits', 'points', ...
        'PaperSize', [760, 620], ...
        'PaperPosition', [0, 0, 760, 620]);

    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);

    outPng = strrep(outPdf, '.pdf', '.png');
    try
        print(fig, outPng, '-dpng', '-r150');
    catch
        exportgraphics(fig, outPng, 'Resolution', 150);
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