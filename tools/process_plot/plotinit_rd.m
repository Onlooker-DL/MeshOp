function plotinit_rd(operatorName, operatorExperiment, testIds)
%PLOTINIT_RD Plot the reaction-diffusion boundary GRF field.
%
% The PDE input of a reaction-diffusion instance is the Dirichlet boundary
% field g(x,y) on the z=0 face, drawn from a Gaussian random field. This
% script reads the exact boundary reconstruction stored in predictions.mat
% (field boundary_test) and plots it as a heatmap. It is independent of any
% neural operator.
%
% Usage:
%   plotinit_rd('fno', 'rd_accu_b3000_mse')
%   plotinit_rd('fno', 'rd_accu_b3000_mse', [1 12 65])
%
% Output:
%   figures/process/ReactionDiffusion/
%       reaction_diffusion_<op>_<exp>_test<k>_id<id>_boundary.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'rd_accu_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 1;
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    predictionFile = fullfile(root, 'result', 'operators', ...
        'reaction_diffusion', operatorName, operatorExperiment, ...
        'predictions.mat');
    if exist(predictionFile, 'file') ~= 2
        error('plotinit_rd:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end

    S = load(predictionFile);
    if ~isfield(S, 'boundary_test')
        error('plotinit_rd:MissingFields', ...
            'predictions.mat must contain boundary_test.');
    end

    boundary = double(S.boundary_test);   % (nx, ny, nTest)
    if ndims(boundary) ~= 3
        error('plotinit_rd:BadRank', ...
            'Expected 3-D boundary array (nx-by-ny-by-nTest).');
    end
    nTest = size(boundary, 3);

    if isfield(S, 'query_x')
        qx = double(S.query_x(:));
        qy = double(S.query_y(:));
    else
        qx = (1:size(boundary, 1)).';
        qy = (1:size(boundary, 2)).';
    end

    if isfield(S, 'source_attempt_id_test')
        sampleIds = double(S.source_attempt_id_test(:));
    elseif isfield(S, 'original_indices')
        sampleIds = double(S.original_indices(:));
    else
        sampleIds = (1:nTest).';
    end

    figDir = fullfile(root, 'figures', 'process', 'ReactionDiffusion');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    for k = testIds(:).'
        if k < 1 || k ~= round(k) || k > nTest
            error('plotinit_rd:BadTestId', ...
                'Test index %g is out of range [1, %d].', k, nTest);
        end
        sampleId = sampleIds(k);

        fig = figure('Color', 'w', 'Units', 'points', ...
            'Position', [1, 1, 514, 409], 'Visible', 'on');
        ax = axes(fig, 'Position', [0.07, 0.07, 0.86, 0.86]);
        imagesc(ax, qx, qy, boundary(:, :, k).');
        axis(ax, 'xy');
        axis(ax, 'tight');
        ax.XTick = [];
        ax.YTick = [];
        caxis(ax, 'auto');
        colormap(ax, parula);

        base = sprintf('reaction_diffusion_%s_%s_test%03d_id%d_boundary', ...
            operatorName, operatorExperiment, k, sampleId);
        plotinit_rd_export(fig, ...
            fullfile(figDir, [base '.pdf']), ...
            fullfile(figDir, [base '.png']));
        fprintf('Saved %s\n', fullfile(figDir, base));
    end
end


function plotinit_rd_export(fig, outPdf, outPng)
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
    try
        print(fig, outPng, '-dpng', '-r150');
    catch
        exportgraphics(fig, outPng, 'Resolution', 150);
    end
    close(fig);
end
