function pltopera(operatorName, operatorExperiment, testIds)
%PLTOPERA Plot target vs predicted refinement-score heatmaps (Burgers).
%
% Usage:
%   pltopera('fno', 'b3000_mse')
%   pltopera('fno', 'b3000_mse', [1 2 3])
%
% Inputs:
%   operatorName       - operator directory name, e.g. 'fno', 'deeponet',
%                        'pod_deeponet', 'transolver'
%   operatorExperiment - experiment directory name, e.g. 'b3000_mse'
%   testIds            - 1-based test-sample indices to plot (default: [1])
%
% Reads
%   <project>/result/operators/burgers/<operatorName>/<operatorExperiment>/predictions.mat
% which contains pred_score and target_score_test as nx-by-nt-by-nTest
% arrays, together with query_x, query_t, source_attempt_id_test and
% max_score.
%
% For every requested test sample, two separate vector PDFs are written:
%   <project>/figures/paper_fig/burgers/<operator>_<experiment>/test<k>_id<id>/burgers_<operator>_<experiment>_test<k>_id<id>_target.pdf
%   <project>/figures/paper_fig/burgers/<operator>_<experiment>/test<k>_id<id>/burgers_<operator>_<experiment>_test<k>_id<id>_predicted.pdf
% A PNG (300 dpi) is written next to each PDF.

    if nargin < 1 || isempty(operatorName)
        operatorName = 'cno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 69;
    end

    % Project root: tools/burgers/pltopera.m -> project root.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

    predictionFile = fullfile(root, 'result', 'operators', 'burgers', ...
        operatorName, operatorExperiment, 'predictions.mat');
    if exist(predictionFile, 'file') ~= 2
        error('pltopera:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end

    S = load(predictionFile);
    if ~isfield(S, 'pred_score') || ~isfield(S, 'target_score_test')
        error('pltopera:MissingFields', ...
            'predictions.mat must contain pred_score and target_score_test.');
    end

    predScore = double(S.pred_score);
    targetScore = double(S.target_score_test);
    if ndims(predScore) ~= 3 || ndims(targetScore) ~= 3
        error('pltopera:BadRank', ...
            'Expected 3-D score arrays (nx-by-nt-by-nTest).');
    end

    nTest = size(predScore, 3);
    if size(targetScore, 3) ~= nTest
        error('pltopera:SizeMismatch', ...
            'pred_score and target_score_test have different sample counts.');
    end

    if isfield(S, 'query_x')
        queryX = double(S.query_x(:));
        queryT = double(S.query_t(:));
    else
        queryX = (1:size(predScore, 1)).';
        queryT = (1:size(predScore, 2)).';
    end

    if isfield(S, 'source_attempt_id_test')
        sampleIds = double(S.source_attempt_id_test(:));
    elseif isfield(S, 'original_indices')
        sampleIds = double(S.original_indices(:));
    else
        sampleIds = (1:nTest).';
    end

    if isfield(S, 'max_score')
        maxScore = double(S.max_score(1));
    else
        maxScore = max([targetScore(:); predScore(:)]);
    end

    methodLabel = pltopera_method_label(operatorName);
    testIds = testIds(:).';
    problemName = 'burgers';

    for k = testIds
        if k < 1 || k ~= round(k) || k > nTest
            error('pltopera:BadTestId', ...
                'Test index %g is out of range [1, %d].', k, nTest);
        end
        sampleId = sampleIds(k);
        caseTitle = sprintf('%s-id:%d', methodLabel, sampleId);

        % Per-sample output folder under figures/paper_fig.
        figDir = fullfile(root, 'figures', 'paper_fig', 'burgers', ...
            sprintf('%s_%s', operatorName, operatorExperiment), ...
            sprintf('test%03d_id%d', k, sampleId));
        if exist(figDir, 'dir') ~= 7
            mkdir(figDir);
        end

        % Target panel (separate figure, vector PDF).
        % Fixed layout so target and predicted PDFs are pixel-identical
        % in size: same figure, same axes position, same colorbar position.
        % Near-square page for side-by-side layout in the paper.
        fig = figure('Color', 'w', 'Units', 'points', ...
            'Position', [1, 1, 514, 409], ...
            'Visible', 'on');
        ax = axes(fig, 'Position', [0.06, 0.08, 0.82, 0.86]);
        imagesc(ax, queryX, queryT, targetScore(:, :, k).');
        axis(ax, 'xy');
        axis(ax, 'tight');
        xlabel(ax, 'x');
        ylabel(ax, 't');
        caxis(ax, [0, maxScore]);
        cb = colorbar(ax, 'Position', [0.90, 0.08, 0.04, 0.86]);
        colormap(ax, parula);
        targetPdf = fullfile(figDir, ...
            sprintf('%s_%s_%s_test%03d_id%d_target.pdf', ...
            problemName, operatorName, operatorExperiment, k, sampleId));
        pltopera_export(fig, targetPdf);
        targetPng = strrep(targetPdf, '.pdf', '.png');
        try
            print(fig, targetPng, '-dpng', '-r300');
        catch
            exportgraphics(fig, targetPng, 'Resolution', 300);
        end
        close(fig);

        % Predicted panel (separate figure, vector PDF).
        fig = figure('Color', 'w', 'Units', 'points', ...
            'Position', [1, 1, 514, 409], ...
            'Visible', 'on');
        ax = axes(fig, 'Position', [0.06, 0.08, 0.82, 0.86]);
        imagesc(ax, queryX, queryT, predScore(:, :, k).');
        axis(ax, 'xy');
        axis(ax, 'tight');
        xlabel(ax, 'x');
        ylabel(ax, 't');
        caxis(ax, [0, maxScore]);
        cb = colorbar(ax, 'Position', [0.90, 0.08, 0.04, 0.86]);
        colormap(ax, parula);
        predictedPdf = fullfile(figDir, ...
            sprintf('%s_%s_%s_test%03d_id%d_predicted.pdf', ...
            problemName, operatorName, operatorExperiment, k, sampleId));
        pltopera_export(fig, predictedPdf);
        predictedPng = strrep(predictedPdf, '.pdf', '.png');
        try
            print(fig, predictedPng, '-dpng', '-r300');
        catch
            exportgraphics(fig, predictedPng, 'Resolution', 300);
        end
        close(fig);

        fprintf('Saved %s\n', targetPdf);
        fprintf('Saved %s\n', predictedPdf);
    end
end

function pltopera_export(fig, outPdf)
%PLTOPERA_EXPORT Save one figure as a vector PDF.
    % Force a fixed 514 x 409 pt page (near-square, consistent across models).
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
end

function label = pltopera_method_label(operatorName)
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
