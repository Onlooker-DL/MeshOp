function pltopera_rd(operatorName, operatorExperiment, testIds)
%PLTOPERA_RD Plot target vs predicted refinement-score slices
%            (reaction-diffusion).
%
% Usage:
%   pltopera_rd('fno', 'rd_accu_b3000_mse')
%   pltopera_rd('cno', 'rd_accu_b3000_mse', [1 2 3])
%
% Reads
%   <project>/result/operators/reaction_diffusion/<operatorName>/<operatorExperiment>/predictions.mat
% which contains pred_score and target_score_test as nx-by-ny-by-nz-by-nTest
% volume fields, together with query_x, query_y, query_z,
% source_attempt_id_test and max_score.
%
% The 3-D volume is plotted as the y=0.5 slice in the x-z plane (the same
% view used by the FEM comparison code). For every requested test sample,
% two vector PDFs and two 300-dpi PNGs are written:
%   <project>/figures/paper_fig/reaction_diffusion/<op>_<exp>/test<k>_id<id>/...
%       reaction_diffusion_<op>_<exp>_test<k>_id<id>_target.pdf / .png
%       reaction_diffusion_<op>_<exp>_test<k>_id<id>_predicted.pdf / .png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'rd_accu_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 1;
    end

    % Project root: tools/reaction_diffusion/pltopera_rd.m.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

    predictionFile = fullfile(root, 'result', 'operators', ...
        'reaction_diffusion', operatorName, operatorExperiment, ...
        'predictions.mat');
    if exist(predictionFile, 'file') ~= 2
        error('pltopera_rd:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end

    S = load(predictionFile);
    if ~isfield(S, 'pred_score') || ~isfield(S, 'target_score_test')
        error('pltopera_rd:MissingFields', ...
            'predictions.mat must contain pred_score and target_score_test.');
    end

    predScore = double(S.pred_score);
    targetScore = double(S.target_score_test);
    if ndims(predScore) ~= 4 || ndims(targetScore) ~= 4
        error('pltopera_rd:BadRank', ...
            'Expected 4-D score arrays (nx-by-ny-by-nz-by-nTest).');
    end

    nTest = size(predScore, 4);
    if size(targetScore, 4) ~= nTest
        error('pltopera_rd:SizeMismatch', ...
            'pred_score and target_score_test have different sample counts.');
    end

    queryX = double(S.query_x(:));
    queryY = double(S.query_y(:));
    queryZ = double(S.query_z(:));

    % Two slices per sample:
    %   z = 0   : x-y plane (the boundary face)
    %   y = 0.5 : x-z plane (interior cut, same view as the FEM plots)
    [~, iy] = min(abs(queryY - 0.5));
    iz0 = 1;  % query_z starts at 0.

    sliceSpecs = { ...
        struct('tag', 'z0', 'ylabelText', 'y', ...
            'xv', queryX, 'yv', queryY, ...
            'fieldFn', @(A, k) squeeze(A(:, :, iz0, k))), ...
        struct('tag', 'y05', 'ylabelText', 'z', ...
            'xv', queryX, 'yv', queryZ, ...
            'fieldFn', @(A, k) squeeze(A(:, iy, :, k)))};

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

    methodLabel = pltopera_rd_label(operatorName);
    testIds = testIds(:).';
    problemName = 'reaction_diffusion';

    for k = testIds
        if k < 1 || k ~= round(k) || k > nTest
            error('pltopera_rd:BadTestId', ...
                'Test index %g is out of range [1, %d].', k, nTest);
        end
        sampleId = sampleIds(k);

        figDir = fullfile(root, 'figures', 'paper_fig', problemName, ...
            sprintf('%s_%s', operatorName, operatorExperiment), ...
            sprintf('test%03d_id%d', k, sampleId));
        if exist(figDir, 'dir') ~= 7
            mkdir(figDir);
        end

        for s = 1:numel(sliceSpecs)
            sp = sliceSpecs{s};
            pltopera_rd_panel(figDir, problemName, operatorName, ...
                operatorExperiment, k, sampleId, sp.tag, sp.xv, sp.yv, ...
                sp.ylabelText, sp.fieldFn(targetScore, k), maxScore, ...
                'target');
            pltopera_rd_panel(figDir, problemName, operatorName, ...
                operatorExperiment, k, sampleId, sp.tag, sp.xv, sp.yv, ...
                sp.ylabelText, sp.fieldFn(predScore, k), maxScore, ...
                'predicted');
        end
    end
end

function pltopera_rd_panel(figDir, problemName, operatorName, ...
        operatorExperiment, k, sampleId, sliceTag, xv, yv, ylabelText, ...
        field, maxScore, kind)
    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.06, 0.08, 0.82, 0.86]);
    imagesc(ax, xv, yv, field.');
    axis(ax, 'xy');
    axis(ax, 'tight');
    xlabel(ax, 'x');
    ylabel(ax, ylabelText);
    caxis(ax, [0, maxScore]);
    cb = colorbar(ax, 'Position', [0.90, 0.08, 0.04, 0.86]);
    colormap(ax, parula);

    outPdf = fullfile(figDir, sprintf( ...
        '%s_%s_%s_test%03d_id%d_%s_%s.pdf', ...
        problemName, operatorName, operatorExperiment, ...
        k, sampleId, sliceTag, kind));
    pltopera_rd_export(fig, outPdf);
    outPng = strrep(outPdf, '.pdf', '.png');
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end
    close(fig);
    fprintf('Saved %s\n', outPdf);
end

function pltopera_rd_export(fig, outPdf)
    % Force a fixed 514 x 409 pt page (near-square, consistent with burgers).
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
end

function label = pltopera_rd_label(operatorName)
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
