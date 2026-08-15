function plot004_nvb(operatorName, operatorExperiment, maxGen, plotRounds, testId)
%PLOT004_NVB Adaptive FEM mesh evolution for a Burgers test sample.
%
%   Runs the REAL residual-driven adaptive FEM (solve -> estimate ->
%   Doerfler mark -> periodic NVB refine -> Newton solve) for maxGen
%   cycles on the common 32x32 mesh, exactly as burgers_fem_comparison.m's
%   fixed-refinement AFEM row. Only one sample is processed, and the mesh
%   after each requested round is saved as an image (no axes/numbers).
%
% Usage:
%   plot004_nvb('fno', 'b3000_mse')
%   plot004_nvb('fno', 'b3000_mse', 14, [0 4 8 12 14], 69)
%
% Output (figures/pltfig/burgers/initial/):
%   burgers_test<id>_round<r>_mesh.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'b3000_mse';
    end
    if nargin < 3 || isempty(maxGen)
        maxGen = 14;
    end
    if nargin < 4 || isempty(plotRounds)
        plotRounds = [0, 4, 8, 12, 14];
    end
    if nargin < 5 || isempty(testId)
        testId = 69;
    end
    validateattributes(maxGen, {'numeric'}, ...
        {'scalar','integer','>=',1});
    validateattributes(testId, {'numeric'}, ...
        {'scalar','integer','>=',1});

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(root, 'src', 'fem'));

    % Dataset sample index for the selected test (same as run_burgers_fem).
    predictionFile = fullfile(root, 'result', 'operators', 'burgers', ...
        operatorName, operatorExperiment, 'predictions.mat');
    if exist(predictionFile, 'file') ~= 2
        error('plot004_nvb:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end
    S = load(predictionFile, 'source_attempt_id_test');
    if isfield(S, 'source_attempt_id_test') && ...
            numel(S.source_attempt_id_test) >= testId
        sourceId = double(S.source_attempt_id_test(testId));
    else
        sourceId = testId;
    end

    datasetFile = fullfile(root, 'data', 'burgers', 'burgers_5100.mat');
    if exist(datasetFile, 'file') ~= 2
        error('plot004_nvb:MissingDataset', ...
            'Burgers dataset not found:\n%s', datasetFile);
    end

    [snapshots, ~] = run_burgers_afem_evolution( ...
        datasetFile, sourceId, maxGen, []);

    figDir = fullfile(root, 'figures', 'pltfig', 'burgers', 'initial');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    nSnap = numel(snapshots);
    for r = plotRounds
        if r < 0 || r > maxGen
            warning('plot004_nvb:BadRound', ...
                'Round %d out of range, skipped.', r);
            continue;
        end
        idx = min(r + 1, nSnap);
        snap = snapshots{idx};
        base = sprintf('burgers_test%03d_round%d_mesh', testId, r);
        plot004_nvb_one(figDir, base, snap.node, snap.elem, ...
            [], maxGen, r);
        fprintf('Saved %s (%d elements)\n', ...
            fullfile(figDir, base), size(snap.elem, 1));
    end
end


function plot004_nvb_one(figDir, base, node, elem, gen, maxGen, roundNo)
    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 800, 400], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.06, 0.06, 0.88, 0.88]);
    patch(ax, 'Faces', elem, 'Vertices', node, ...
        'FaceColor', 'w', 'EdgeColor', [0.4 0.4 0.4], ...
        'LineWidth', 0.25);
    axis(ax, 'equal');
    axis(ax, 'tight');
    axis(ax, 'off');
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    plot004_nvb_export(fig, ...
        fullfile(figDir, [base '.pdf']), ...
        fullfile(figDir, [base '.png']));
end


function plot004_nvb_export(fig, outPdf, outPng)
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [800, 400], ...
        'PaperPosition', [0, 0, 800, 400]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
    try
        print(fig, outPng, '-dpng', '-r150');
    catch
        exportgraphics(fig, outPng, 'Resolution', 150);
    end
    close(fig);
end
