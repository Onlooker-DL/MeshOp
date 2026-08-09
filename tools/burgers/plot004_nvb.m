function plot004_nvb(operatorName, operatorExperiment, maxGen, plotRounds)
%PLOT004_NVB Adaptive NVB mesh evolution for Burgers test 004.
%
%   Reads the refinement-demand field of Burgers test 004 from
%   predictions.mat, truncates it to maxGen (=8) levels, and grows the
%   initial 32x32 uniform triangular mesh by adaptive newest-vertex
%   bisection: each element is refined until its generation reaches the
%   local demand, with the NVB conformity closure applied.
%
%   Three figures are saved: the initial mesh, the mesh after round 4, and
%   the mesh after round maxGen (8).
%
% Usage:
%   plot004_nvb('fno', 'b3000_mse')
%   plot004_nvb('fno', 'b3000_mse', 8, [0 4 8])
%
% Output (figures/pltfig/burgers/initial/):
%   burgers_test004_round<r>_mesh.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'b3000_mse';
    end
    if nargin < 3 || isempty(maxGen)
        maxGen = 8;
    end
    if nargin < 4 || isempty(plotRounds)
        plotRounds = [0, 4, maxGen];
    end
    validateattributes(maxGen, {'numeric'}, {'scalar','integer','>=',1});

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    predictionFile = fullfile(root, 'result', 'operators', 'burgers', ...
        operatorName, operatorExperiment, 'predictions.mat');
    if exist(predictionFile, 'file') ~= 2
        error('plot004_nvb:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end

    S = load(predictionFile);
    if ~isfield(S, 'target_score_test') || ~isfield(S, 'query_x') || ...
            ~isfield(S, 'query_t')
        error('plot004_nvb:MissingFields', ...
            'predictions.mat must contain target_score_test, query_x, query_t.');
    end

    % Test 004 is the 4th entry of the exported test set.
    testId = 4;
    targetScore = double(S.target_score_test(:, :, testId));   % (nx, nt)
    targetScore = min(max(targetScore, 0), double(maxGen));

    qx = double(S.query_x(:));
    qt = double(S.query_t(:));
    [node0, elem0, opp0] = uniform_tri_mesh(32);
    gen0 = zeros(size(elem0, 1), 1);

    snapshots = cell(maxGen + 1, 1);
    snapshots{1} = struct('node', node0, 'elem', elem0, 'gen', gen0);

    node = node0;
    elem = elem0;
    opp = opp0;
    gen = gen0;

    for r = 1:maxGen
        [node, elem, opp, gen] = nvb_adaptive_round( ...
            node, elem, opp, gen, targetScore, qx, qt, maxGen);
        snapshots{r + 1} = struct('node', node, 'elem', elem, 'gen', gen);
    end

    figDir = fullfile(root, 'figures', 'pltfig', 'burgers', 'initial');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    for r = plotRounds
        if r < 0 || r > maxGen
            warning('plot004_nvb:BadRound', ...
                'Round %d out of range, skipped.', r);
            continue;
        end
        snap = snapshots{r + 1};
        base = sprintf('burgers_test004_round%d_mesh', r);
        plot004_nvb_one(figDir, base, snap.node, snap.elem, ...
            snap.gen, maxGen, r);
        fprintf('Saved %s (%d elements)\n', ...
            fullfile(figDir, base), size(snap.elem, 1));
    end
end


function [node, elem, opp, gen] = nvb_adaptive_round( ...
        node, elem, opp, gen, score, qx, qt, maxGen)
    % One adaptive NVB round: split every element below its local demand,
    % then propagate the conformity closure until no hanging edges remain.
    cx = (node(elem(:, 1), 1) + node(elem(:, 2), 1) + ...
          node(elem(:, 3), 1)) / 3;
    ct = (node(elem(:, 1), 2) + node(elem(:, 2), 2) + ...
          node(elem(:, 3), 2)) / 3;
    sc = interp2(qt, qx, score, ct, cx, 'linear', 0.0);
    gK = floor(sc + 0.5);
    gK = min(max(gK, 0), double(maxGen));

    marked = find(gen < gK);
    while ~isempty(marked)
        [node, elem, opp, gen, splitKeys] = bisect_marked( ...
            node, elem, opp, gen, marked);
        marked = find_closure_units(elem, splitKeys);
    end
end


function [node, elem, opp, gen, splitKeys] = bisect_marked( ...
        node, elem, opp, gen, marked)
    % Bisect the marked elements along their tagged edges.
    marked = marked(:);
    N = numel(marked);
    splitKeys = zeros(N, 1, 'uint64');

    newElem = cell(N, 1);
    newOpp = cell(N, 1);
    newGen = cell(N, 1);

    for i = 1:N
        idx = marked(i);
        others = elem(idx, elem(idx, :) ~= opp(idx));
        a = others(1);
        b = others(2);
        c = opp(idx);
        mid = 0.5 * (node(a, :) + node(b, :));
        node = [node; mid]; %#ok<AGROW>
        m = size(node, 1);

        child1 = [a, m, c];
        child2 = [b, c, m];
        newElem{i} = [child1; child2];
        newOpp{i} = [m; m];
        newGen{i} = [gen(idx) + 1; gen(idx) + 1];
        splitKeys(i) = edge_key(a, b);
    end

    keep = true(size(elem, 1), 1);
    keep(marked) = false;
    elem = [elem(keep, :); vertcat(newElem{:})];
    opp = [opp(keep); vertcat(newOpp{:})];
    gen = [gen(keep); vertcat(newGen{:})];
end


function extra = find_closure_units(elem, splitKeys)
    % Any element with an edge in splitKeys must split for conformity.
    n = size(elem, 1);
    extra = zeros(0, 1);
    for i = 1:n
        e12 = edge_key(elem(i, 1), elem(i, 2));
        e23 = edge_key(elem(i, 2), elem(i, 3));
        e31 = edge_key(elem(i, 3), elem(i, 1));
        if any(e12 == splitKeys) || any(e23 == splitKeys) || ...
                any(e31 == splitKeys)
            extra(end + 1) = i; %#ok<AGROW>
        end
    end
end


function key = edge_key(a, b)
    if a < b
        key = uint64(a) * uint64(4294967296) + uint64(b);
    else
        key = uint64(b) * uint64(4294967296) + uint64(a);
    end
end


function [node, elem, opp] = uniform_tri_mesh(N)
    x = linspace(-1, 1, N + 1);
    t = linspace(0, 1, N + 1);
    [X, T] = meshgrid(x, t);
    node = [X(:), T(:)];

    elem = zeros(2 * N * N, 3);
    opp = zeros(2 * N * N, 1);
    idx = 0;
    for j = 1:N
        for i = 1:N
            v1 = (j - 1) * (N + 1) + i;
            v2 = v1 + 1;
            v3 = v1 + (N + 1);
            v4 = v3 + 1;
            idx = idx + 1;
            elem(idx, :) = [v1, v2, v4];
            opp(idx) = v2;
            idx = idx + 1;
            elem(idx, :) = [v1, v4, v3];
            opp(idx) = v3;
        end
    end
end


function plot004_nvb_one(figDir, base, node, elem, gen, maxGen, roundNo)
    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 800, 400], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.06, 0.06, 0.88, 0.88]);
    patch(ax, 'Faces', elem, 'Vertices', node, ...
        'FaceColor', 'w', 'EdgeColor', [0.25 0.25 0.25], ...
        'LineWidth', 0.3);
    axis(ax, 'equal');
    axis(ax, 'tight');
    box(ax, 'on');
    xlabel(ax, 'x');
    ylabel(ax, 't');
    plot004_nvb_export(fig, ...
        fullfile(figDir, [base '.pdf']), ...
        fullfile(figDir, [base '.png']));
end


function plot004_nvb_export(fig, outPdf, outPng)
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [800, 400], ...
        'PaperPosition', [0, 0, 800, 400]);
    print(fig, outPdf, '-dpdf', '-painters');
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end
    close(fig);
end
