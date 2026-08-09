function plotmeshrd(operatorName, operatorExperiment, testIds)
%PLOTMESHRD Draw tetrahedral FEM mesh slices for reaction-diffusion.
%
% Usage:
%   plotmeshrd('fno', 'rd_accu_b3000_mse')
%   plotmeshrd('fno', 'rd_accu_b3000_mse', [1 2])
%
% The 3-D tetrahedral meshes saved by run_reaction_diffusion_accu_fem are
% sliced at z = 0 (x-y plane) and y = 0.5 (x-z plane); each slice is drawn
% as a 2-D triangular mesh. Enabled methods (fno / target / standard) are
% taken from the saved visualization data.
%
% Output:
%   figures/pltfig/reaction_diffusion/femmesh/<op>_<exp>/test<k>_id<id>/
%       reaction_diffusion_<op>_<exp>_test<k>_id<id>_<z0|y05>_<method>_mesh.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'cno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'rd_accu_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 65;
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    femRoot = fullfile(root, 'result', 'fem', 'reaction_diffusion_accu', ...
        operatorName, operatorExperiment);

    if exist(femRoot, 'dir') ~= 7
        error('plotmeshrd:MissingResults', ...
            'FEM result folder not found:\n%s', femRoot);
    end

    visFiles = dir(fullfile(femRoot, '**', 'visualization_data_*.mat'));

    if isempty(visFiles)
        error('plotmeshrd:NoVisualization', ...
            ['No visualization_data_*.mat found under:\n%s\n', ...
             'Set cfg.saveVisualizationData=true and rerun the FEM entry.'], ...
            femRoot);
    end

    visFile = fullfile(visFiles(1).folder, visFiles(1).name);
    fprintf('[plotmeshrd] using %s\n', visFile);

    S = load(visFile);

    if ~isfield(S, 'visualizationSamples')
        error('plotmeshrd:MissingField', ...
            'visualizationSamples not found in the MAT file.');
    end

    methodFields = {'fno', 'target', 'standard'};

    methodLabels = struct( ...
        'fno', operatorLabel(operatorName), ...
        'target', 'Target', ...
        'standard', 'Standard AFEM');

    nVis = numel(S.visualizationSamples);

    if isempty(testIds)
        testIds = zeros(1, nVis);

        for v = 1:nVis
            testIds(v) = double( ...
                S.visualizationSamples{v}.meta.testId(1));
        end

        fprintf('[plotmeshrd] plotting all visualization samples: %s\n', ...
            mat2str(testIds));
    end

    plotted = false;

    for k = testIds(:).'
        found = false;

        for v = 1:nVis
            Gc = S.visualizationSamples{v};

            testId = double(Gc.meta.testId(1));

            if testId ~= k
                continue;
            end

            found = true;
            plotted = true;

            datasetIndex = double(Gc.meta.datasetIndex(1));

            figDir = fullfile(root, 'figures', 'pltfig', ...
                'reaction_diffusion', 'femmesh', ...
                sprintf('%s_%s', operatorName, operatorExperiment), ...
                sprintf('test%03d_id%d', k, datasetIndex));

            if exist(figDir, 'dir') ~= 7
                mkdir(figDir);
            end

            % -------------------------------------------------------------
            % Slices:
            %   z = 0   -> x-y plane
            %   y = 0.5 -> x-z plane
            % -------------------------------------------------------------
            sliceSpecs = { ...
                struct('tag', 'z0', 'dim', 3, 'val', 0.0, ...
                    'xdim', 1, 'ydim', 2), ...
                struct('tag', 'y05', 'dim', 2, 'val', 0.5, ...
                    'xdim', 1, 'ydim', 3)};

            for m = 1:numel(methodFields)

                mf = methodFields{m};

                if ~isfield(Gc, mf)
                    continue;
                end

                Sf = Gc.(mf);

                if ~isfield(Sf, 'final')
                    continue;
                end

                P = double(Sf.final.P);
                T = double(Sf.final.T);

                % v7.3 load can add singleton dimensions; normalize.
                P = squeeze(P);
                T = squeeze(T);

                fprintf('[plotmeshrd] test %d, %s: size(P)=%s, size(T)=%s\n', ...
                    k, mf, mat2str(size(P)), mat2str(size(T)));

                if size(P, 1) == 3
                    % already 3 x N
                elseif size(P, 2) == 3
                    P = P.';
                else
                    warning('plotmeshrd:BadMesh', ...
                        'Unexpected mesh size for %s.', mf);
                    continue;
                end

                if size(T, 1) == 4
                    % already 4 x M
                elseif size(T, 2) == 4
                    T = T.';
                else
                    warning('plotmeshrd:BadMesh', ...
                        'Unexpected tetrahedron size for %s.', mf);
                    continue;
                end

                for s = 1:numel(sliceSpecs)

                    sp = sliceSpecs{s};

                    [sp2, st2] = slice_tetra_mesh(P, T, sp.dim, sp.val);

                    if isempty(st2)
                        fprintf('[plotmeshrd] test %d, %s, %s: empty slice\n', ...
                            k, sp.tag, mf);
                        continue;
                    end

                    outPdf = fullfile(figDir, sprintf( ...
                        'reaction_diffusion_%s_%s_test%03d_id%d_%s_%s_mesh.pdf', ...
                        operatorName, operatorExperiment, k, datasetIndex, ...
                        sp.tag, mf));

                    plotmeshrd_panel( ...
                        figDir, sp, sp2, st2, ...
                        methodLabels.(mf), outPdf);

                    fprintf('[plotmeshrd] test %d, %s, %s -> %s\n', ...
                        k, sp.tag, mf, outPdf);
                end
            end
        end

        if ~found
            warning('plotmeshrd:MissingSample', ...
                'Test %d not present in the visualization data.', k);
        end
    end

    if ~plotted
        error('plotmeshrd:NothingPlotted', ...
            'No requested test matched the visualization samples.');
    end
end


function plotmeshrd_panel(figDir, sp, node2, tri2, methodLabel, outPdf)
%PLOTMESHRD_PANEL Plot one reaction-diffusion mesh slice.
% Keep the original 800x400 stretched layout.
% Only adjust mesh line darkness and thickness.

    %#ok<INUSD>

    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 800, 400], 'Visible', 'on');

    ax = axes(fig, 'Position', [0.06, 0.06, 0.88, 0.88]);

    if strcmpi(sp.tag, 'z0')
        % Dense bottom-face mesh:
        % thinner/lighter than original, but still visible after LaTeX scaling.
        edgeColor = [0.45 0.45 0.45];
        lineWidth = 0.22;
    else
        % Interior slice is less dense.
        edgeColor = [0.35 0.35 0.35];
        lineWidth = 0.28;
    end

    patch(ax, ...
        'Faces', tri2, ...
        'Vertices', node2.', ...
        'FaceColor', 'w', ...
        'EdgeColor', edgeColor, ...
        'LineWidth', lineWidth);

    % Keep the stretched appearance.
    axis(ax, 'tight');
    box(ax, 'on');

    if sp.dim == 3
        xlabel(ax, 'x');
        ylabel(ax, 'y');
    else
        xlabel(ax, 'x');
        ylabel(ax, 'z');
    end

    set(ax, ...
        'FontSize', 11, ...
        'LineWidth', 0.75, ...
        'TickDir', 'out', ...
        'Layer', 'top');

    plotmeshrd_export(fig, outPdf);

    outPng = strrep(outPdf, '.pdf', '.png');
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end

    close(fig);
end


function [sp, st] = slice_tetra_mesh(P, T, dim, val)
%SLICE_TETRA_MESH Intersect tetrahedral mesh with coordinate = val plane.
%
% P: 3 x N node coordinates
% T: 4 x M tetrahedra
%
% Returns:
%   sp : 2 x K slice node coordinates
%   st : L x 3 triangular connectivity

    tol = 1.0e-8;

    Pv = P(dim, :);

    d = Pv(T);

    above = d > val + tol;
    below = d < val - tol;
    onPlane = abs(d - val) <= tol;

    nAbove = sum(above, 1);
    nBelow = sum(below, 1);
    nOn = sum(onPlane, 1);

    active = find((nAbove > 0 & nBelow > 0) | nOn > 0);

    if isempty(active)
        sp = zeros(2, 0);
        st = zeros(0, 3);
        return;
    end

    T = T(:, active);
    d = d(:, active);
    above = above(:, active);
    below = below(:, active);
    onPlane = onPlane(:, active);

    M = size(T, 2);

    edges = [ ...
        1 2; ...
        1 3; ...
        1 4; ...
        2 3; ...
        2 4; ...
        3 4].';

    e1 = T(edges(1, :), :);
    e2 = T(edges(2, :), :);

    a1 = above(edges(1, :), :);
    b1 = below(edges(1, :), :);

    a2 = above(edges(2, :), :);
    b2 = below(edges(2, :), :);

    crossMask = (a1 & b2) | (b1 & a2);

    otherDims = setdiff(1:3, dim);

    % -------------------------------------------------------------
    % Intersection points on strictly crossing edges
    % -------------------------------------------------------------
    edgePts = zeros(0, 2);
    edgeTet = zeros(0, 1);

    [eid, tid] = find(crossMask);

    if ~isempty(eid)

        linE = sub2ind([6, M], eid, tid);

        n1 = e1(linE);
        n2 = e2(linE);

        d1e = d(sub2ind( ...
            [4, M], edges(1, eid).', tid));

        d2e = d(sub2ind( ...
            [4, M], edges(2, eid).', tid));

        t = d1e ./ (d1e - d2e);

        u = ...
            P(otherDims(1), n1) .* (1 - t) + ...
            P(otherDims(1), n2) .* t;

        v = ...
            P(otherDims(2), n1) .* (1 - t) + ...
            P(otherDims(2), n2) .* t;

        edgePts = [u(:), v(:)];
        edgeTet = tid;
    end

    % -------------------------------------------------------------
    % Vertices lying exactly on the slicing plane
    % -------------------------------------------------------------
    vertPts = zeros(0, 2);
    vertTet = zeros(0, 1);

    [vid, vtid] = find(onPlane);

    if ~isempty(vid)

        linV = sub2ind([4, M], vid, vtid);

        vn = T(linV);

        vertPts = [ ...
            P(otherDims(1), vn).', ...
            P(otherDims(2), vn).'];

        vertTet = vtid;
    end

    allPts = [edgePts; vertPts];
    allTet = [edgeTet; vertTet];

    if isempty(allPts)
        sp = zeros(2, 0);
        st = zeros(0, 3);
        return;
    end

    % -------------------------------------------------------------
    % Merge nearly identical intersection points
    % -------------------------------------------------------------
    [uniqPts, ~, ic] = uniquetol( ...
        allPts, ...
        tol, ...
        'ByRows', true, ...
        'DataScale', max(1.0, max(abs(allPts(:)))));

    sp = uniqPts.';

    % -------------------------------------------------------------
    % Group intersection points by tetrahedron
    % -------------------------------------------------------------
    [~, ord] = sort(allTet);

    sTet = allTet(ord);
    sIc = ic(ord);

    counts = accumarray(sTet, 1, [M, 1]);

    starts = cumsum([1; counts(1:end - 1)]);
    ends = starts + counts - 1;

    valid = find(counts >= 3);

    numValid = numel(valid);

    % Upper bound: each polygon produces at most fan-size - 1 triangles.
    tris = zeros(2 * numValid, 3);

    nTri = 0;

    for r = 1:numValid

        j = valid(r);

        ids = sIc(starts(j):ends(j));

        if numel(ids) == 3

            nTri = nTri + 1;
            tris(nTri, :) = ids(:).';

        else

            c = mean(sp(:, ids).', 1);

            ang = atan2( ...
                sp(2, ids).' - c(2), ...
                sp(1, ids).' - c(1));

            [~, ord2] = sort(ang);

            ids = ids(ord2);

            for q = 2:numel(ids) - 1

                nTri = nTri + 1;

                tris(nTri, :) = ...
                    ids([1, q, q + 1]);

            end
        end
    end

    st = tris(1:nTri, :);
end


function plotmeshrd_export(fig, outPdf)
%PLOTMESHRD_EXPORT Export wide 800x400 PDF.

    set(fig, ...
        'PaperUnits', 'points', ...
        'PaperSize', [800, 400], ...
        'PaperPosition', [0, 0, 800, 400]);

    print(fig, outPdf, '-dpdf', '-painters');
end


function label = operatorLabel(operatorName)
%OPERATORLABEL Convert operator identifier to display label.

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