function plotquery()
%PLOTQUERY Draw the 7-point score-query pattern on a triangle (Burgers).
%
%   Pattern 'seven': centroid + three edge midpoints + three vertices,
%   exactly as implemented in burgers_fem_comparison.m. The element is a
%   right triangle matching the bottom face of the right tetrahedron used in
%   the reaction-diffusion plot.
%
% Output (figures/pltfig/burgers/query/):
%   burgers_query_7points.pdf/.png

    p1 = [0.0, 0.0];
    p2 = [1.0, 0.0];
    p3 = [0.0, 1.0];                 % right isosceles triangle, right angle at p1

    centroid = (p1 + p2 + p3) / 3;
    m12 = 0.5 * (p1 + p2);
    m23 = 0.5 * (p2 + p3);
    m31 = 0.5 * (p3 + p1);
    pts = [centroid; m12; m23; m31; p1; p2; p3];

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    figDir = fullfile(root, 'figures', 'pltfig', 'burgers', 'query');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.02, 0.02, 0.96, 0.96]);
    patch(ax, 'Faces', [1 2 3], 'Vertices', [p1; p2; p3], ...
        'FaceColor', [0.93 0.93 0.93], 'FaceAlpha', 0.55, ...
        'EdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.4);
    hold(ax, 'on');
    plot(ax, pts(:, 1), pts(:, 2), 'o', 'MarkerSize', 9, ...
        'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
    hold(ax, 'off');
    axis(ax, 'equal');
    axis(ax, 'off');
    box(ax, 'off');
    xlim(ax, [-0.083, 1.083]);
    ylim(ax, [-0.154, 1.012]);
    plotquery_export(fig, ...
        fullfile(figDir, 'burgers_query_7points.pdf'), ...
        fullfile(figDir, 'burgers_query_7points.png'));
    fprintf('Saved burgers query pattern (7 points)\n');
end


function plotquery_export(fig, outPdf, outPng)
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
