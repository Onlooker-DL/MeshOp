function plotquery_rd()
%PLOTQUERY_RD Draw the reaction-diffusion query pattern as a 2-D triangle.
%
%   Draws the right tetrahedron used by the FEM (three mutually orthogonal
%   edges meet at (1,0,0)) and marks the 15 query points of the 'fifteen'
%   pattern: 4 vertices + 6 edge midpoints + 4 face centroids + the
%   tetrahedron centroid.
%
% Output (figures/pltfig/reaction_diffusion/query/):
%   reaction_diffusion_query_15points.pdf/.png

    % Standard right tetrahedron: right angle at the origin, the three
    % orthogonal edges along +x, +y, +z. The bottom face (z=0) is exactly
    % the right triangle used in the Burgers plot.
    p1 = [0.0, 0.0, 0.0];
    p2 = [1.0, 0.0, 0.0];
    p3 = [0.0, 1.0, 0.0];
    p4 = [0.0, 0.0, 1.0];
    V = [p1; p2; p3; p4];

    m12 = 0.5 * (p1 + p2);
    m13 = 0.5 * (p1 + p3);
    m14 = 0.5 * (p1 + p4);
    m23 = 0.5 * (p2 + p3);
    m24 = 0.5 * (p2 + p4);
    m34 = 0.5 * (p3 + p4);
    f123 = (p1 + p2 + p3) / 3;
    f124 = (p1 + p2 + p4) / 3;
    f134 = (p1 + p3 + p4) / 3;
    f234 = (p2 + p3 + p4) / 3;
    centroid = 0.25 * (p1 + p2 + p3 + p4);

    pts = [ ...
        p1; p2; p3; p4; ...
        m12; m13; m14; m23; m24; m34; ...
        f123; f124; f134; f234; ...
        centroid];

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    figDir = fullfile(root, 'figures', 'pltfig', ...
        'reaction_diffusion', 'query');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    fig = figure('Color', 'w', 'Units', 'points', ...
        'Position', [1, 1, 514, 409], 'Visible', 'on');
    ax = axes(fig, 'Position', [0.02, 0.02, 0.96, 0.96]);
    faces = [1 2 3; 1 2 4; 1 3 4; 2 3 4];
    patch(ax, 'Faces', faces, 'Vertices', V, ...
        'FaceColor', [0.93 0.93 0.93], 'FaceAlpha', 0.55, ...
        'EdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.4);
    hold(ax, 'on');
    scatter3(ax, pts(:, 1), pts(:, 2), pts(:, 3), 40, 'r', 'filled');
    hold(ax, 'off');
    axis(ax, 'equal');
    axis(ax, 'off');
    box(ax, 'off');
    view(ax, [1, 1, 1]);
    camorbit(ax, 0, -22);   % tilt the view so the tetrahedron sits lower
    plotquery_rd_export(fig, ...
        fullfile(figDir, 'reaction_diffusion_query_15points.pdf'), ...
        fullfile(figDir, 'reaction_diffusion_query_15points.png'));
    fprintf('Saved reaction-diffusion query pattern (2-D triangle)\n');
end


function plotquery_rd_export(fig, outPdf, outPng)
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
