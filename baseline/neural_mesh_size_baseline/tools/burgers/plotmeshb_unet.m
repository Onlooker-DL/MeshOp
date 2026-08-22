function plotmeshb_unet(testIds, keepOpen)
%PLOTMESHB_UNET Draw U-Net/Gmsh Burgers meshes saved for MATLAB FEM.
%
% Usage:
%   plotmeshb_unet()             % tests 4 and 69
%   plotmeshb_unet([4 69])
%   plotmeshb_unet(4, true)      % keep the MATLAB figure open
%
% The function first uses a selected FEM artifact, which contains node and
% elem. If that file is unavailable, it falls back to the corresponding
% Gmsh export containing nodes and triangles. Thus, any of the 100 saved
% test meshes can be plotted without rerunning Gmsh or the FEM solver.
%
% Output:
%   figures/pltfig/burgers/femmesh/
%     burgers_unet_gmsh_t<test>_i<source>_mesh.pdf
%     burgers_unet_gmsh_t<test>_i<source>_mesh.png

    if nargin < 1 || isempty(testIds)
        testIds = [4, 69];
    end
    if nargin < 2 || isempty(keepOpen)
        keepOpen = false;
    end

    validateattributes(testIds, {'numeric'}, ...
        {'vector', 'integer', 'positive', '<=', 100}, ...
        mfilename, 'testIds');
    validateattributes(keepOpen, {'logical', 'numeric'}, ...
        {'scalar'}, mfilename, 'keepOpen');
    keepOpen = logical(keepOpen);

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    experiment = 'b3000_continuous_mesh_size_mse';
    resultRoot = fullfile(root, 'results', 'burgers', 'unet', experiment);
    selectedDir = fullfile(resultRoot, 'matlab_fem_normalized_xt', ...
        'selected_samples');
    gmshDir = fullfile(resultRoot, 'gmsh_meshes_matlab_normalized_xt');
    figureDir = fullfile(root, 'figures', 'pltfig', 'burgers', 'femmesh');

    if exist(resultRoot, 'dir') ~= 7
        error('plotmeshb_unet:MissingResults', ...
            'Result directory not found:\n%s', resultRoot);
    end
    if exist(figureDir, 'dir') ~= 7
        mkdir(figureDir);
    end

    for testId = testIds(:).'
        [node, elem, sourceId, inputFile] = load_saved_mesh( ...
            selectedDir, gmshDir, testId);
        validate_mesh(node, elem, inputFile);

        fig = figure('Color', 'w', 'Units', 'points', ...
            'Position', [1, 1, 514, 409], 'Visible', 'on');
        ax = axes(fig, 'Position', [0.07, 0.07, 0.86, 0.86]);

        patch(ax, 'Faces', elem, 'Vertices', node, ...
            'FaceColor', 'w', 'EdgeColor', [0.4 0.4 0.4], ...
            'LineWidth', 0.25);
        axis(ax, 'tight');
        axis(ax, 'off');
        if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
            ax.Toolbar.Visible = 'off';
        end
        drawnow;

        stem = sprintf('burgers_unet_gmsh_t%03d_i%04d_mesh', ...
            testId, sourceId);
        outPdf = fullfile(figureDir, [stem, '.pdf']);
        outPng = fullfile(figureDir, [stem, '.png']);
        export_mesh_figure(fig, outPdf, outPng);

        fprintf('[plotmeshb_unet] test %d, source %d\n', testId, sourceId);
        fprintf('  input: %s\n', inputFile);
        fprintf('  PDF  : %s\n', outPdf);
        fprintf('  PNG  : %s\n', outPng);

        if ~keepOpen
            close(fig);
        end
    end
end


function [node, elem, sourceId, inputFile] = load_saved_mesh( ...
        selectedDir, gmshDir, testId)
    pattern = sprintf('test_%03d_source_*.mat', testId);
    files = dir(fullfile(selectedDir, pattern));
    useSelected = ~isempty(files);
    if ~useSelected
        files = dir(fullfile(gmshDir, pattern));
    end
    if isempty(files)
        error('plotmeshb_unet:MissingSample', ...
            ['No saved mesh was found for test %d under:\n%s\n', ...
             'or:\n%s'], testId, selectedDir, gmshDir);
    end
    if numel(files) > 1
        error('plotmeshb_unet:AmbiguousSample', ...
            'Multiple saved meshes were found for test %d.', testId);
    end

    inputFile = fullfile(files(1).folder, files(1).name);
    S = load(inputFile);
    if useSelected
        required = {'node', 'elem'};
        if ~all(isfield(S, required))
            error('plotmeshb_unet:InvalidSelectedFile', ...
                'Selected FEM file lacks node/elem:\n%s', inputFile);
        end
        node = double(S.node);
        elem = double(S.elem);
    else
        required = {'nodes', 'triangles'};
        if ~all(isfield(S, required))
            error('plotmeshb_unet:InvalidGmshFile', ...
                'Gmsh file lacks nodes/triangles:\n%s', inputFile);
        end
        node = double(S.nodes);
        elem = double(S.triangles);
    end

    token = regexp(files(1).name, 'source_(\d+)', 'tokens', 'once');
    if isempty(token)
        error('plotmeshb_unet:MissingSourceId', ...
            'Cannot read source ID from file name:\n%s', inputFile);
    end
    sourceId = str2double(token{1});
end


function validate_mesh(node, elem, inputFile)
    if size(node, 2) ~= 2 || size(elem, 2) ~= 3
        error('plotmeshb_unet:InvalidShape', ...
            'Expected node Nx2 and elem Mx3 in:\n%s', inputFile);
    end
    if isempty(node) || isempty(elem) || any(~isfinite(node(:)))
        error('plotmeshb_unet:InvalidMesh', ...
            'Mesh is empty or contains non-finite nodes:\n%s', inputFile);
    end
    if any(elem(:) ~= round(elem(:))) || min(elem(:)) < 1 || ...
            max(elem(:)) > size(node, 1)
        error('plotmeshb_unet:InvalidConnectivity', ...
            'Triangle connectivity is invalid in:\n%s', inputFile);
    end
end


function export_mesh_figure(fig, outPdf, outPng)
    % Match plotmeshb: fixed 514 x 409 pt window/page and 150 dpi output.
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    exportgraphics(fig, outPdf, ...
        'ContentType', 'image', 'Resolution', 150);
    try
        print(fig, outPng, '-dpng', '-r150');
    catch
        exportgraphics(fig, outPng, 'Resolution', 150);
    end
end
