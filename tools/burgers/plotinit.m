function plotinit(operatorName, operatorExperiment, testIds)
%PLOTINIT Plot the Burgers initial condition u0(x) and forcing f(x).
%
% These are the PDE inputs of a test instance, independent of any neural
% operator. The exact Fourier reconstruction stored in predictions.mat
% (fields U0_test / F_test / x_input) is used.
%
% Usage:
%   plotinit('fno', 'b3000_mse')          % default test 1
%   plotinit('fno', 'b3000_mse', [1 69])
%
% Output:
%   figures/pltfig/burgers/initial/
%       burgers_<op>_<exp>_test<k>_id<id>_u0.pdf/.png
%       burgers_<op>_<exp>_test<k>_id<id>_f.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 4;
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    predictionFile = fullfile(root, 'result', 'operators', 'burgers', ...
        operatorName, operatorExperiment, 'predictions.mat');
    if exist(predictionFile, 'file') ~= 2
        error('plotinit:MissingPredictions', ...
            'Predictions file not found:\n%s', predictionFile);
    end

    S = load(predictionFile);
    if ~isfield(S, 'U0_test') || ~isfield(S, 'F_test') || ...
            ~isfield(S, 'x_input')
        error('plotinit:MissingFields', ...
            'predictions.mat must contain U0_test, F_test and x_input.');
    end

    u0 = double(S.U0_test);      % (nx, nTest)
    f = double(S.F_test);        % (nx, nTest)
    x = double(S.x_input(:));    % (nx, 1)

    nTest = size(u0, 2);
    if size(f, 2) ~= nTest
        error('plotinit:SizeMismatch', ...
            'U0_test and F_test have different sample counts.');
    end

    if isfield(S, 'source_attempt_id_test')
        sampleIds = double(S.source_attempt_id_test(:));
    elseif isfield(S, 'original_indices')
        sampleIds = double(S.original_indices(:));
    else
        sampleIds = (1:nTest).';
    end

    figDir = fullfile(root, 'figures', 'pltfig', 'burgers', 'initial');
    if exist(figDir, 'dir') ~= 7
        mkdir(figDir);
    end

    for k = testIds(:).'
        if k < 1 || k ~= round(k) || k > nTest
            error('plotinit:BadTestId', ...
                'Test index %g is out of range [1, %d].', k, nTest);
        end
        sampleId = sampleIds(k);
        base = sprintf('burgers_%s_%s_test%03d_id%d', ...
            operatorName, operatorExperiment, k, sampleId);

        % u0(x)
        fig = figure('Color', 'w', 'Units', 'points', ...
            'Position', [1, 1, 514, 409], 'Visible', 'on');
        ax = axes(fig, 'Position', [0.08, 0.10, 0.86, 0.82]);
        plot(ax, x, u0(:, k), 'LineWidth', 1.6, 'Color', [0.00 0.45 0.74]);
        box(ax, 'on');
        xlabel(ax, 'x');
        ylabel(ax, 'u_0(x)');
        xlim(ax, [x(1), x(end)]);
        plotinit_export(fig, fullfile(figDir, [base '_u0.pdf']), ...
            fullfile(figDir, [base '_u0.png']));

        % f(x)
        fig = figure('Color', 'w', 'Units', 'points', ...
            'Position', [1, 1, 514, 409], 'Visible', 'on');
        ax = axes(fig, 'Position', [0.08, 0.10, 0.86, 0.82]);
        plot(ax, x, f(:, k), 'LineWidth', 1.6, 'Color', [0.85 0.33 0.10]);
        box(ax, 'on');
        xlabel(ax, 'x');
        ylabel(ax, 'f(x)');
        xlim(ax, [x(1), x(end)]);
        plotinit_export(fig, fullfile(figDir, [base '_f.pdf']), ...
            fullfile(figDir, [base '_f.png']));

        fprintf('Saved %s_u0 / %s_f\n', ...
            fullfile(figDir, base), fullfile(figDir, base));
    end
end


function plotinit_export(fig, outPdf, outPng)
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [514, 409], ...
        'PaperPosition', [0, 0, 514, 409]);
    print(fig, outPdf, '-dpdf', '-painters');
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end
    close(fig);
end
