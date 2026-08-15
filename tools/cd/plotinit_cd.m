function plotinit_cd(operatorName, operatorExperiment, testIds)
%PLOTINIT_CD Plot the CD-disk PDE input: the localized random forcing f(x,y).
%
% Usage:
%   plotinit_cd('fno','cd_b3000_mse')
%   plotinit_cd('fno','cd_b3000_mse',[1 69])
%
% The exact Fourier coefficients are read from
%   data/cd_disk/cd_disk_3100.mat
% and the forcing is reconstructed with the shared cd_grf machinery used by
% data generation and FEM evaluation.
%
% Output:
%   figures/pltfig/cd/initial/cd_<op>_<exp>_test<k>_id<id>_f.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'cd_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 4;
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(root,'src','fem','cd_disk'));

    predictionFile = fullfile(root,'result','operators','cd_disk', ...
        operatorName,operatorExperiment,'predictions.mat');
    if exist(predictionFile,'file') ~= 2
        error('plotinit_cd:MissingPredictions', ...
            'Predictions file not found:\n%s',predictionFile);
    end
    P = cd_plot_load_predictions(predictionFile);
    nTest = size(P.pred_score,3);
    if isfield(P,'source_attempt_id_test')
        sampleIds = double(P.source_attempt_id_test(:));
    else
        sampleIds = (1:nTest).';
    end

    datasetFile = fullfile(root,'data','cd_disk','cd_disk_3100.mat');
    if exist(datasetFile,'file') ~= 2
        error('plotinit_cd:MissingDataset', ...
            'Dataset file not found:\n%s',datasetFile);
    end
    D = cd_plot_load_dataset(datasetFile, ...
        {'beta_angle','forcing_center','forcing_width', ...
         'grf_xi_cos','grf_xi_sin'});

    cfg = cd_config();
    nGrid = 256;
    xg = linspace(-1,1,nGrid);
    [X,Y] = meshgrid(xg);
    mask = (X.^2 + Y.^2) <= 1;

    figDir = fullfile(root,'figures','pltfig','cd','initial');
    if exist(figDir,'dir') ~= 7, mkdir(figDir); end

    for k = testIds(:).'
        if k < 1 || k ~= round(k) || k > nTest
            error('plotinit_cd:BadTestId', ...
                'Test index %g is out of range [1,%d].',k,nTest);
        end
        grf = cd_grf('fromcoeffs',cfg, ...
            double(D.beta_angle(k)), ...
            double(D.forcing_center(k)), ...
            double(D.forcing_width(k)), ...
            double(D.grf_xi_cos(:,k)), ...
            double(D.grf_xi_sin(:,k)));
        F = nan(nGrid,nGrid);
        F(mask) = cd_grf('eval',grf,X(mask),Y(mask));

        sampleId = sampleIds(k);
        base = sprintf('cd_%s_%s_test%03d_id%d_f', ...
            operatorName,operatorExperiment,k,sampleId);

        fig = figure('Color','w','Units','points', ...
            'Position',[1,1,514,409],'Visible','on');
        ax = axes(fig,'Position',[0.05,0.06,0.80,0.86]);
        imagesc(ax,xg,xg,F);
        axis(ax,'xy');
        axis(ax,'equal');
        axis(ax,'tight');
        xlim(ax,[-1,1]);
        ylim(ax,[-1,1]);
        cb = colorbar(ax,'Position',[0.88,0.06,0.04,0.86]);
        colormap(ax,parula);
        plotinit_cd_export(fig, ...
            fullfile(figDir,[base '.pdf']), ...
            fullfile(figDir,[base '.png']));
        fprintf('Saved %s\n',fullfile(figDir,base));
    end
end

function plotinit_cd_export(fig,outPdf,outPng)
    set(fig,'PaperUnits','points', ...
        'PaperSize',[514,409], ...
        'PaperPosition',[0,0,514,409]);
    exportgraphics(fig,outPdf,'ContentType','image','Resolution',150);
    try
        print(fig,outPng,'-dpng','-r300');
    catch
        exportgraphics(fig,outPng,'Resolution',300);
    end
    close(fig);
end

function D = cd_plot_load_dataset(fileName,fields)
    D = struct();
    info = h5info(fileName);
    names = {info.Datasets.Name};
    for i = 1:numel(fields)
        fld = fields{i};
        if ~any(strcmp(names,fld))
            error('plotinit_cd:MissingField', ...
                'Dataset %s is missing %s.',fileName,fld);
        end
        D.(fld) = h5read(fileName,['/' fld]);
    end
end

function P = cd_plot_load_predictions(fileName)
    P = struct();
    info = h5info(fileName);
    names = {info.Datasets.Name};
    need = {'pred_score','target_score_test','query_r','query_theta'};
    for i = 1:numel(need)
        if ~any(strcmp(names,need{i}))
            error('plotinit_cd:MissingField', ...
                'predictions.mat is missing dataset %s.',need{i});
        end
        P.(need{i}) = h5read(fileName,['/' need{i}]);
    end
    for opt = {'source_attempt_id_test','max_score'}
        fld = opt{1};
        if any(strcmp(names,fld))
            P.(fld) = h5read(fileName,['/' fld]);
        end
    end
    P.query_r = double(P.query_r(:));
    P.query_theta = double(P.query_theta(:));
    nr = numel(P.query_r);
    nth = numel(P.query_theta);
    if size(P.pred_score,1) ~= nr || size(P.pred_score,2) ~= nth
        P.pred_score = permute(P.pred_score,ndims(P.pred_score):-1:1);
        P.target_score_test = permute(P.target_score_test, ...
            ndims(P.target_score_test):-1:1);
    end
end