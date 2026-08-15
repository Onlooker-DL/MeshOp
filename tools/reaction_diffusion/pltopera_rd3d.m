function pltopera_rd3d(operatorName, operatorExperiment, testIds)
%PLTOPERA_RD3D 3-D colored score surfaces for reaction-diffusion.
%
% Same inputs as pltopera_rd, but renders the z=0 boundary plane and the
% y=0.5 interior cut as two colored 3-D surfaces inside the unit cube,
% using the same parula colormap as the 2-D heatmaps.
%
% Usage:
%   pltopera_rd3d('fno','rd_accu_b3000_mse')
%   pltopera_rd3d('fno','rd_accu_b3000_mse',[12 65])
%
% Output (figures/paper_fig/reaction_diffusion/<op>_<exp>/test<k>_id<id>/):
%   reaction_diffusion_<op>_<exp>_test<k>_id<id>_3d_target.pdf/.png
%   reaction_diffusion_<op>_<exp>_test<k>_id<id>_3d_predicted.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'rd_accu_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 12;
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    predictionFile = fullfile(root,'result','operators', ...
        'reaction_diffusion',operatorName,operatorExperiment, ...
        'predictions.mat');
    if exist(predictionFile,'file') ~= 2
        error('pltopera_rd3d:MissingPredictions', ...
            'Predictions file not found:\n%s',predictionFile);
    end

    S = load(predictionFile);
    if ~isfield(S,'pred_score') || ~isfield(S,'target_score_test')
        error('pltopera_rd3d:MissingFields', ...
            'predictions.mat must contain pred_score and target_score_test.');
    end
    predScore = double(S.pred_score);
    targetScore = double(S.target_score_test);
    if ndims(predScore) ~= 4 || ndims(targetScore) ~= 4
        error('pltopera_rd3d:BadRank', ...
            'Expected 4-D score arrays (nx-by-ny-by-nz-by-nTest).');
    end
    nTest = size(predScore,4);
    if size(targetScore,4) ~= nTest
        error('pltopera_rd3d:SizeMismatch', ...
            'pred_score and target_score_test have different sample counts.');
    end

    queryX = double(S.query_x(:));
    queryY = double(S.query_y(:));
    queryZ = double(S.query_z(:));
    [~,iy] = min(abs(queryY - 0.5));
    iz0 = 1;

    if isfield(S,'source_attempt_id_test')
        sampleIds = double(S.source_attempt_id_test(:));
    else
        sampleIds = (1:nTest).';
    end
    if isfield(S,'max_score')
        maxScore = double(S.max_score(1));
    else
        maxScore = max([targetScore(:);predScore(:)]);
    end

    methodLabel = pltopera_rd3d_label(operatorName);
    problemName = 'reaction_diffusion';

    for k = testIds(:).'
        if k < 1 || k ~= round(k) || k > nTest
            error('pltopera_rd3d:BadTestId', ...
                'Test index %g is out of range [1,%d].',k,nTest);
        end
        sampleId = sampleIds(k);
        figDir = fullfile(root,'figures','paper_fig',problemName, ...
            sprintf('%s_%s',operatorName,operatorExperiment), ...
            sprintf('test%03d_id%d',k,sampleId));
        if exist(figDir,'dir') ~= 7, mkdir(figDir); end

        pltopera_rd3d_one(figDir,problemName,operatorName, ...
            operatorExperiment,k,sampleId,methodLabel, ...
            queryX,queryY,queryZ,iy,iz0, ...
            targetScore(:,:,:,k),maxScore,'target');
        pltopera_rd3d_one(figDir,problemName,operatorName, ...
            operatorExperiment,k,sampleId,methodLabel, ...
            queryX,queryY,queryZ,iy,iz0, ...
            predScore(:,:,:,k),maxScore,'predicted');
    end
end

function pltopera_rd3d_one(figDir,problemName,operatorName, ...
        operatorExperiment,k,sampleId,methodLabel, ...
        queryX,queryY,queryZ,iy,iz0,vol,maxScore,kind)
    Fz0 = squeeze(vol(:,:,iz0));      % (nx,ny)
    Fy05 = squeeze(vol(:,iy,:));      % (nx,nz)

    fig = figure('Color','w','Units','points', ...
        'Position',[10 10 760 620],'Visible','on');
    ax = axes(fig,'Position',[0.04 0.04 0.86 0.88]);
    hold(ax,'on');

    surf(ax,queryX,queryY,zeros(size(Fz0.')),Fz0.', ...
        'EdgeColor','none','FaceAlpha',0.92);
    surf(ax,queryX, ...
        0.5*ones(numel(queryZ),numel(queryX)), ...
        repmat(queryZ(:),1,numel(queryX)),Fy05.', ...
        'EdgeColor','none','FaceAlpha',0.92);

    pltopera_rd3d_cube(ax);
    view(ax,3);
    axis(ax,'equal');
    axis(ax,'off');
    caxis(ax,[0,maxScore]);
    cb = colorbar(ax,'Position',[0.92 0.04 0.03 0.88]);
    colormap(ax,parula);
    if isprop(ax,'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end

    outPdf = fullfile(figDir,sprintf( ...
        '%s_%s_%s_test%03d_id%d_3d_%s.pdf', ...
        problemName,operatorName,operatorExperiment,k,sampleId,kind));
    pltopera_rd3d_export(fig,outPdf);
    outPng = strrep(outPdf,'.pdf','.png');
    try
        print(fig,outPng,'-dpng','-r150');
    catch
        exportgraphics(fig,outPng,'Resolution',150);
    end
    close(fig);
    fprintf('Saved %s\n',outPdf);
end

function pltopera_rd3d_cube(ax)
    v = [0 0 0;1 0 0;1 1 0;0 1 0;0 0 1;1 0 1;1 1 1;0 1 1];
    e = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
    line(ax,v(e(:,1),1),v(e(:,1),2),v(e(:,1),3), ...
        'Color',[0.2 0.2 0.2],'LineWidth',0.8);
    line(ax,v(e(:,2),1),v(e(:,2),2),v(e(:,2),3), ...
        'Color',[0.2 0.2 0.2],'LineWidth',0.8);
end

function pltopera_rd3d_export(fig,outPdf)
    set(fig,'PaperUnits','points', ...
        'PaperSize',[760 620], ...
        'PaperPosition',[0 0 760 620]);
    exportgraphics(fig,outPdf,'ContentType','image','Resolution',150);
end

function label = pltopera_rd3d_label(operatorName)
    switch lower(operatorName)
        case 'fno', label = 'FNO';
        case 'cno', label = 'CNO';
        case 'deeponet', label = 'DeepONet';
        case 'pod_deeponet', label = 'POD-DeepONet';
        case 'transolver', label = 'Transolver';
        otherwise, label = upper(operatorName);
    end
end