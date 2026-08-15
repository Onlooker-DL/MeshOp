function pltopera_cd(operatorName, operatorExperiment, testIds)
%PLTOPERA_CD Plot target vs predicted refinement-score heatmaps on the disk.
%
% Usage:
%   pltopera_cd('fno','cd_b3000_mse')
%   pltopera_cd('fno','cd_b3000_mse',[1 4 69])
%
% Reads
%   result/operators/cd_disk/<operatorName>/<operatorExperiment>/predictions.mat
% (HDF5) which contains pred_score / target_score_test as (nr,nth,nTest)
% after orientation fix, plus query_r, query_theta, source_attempt_id_test
% and max_score.
%
% For each requested test sample, two figures are written:
%   figures/paper_fig/cd/<op>_<exp>/test<k>_id<id>/cd_*_target.pdf/.png
%   figures/paper_fig/cd/<op>_<exp>/test<k>_id<id>/cd_*_predicted.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'cd_b3000_mse';
    end
    if nargin < 3 || isempty(testIds)
        testIds = 3;
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    predictionFile = fullfile(root,'result','operators','cd_disk', ...
        operatorName,operatorExperiment,'predictions.mat');
    if exist(predictionFile,'file') ~= 2
        error('pltopera_cd:MissingPredictions', ...
            'Predictions file not found:\n%s',predictionFile);
    end

    P = cd_plot_load_predictions(predictionFile);
    nr = numel(P.query_r);
    nth = numel(P.query_theta);
    nTest = size(P.pred_score,3);

    r = double(P.query_r(:));
    th = double(P.query_theta(:));
    % Close the 2*pi seam: thetaVec stops at 2*pi*(nth-1)/nth, so append
    % one wrapped column at 2*pi (same geometry as theta=0).
    if abs(th(end) - 2*pi) > 1e-12
        th = [th; 2*pi];
    end
    nthW = numel(th);
    [R,TH] = ndgrid(r,th);
    X = R.*cos(TH);
    Y = R.*sin(TH);
    % Center vertex fills the small hole at r=0.
    Xc = [0; X(:)];
    Yc = [0; Y(:)];
    [I,J] = ndgrid(1:nr-1,1:nthW-1);
    quads = [sub2ind([nr,nthW],I(:),J(:)), ...
             sub2ind([nr,nthW],I(:)+1,J(:)), ...
             sub2ind([nr,nthW],I(:)+1,J(:)+1), ...
             sub2ind([nr,nthW],I(:),J(:)+1)] + 1;
    tri = [ones(nthW-1,1), ...
           sub2ind([nr,nthW],ones(1,nthW-1),1:nthW-1).' + 1, ...
           sub2ind([nr,nthW],ones(1,nthW-1),2:nthW).' + 1];
    faces = [quads; [tri, nan(size(tri,1),1)]];

    if isfield(P,'max_score') && ~isempty(P.max_score)
        maxScore = double(P.max_score(1));
    else
        maxScore = max([P.target_score_test(:);P.pred_score(:)]);
    end

    if isfield(P,'source_attempt_id_test')
        sampleIds = double(P.source_attempt_id_test(:));
    else
        sampleIds = (1:nTest).';
    end
    methodLabel = pltopera_cd_label(operatorName);

    for k = testIds(:).'
        if k < 1 || k ~= round(k) || k > nTest
            error('pltopera_cd:BadTestId', ...
                'Test index %g is out of range [1,%d].',k,nTest);
        end
        sampleId = sampleIds(k);
        figDir = fullfile(root,'figures','paper_fig','cd', ...
            sprintf('%s_%s',operatorName,operatorExperiment), ...
            sprintf('test%03d_id%d',k,sampleId));
        if exist(figDir,'dir') ~= 7, mkdir(figDir); end

        pltopera_cd_one(figDir,'target',methodLabel, ...
            k,sampleId,operatorName,operatorExperiment, ...
            P.target_score_test(:,:,k),faces,Xc,Yc,maxScore);
        pltopera_cd_one(figDir,'predicted',methodLabel, ...
            k,sampleId,operatorName,operatorExperiment, ...
            P.pred_score(:,:,k),faces,Xc,Yc,maxScore);
    end
end

function pltopera_cd_one(figDir,kind,methodLabel,k,sampleId, ...
        operatorName,operatorExperiment,score,faces,Xc,Yc,maxScore)
    fig = figure('Color','w','Units','points', ...
        'Position',[1,1,514,409],'Visible','on');
    ax = axes(fig,'Position',[0.05,0.06,0.80,0.86]);
    score = double(score);
    % Wrap the periodic angle: column theta=2*pi duplicates theta=0.
    scoreWrap = [score, score(:,1)];
    % Center value: average of the innermost radial ring.
    cdata = [mean(score(1,:)); scoreWrap(:)];
    patch(ax,'Faces',faces,'Vertices',[Xc,Yc], ...
        'FaceVertexCData',cdata,'FaceColor','interp', ...
        'EdgeColor','none');
    axis(ax,'equal');
    axis(ax,'tight');
    axis(ax,'off');
    caxis(ax,[0,maxScore]);
    cb = colorbar(ax,'Position',[0.88,0.06,0.04,0.86]);
    colormap(ax,parula);
    if strcmp(kind,'target')
        title(ax,sprintf('%s target id:%d',methodLabel,sampleId), ...
            'FontSize',9);
    else
        title(ax,sprintf('%s predicted id:%d',methodLabel,sampleId), ...
            'FontSize',9);
    end
    outPdf = fullfile(figDir,sprintf('cd_%s_%s_test%03d_id%d_%s.pdf', ...
        operatorName,operatorExperiment,k,sampleId,kind));
    pltopera_cd_export(fig,outPdf);
    outPng = strrep(outPdf,'.pdf','.png');
    try
        print(fig,outPng,'-dpng','-r300');
    catch
        exportgraphics(fig,outPng,'Resolution',300);
    end
    close(fig);
    fprintf('Saved %s\n',outPdf);
end

function pltopera_cd_export(fig,outPdf)
    set(fig,'PaperUnits','points', ...
        'PaperSize',[514,409], ...
        'PaperPosition',[0,0,514,409]);
    exportgraphics(fig,outPdf,'ContentType','image','Resolution',150);
end

function P = cd_plot_load_predictions(fileName)
    P = struct();
    info = h5info(fileName);
    names = {info.Datasets.Name};
    need = {'pred_score','target_score_test','query_r','query_theta'};
    for i = 1:numel(need)
        if ~any(strcmp(names,need{i}))
            error('pltopera_cd:MissingField', ...
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

function label = pltopera_cd_label(operatorName)
    switch lower(operatorName)
        case 'fno', label = 'FNO';
        case 'cno', label = 'CNO';
        case 'deeponet', label = 'DeepONet';
        case 'pod_deeponet', label = 'POD-DeepONet';
        otherwise, label = upper(operatorName);
    end
end