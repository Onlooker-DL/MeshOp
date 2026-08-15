function plot004_nvb_cd(operatorName, operatorExperiment, testId, maxGen, plotRounds)
%PLOT004_NVB_CD Adaptive NVB mesh evolution for a CD-disk test sample.
%
% Reads the target refinement score of a test sample from
%   result/operators/cd_disk/<op>/<exp>/predictions.mat,
% and the shared initial disk mesh from data/cd_disk/cd_disk_3100.mat.
% It then grows the initial mesh by the SAME 7-point rule and NVB used in
% data generation and cd_disk_fem_comparison:
%   desired = floor(max7(s) + 1 - cfg.generationThreshold), capped at
%   cfg.scoreMaximum, with cd_nvb_refine_conforming_local closure.
%
% Usage:
%   plot004_nvb_cd('fno','cd_b3000_mse')
%   plot004_nvb_cd('fno','cd_b3000_mse',4,12,[0 4 8 12])
%
% Output (figures/pltfig/cd/initial/):
%   cd_test<testId>_round<r>_mesh.pdf/.png

    if nargin < 1 || isempty(operatorName)
        operatorName = 'fno';
    end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'cd_b3000_mse';
    end
    if nargin < 3 || isempty(testId)
        testId = 4;
    end
    if nargin < 4 || isempty(maxGen)
        maxGen = 12;
    end
    if nargin < 5 || isempty(plotRounds)
        plotRounds = [0,4,8,maxGen];
    end
    validateattributes(maxGen,{'numeric'},{'scalar','integer','>=',1});

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(root,'src','fem','cd_disk'));

    predictionFile = fullfile(root,'result','operators','cd_disk', ...
        operatorName,operatorExperiment,'predictions.mat');
    if exist(predictionFile,'file') ~= 2
        error('plot004_nvb_cd:MissingPredictions', ...
            'Predictions file not found:\n%s',predictionFile);
    end
    P = cd_plot_load_predictions(predictionFile);
    nTest = size(P.pred_score,3);
    if testId < 1 || testId ~= round(testId) || testId > nTest
        error('plot004_nvb_cd:BadTestId', ...
            'Test index %g is out of range [1,%d].',testId,nTest);
    end

    datasetFile = fullfile(root,'data','cd_disk','cd_disk_3100.mat');
    if exist(datasetFile,'file') ~= 2
        error('plot004_nvb_cd:MissingDataset', ...
            'Dataset file not found:\n%s',datasetFile);
    end
    D = cd_plot_load_dataset(datasetFile,{'initial_node','initial_elem'});

    cfg = cd_config();
    r = double(P.query_r(:));
    th = double(P.query_theta(:));
    score = double(P.target_score_test(:,:,testId));
    Fscore = griddedInterpolant({r,th},score,'linear','nearest');

    node = double(D.initial_node);
    elem = double(D.initial_elem);
    gen = zeros(size(elem,1),1);
    snapshots = cell(maxGen+1,1);
    snapshots{1} = struct('node',node,'elem',elem,'gen',gen);

    for rnd = 1:maxGen
        p1 = node(elem(:,1),:);
        p2 = node(elem(:,2),:);
        p3 = node(elem(:,3),:);
        q = [ (p1+p2+p3)/3; (p1+p2)/2; (p2+p3)/2; (p3+p1)/2; p1; p2; p3 ];
        rq = sqrt(sum(q.^2,2));
        tq = atan2(q(:,2),q(:,1));
        tq(tq<0) = tq(tq<0) + 2*pi;
        s = Fscore(rq,tq);
        sElem = max(reshape(s,size(elem,1),7),[],2);
        desired = floor(sElem + 1 - cfg.generationThreshold);
        desired = min(max(desired,0),double(cfg.scoreMaximum));
        marked = find(double(gen) < desired);
        if isempty(marked)
            snapshots{rnd+1} = snapshots{rnd};
            continue;
        end
        [node,elem,gen,~] = cd_nvb_refine_conforming_local( ...
            node,elem,gen,marked,cfg);
        snapshots{rnd+1} = struct('node',node,'elem',elem,'gen',gen);
    end

    figDir = fullfile(root,'figures','pltfig','cd','initial');
    if exist(figDir,'dir') ~= 7, mkdir(figDir); end

    for r = plotRounds
        if r < 0 || r > maxGen
            warning('plot004_nvb_cd:BadRound', ...
                'Round %d out of range, skipped.',r);
            continue;
        end
        snap = snapshots{r+1};
        base = sprintf('cd_test%03d_round%d_mesh',testId,r);
        plot004_nvb_cd_one(figDir,base,snap.node,snap.elem);
        fprintf('Saved %s (%d elements)\n', ...
            fullfile(figDir,base),size(snap.elem,1));
    end
end

function plot004_nvb_cd_one(figDir,base,node,elem)
    fig = figure('Color','w','Units','points', ...
        'Position',[1,1,514,409],'Visible','on');
    ax = axes(fig,'Position',[0.03,0.03,0.94,0.94]);
    patch(ax,'Faces',elem,'Vertices',node, ...
        'FaceColor','w','EdgeColor',[0.4 0.4 0.4], ...
        'LineWidth',0.25);
    axis(ax,'equal');
    axis(ax,'tight');
    axis(ax,'off');
    if isprop(ax,'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    plot004_nvb_cd_export(fig, ...
        fullfile(figDir,[base '.pdf']), ...
        fullfile(figDir,[base '.png']));
end

function plot004_nvb_cd_export(fig,outPdf,outPng)
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
            error('plot004_nvb_cd:MissingField', ...
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
            error('plot004_nvb_cd:MissingField', ...
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