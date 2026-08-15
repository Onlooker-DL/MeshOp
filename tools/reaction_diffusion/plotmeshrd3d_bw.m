function plotmeshrd3d_bw(operatorName, operatorExperiment, testIds, methodNames)
%PLOTMESHRD3D 3-D view of the RD disk: z=0 bottom face + y=0.5 interior cut.
%
%   plotmeshrd3d('fno','rd_accu_b3000_mse',65)
%   plotmeshrd3d('fno','rd_accu_b3000_mse',[65 12])
%   plotmeshrd3d('fno','rd_accu_b3000_mse',65,{'fno','standard'})
%
% Reads the same visualization_data_*.mat as plotmeshrd, extracts the
% z=0 boundary triangles (bottom face) and the y=0.5 tetra slice, and
% renders both in one 3-D axes with the unit-cube wireframe.
%
% Output: figures/pltfig/reaction_diffusion/femmesh3d/
%   rd3d_<method>_test<k>_id<id>.pdf/.png

    if nargin < 1 || isempty(operatorName), operatorName = 'fno'; end
    if nargin < 2 || isempty(operatorExperiment)
        operatorExperiment = 'rd_accu_b3000_mse';
    end
    if nargin < 3 || isempty(testIds), testIds = 12; end
    if nargin < 4 || isempty(methodNames)
        methodNames = {'fno','target','standard'};
    end

    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    femRoot = fullfile(root,'result','fem','reaction_diffusion_accu', ...
        operatorName,operatorExperiment);
    visFiles = dir(fullfile(femRoot,'**','visualization_data_*.mat'));
    if isempty(visFiles)
        error('plotmeshrd3d:NoVisualization','No visualization_data_*.mat found.');
    end
    S = load(fullfile(visFiles(1).folder,visFiles(1).name));

    outDir = fullfile(root,'figures','pltfig','reaction_diffusion','femmesh3d','bw_preview');
    if exist(outDir,'dir') ~= 7, mkdir(outDir); end

    for k = testIds(:).'
        for v = 1:numel(S.visualizationSamples)
            Gc = S.visualizationSamples{v};
            if double(Gc.meta.testId(1)) ~= k, continue; end
            datasetIndex = double(Gc.meta.datasetIndex(1));
            for m = 1:numel(methodNames)
                mf = methodNames{m};
                if ~isfield(Gc,mf) || ~isfield(Gc.(mf),'final')
                    warning('plotmeshrd3d:NoMethod','No %s for test %d.',mf,k);
                    continue;
                end
                P = squeeze(double(Gc.(mf).final.P));
                T = squeeze(double(Gc.(mf).final.T));
                if size(P,1) == 3, P = P.'; end
                if size(T,1) == 4, T = T.'; end

                % bottom face: boundary triangles with all z ~ 0
                [bt,~] = rd_bottom_face(P,T,1e-8);
                % y=0.5 interior slice
                [sp2,st2] = rd_slice_tetra_mesh(P.',T.',2,0.5);

                fig = figure('Color','w','Units','points', ...
                    'Position',[10 10 760 620],'Visible','on');
                ax = axes(fig,'Position',[0.05 0.05 0.9 0.9]);
                hold(ax,'on');
                patch(ax,'Faces',bt,'Vertices',P, ...
                    'FaceColor',[1 1 1], ...
                    'EdgeColor',[0 0 0],'LineWidth',0.15);
                if ~isempty(st2)
                    P3 = zeros(size(sp2,2),3);
                    P3(:,1) = sp2(1,:).';
                    P3(:,2) = 0.5;
                    P3(:,3) = sp2(2,:).';
                    patch(ax,'Faces',st2,'Vertices',P3, ...
                        'FaceColor',[0.88 0.88 0.88], ...
                        'EdgeColor',[0 0 0],'LineWidth',0.15);
                end
                % unit-cube wireframe
                rd_draw_cube(ax);
                view(ax,3);
                axis(ax,'equal');
                axis(ax,'off');
                if isprop(ax,'Toolbar') && ~isempty(ax.Toolbar)
                    ax.Toolbar.Visible = 'off';
                end

                base = sprintf('rd3d_bw_%s_test%03d_id%05d',mf,k,datasetIndex);
                rd_export(fig,outDir,base);
            end
        end
    end
end

function [faces,ok] = rd_bottom_face(P,T,tol)
%RD_BOTTOM_FACE Boundary triangles lying in the z=0 plane.
    allF = [T(:,[1 2 3]);T(:,[1 2 4]);T(:,[1 3 4]);T(:,[2 3 4])];
    allF = sort(allF,2);
    [uF,~,ic] = unique(allF,'rows');
    mult = accumarray(ic,1);
    bndF = uF(mult==1,:);
    zz = reshape(abs(P(bndF,3)),size(bndF,1),3);
    ok = all(zz <= tol,2);
    faces = bndF(ok,:);
end

function [sp, st] = rd_slice_tetra_mesh(P, T, dim, val)
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




function rd_draw_cube(ax)
    v = [0 0 0;1 0 0;1 1 0;0 1 0;0 0 1;1 0 1;1 1 1;0 1 1];
    e = [1 2;2 3;3 4;4 1;5 6;6 7;7 8;8 5;1 5;2 6;3 7;4 8];
    line(ax,v(e(:,1),1),v(e(:,1),2),v(e(:,1),3), ...
        'Color',[0 0 0],'LineWidth',0.8);
    line(ax,v(e(:,2),1),v(e(:,2),2),v(e(:,2),3), ...
        'Color',[0 0 0],'LineWidth',0.8);
end

function rd_export(fig,outDir,base)
    set(fig,'PaperUnits','points','PaperSize',[760 620],'PaperPosition',[0 0 760 620]);
    exportgraphics(fig,fullfile(outDir,[base '.pdf']),'ContentType','image','Resolution',150);
    try
        print(fig,fullfile(outDir,[base '.png']),'-dpng','-r150');
    catch
        exportgraphics(fig,fullfile(outDir,[base '.png']),'Resolution',150);
    end
    close(fig);
    fprintf('saved %s\n',fullfile(outDir,[base '.pdf']));
end