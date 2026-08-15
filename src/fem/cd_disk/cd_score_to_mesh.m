function [node,elem,generation] = cd_score_to_mesh( ...
        cfg,scoreGrid,queryR,queryTheta,node0,elem0,gen0)
%CD_SCORE_TO_MESH Realize a continuous score field on the polar query grid
% into an adaptive mesh, using the SAME 7-point rule as the Burgers code:
% for every element evaluate the score at
%     centroid + 3 edge midpoints + 3 vertices
% through the polar-grid interpolant and take the maximum, then
%     desired = floor(s + 1 - generationThreshold), capped at scoreMaximum.
% Uses the SAME initial mesh and NVB as data generation.
    if nargin < 7 || isempty(gen0)
        gen0 = zeros(size(elem0,1),1);
    end
    node = node0;
    elem = elem0;
    generation = double(gen0(:));

    rVec = double(queryR(:));
    tVec = double(queryTheta(:));
    if any(diff(rVec)<=0) || any(diff(tVec)<=0)
        error('cd_score_to_mesh:GridOrder','Query grid must be increasing.');
    end
    Fscore = griddedInterpolant({rVec,tVec},double(scoreGrid),'linear','nearest');

    for pass = 1:cfg.scoreRefineMaxCalls
        % 7 query points per element: centroid, 3 edge midpoints, 3 vertices.
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
        marked = find(double(generation) < desired);
        if isempty(marked)
            break;
        end
        [node,elem,generation,~] = cd_nvb_refine_conforming_local( ...
            node,elem,generation,marked,cfg);
        if size(elem,1) >= cfg.maxElements
            error('cd_score_to_mesh:MaxElements', ...
                'Realization exceeded cfg.maxElements=%d.',cfg.maxElements);
        end
    end
end