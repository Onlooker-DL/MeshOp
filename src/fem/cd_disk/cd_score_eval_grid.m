function scoreGrid = cd_score_eval_grid(node,elem,generation,queryR,queryTheta,scoreMax)
%CD_SCORE_EVAL_GRID Sample the leaf-generation field of a mesh on the
% polar query grid, capped at scoreMax (same rule as burgers).
    if nargin < 6 || isempty(scoreMax)
        scoreMax = 12;
    end
    [RR,TT] = ndgrid(double(queryR(:)),double(queryTheta(:)));
    X = RR(:).*cos(TT(:));
    Y = RR(:).*sin(TT(:));
    T = tsearchn(double(node),double(elem),[X,Y]);
    score = zeros(numel(X),1);
    inside = ~isnan(T);
    score(inside) = double(generation(T(inside)));
    score = min(max(score,0),double(scoreMax));
    scoreGrid = reshape(score,[numel(queryR),numel(queryTheta)]);
end