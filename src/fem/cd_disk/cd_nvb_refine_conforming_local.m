%CD_NVB_REFINE_CONFORMING_LOCAL Shared disk NVB refinement.
% Verbatim from one_sample_cd_disk_random_localized_grf_K16.m.
function [node,elem,generation,stats]= ...
    cd_nvb_refine_conforming_local(node,elem,generation,markedElem,cfg)

markedElem=unique(round(markedElem(:)));
markedElem=markedElem(markedElem>=1 & markedElem<=size(elem,1));

stats.totalBisectedParents=0;
stats.completionSubsteps=0;
stats.createdNodes=0;
stats.addedElements=0;

if isempty(markedElem)
    return;
end

oldElementCount=size(elem,1);

% Every marked triangle requests its current reference edge [z1,z2].
pending=unique(sort(elem(markedElem,[2,3]),2),'rows');

% Persistent midpoint lookup during this single refinement call.
midpointEdges=zeros(0,2);
midpointIds=zeros(0,1);

substep=0;

while ~isempty(pending)

    substep=substep+1;

    if substep>cfg.nvbMaxCompletionSteps
        error('NVB completion exceeded %d substeps.', ...
            cfg.nvbMaxCompletionSteps);
    end

    % Completion:
    % If a requested geometric edge belongs to a triangle but is not that
    % triangle's reference edge, request that triangle's reference edge too.
    changed=true;

    while changed

        NT=size(elem,1);

        e12=sort(elem(:,[1,2]),2);
        e23=sort(elem(:,[2,3]),2); % reference edges
        e31=sort(elem(:,[3,1]),2);

        allEdges=[e12;e23;e31];
        owner=[(1:NT)';(1:NT)';(1:NT)'];

        hit=ismember(allEdges,pending,'rows');
        incident=unique(owner(hit));

        if isempty(incident)
            changed=false;
            break;
        end

        requiredReferenceEdges=unique(e23(incident,:),'rows');

        newPending=unique([pending;requiredReferenceEdges],'rows');

        changed=size(newPending,1)>size(pending,1);
        pending=newPending;
    end

    referenceEdges=sort(elem(:,[2,3]),2);

    splitMask=ismember(referenceEdges,pending,'rows');
    splitIds=find(splitMask);

    if isempty(splitIds)
        error('NVB completion stalled with pending edges.');
    end

    splitEdges=unique(referenceEdges(splitIds,:),'rows');

    [known,locKnown]=ismember(splitEdges,midpointEdges,'rows');

    splitMidIds=zeros(size(splitEdges,1),1);

    if any(known)
        splitMidIds(known)=midpointIds(locKnown(known));
    end

    newMask=~known;

    if any(newMask)

        newEdges=splitEdges(newMask,:);

        newPts=0.5*( ...
            node(newEdges(:,1),:)+node(newEdges(:,2),:));

        firstId=size(node,1)+1;
        newIds=(firstId:firstId+size(newPts,1)-1).';

        node=[node;newPts]; %#ok<AGROW>

        midpointEdges=[midpointEdges;newEdges]; %#ok<AGROW>
        midpointIds=[midpointIds;newIds]; %#ok<AGROW>

        splitMidIds(newMask)=newIds;

        stats.createdNodes=stats.createdNodes+size(newPts,1);
    end

    [tf,loc]=ismember(referenceEdges(splitIds,:),splitEdges,'rows');

    if ~all(tf)
        error('Internal NVB midpoint lookup failed.');
    end

    mids=splitMidIds(loc);

    unsplitIds=find(~splitMask);

    oldGen=generation(splitIds);

    newElem=zeros(numel(unsplitIds)+2*numel(splitIds),3);
    newGen=zeros(size(newElem,1),1);

    ptr=0;

    if ~isempty(unsplitIds)

        nr=numel(unsplitIds);

        newElem(1:nr,:)=elem(unsplitIds,:);
        newGen(1:nr)=generation(unsplitIds);

        ptr=nr;
    end

    z0=elem(splitIds,1);
    z1=elem(splitIds,2);
    z2=elem(splitIds,3);

    % Standard newest-vertex-bisection children.
    child1=[mids,z0,z1];
    child2=[mids,z2,z0];

    nr=numel(splitIds);

    newElem(ptr+(1:nr),:)=child1;
    newGen(ptr+(1:nr))=oldGen+1;

    newElem(ptr+nr+(1:nr),:)=child2;
    newGen(ptr+nr+(1:nr))=oldGen+1;

    elem=newElem;
    generation=newGen;

    stats.totalBisectedParents= ...
        stats.totalBisectedParents+numel(splitIds);
    stats.completionSubsteps=substep;

    % A requested edge is finished once it no longer appears as a full
    % active mesh edge.
    currentEdges=unique(sort([ ...
        elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])],2),'rows');

    pending=intersect(pending,currentEdges,'rows','stable');
end

stats.addedElements=size(elem,1)-oldElementCount;

checkMeshConformingBasic(node,elem);

end

%% =====================================================================
%                     Fourier-Chebyshev disk reference
% ======================================================================

function checkMeshConformingBasic(node,elem)

if isempty(node) || isempty(elem)
    error('Empty mesh.');
end

if any(~isfinite(node(:))) || any(~isfinite(elem(:)))
    error('Mesh contains NaN or Inf.');
end

if any(elem(:)<1) || any(elem(:)>size(node,1))
    error('Invalid mesh node index.');
end

P1=node(elem(:,1),:);
P2=node(elem(:,2),:);
P3=node(elem(:,3),:);

area=0.5*abs( ...
    (P2(:,1)-P1(:,1)).*(P3(:,2)-P1(:,2)) - ...
    (P2(:,2)-P1(:,2)).*(P3(:,1)-P1(:,1)));

if any(area<=1e-14)
    error('Mesh contains degenerate triangles.');
end

edges=sort([ ...
    elem(:,[1 2]);
    elem(:,[2 3]);
    elem(:,[3 1])],2);

[~,~,ic]=unique(edges,'rows');
multiplicity=accumarray(ic,1);

if any(multiplicity>2)
    error('Non-manifold edge found after NVB.');
end

end
