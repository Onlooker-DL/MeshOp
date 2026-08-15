function out = cd_estimator(cmd, varargin)
%CD_ESTIMATOR Shared residual estimator + Dorfler marking.
%   eta2 = cd_estimator('estimate', node, elem, u, cfg, grf, beta)
%   marked = cd_estimator('mark', eta2, theta)
switch lower(cmd)
    case 'estimate'
        out = residualEstimatorCD(varargin{1}, varargin{2}, varargin{3}, ...
            varargin{4}.epsilon, varargin{6}, varargin{5});
    case 'mark'
        out = dorflerMark(varargin{1}, varargin{2});
    otherwise
        error('cd_estimator:UnknownCmd','Unknown command %s.',cmd);
end
end
function eta2=residualEstimatorCD(node,elem,u,epsilon,beta,grf)

nT=size(elem,1);

[area,gradX,gradY,hK]=allElementGeometry(node,elem);

u1=u(elem(:,1));
u2=u(elem(:,2));
u3=u(elem(:,3));

gradUx=gradX(:,1).*u1+gradX(:,2).*u2+gradX(:,3).*u3;
gradUy=gradY(:,1).*u1+gradY(:,2).*u2+gradY(:,3).*u3;

advU=beta(1)*gradUx+beta(2)*gradUy;

lambdaQ=[2/3 1/6 1/6;
         1/6 2/3 1/6;
         1/6 1/6 2/3];

P1=node(elem(:,1),:);
P2=node(elem(:,2),:);
P3=node(elem(:,3),:);

resL2=zeros(nT,1);

for q=1:3

    Xq=lambdaQ(q,1)*P1+lambdaQ(q,2)*P2+lambdaQ(q,3)*P3;

    fq=cd_grf('eval',grf,Xq(:,1),Xq(:,2));

    Rq=fq-advU;

    resL2=resL2+area/3.*Rq.^2;
end

eta2=hK.^2.*resL2;

[edges,adj1,adj2]=edgeAdjacency(elem);

for m=1:size(edges,1)

    K1=adj1(m);
    K2=adj2(m);

    if K2==0
        continue;
    end

    ia=edges(m,1);
    ib=edges(m,2);

    tangent=node(ib,:)-node(ia,:);
    hE=norm(tangent);

    n=[tangent(2);-tangent(1)]/hE;

    jump=epsilon*( ...
        (gradUx(K1)-gradUx(K2))*n(1)+ ...
        (gradUy(K1)-gradUy(K2))*n(2));

    contribution=hE^2*jump^2;

    eta2(K1)=eta2(K1)+0.5*contribution;
    eta2(K2)=eta2(K2)+0.5*contribution;
end

if any(~isfinite(eta2))
    error('Residual estimator contains NaN or Inf.');
end

eta2=max(eta2,0);

end

%% =====================================================================
%                          Dörfler marking
% ======================================================================


function marked=dorflerMark(eta2,theta)
% Same descending-prefix design:
% sum_{K in M} eta_K^2 >= theta sum_K eta_K^2.

if any(~isfinite(eta2))
    error('Dorfler input contains NaN or Inf.');
end

total=sum(eta2);
marked=false(size(eta2));

if total<=0 || ~isfinite(total)
    [~,id]=max(eta2);
    marked(id)=true;
    return;
end

[vals,ids]=sort(eta2,'descend');
m=find(cumsum(vals)>=theta*total,1,'first');

marked(ids(1:m))=true;

end

%% =====================================================================
%                             TRUE NVB
% ======================================================================


function [edges,adj1,adj2]=edgeAdjacency(elem)

nT=size(elem,1);

E=[sort(elem(:,[1 2]),2);
   sort(elem(:,[2 3]),2);
   sort(elem(:,[3 1]),2)];

owner=[(1:nT)';(1:nT)';(1:nT)'];

[edges,~,ic]=unique(E,'rows');

adj1=zeros(size(edges,1),1);
adj2=zeros(size(edges,1),1);

for q=1:numel(ic)

    id=ic(q);
    K=owner(q);

    if adj1(id)==0
        adj1(id)=K;
    elseif adj2(id)==0 && adj1(id)~=K
        adj2(id)=K;
    end
end

end

%% =====================================================================

function bnd=boundaryNodesFromElem(elem)

edges=sort([ ...
    elem(:,[1 2]);
    elem(:,[2 3]);
    elem(:,[3 1])],2);

[uniqueEdges,~,ic]=unique(edges,'rows');

count=accumarray(ic,1);

bndEdges=uniqueEdges(count==1,:);

bnd=unique(bndEdges(:));

end

%% =====================================================================

function [area,gradX,gradY,hK]=allElementGeometry(node,elem)

P1=node(elem(:,1),:);
P2=node(elem(:,2),:);
P3=node(elem(:,3),:);

detJ=(P2(:,1)-P1(:,1)).*(P3(:,2)-P1(:,2)) - ...
     (P3(:,1)-P1(:,1)).*(P2(:,2)-P1(:,2));

area=0.5*abs(detJ);

if any(area<=1e-15)
    error('Degenerate triangle detected.');
end

gradX=[ ...
    (P2(:,2)-P3(:,2))./detJ,...
    (P3(:,2)-P1(:,2))./detJ,...
    (P1(:,2)-P2(:,2))./detJ];

gradY=[ ...
    (P3(:,1)-P2(:,1))./detJ,...
    (P1(:,1)-P3(:,1))./detJ,...
    (P2(:,1)-P1(:,1))./detJ];

L12=sqrt(sum((P2-P1).^2,2));
L23=sqrt(sum((P3-P2).^2,2));
L31=sqrt(sum((P1-P3).^2,2));

hK=max([L12,L23,L31],[],2);

end

%% =====================================================================
