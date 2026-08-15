function [u,info] = cd_solve_p1_supg(node,elem,cfg,grf,beta)
%CD_SOLVE_P1_SUPG Shared P1 SUPG solver (verbatim logic from one-sample).
% Returns assembly/solve wall-clock seconds for the timing breakdown.
    info = struct('assemblyTimeSec',0.0,'solveTimeSec',0.0);
    [u,info.assemblyTimeSec,info.solveTimeSec] = ...
        solveP1SUPGCD(node,elem,cfg.epsilon,beta,grf,true);
end
function [u,tAsmOut,tSolOut]=solveP1SUPGCD(node,elem,epsilon,beta,grf,useSUPG)

nP=size(node,1);
nT=size(elem,1);
tAsmTimer=tic;

[area,gradX,gradY,hK]=allElementGeometry(node,elem);

betaGrad=beta(1)*gradX+beta(2)*gradY;

if useSUPG
    tau=computeSUPGTau(hK,epsilon,norm(beta));
else
    tau=zeros(nT,1);
end

lambdaQ=[2/3 1/6 1/6;
         1/6 2/3 1/6;
         1/6 1/6 2/3];

P1=node(elem(:,1),:);
P2=node(elem(:,2),:);
P3=node(elem(:,3),:);

fQ=zeros(nT,3);

for q=1:3
    Xq=lambdaQ(q,1)*P1+lambdaQ(q,2)*P2+lambdaQ(q,3)*P3;
    fQ(:,q)=cd_grf('eval',grf,Xq(:,1),Xq(:,2));
end

intF=area/3.*sum(fQ,2);

I=zeros(9*nT,1);
J=zeros(9*nT,1);
V=zeros(9*nT,1);
F=zeros(nP,1);

ptr=0;

for i=1:3

    ni=elem(:,i);

    loadStd=zeros(nT,1);

    for q=1:3
        loadStd=loadStd+area/3.*fQ(:,q).*lambdaQ(q,i);
    end

    loadSUPG=tau.*betaGrad(:,i).*intF;

    F=F+accumarray(ni,loadStd+loadSUPG,[nP,1],@sum,0);

    for j=1:3

        nj=elem(:,j);

        diffTerm=epsilon*area.*( ...
            gradX(:,i).*gradX(:,j)+gradY(:,i).*gradY(:,j));

        advTerm=area/3.*betaGrad(:,j);

        supgTerm=tau.*area.*betaGrad(:,i).*betaGrad(:,j);

        ids=ptr+(1:nT);

        I(ids)=ni;
        J(ids)=nj;
        V(ids)=diffTerm+advTerm+supgTerm;

        ptr=ptr+nT;
    end
end

A=sparse(I,J,V,nP,nP);
tAsmOut=toc(tAsmTimer);

boundaryNodes=boundaryNodesFromElem(elem);

isDir=false(nP,1);
isDir(boundaryNodes)=true;

free=find(~isDir);

u=zeros(nP,1);
tSolTimer=tic;
u(free)=A(free,free)\F(free);
tSolOut=toc(tSolTimer);

if any(~isfinite(u))
    error('FEM solution contains NaN or Inf.');
end

end

%% =====================================================================
%                           Residual estimator
% ======================================================================


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

function tau=computeSUPGTau(hK,epsilon,betaNorm)

if betaNorm<=0
    tau=zeros(size(hK));
    return;
end

Pe=betaNorm*hK/(2*epsilon);

tau=zeros(size(hK));

small=Pe<1e-3;
large=Pe>50;
mid=~(small|large);

tau(small)=hK(small).^2/(12*epsilon);

tau(large)=hK(large)/(2*betaNorm).* ...
    (1-1./Pe(large));

q=Pe(mid);

tau(mid)=hK(mid)/(2*betaNorm).* ...
    (coth(q)-1./q);

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
