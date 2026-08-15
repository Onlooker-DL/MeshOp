function varargout = cd_reference(cmd, varargin)
%CD_REFERENCE Shared Fourier-Chebyshev disk reference + FEM error.
%   [refCoarse,refFine,specDiff] = cd_reference('solve',grf,beta,cfg)
%   [relL2,absL2]               = cd_reference('error',node,elem,u,refFine)
%   [relL2,absL2,specDiff,refFine] = cd_reference('full',grf,beta,cfg,node,elem,u)
switch lower(cmd)
    case 'solve'
        grf = varargin{1};
        beta = varargin{2};
        cfg = varargin{3};
        refCoarse = solveDiskFourierChebyshev( ...
            grf,beta,cfg.epsilon, ...
            cfg.specCoarse.NrIntervals,cfg.specCoarse.M, ...
            cfg.specInterpolationNtheta);
        refFine = solveDiskFourierChebyshev( ...
            grf,beta,cfg.epsilon, ...
            cfg.specFine.NrIntervals,cfg.specFine.M, ...
            cfg.specInterpolationNtheta);
        specDiff = compareSpectralReferences( ...
            refCoarse,refFine,cfg.specCheckGrid);
        if specDiff > 2e-3
            warning(['cd_reference: coarse/fine spectral difference %.3e; ', ...
                     'increase cfg.specFine before treating the fine solve ', ...
                     'as a converged reference.'],specDiff);
        end
        varargout = {refCoarse,refFine,specDiff};
    case 'error'
        [relL2,absL2] = femErrorAgainstSpectralReference( ...
            varargin{1},varargin{2},varargin{3},varargin{4});
        varargout = {relL2,absL2};
    case 'full'
        [~,refFine,specDiff] = cd_reference('solve', ...
            varargin{1},varargin{2},varargin{3});
        [relL2,absL2] = cd_reference('error', ...
            varargin{4},varargin{5},varargin{6},refFine);
        varargout = {relL2,absL2,specDiff,refFine};
    otherwise
        error('cd_reference:UnknownCmd','Unknown command %s.',cmd);
end
end
function ref=solveDiskFourierChebyshev( ...
    grf,beta,epsilon,NrIntervals,M,NthetaInterp)
% Fourier in theta + Chebyshev on the doubled radial interval [-1,1].
%
% Modal PDE:
%
% -eps [u_m'' + r^{-1}u_m' - m^2 r^{-2}u_m]
%
% plus convection couplings
%
% source mode m -> output m+1:
%   0.5 exp(-i alpha) [u_m' - m r^{-1}u_m]
%
% source mode m -> output m-1:
%   0.5 exp(+i alpha) [u_m' + m r^{-1}u_m].
%
% BMC/disk symmetry:
%   u_m(-r)=(-1)^m u_m(r).
%
% NrIntervals is odd, hence r=0 is omitted.

if mod(NrIntervals,2)~=1
    error('NrIntervals must be odd so r=0 is omitted.');
end

if NthetaInterp<2*M+1
    NthetaInterp=2^(nextpow2(2*M+1));
end

alpha=atan2(beta(2),beta(1));

% Chebyshev-Lobatto nodes x in [1,-1] and differentiation matrix.
[D,x]=chebDiffMatrix(NrIntervals);
D2=D*D;

nFull=NrIntervals+1;
nPos=(NrIntervals+1)/2;

posIdx=(1:nPos).';
posInterior=(2:nPos).';

rPos=x(posIdx);
rI=x(posInterior);

if any(rI<=0)
    error('Internal radial-grid construction failure.');
end

nrI=numel(rI);
modes=(-M:M).';
nModes=numel(modes);
nUnknown=nrI*nModes;

R1=diag(1./rI);
R2=diag(1./(rI.^2));
Id=eye(nrI);

% Forcing Fourier coefficients on the positive radial collocation rows.
Nfft=2^nextpow2(max(512,4*(2*M+1)));
thetaFFT=2*pi*(0:Nfft-1)/Nfft;

[Rmat,Tmat]=ndgrid(rI,thetaFFT);

X=Rmat.*cos(Tmat);
Y=Rmat.*sin(Tmat);

Fphys=cd_grf('eval',grf,X,Y);
Fhat=fft(Fphys,[],2)/Nfft;

rhs=zeros(nUnknown,1);

for jm=1:nModes

    m=modes(jm);
    idxMode=mod(m,Nfft)+1;

    rows=(jm-1)*nrI+(1:nrI);

    rhs(rows)=Fhat(:,idxMode);
end

% Assemble block-sparse complex operator.
Iblocks=cell(3*nModes,1);
Jblocks=cell(3*nModes,1);
Vblocks=cell(3*nModes,1);
blockCounter=0;

for jm=1:nModes

    m=modes(jm);

    parity=(-1)^abs(m);

    E=radialParityExtensionMatrix(nFull,nPos,parity);

    % Boundary r=1 is column 1 of positive values and equals zero.
    D1=D(posInterior,:)*E(:,2:nPos);
    Drr=D2(posInterior,:)*E(:,2:nPos);

    Lm=-epsilon*(Drr+R1*D1-(m^2)*R2);

    rows=(jm-1)*nrI+(1:nrI);

    % diagonal block
    blockCounter=blockCounter+1;
    [rr,cc]=ndgrid(rows,rows);
    Iblocks{blockCounter}=rr(:);
    Jblocks{blockCounter}=cc(:);
    Vblocks{blockCounter}=Lm(:);

    % source m -> output m+1
    if jm<nModes

        outRows=jm*nrI+(1:nrI);

        Cplus=0.5*exp(-1i*alpha)*(D1-m*R1);

        blockCounter=blockCounter+1;
        [rr,cc]=ndgrid(outRows,rows);
        Iblocks{blockCounter}=rr(:);
        Jblocks{blockCounter}=cc(:);
        Vblocks{blockCounter}=Cplus(:);
    end

    % source m -> output m-1
    if jm>1

        outRows=(jm-2)*nrI+(1:nrI);

        Cminus=0.5*exp(1i*alpha)*(D1+m*R1);

        blockCounter=blockCounter+1;
        [rr,cc]=ndgrid(outRows,rows);
        Iblocks{blockCounter}=rr(:);
        Jblocks{blockCounter}=cc(:);
        Vblocks{blockCounter}=Cminus(:);
    end
end

I=vertcat(Iblocks{1:blockCounter});
J=vertcat(Jblocks{1:blockCounter});
V=vertcat(Vblocks{1:blockCounter});

A=sparse(I,J,V,nUnknown,nUnknown);

fprintf('  spectral unknowns = %d\n',nUnknown);
fprintf('  sparse nnz        = %d\n',nnz(A));
fprintf('  direct sparse solve...\n');

coeffVector=A\rhs;

linearResidual=norm(A*coeffVector-rhs)/max(norm(rhs),1e-30);

fprintf('  relative linear residual = %.3e\n',linearResidual);

% Modal coefficients on positive radial nodes, including boundary r=1.
Ucoeff=zeros(nPos,nModes);

for jm=1:nModes
    rows=(jm-1)*nrI+(1:nrI);
    Ucoeff(2:end,jm)=coeffVector(rows);
end

% Build a high-resolution physical polar representation for convenient
% postprocessing/interpolation. This does not change the spectral solve.
thetaInterp=2*pi*(0:NthetaInterp-1)/NthetaInterp;

phase=exp(1i*(modes*thetaInterp));

Upolar=real(Ucoeff*phase);

% Construct interpolation grid in increasing r.
% Add r=0 with a single angle-independent value from the m=0 mode.
zeroMode=find(modes==0,1);
centerValue=real(Ucoeff(end,zeroMode));

rAscending=[0;flipud(rPos)];
Uascending=[ ...
    centerValue*ones(1,NthetaInterp); ...
    flipud(Upolar)];

thetaExtended=[thetaInterp,2*pi];
Uextended=[Uascending,Uascending(:,1)];

Finterp=griddedInterpolant( ...
    {rAscending,thetaExtended},Uextended, ...
    'spline','nearest');

ref.NrIntervals=NrIntervals;
ref.M=M;
ref.modes=modes;
ref.rPos=rPos;
ref.Ucoeff=Ucoeff;
ref.thetaInterp=thetaInterp;
ref.Upolar=Upolar;
ref.interpolant=Finterp;
ref.linearResidual=linearResidual;
ref.numUnknowns=nUnknown;

end


function E=radialParityExtensionMatrix(nFull,nPos,parity)
% Map positive-r nodal values to all doubled Chebyshev nodes:
%
%   u(-r)=parity*u(r), parity=(-1)^m.

E=zeros(nFull,nPos);

E(1:nPos,:)=eye(nPos);

for j=nPos+1:nFull
    jp=nFull+1-j;
    E(j,jp)=parity;
end

end

%% =====================================================================

function [D,x]=chebDiffMatrix(N)
% Standard Chebyshev-Lobatto differentiation matrix on [-1,1].
% x runs from 1 down to -1.

if N==0
    D=0;
    x=1;
    return;
end

x=cos(pi*(0:N)'/N);

c=[2;ones(N-1,1);2].*(-1).^(0:N)';

X=repmat(x,1,N+1);
dX=X-X';

D=(c*(1./c)')./(dX+eye(N+1));
D=D-diag(sum(D,2));

end

%% =====================================================================

function values=evalSpectralReference(ref,x,y)

originalSize=size(x);

x=x(:);
y=y(:);

r=sqrt(x.^2+y.^2);
theta=mod(atan2(y,x),2*pi);

% Numerical roundoff only.
r=min(max(r,0),1);

values=ref.interpolant(r,theta);
values=reshape(values,originalSize);

end

%% =====================================================================

function rel=compareSpectralReferences(refCoarse,refFine,nGrid)

x=linspace(-0.995,0.995,nGrid);
y=linspace(-0.995,0.995,nGrid);

[X,Y]=meshgrid(x,y);

mask=X.^2+Y.^2<0.995^2;

xc=X(mask);
yc=Y(mask);

uc=evalSpectralReference(refCoarse,xc,yc);
uf=evalSpectralReference(refFine,xc,yc);

rel=norm(uc-uf)/max(norm(uf),1e-30);

end

%% =====================================================================

function [relL2,absL2,elemErr2]= ...
    femErrorAgainstSpectralReference(node,elem,u,ref)

lambdaQ=[2/3 1/6 1/6;
         1/6 2/3 1/6;
         1/6 1/6 2/3];

[area,~,~,~]=allElementGeometry(node,elem);

P1=node(elem(:,1),:);
P2=node(elem(:,2),:);
P3=node(elem(:,3),:);

Ue=[u(elem(:,1)),u(elem(:,2)),u(elem(:,3))];

nT=size(elem,1);

elemErr2=zeros(nT,1);
refNorm2=0;

for q=1:3

    Xq=lambdaQ(q,1)*P1+lambdaQ(q,2)*P2+lambdaQ(q,3)*P3;

    uhq=Ue*lambdaQ(q,:)';

    urq=evalSpectralReference(ref,Xq(:,1),Xq(:,2));

    contribution=area/3.*(uhq-urq).^2;

    elemErr2=elemErr2+contribution;

    refNorm2=refNorm2+sum(area/3.*urq.^2);
end

absL2=sqrt(sum(elemErr2));
relL2=absL2/sqrt(max(refNorm2,1e-30));

end

%% =====================================================================
%                            2-D Fourier GRF
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
