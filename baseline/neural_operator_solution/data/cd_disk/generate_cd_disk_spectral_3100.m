function generate_cd_disk_spectral_3100(varargin)
%GENERATE_CD_DISK_SPECTRAL_3100 Build 3100 Fourier-Chebyshev reference solutions.
%
% Put cd_disk_3100.mat in this folder and run:
%
%   generate_cd_disk_spectral_3100
%
% The first 3000 samples are training data and samples 3001:3100 are test
% data.  Output is cd_disk_spectral_3100.h5.  Completed finite batches are
% resumable.  The default spatial reference matches cd_config.m:
% NrIntervals=255, angular modes=-128:128, epsilon=1e-3.

baseDir = fileparts(mfilename('fullpath'));
p = inputParser;
addParameter(p,'SourceFile',fullfile(baseDir,'cd_disk_3100.mat'),@(x)ischar(x)||isstring(x));
addParameter(p,'OutputFile',fullfile(baseDir,'cd_disk_spectral_3100.h5'),@(x)ischar(x)||isstring(x));
addParameter(p,'BatchSize',4,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'UseParallel',true,@(x)islogical(x)||isnumeric(x));
addParameter(p,'NumWorkers',4,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(p,'MaxBatches',inf,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'Overwrite',false,@(x)islogical(x)||isnumeric(x));
addParameter(p,'NrIntervals',255,@(x)isnumeric(x)&&isscalar(x)&&x>=15);
addParameter(p,'AngularModes',128,@(x)isnumeric(x)&&isscalar(x)&&x>=4);
addParameter(p,'InterpolationAngles',512,@(x)isnumeric(x)&&isscalar(x)&&x>=16);
parse(p,varargin{:});
cfg = p.Results;
cfg.SourceFile = char(cfg.SourceFile);
cfg.OutputFile = char(cfg.OutputFile);
cfg.BatchSize = round(double(cfg.BatchSize));
cfg.NumWorkers = round(double(cfg.NumWorkers));
cfg.MaxBatches = double(cfg.MaxBatches);
cfg.NrIntervals = round(double(cfg.NrIntervals));
cfg.AngularModes = round(double(cfg.AngularModes));
cfg.InterpolationAngles = round(double(cfg.InterpolationAngles));
cfg.UseParallel = logical(cfg.UseParallel);
cfg.Overwrite = logical(cfg.Overwrite);
cfg.NumSamples = 3100;
cfg.Epsilon = 1.0e-3;

if mod(cfg.NrIntervals,2)~=1
    error('NrIntervals must be odd so that the doubled radial grid omits r=0.');
end
if ~isfile(cfg.SourceFile)
    error('Missing source file: %s',cfg.SourceFile);
end
if cfg.Overwrite && isfile(cfg.OutputFile)
    delete(cfg.OutputFile);
end

M = matfile(cfg.SourceFile);
r = double(M.rVec(:,1));
theta = double(M.thetaVec(:,1));
modes = double(M.grf_modes(:,1:2));
spectralStd = double(M.grf_spectral_std(:,1));
betaAngle = double(M.beta_angle(1:cfg.NumSamples,1));
forcingCenter = double(M.forcing_center(1:cfg.NumSamples,1));
forcingWidth = double(M.forcing_width(1:cfg.NumSamples,1));
sourceSize = size(M,'grf_xi_cos');
if sourceSize(2)<cfg.NumSamples || numel(betaAngle)<cfg.NumSamples
    error('The source MAT file contains fewer than 3100 samples.');
end
if size(modes,1)~=numel(spectralStd)
    error('grf_modes and grf_spectral_std are inconsistent.');
end

fprintf('Source : %s\n',cfg.SourceFile);
fprintf('Output : %s\n',cfg.OutputFile);
fprintf('Split  : samples 1:3000 train, 3001:3100 test\n');
fprintf('Solver : Fourier-Chebyshev, NrIntervals=%d, M=%d, epsilon=%.3g\n', ...
    cfg.NrIntervals,cfg.AngularModes,cfg.Epsilon);

completed = initialize_output(cfg.OutputFile,r,theta,betaAngle, ...
    forcingCenter,forcingWidth,cfg);
if completed>=cfg.NumSamples
    fprintf('Output already contains all %d samples.\n',cfg.NumSamples);
    return;
end

useParallel = cfg.UseParallel && license('test','Distrib_Computing_Toolbox');
if useParallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            if cfg.NumWorkers>0
                pool = parpool('threads',cfg.NumWorkers);
            else
                pool = parpool('threads');
            end
        end
        fprintf('Parallel pool: %d workers\n',pool.NumWorkers);
    catch poolError
        warning('Parallel pool failed (%s). Falling back to serial.',poolError.message);
        useParallel = false;
    end
end
if ~useParallel
    fprintf('Serial generation\n');
end

wallStart = tic;
batchCounter = 0;
for first = completed+1:cfg.BatchSize:cfg.NumSamples
    last = min(first+cfg.BatchSize-1,cfg.NumSamples);
    ids = first:last;
    count = numel(ids);
    xiCos = double(M.grf_xi_cos(:,first:last));
    xiSin = double(M.grf_xi_sin(:,first:last));
    solutionBatch = zeros(numel(r),numel(theta),count,'single');
    forcingBatch = zeros(numel(r),numel(theta),count,'single');
    residualBatch = zeros(count,1);

    if useParallel
        parfor localId = 1:count
            sampleId = ids(localId);
            grf = make_grf(modes,spectralStd,xiCos(:,localId),xiSin(:,localId), ...
                betaAngle(sampleId),forcingCenter(sampleId),forcingWidth(sampleId));
            [solutionBatch(:,:,localId),forcingBatch(:,:,localId), ...
                residualBatch(localId)] = solve_one(grf,r,theta,cfg);
        end
    else
        for localId = 1:count
            sampleId = ids(localId);
            grf = make_grf(modes,spectralStd,xiCos(:,localId),xiSin(:,localId), ...
                betaAngle(sampleId),forcingCenter(sampleId),forcingWidth(sampleId));
            [solutionBatch(:,:,localId),forcingBatch(:,:,localId), ...
                residualBatch(localId)] = solve_one(grf,r,theta,cfg);
        end
    end

    % MATLAB dimensions are reversed by h5py.  These writes yield
    % [sample,r,theta] in Python.
    h5write(cfg.OutputFile,'/solution',permute(solutionBatch,[2,1,3]), ...
        [1,1,first],[numel(theta),numel(r),count]);
    h5write(cfg.OutputFile,'/forcing',permute(forcingBatch,[2,1,3]), ...
        [1,1,first],[numel(theta),numel(r),count]);
    h5write(cfg.OutputFile,'/linear_residual',single(residualBatch(:).'), ...
        [1,first],[1,count]);
    h5writeatt(cfg.OutputFile,'/','completed_samples',int32(last));

    batchCounter = batchCounter+1;
    fprintf('[cd spectral] %4d/%d | max residual %.3e | elapsed %.1f s\n', ...
        last,cfg.NumSamples,max(residualBatch),toc(wallStart));
    if batchCounter>=cfg.MaxBatches
        fprintf('Stopped intentionally after %d new batch(es); resume is safe.\n',batchCounter);
        break;
    end
end
end

function completed = initialize_output(fileName,r,theta,beta,qc,sw,cfg)
if isfile(fileName)
    completed = double(h5readatt(fileName,'/','completed_samples'));
    if double(h5readatt(fileName,'/','num_samples'))~=cfg.NumSamples || ...
            double(h5readatt(fileName,'/','nr_intervals'))~=cfg.NrIntervals || ...
            double(h5readatt(fileName,'/','angular_modes'))~=cfg.AngularModes
        error('Existing output solver settings differ; use Overwrite=true intentionally.');
    end
    return;
end
nr = numel(r); nth = numel(theta); n = cfg.NumSamples;
h5create(fileName,'/solution',[nth,nr,n],'Datatype','single', ...
    'ChunkSize',[nth,nr,1]);
h5create(fileName,'/forcing',[nth,nr,n],'Datatype','single', ...
    'ChunkSize',[nth,nr,1]);
h5create(fileName,'/linear_residual',[1,n],'Datatype','single','ChunkSize',[1,min(n,128)]);
h5create(fileName,'/r',[1,nr],'Datatype','single');
h5create(fileName,'/theta',[1,nth],'Datatype','single');
h5create(fileName,'/beta_angle',[1,n],'Datatype','single');
h5create(fileName,'/forcing_center',[1,n],'Datatype','single');
h5create(fileName,'/forcing_width',[1,n],'Datatype','single');
h5create(fileName,'/source_sample_id',[1,n],'Datatype','int32');
h5write(fileName,'/r',single(r(:).'));
h5write(fileName,'/theta',single(theta(:).'));
h5write(fileName,'/beta_angle',single(beta(:).'));
h5write(fileName,'/forcing_center',single(qc(:).'));
h5write(fileName,'/forcing_width',single(sw(:).'));
h5write(fileName,'/source_sample_id',int32(1:n));
h5writeatt(fileName,'/','completed_samples',int32(0));
h5writeatt(fileName,'/','num_samples',int32(n));
h5writeatt(fileName,'/','train_samples',int32(3000));
h5writeatt(fileName,'/','test_samples',int32(100));
h5writeatt(fileName,'/','epsilon',cfg.Epsilon);
h5writeatt(fileName,'/','nr_intervals',int32(cfg.NrIntervals));
h5writeatt(fileName,'/','angular_modes',int32(cfg.AngularModes));
h5writeatt(fileName,'/','method','Fourier-Chebyshev disk reference');
completed = 0;
end

function grf = make_grf(modes,weight,a,b,beta,qc,sw)
grf.kx = modes(:,1);
grf.ky = modes(:,2);
grf.weight = weight(:);
grf.a = a(:);
grf.b = b(:);
grf.betaAngle = beta;
grf.forcingCenter = qc;
grf.forcingWidth = sw;
end

function [uq,fq,residual] = solve_one(grf,queryR,queryTheta,cfg)
N = cfg.NrIntervals;
M = cfg.AngularModes;
[D,x] = cheb_diff_matrix(N);
D2 = D*D;
nFull = N+1;
nPos = (N+1)/2;
posInterior = (2:nPos).';
rPos = x(1:nPos);
rI = x(posInterior);
nrI = numel(rI);
modes = (-M:M).';
nModes = numel(modes);
nUnknown = nrI*nModes;
R1 = diag(1./rI);
R2 = diag(1./(rI.^2));

Nfft = 2^nextpow2(max(512,4*(2*M+1)));
thetaFft = 2*pi*(0:Nfft-1)/Nfft;
[Rmat,Tmat] = ndgrid(rI,thetaFft);
Fphys = eval_forcing(grf,Rmat.*cos(Tmat),Rmat.*sin(Tmat));
Fhat = fft(Fphys,[],2)/Nfft;
rhs = zeros(nUnknown,1);
for jm = 1:nModes
    rows = (jm-1)*nrI+(1:nrI);
    rhs(rows) = Fhat(:,mod(modes(jm),Nfft)+1);
end

ib = cell(3*nModes,1); jb = cell(3*nModes,1); vb = cell(3*nModes,1);
counter = 0;
alpha = grf.betaAngle;
for jm = 1:nModes
    m = modes(jm);
    E = radial_parity_extension(nFull,nPos,(-1)^abs(m));
    D1 = D(posInterior,:)*E(:,2:nPos);
    Drr = D2(posInterior,:)*E(:,2:nPos);
    Lm = -cfg.Epsilon*(Drr+R1*D1-(m^2)*R2);
    rows = (jm-1)*nrI+(1:nrI);
    counter=counter+1; [rr,cc]=ndgrid(rows,rows);
    ib{counter}=rr(:); jb{counter}=cc(:); vb{counter}=Lm(:);
    if jm<nModes
        outRows=jm*nrI+(1:nrI);
        Cplus=0.5*exp(-1i*alpha)*(D1-m*R1);
        counter=counter+1; [rr,cc]=ndgrid(outRows,rows);
        ib{counter}=rr(:); jb{counter}=cc(:); vb{counter}=Cplus(:);
    end
    if jm>1
        outRows=(jm-2)*nrI+(1:nrI);
        Cminus=0.5*exp(1i*alpha)*(D1+m*R1);
        counter=counter+1; [rr,cc]=ndgrid(outRows,rows);
        ib{counter}=rr(:); jb{counter}=cc(:); vb{counter}=Cminus(:);
    end
end
A = sparse(vertcat(ib{1:counter}),vertcat(jb{1:counter}), ...
    vertcat(vb{1:counter}),nUnknown,nUnknown);
coeff = A\rhs;
residual = norm(A*coeff-rhs)/max(norm(rhs),1e-30);

Ucoeff = zeros(nPos,nModes);
for jm=1:nModes
    Ucoeff(2:end,jm)=coeff((jm-1)*nrI+(1:nrI));
end
nThetaInterp=max(cfg.InterpolationAngles,2^(nextpow2(2*M+1)));
thetaInterp=2*pi*(0:nThetaInterp-1)/nThetaInterp;
Upolar=real(Ucoeff*exp(1i*(modes*thetaInterp)));
centerValue=real(Ucoeff(end,modes==0));
rAscending=[0;flipud(rPos)];
uAscending=[centerValue*ones(1,nThetaInterp);flipud(Upolar)];
F=griddedInterpolant({rAscending,[thetaInterp,2*pi]}, ...
    [uAscending,uAscending(:,1)],'spline','nearest');
[RR,TT]=ndgrid(queryR,queryTheta);
uq=single(F(RR,TT));
fq=single(eval_forcing(grf,RR.*cos(TT),RR.*sin(TT)));
end

function value = eval_forcing(grf,x,y)
shape=size(x); xv=x(:); yv=y(:); value=zeros(size(xv));
for k=1:numel(grf.weight)
    phase=pi*(grf.kx(k)*xv+grf.ky(k)*yv);
    value=value+grf.weight(k)*(grf.a(k)*cos(phase)+grf.b(k)*sin(phase));
end
q=-sin(grf.betaAngle).*xv+cos(grf.betaAngle).*yv;
window=exp(-0.5*((q-grf.forcingCenter)./grf.forcingWidth).^2);
value=reshape(window.*value,shape);
end

function E = radial_parity_extension(nFull,nPos,parity)
E=zeros(nFull,nPos); E(1:nPos,:)=eye(nPos);
for j=nPos+1:nFull
    E(j,nFull+1-j)=parity;
end
end

function [D,x] = cheb_diff_matrix(N)
x=cos(pi*(0:N)'/N);
c=[2;ones(N-1,1);2].*(-1).^(0:N)';
X=repmat(x,1,N+1); dX=X-X';
D=(c*(1./c)')./(dX+eye(N+1));
D=D-diag(sum(D,2));
end
