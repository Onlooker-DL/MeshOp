function generate_reaction_diffusion_exact_3100(varargin)
%GENERATE_REACTION_DIFFUSION_EXACT_3100 Reconstruct all 3100 exact solutions.
%
% Put reaction_diffusion_accu_3100.mat in this folder and run:
%
%   generate_reaction_diffusion_exact_3100
%
% By default, the saved 65x65 boundary is expanded in one vectorized batch
% with the exact discrete sine-series solution
% of
%
%   -epsilon^2*Delta(u) + u = 0  on (0,1)^3,
%
% with the random Dirichlet field at z=0 and zero data on the other faces.
% The first 3000 samples train; samples 3001:3100 test.  Output is a
% resumable HDF5 file named reaction_diffusion_exact_3100.h5.  Set
% ReferenceCellsXY=256 only for the slower high-resolution audit reference.

baseDir = fileparts(mfilename('fullpath'));
p = inputParser;
addParameter(p,'SourceFile',fullfile(baseDir,'reaction_diffusion_accu_3100.mat'),@(x)ischar(x)||isstring(x));
addParameter(p,'OutputFile',fullfile(baseDir,'reaction_diffusion_exact_3100.h5'),@(x)ischar(x)||isstring(x));
addParameter(p,'BatchSize',16,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'UseParallel',true,@(x)islogical(x)||isnumeric(x));
addParameter(p,'NumWorkers',4,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(p,'MaxBatches',inf,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'Overwrite',false,@(x)islogical(x)||isnumeric(x));
addParameter(p,'Epsilon',0.02,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'ReferenceCellsXY',256,@(x)isnumeric(x)&&isscalar(x)&&x>=8);
addParameter(p,'BoundaryMultiplier',30,@(x)isnumeric(x)&&isscalar(x)&&x>0);
parse(p,varargin{:});
cfg=p.Results;
cfg.SourceFile=char(cfg.SourceFile);
cfg.OutputFile=char(cfg.OutputFile);
cfg.BatchSize=round(double(cfg.BatchSize));
cfg.NumWorkers=round(double(cfg.NumWorkers));
cfg.MaxBatches=double(cfg.MaxBatches);
cfg.UseParallel=logical(cfg.UseParallel);
cfg.Overwrite=logical(cfg.Overwrite);
cfg.Epsilon=double(cfg.Epsilon);
cfg.ReferenceCellsXY=round(double(cfg.ReferenceCellsXY));
cfg.BoundaryMultiplier=double(cfg.BoundaryMultiplier);
cfg.NumSamples=3100;

if ~isfile(cfg.SourceFile)
    error('Missing source file: %s',cfg.SourceFile);
end
if cfg.Overwrite && isfile(cfg.OutputFile)
    delete(cfg.OutputFile);
end

M=matfile(cfg.SourceFile);
queryX=double(M.query_x(:,1));
queryY=double(M.query_y(:,1));
queryZ=double(M.query_z(:,1));
boundarySize=size(M,'boundary_input');
if numel(boundarySize)~=3 || boundarySize(3)<cfg.NumSamples
    error('boundary_input must contain at least 3100 samples.');
end
if boundarySize(1)~=numel(queryX) || boundarySize(2)~=numel(queryY)
    error('boundary_input dimensions do not match query_x/query_y.');
end
if numel(queryX)~=numel(queryY) || numel(queryX)<3
    error('The sine solver requires equal x/y grids with at least 3 points.');
end
modes=double(M.grf_modes(:,1:2));
spectralStd=double(M.grf_spectral_std(:,1));
coefficientSize=size(M,'grf_xi_cos');
if coefficientSize(1)~=size(modes,1) || coefficientSize(2)<cfg.NumSamples || ...
        numel(spectralStd)~=size(modes,1)
    error('Saved RD Fourier coefficients are inconsistent.');
end
fastOutputGrid = cfg.ReferenceCellsXY==numel(queryX)-1;

fprintf('Source : %s\n',cfg.SourceFile);
fprintf('Output : %s\n',cfg.OutputFile);
fprintf('Split  : samples 1:3000 train, 3001:3100 test\n');
fprintf(['Solver : exact sine expansion, boundary reference=%d^2, ', ...
    'output=%dx%dx%d, epsilon=%.4g\n'],cfg.ReferenceCellsXY, ...
    numel(queryX),numel(queryY),numel(queryZ),cfg.Epsilon);
if fastOutputGrid
    fprintf('Mode   : vectorized exact solve on the saved 65-point output grid\n');
else
    fprintf('Mode   : high-resolution Fourier-boundary reconstruction\n');
end

completed=initialize_output(cfg.OutputFile,queryX,queryY,queryZ,cfg);
if completed>=cfg.NumSamples
    fprintf('Output already contains all %d samples.\n',cfg.NumSamples);
    return;
end

useParallel=~fastOutputGrid && cfg.UseParallel && ...
    license('test','Distrib_Computing_Toolbox');
if useParallel
    try
        pool=gcp('nocreate');
        if isempty(pool)
            if cfg.NumWorkers>0
                pool=parpool('threads',cfg.NumWorkers);
            else
                pool=parpool('threads');
            end
        end
        fprintf('Parallel pool: %d workers\n',pool.NumWorkers);
    catch poolError
        warning('Parallel pool failed (%s). Falling back to serial.',poolError.message);
        useParallel=false;
    end
end
if ~useParallel
    if fastOutputGrid
        fprintf('Vectorized batch generation (MATLAB FFT is internally threaded)\n');
    else
        fprintf('Serial generation\n');
    end
end

wallStart=tic;
batchCounter=0;
nx=numel(queryX); ny=numel(queryY); nz=numel(queryZ);
for first=completed+1:cfg.BatchSize:cfg.NumSamples
    last=min(first+cfg.BatchSize-1,cfg.NumSamples);
    count=last-first+1;
    sourceBoundaryBatch=single(M.boundary_input(:,:,first:last));
    if fastOutputGrid
        boundaryBatch=sourceBoundaryBatch;
        [solutionBatch,reconstructionError]=solve_batch_exact( ...
            boundaryBatch,queryZ,cfg.Epsilon);
    else
        xiCos=double(M.grf_xi_cos(:,first:last));
        xiSin=double(M.grf_xi_sin(:,first:last));
        boundaryBatch=zeros(nx,ny,count,'single');
        solutionBatch=zeros(nx,ny,nz,count,'single');
        reconstructionError=zeros(count,1);
    end
    if ~fastOutputGrid && useParallel
        parfor localId=1:count
            [solutionBatch(:,:,:,localId),boundaryBatch(:,:,localId), ...
                reconstructionError(localId)] = solve_one_exact( ...
                xiCos(:,localId),xiSin(:,localId),modes,spectralStd, ...
                queryX,queryY,queryZ,double(sourceBoundaryBatch(:,:,localId)),cfg);
        end
    elseif ~fastOutputGrid
        for localId=1:count
            [solutionBatch(:,:,:,localId),boundaryBatch(:,:,localId), ...
                reconstructionError(localId)] = solve_one_exact( ...
                xiCos(:,localId),xiSin(:,localId),modes,spectralStd, ...
                queryX,queryY,queryZ,double(sourceBoundaryBatch(:,:,localId)),cfg);
        end
    end

    % h5py sees [sample,x,y,z] and [sample,x,y].
    h5write(cfg.OutputFile,'/solution',permute(solutionBatch,[3,2,1,4]), ...
        [1,1,1,first],[nz,ny,nx,count]);
    h5write(cfg.OutputFile,'/boundary',permute(boundaryBatch,[2,1,3]), ...
        [1,1,first],[ny,nx,count]);
    h5write(cfg.OutputFile,'/boundary_reconstruction_error', ...
        single(reconstructionError(:).'),[1,first],[1,count]);
    h5writeatt(cfg.OutputFile,'/','completed_samples',int32(last));
    batchCounter=batchCounter+1;
    fprintf('[rd exact] %4d/%d | max boundary error %.3e | elapsed %.1f s\n', ...
        last,cfg.NumSamples,max(reconstructionError),toc(wallStart));
    if batchCounter>=cfg.MaxBatches
        fprintf('Stopped intentionally after %d new batch(es); resume is safe.\n',batchCounter);
        break;
    end
end
end

function completed=initialize_output(fileName,x,y,z,cfg)
if isfile(fileName)
    completed=double(h5readatt(fileName,'/','completed_samples'));
    if double(h5readatt(fileName,'/','num_samples'))~=cfg.NumSamples || ...
            abs(double(h5readatt(fileName,'/','epsilon'))-cfg.Epsilon)>1e-14 || ...
            double(h5readatt(fileName,'/','reference_cells_xy'))~=cfg.ReferenceCellsXY
        error('Existing output settings differ; use Overwrite=true intentionally.');
    end
    return;
end
nx=numel(x); ny=numel(y); nz=numel(z); n=cfg.NumSamples;
h5create(fileName,'/solution',[nz,ny,nx,n],'Datatype','single', ...
    'ChunkSize',[nz,ny,nx,1]);
h5create(fileName,'/boundary',[ny,nx,n],'Datatype','single', ...
    'ChunkSize',[ny,nx,1]);
h5create(fileName,'/boundary_reconstruction_error',[1,n], ...
    'Datatype','single','ChunkSize',[1,min(n,128)]);
h5create(fileName,'/x',[1,nx],'Datatype','single');
h5create(fileName,'/y',[1,ny],'Datatype','single');
h5create(fileName,'/z',[1,nz],'Datatype','single');
h5create(fileName,'/source_sample_id',[1,n],'Datatype','int32');
h5write(fileName,'/x',single(x(:).'));
h5write(fileName,'/y',single(y(:).'));
h5write(fileName,'/z',single(z(:).'));
h5write(fileName,'/source_sample_id',int32(1:n));
h5writeatt(fileName,'/','completed_samples',int32(0));
h5writeatt(fileName,'/','num_samples',int32(n));
h5writeatt(fileName,'/','train_samples',int32(3000));
h5writeatt(fileName,'/','test_samples',int32(100));
h5writeatt(fileName,'/','epsilon',cfg.Epsilon);
h5writeatt(fileName,'/','reference_cells_xy',int32(cfg.ReferenceCellsXY));
h5writeatt(fileName,'/','boundary_multiplier',cfg.BoundaryMultiplier);
if cfg.ReferenceCellsXY==nx-1
    method='vectorized exact sine-series solution on output grid';
else
    method='high-resolution Fourier-boundary reconstruction plus sine series';
end
h5writeatt(fileName,'/','method',method);
h5writeatt(fileName,'/','pde','-epsilon^2 Laplacian(u) + u = 0');
completed=0;
end

function [solution,errorAtBoundary]=solve_batch_exact(boundary,z,epsilon)
% One batched DST for all samples and all interior z levels.  This is the
% exact discrete sine-series solution on the saved 65-point x/y grid.
nx=size(boundary,1);
ny=size(boundary,2);
count=size(boundary,3);
if nx~=ny
    error('Only equal x/y sine grids are supported.');
end
N=nx-1;
gInterior=double(boundary(2:end-1,2:end-1,:));
gHat=dst2_type1(gInterior);
bCoeff=(2/N)^2*gHat;
[m1,m2]=ndgrid(1:N-1,1:N-1);
mu=sqrt(epsilon^(-2)+pi^2*(m1.^2+m2.^2));
solution=zeros(nx,ny,numel(z),count,'single');
solution(:,:,1,:)=reshape(boundary,nx,ny,1,count);
if numel(z)>2
    zMiddle=reshape(z(2:end-1),1,1,[],1);
    factor=exp(-mu.*zMiddle).* ...
        (1-exp(-2*mu.*(1-zMiddle)))./(1-exp(-2*mu));
    modal=reshape(bCoeff,N-1,N-1,1,count).*factor;
    interior=dst2_type1(modal);
    solution(2:end-1,2:end-1,2:end-1,:)=single(interior);
end
reconstructed=dst2_type1(bCoeff);
difference=reshape(reconstructed-gInterior,[],count);
errorAtBoundary=max(abs(difference),[],1).';
end

function [solution,boundaryQuery,errorAtBoundary]=solve_one_exact( ...
    xiCos,xiSin,modes,spectralStd,queryX,queryY,z,sourceBoundary,cfg)
N=cfg.ReferenceCellsXY;
if any(abs(queryX*N-round(queryX*N))>1e-10) || ...
        any(abs(queryY*N-round(queryY*N))>1e-10)
    error('Every x/y query coordinate must lie on the reference grid.');
end
gPeriodic=render_periodic_grf(N,modes,spectralStd,xiCos,xiSin);
gExtended=zeros(N+1,N+1);
gExtended(1:N,1:N)=gPeriodic;
gExtended(N+1,1:N)=gPeriodic(1,:);
gExtended(1:N,N+1)=gPeriodic(:,1);
gExtended(N+1,N+1)=gPeriodic(1,1);
fine=(0:N)'/N;
[xx,yy]=ndgrid(fine,fine);
boundaryFine=cfg.BoundaryMultiplier*16*xx.*(1-xx).*yy.*(1-yy).*gExtended;
xIds=round(queryX*N)+1;
yIds=round(queryY*N)+1;
boundaryQuery=single(boundaryFine(xIds,yIds));
gInterior=boundaryFine(2:end-1,2:end-1);
gHat=dst2_type1(gInterior);
bCoeff=(2/N)^2*gHat;
[m1,m2]=ndgrid(1:N-1,1:N-1);
mu=sqrt(cfg.Epsilon^(-2)+pi^2*(m1.^2+m2.^2));
solution=zeros(numel(queryX),numel(queryY),numel(z),'single');
solution(:,:,1)=boundaryQuery;
interiorX=xIds(2:end-1)-1;
interiorY=yIds(2:end-1)-1;
if numel(z)>2
    zMiddle=reshape(z(2:end-1),1,1,[]);
    factor=exp(-mu.*zMiddle).* ...
        (1-exp(-2*mu.*(1-zMiddle)))./(1-exp(-2*mu));
    interior=dst2_type1(bCoeff.*factor);
    solution(2:end-1,2:end-1,2:end-1)= ...
        single(interior(interiorX,interiorY,:));
end
errorAtBoundary=max(abs(double(boundaryQuery(:))-sourceBoundary(:)));
end

function g=render_periodic_grf(N,modes,spectralStd,xiCos,xiSin)
ip=mod(modes(:,1),N)+1;
jp=mod(modes(:,2),N)+1;
im=mod(-modes(:,1),N)+1;
jm=mod(-modes(:,2),N)+1;
idxPlus=sub2ind([N,N],ip,jp);
idxMinus=sub2ind([N,N],im,jm);
coeff=0.5*N^2.*spectralStd(:).*(xiCos(:)-1i*xiSin(:));
Fhat=complex(zeros(N,N));
Fhat(idxPlus)=coeff;
Fhat(idxMinus)=conj(coeff);
g=real(ifft2(Fhat));
end

function Y=dst2_type1(X)
Y=dst1_dimension(X,1);
Y=dst1_dimension(Y,2);
end

function Y=dst1_dimension(X,dim)
nd=ndims(X); perm=[dim,1:dim-1,dim+1:nd];
Xp=permute(X,perm); sz=size(Xp); n=sz(1); X2=reshape(Xp,n,[]);
Z=zeros(2*(n+1),size(X2,2),'like',X2);
Z(2:n+1,:)=X2; Z(n+3:end,:)=-flipud(X2);
F=fft(Z,[],1); Y2=-imag(F(2:n+1,:))/2;
Y=ipermute(reshape(Y2,sz),perm);
end
