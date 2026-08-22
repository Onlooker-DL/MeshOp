function results = evaluate_burgers_matlab(cfg)
%EVALUATE_BURGERS_MATLAB MATLAB half of the U-Net/Gmsh/FEM baseline.
%
% The online path is
%   saved U-Net inference time + Gmsh time + common 32x32 FEM solve
%   + coarse-to-generated-mesh interpolation + final Newton FEM solve.
% The solver, warm start, periodic reduction, quadrature, Newton safeguards,
% spectral reference and physical L2 metric mirror MeshOp v1.1.1.

validate_inputs(cfg);
if ~isfolder(cfg.outputDirectory), mkdir(cfg.outputDirectory); end
scalarDir = fullfile(cfg.outputDirectory,'sample_metrics');
artifactDir = fullfile(cfg.outputDirectory,'selected_samples');
if ~isfolder(scalarDir), mkdir(scalarDir); end
if ~isfolder(artifactDir), mkdir(artifactDir); end

data = read_burgers_data(cfg.dataFile);
if isempty(cfg.testIds)
    testIds = 1:data.testSamples;
else
    testIds = unique(round(cfg.testIds),'stable');
end
if any(testIds<1 | testIds>data.testSamples)
    error('TestIds must lie in [1,%d].',data.testSamples);
end

metrics = repmat(empty_metric(),numel(testIds),1);
fprintf('[MATLAB FEM] samples=%d | common coarse grid=%dx%d\n', ...
    numel(testIds),cfg.initialGrid,cfg.initialGrid);
fprintf(['[MATLAB FEM] matched online = inference + Gmsh + coarse solve ', ...
    '+ final solve; strict online also includes interpolation.\n']);

for kk=1:numel(testIds)
    testId = testIds(kk);
    row = data.trainSamples+testId;
    sourceId = double(data.sourceIds(row));
    metricFile = fullfile(scalarDir,sprintf('test_%03d.mat',testId));
    if cfg.resume && isfile(metricFile)
        saved = load(metricFile,'metric');
        if isfield(saved,'metric') && saved.metric.source_dataset_index==sourceId
            metrics(kk)=saved.metric;
            fprintf('[%3d/%3d] test=%03d source=%d | resumed | relL2=%.4e\n', ...
                kk,numel(testIds),testId,sourceId,metrics(kk).relative_l2);
            continue;
        end
    end

    meshFile = fullfile(cfg.meshDirectory, ...
        sprintf('test_%03d_source_%d.mat',testId,sourceId));
    if ~isfile(meshFile)
        error(['Missing exported mesh %s. Run python -m ', ...
            'src.meshing.export_burgers_meshes first.'],meshFile);
    end
    meshData = load(meshFile);
    [node,elem] = validate_mesh(meshData,cfg,testId,sourceId);
    [u0fun,ffun] = make_sample_functions(data,row);
    sampleCfg = cfg;
    sampleCfg.forcingFun = ffun;

    % Reference and error work are deliberately outside online timing.
    referenceStart = tic;
    ref = solve_burgers_fourier_reference(u0fun,sampleCfg);
    Fref = make_ref_interpolant(ref);
    referenceSec = toc(referenceStart);

    [coarseNode,coarseElem] = make_uniform_spacetime_mesh( ...
        cfg.initialGrid,sampleCfg);
    coarseStart = tic;
    [coarseU,coarseInfo] = solve_burgers_newton_st( ...
        coarseNode,coarseElem,u0fun,sampleCfg,'initial_condition');
    coarseSec = toc(coarseStart);

    interpolationStart = tic;
    initialU = interpolate_to_new_nodes( ...
        coarseNode,coarseElem,coarseU,node);
    interpolationSec = toc(interpolationStart);

    finalStart = tic;
    [u,finalInfo] = solve_burgers_newton_st( ...
        node,elem,u0fun,sampleCfg,initialU);
    finalSec = toc(finalStart);

    errorStart = tic;
    err = compute_fem_vs_reference_error(node,elem,u,ref,sampleCfg,Fref);
    errorSec = toc(errorStart);
    map = build_periodic_reduction(node,sampleCfg);

    inferenceSec = read_scalar(meshData,'inference_time_sec');
    meshSec = read_scalar(meshData,'mesh_time_sec');
    metric = empty_metric();
    metric.test_id = testId;
    metric.source_dataset_index = sourceId;
    metric.nodes = size(node,1);
    metric.elements = size(elem,1);
    metric.dof = numel(map.freeReduced);
    metric.relative_l2 = err.spaceTimeL2Rel;
    metric.absolute_l2 = err.spaceTimeL2Abs;
    metric.final_time_relative_l2 = err.finalTimeL2Rel;
    metric.nodal_linf = err.nodalLinf;
    metric.inference_time_sec = inferenceSec;
    metric.gmsh_mesh_time_sec = meshSec;
    metric.coarse_fem_time_sec = coarseSec;
    metric.interpolation_time_sec = interpolationSec;
    metric.final_fem_time_sec = finalSec;
    metric.meshop_matched_online_sec = ...
        inferenceSec+meshSec+coarseSec+finalSec;
    metric.strict_end_to_end_online_sec = ...
        metric.meshop_matched_online_sec+interpolationSec;
    metric.coarse_assembly_time_sec = coarseInfo.assemblyTimeSec;
    metric.coarse_linear_solve_time_sec = coarseInfo.solveTimeSec;
    metric.final_assembly_time_sec = finalInfo.assemblyTimeSec;
    metric.final_linear_solve_time_sec = finalInfo.solveTimeSec;
    metric.coarse_newton_iterations = coarseInfo.iter;
    metric.final_newton_iterations = finalInfo.iter;
    metric.coarse_newton_converged = coarseInfo.converged;
    metric.final_newton_converged = finalInfo.converged;
    metric.reference_time_sec_excluded = referenceSec;
    metric.error_time_sec_excluded = errorSec;
    metric.mesh_background_field_time_sec = ...
        read_scalar(meshData,'mesh_background_field_time_sec');
    metric.mesh_gmsh_generate_time_sec = ...
        read_scalar(meshData,'mesh_gmsh_generate_time_sec');
    metric.mesh_extraction_time_sec = ...
        read_scalar(meshData,'mesh_extraction_time_sec');
    metrics(kk)=metric;
    save(metricFile,'metric');

    if ismember(testId,cfg.saveSampleIds)
        save_selected_sample(artifactDir,testId,sourceId,node,elem,u, ...
            initialU,coarseNode,coarseElem,coarseU,meshData,metric,cfg);
    end
    write_outputs(metrics(1:kk),cfg);
    fprintf(['[%3d/%3d] test=%03d source=%d | dof=%d elem=%d | ', ...
        'relL2=%.4e | matched=%.3fs strict=%.3fs | Newton=%d\n'], ...
        kk,numel(testIds),testId,sourceId,metric.dof,metric.elements, ...
        metric.relative_l2,metric.meshop_matched_online_sec, ...
        metric.strict_end_to_end_online_sec,metric.final_newton_iterations);
end

results = write_outputs(metrics,cfg);
fprintf('Wrote %s\n',fullfile(cfg.outputDirectory, ...
    'per_sample_matlab_fem_metrics.csv'));
end


function validate_inputs(cfg)
required = {'dataFile','meshDirectory','outputDirectory','meshCoordinateMode','initialGrid', ...
    'xmin','xmax','tmin','tmax','nu'};
for k=1:numel(required)
    if ~isfield(cfg,required{k}), error('cfg.%s is required.',required{k}); end
end
if ~isfile(cfg.dataFile), error('Missing data file: %s',cfg.dataFile); end
if ~isfolder(cfg.meshDirectory), error('Missing mesh directory: %s',cfg.meshDirectory); end
end


function data = read_burgers_data(fileName)
data.trainSamples = double(h5readatt(fileName,'/','train_samples'));
data.testSamples = double(h5readatt(fileName,'/','test_samples'));
data.totalSamples = double(h5readatt(fileName,'/','total_samples'));
data.sourceIds = double(h5read(fileName,'/source_dataset_index'));
data.sourceIds = data.sourceIds(:);
data.ic = orient_coefficients(h5read(fileName,'/ic_coefficients'),data.totalSamples);
data.forcingCos = orient_coefficients( ...
    h5read(fileName,'/forcing_xi_cos'),data.totalSamples);
data.forcingSin = orient_coefficients( ...
    h5read(fileName,'/forcing_xi_sin'),data.totalSamples);
data.grfModes = double(h5read(fileName,'/grf_modes')); data.grfModes=data.grfModes(:);
data.grfStd = double(h5read(fileName,'/grf_spectral_std')); data.grfStd=data.grfStd(:);
data.forcingModes = double(h5read(fileName,'/forcing_modes')); data.forcingModes=data.forcingModes(:);
data.forcingStd = double(h5read(fileName,'/forcing_spectral_std')); data.forcingStd=data.forcingStd(:);
expected = 2+2*numel(data.grfModes);
if size(data.ic,1)~=expected
    error('ic_coefficients has %d rows; expected %d.',size(data.ic,1),expected);
end
if size(data.forcingCos,1)~=numel(data.forcingModes) || ...
        ~isequal(size(data.forcingCos),size(data.forcingSin))
    error('Forcing coefficient arrays are inconsistent.');
end
end


function A = orient_coefficients(A,nSamples)
A = double(A);
if size(A,2)==nSamples
    return;
elseif size(A,1)==nSamples
    A=A.';
else
    error('Coefficient array has no sample dimension of length %d.',nSamples);
end
end


function [u0fun,ffun] = make_sample_functions(data,row)
nGrf = numel(data.grfModes);
coeff = data.ic(:,row);
amplitude = coeff(1);
shift = coeff(2);
grfCos = coeff(3:2+nGrf);
grfSin = coeff(3+nGrf:2+2*nGrf);
forcingCos = data.forcingCos(:,row);
forcingSin = data.forcingSin(:,row);
u0fun = @(x)evaluate_initial_condition(x,amplitude,shift, ...
    data.grfModes,data.grfStd,grfCos,grfSin);
ffun = @(x)evaluate_fourier_field(x,data.forcingModes, ...
    data.forcingStd,forcingCos,forcingSin);
end


function values = evaluate_initial_condition(x,a,b,modes,sigma,xiCos,xiSin)
originalSize=size(x); xv=double(x(:));
values=-a*sin(pi*(xv+b))+cos(pi*xv*modes.')*(sigma.*xiCos)+ ...
    sin(pi*xv*modes.')*(sigma.*xiSin);
values=reshape(values,originalSize);
end


function values = evaluate_fourier_field(x,modes,sigma,xiCos,xiSin)
originalSize=size(x); xv=double(x(:));
values=cos(pi*xv*modes.')*(sigma.*xiCos)+ ...
    sin(pi*xv*modes.')*(sigma.*xiSin);
values=reshape(values,originalSize);
end


function [node,elem] = validate_mesh(M,cfg,testId,sourceId)
required={'nodes','triangles','test_id','source_dataset_index','coordinate_mode'};
for k=1:numel(required)
    if ~isfield(M,required{k}), error('Mesh MAT is missing %s.',required{k}); end
end
node=double(M.nodes); elem=double(M.triangles);
if size(node,2)~=2 || size(elem,2)~=3
    error('Expected nodes Nx2 and triangles Mx3.');
end
if round(double(M.test_id(1)))~=testId || ...
        round(double(M.source_dataset_index(1)))~=sourceId
    error('Mesh identity does not match test/source ID.');
end
actualMode=strtrim(char(string(M.coordinate_mode)));
if ~strcmp(actualMode,cfg.meshCoordinateMode)
    error('Mesh coordinate_mode="%s"; expected "%s".', ...
        actualMode,cfg.meshCoordinateMode);
end
if size(elem,1)>cfg.maxElements
    error('Mesh has %d elements; cfg.maxElements=%d.',size(elem,1),cfg.maxElements);
end
if any(elem(:)~=round(elem(:))) || min(elem(:))<1 || max(elem(:))>size(node,1)
    error('Invalid one-based triangle connectivity.');
end
if any(~isfinite(node(:)))
    error('Mesh nodes contain NaN/Inf.');
end
elem=round(elem);
end


function value = read_scalar(S,name)
if ~isfield(S,name), error('Mesh MAT is missing timing field %s.',name); end
value=double(S.(name)(1));
if ~isfinite(value) || value<0, error('Invalid timing field %s.',name); end
end


function metric = empty_metric()
metric=struct( ...
    'test_id',NaN,'source_dataset_index',NaN,'nodes',NaN,'elements',NaN, ...
    'dof',NaN,'relative_l2',NaN,'absolute_l2',NaN, ...
    'final_time_relative_l2',NaN,'nodal_linf',NaN, ...
    'inference_time_sec',NaN,'gmsh_mesh_time_sec',NaN, ...
    'coarse_fem_time_sec',NaN,'interpolation_time_sec',NaN, ...
    'final_fem_time_sec',NaN,'meshop_matched_online_sec',NaN, ...
    'strict_end_to_end_online_sec',NaN,'coarse_assembly_time_sec',NaN, ...
    'coarse_linear_solve_time_sec',NaN,'final_assembly_time_sec',NaN, ...
    'final_linear_solve_time_sec',NaN,'coarse_newton_iterations',NaN, ...
    'final_newton_iterations',NaN,'coarse_newton_converged',false, ...
    'final_newton_converged',false,'reference_time_sec_excluded',NaN, ...
    'error_time_sec_excluded',NaN,'mesh_background_field_time_sec',NaN, ...
    'mesh_gmsh_generate_time_sec',NaN,'mesh_extraction_time_sec',NaN);
end


function results = write_outputs(metrics,cfg)
valid=arrayfun(@(m)isfinite(m.test_id),metrics);
metrics=metrics(valid);
if isempty(metrics), results=struct(); return; end
T=struct2table(metrics);
writetable(T,fullfile(cfg.outputDirectory,'per_sample_matlab_fem_metrics.csv'));
summary=struct();
summary.problem='burgers';
summary.method='unet_mesh_size_gmsh_periodic_p1_spacetime_fem';
summary.samples=height(T);
summary.initial_grid=cfg.initialGrid;
summary.mesh_coordinate_mode=cfg.meshCoordinateMode;
summary.mesh_coordinate_map='xi=(x+1)/2, tau=t; Gmsh on unit square; return x=2*xi-1';
summary.warm_start='32x32 coarse FEM solution interpolated to the Gmsh mesh';
summary.mean_relative_l2=mean(T.relative_l2);
summary.median_relative_l2=median(T.relative_l2);
summary.mean_dof=mean(T.dof);
summary.mean_elements=mean(T.elements);
summary.mean_inference_time_sec=mean(T.inference_time_sec);
summary.mean_gmsh_mesh_time_sec=mean(T.gmsh_mesh_time_sec);
summary.mean_coarse_fem_time_sec=mean(T.coarse_fem_time_sec);
summary.mean_interpolation_time_sec=mean(T.interpolation_time_sec);
summary.mean_final_fem_time_sec=mean(T.final_fem_time_sec);
summary.mean_meshop_matched_online_sec=mean(T.meshop_matched_online_sec);
summary.mean_strict_end_to_end_online_sec=mean(T.strict_end_to_end_online_sec);
summary.meshop_matched_timing_definition= ...
    'U-Net inference + full Gmsh mesh generation + 32x32 coarse FEM solve + final FEM solve';
summary.strict_timing_definition= ...
    'MeshOp-matched timing + coarse-to-generated-mesh interpolation';
summary.excluded_from_online_time= ...
    'spectral reference, error evaluation, plotting, and file I/O';
summary.relative_l2_definition= ...
    'per-sample physical-domain P1 quadrature relative L2 error';
summary.reference_method= ...
    'periodic Fourier pseudo-spectral ETDRK4, N=512, dt=2.5e-4, 2/3 dealiasing';
summary.newton=struct('max_iterations',cfg.newtonMaxIt, ...
    'absolute_tolerance',cfg.newtonTol,'relative_tolerance',cfg.newtonRelTol, ...
    'line_search_max',cfg.newtonLineSearchMax, ...
    'regularization_trials',cfg.newtonRegularizationTrials, ...
    'max_update_factor',cfg.newtonMaxUpdateFactor);
jsonText=jsonencode(summary,'PrettyPrint',true);
fid=fopen(fullfile(cfg.outputDirectory,'final_metrics_matlab.json'),'w');
if fid<0, error('Cannot create final_metrics_matlab.json.'); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonText,'char');
save(fullfile(cfg.outputDirectory,'matlab_fem_results.mat'),'T','summary','cfg');
results=struct('table',T,'summary',summary);
end


function save_selected_sample(outDir,testId,sourceId,node,elem,u,initialU, ...
        coarseNode,coarseElem,coarseU,meshData,metric,cfg)
matFile=fullfile(outDir,sprintf('test_%03d_source_%d.mat',testId,sourceId));
save(matFile,'node','elem','u','initialU','coarseNode','coarseElem', ...
    'coarseU','meshData','metric','-v7.3');
if ~cfg.makePlots, return; end
fig=figure('Visible','off','Color','w','Position',[50,50,1500,430]);
cleanup=onCleanup(@()close(fig)); %#ok<NASGU>
subplot(1,3,1);
if isfield(meshData,'predicted_log_h') && isfield(meshData,'query_x') && ...
        isfield(meshData,'query_t')
    imagesc(meshData.query_x(:),meshData.query_t(:), ...
        double(meshData.predicted_log_h)); axis xy tight; colorbar;
else
    axis off; text(0.5,0.5,'size field unavailable','HorizontalAlignment','center');
end
xlabel('x'); ylabel('t');
subplot(1,3,2);
patch('Faces',elem,'Vertices',node,'FaceColor','none', ...
    'EdgeColor',[0.1 0.1 0.1],'LineWidth',0.1);
axis equal tight; xlabel('x'); ylabel('t');
subplot(1,3,3);
patch('Faces',elem,'Vertices',node,'FaceVertexCData',u, ...
    'FaceColor','interp','EdgeColor','none');
axis equal tight; xlabel('x'); ylabel('t'); colorbar;
exportgraphics(fig,fullfile(outDir, ...
    sprintf('test_%03d_source_%d.png',testId,sourceId)),'Resolution',220);
end


function [node,elem] = make_uniform_spacetime_mesh(n,cfg)
xg=linspace(cfg.xmin,cfg.xmax,n+1);
tg=linspace(cfg.tmin,cfg.tmax,n+1);
[XX,TT]=meshgrid(xg,tg);
node=[XX(:),TT(:)];
id=@(i,j)j*(n+1)+i+1;
elem=zeros(2*n*n,3); cnt=0;
for j=0:n-1
    for i=0:n-1
        n00=id(i,j); n10=id(i+1,j); n01=id(i,j+1); n11=id(i+1,j+1);
        cnt=cnt+1; elem(cnt,:)=[n00,n10,n11];
        cnt=cnt+1; elem(cnt,:)=[n00,n11,n01];
    end
end
end


function map = build_periodic_reduction(node,cfg)
N=size(node,1); tol=periodic_tolerance(cfg);
x=node(:,1); t=node(:,2);
left=find(abs(x-cfg.xmin)<tol); right=find(abs(x-cfg.xmax)<tol);
[tLeft,oLeft]=sort(t(left)); [tRight,oRight]=sort(t(right));
left=left(oLeft); right=right(oRight);
if numel(left)~=numel(right) || ...
        (~isempty(left) && max(abs(tLeft-tRight))>20*tol)
    error(['Periodic boundary node sets do not match. The Gmsh exporter ', ...
        'must use setPeriodic on the two vertical sides.']);
end
master=(1:N).'; master(right)=left;
[~,~,rid]=unique(master); nReduced=max(rid);
P=sparse((1:N).',rid,1,N,nReduced);
representative=accumarray(rid,(1:N).',[],@min);
initFull=find(abs(t-cfg.tmin)<tol);
fixedReduced=unique(rid(initFull));
freeReduced=setdiff((1:nReduced).',fixedReduced);
map=struct('P',P,'rid',rid,'representative',representative, ...
    'nReduced',nReduced,'fixedReduced',fixedReduced, ...
    'freeReduced',freeReduced,'leftNodes',left,'rightNodes',right);
end


function tol = periodic_tolerance(cfg)
tol=1e-10*max([1,abs(cfg.xmin),abs(cfg.xmax),abs(cfg.tmin),abs(cfg.tmax)]);
end


function [uFull,ninfo] = solve_burgers_newton_st(node,elem,u0fun,cfg,uInit)
% Same damped Newton path used by MeshOp v1.1.1.
tAsmSetup=tic;
map=build_periodic_reduction(node,cfg);
geom=precompute_fem_assembly_geometry(node,elem);
geom.fq=cfg.forcingFun(geom.xq);
tAsmTotal=toc(tAsmSetup); tSolveTotal=0;
P=map.P; rep=map.representative; free=map.freeReduced; fixed=map.fixedReduced;
N=size(node,1); x=node(:,1); t=node(:,2); tol=periodic_tolerance(cfg);
gFull=zeros(N,1); initMask=abs(t-cfg.tmin)<tol;
gFull(initMask)=u0fun(x(initMask));
gRed=zeros(map.nReduced,1); gRed(fixed)=gFull(rep(fixed));

if ischar(uInit) || (isstring(uInit) && isscalar(uInit))
    token=lower(strtrim(char(uInit)));
    if strcmp(token,'initial_condition')
        initialFull=u0fun(x); initializationMode='initial_condition';
    elseif strcmp(token,'zero')
        initialFull=zeros(N,1); initializationMode='zero';
    else
        error('Unsupported Newton initialization mode: %s',token);
    end
elseif isnumeric(uInit) && isvector(uInit) && numel(uInit)==N
    initialFull=double(uInit(:));
    initializationMode='provided_interpolated_solution';
else
    error('uInit must be initial_condition, zero, or a vector per node.');
end
if any(~isfinite(initialFull)), error('Newton initial guess contains NaN/Inf.'); end
uRed=initialFull(rep); uRed(fixed)=gRed(fixed); uFull=P*uRed;
tAsm=tic;
[Rfull,Jfull]=assemble_burgers_res_jac_precomputed(geom,uFull,cfg.nu);
tAsmTotal=tAsmTotal+toc(tAsm);
Rred=P.'*Rfull; Jred=P.'*Jfull*P;
initialResidual=norm(Rred(free))/sqrt(max(numel(free),1));
target=max(cfg.newtonTol,cfg.newtonRelTol*initialResidual);
ninfo=struct('initialResidual',initialResidual,'targetResidual',target, ...
    'initializationMode',initializationMode,'acceptedSteps',[], ...
    'regularization',[],'converged',false,'usedLastIterate',false, ...
    'exitReason','running','assemblyTimeSec',tAsmTotal, ...
    'solveTimeSec',tSolveTotal,'iter',0,'finalResidual',initialResidual);

for it=1:cfg.newtonMaxIt
    residualVector=Rred(free);
    resNorm=norm(residualVector)/sqrt(max(numel(free),1));
    if cfg.newtonVerbose
        fprintf('      Newton %02d residual %.3e\n',it-1,resNorm);
    end
    if resNorm<=target
        ninfo.iter=it-1; ninfo.finalResidual=resNorm; ninfo.converged=true;
        ninfo.exitReason='converged'; uFull=P*uRed; return;
    end
    A=Jred(free,free); rhs=-residualVector;
    accepted=false; baseNorm=norm(residualVector); bestNorm=baseNorm;
    bestRed=uRed; bestStep=0; bestMu=NaN;
    matrixScale=max(1,norm(A,1));
    baseMu=max(cfg.newtonRegularizationBase*matrixScale,eps(matrixScale));
    for regTry=0:cfg.newtonRegularizationTrials
        if regTry==0, mu=0; else, mu=baseMu*10^(regTry-1); end
        Atry=A;
        if mu>0, Atry=Atry+mu*speye(size(Atry)); end
        tSol=tic;
        try
            duFree=Atry\rhs;
        catch
            continue;
        end
        ninfo.solveTimeSec=ninfo.solveTimeSec+toc(tSol);
        if any(~isfinite(duFree)), continue; end
        maxAllowed=cfg.newtonMaxUpdateFactor*max(1,max(abs(uRed(free))));
        duInf=norm(duFree,inf);
        if duInf>maxAllowed, duFree=duFree*(maxAllowed/duInf); end
        if cfg.newtonLineSearch, maxLS=cfg.newtonLineSearchMax; else, maxLS=0; end
        for ls=0:maxLS
            step=2^(-ls); trialRed=uRed;
            trialRed(free)=uRed(free)+step*duFree; trialRed(fixed)=gRed(fixed);
            trialFull=P*trialRed;
            tAsm=tic;
            RtrialFull=assemble_burgers_res_only_precomputed(geom,trialFull,cfg.nu);
            ninfo.assemblyTimeSec=ninfo.assemblyTimeSec+toc(tAsm);
            RtrialRed=P.'*RtrialFull; trialNorm=norm(RtrialRed(free));
            if isfinite(trialNorm) && trialNorm<bestNorm
                bestNorm=trialNorm; bestRed=trialRed; bestStep=step; bestMu=mu;
            end
            if isfinite(trialNorm) && trialNorm<=(1-1e-4*step)*baseNorm
                uRed=trialRed; Rred=RtrialRed; accepted=true;
                ninfo.acceptedSteps(end+1,1)=step; %#ok<AGROW>
                ninfo.regularization(end+1,1)=mu; %#ok<AGROW>
                break;
            end
        end
        if accepted, break; end
    end
    if ~accepted && bestNorm<baseNorm*(1-1e-12)
        uRed=bestRed; uFull=P*uRed; tAsm=tic;
        Rfull=assemble_burgers_res_only_precomputed(geom,uFull,cfg.nu);
        ninfo.assemblyTimeSec=ninfo.assemblyTimeSec+toc(tAsm);
        Rred=P.'*Rfull; accepted=true;
        ninfo.acceptedSteps(end+1,1)=bestStep; %#ok<AGROW>
        ninfo.regularization(end+1,1)=bestMu; %#ok<AGROW>
    end
    if ~accepted
        ninfo.iter=it-1; ninfo.finalResidual=resNorm;
        ninfo.usedLastIterate=true; ninfo.exitReason='stalled_return_last_iterate';
        uFull=P*uRed;
        warning(['Periodic Newton stalled at iteration %d: RMS residual ', ...
            '%.3e > target %.3e. Keeping the last finite iterate.'], ...
            it-1,resNorm,target);
        return;
    end
    uFull=P*uRed; tAsm=tic;
    [Rfull,Jfull]=assemble_burgers_res_jac_precomputed(geom,uFull,cfg.nu);
    ninfo.assemblyTimeSec=ninfo.assemblyTimeSec+toc(tAsm);
    Rred=P.'*Rfull; Jred=P.'*Jfull*P;
end
finalResidual=norm(Rred(free))/sqrt(max(numel(free),1));
ninfo.iter=cfg.newtonMaxIt; ninfo.finalResidual=finalResidual;
ninfo.converged=finalResidual<=target; uFull=P*uRed;
if ninfo.converged
    ninfo.exitReason='converged_at_max_iterations';
else
    ninfo.usedLastIterate=true; ninfo.exitReason='max_iterations_return_last_iterate';
    warning(['Periodic Newton reached maxIt=%d with RMS residual %.3e > ', ...
        'target %.3e. Keeping the last finite iterate.'], ...
        cfg.newtonMaxIt,finalResidual,target);
end
end


function geom = precompute_fem_assembly_geometry(node,elem)
geom.N=size(node,1); geom.elem=elem;
p1=node(elem(:,1),:); p2=node(elem(:,2),:); p3=node(elem(:,3),:);
detJ=(p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2))- ...
    (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2));
geom.area=0.5*abs(detJ);
if any(geom.area<=1e-15), error('FEM geometry contains degenerate triangles.'); end
NT=size(elem,1); geom.dphidx=zeros(NT,3); geom.dphidt=zeros(NT,3);
geom.dphidx(:,1)=(p2(:,2)-p3(:,2))./detJ;
geom.dphidx(:,2)=(p3(:,2)-p1(:,2))./detJ;
geom.dphidx(:,3)=(p1(:,2)-p2(:,2))./detJ;
geom.dphidt(:,1)=(p3(:,1)-p2(:,1))./detJ;
geom.dphidt(:,2)=(p1(:,1)-p3(:,1))./detJ;
geom.dphidt(:,3)=(p2(:,1)-p1(:,1))./detJ;
[geom.lambda,geom.w]=tri_quad_3(); geom.xq=zeros(NT,3);
for q=1:3
    phi=geom.lambda(q,:);
    geom.xq(:,q)=phi(1)*p1(:,1)+phi(2)*p2(:,1)+phi(3)*p3(:,1);
end
geom.I=zeros(NT,9); geom.JJ=zeros(NT,9); col=0;
for a=1:3
    for b=1:3
        col=col+1; geom.I(:,col)=elem(:,a); geom.JJ(:,col)=elem(:,b);
    end
end
end


function [R,J] = assemble_burgers_res_jac_precomputed(geom,u,nu)
elem=geom.elem; Ue=u(elem); ux=sum(Ue.*geom.dphidx,2);
ut=sum(Ue.*geom.dphidt,2); Uq=Ue*geom.lambda.';
NT=size(elem,1); Re=zeros(NT,3); Je=zeros(NT,9);
for q=1:3
    phi=geom.lambda(q,:); uq=Uq(:,q);
    strong=ut+uq.*ux-geom.fq(:,q); weight=geom.area*geom.w(q);
    for a=1:3
        Re(:,a)=Re(:,a)+weight.*(strong*phi(a)+nu*ux.*geom.dphidx(:,a));
        for b=1:3
            col=(a-1)*3+b;
            dstrong=geom.dphidt(:,b)+phi(b)*ux+uq.*geom.dphidx(:,b);
            Je(:,col)=Je(:,col)+weight.*(dstrong*phi(a)+ ...
                nu*geom.dphidx(:,b).*geom.dphidx(:,a));
        end
    end
end
R=accumarray(elem(:),Re(:),[geom.N,1],@sum,0);
J=sparse(geom.I(:),geom.JJ(:),Je(:),geom.N,geom.N);
end


function R = assemble_burgers_res_only_precomputed(geom,u,nu)
elem=geom.elem; Ue=u(elem); ux=sum(Ue.*geom.dphidx,2);
ut=sum(Ue.*geom.dphidt,2); Uq=Ue*geom.lambda.';
NT=size(elem,1); Re=zeros(NT,3);
for q=1:3
    phi=geom.lambda(q,:); uq=Uq(:,q);
    strong=ut+uq.*ux-geom.fq(:,q); weight=geom.area*geom.w(q);
    for a=1:3
        Re(:,a)=Re(:,a)+weight.*(strong*phi(a)+nu*ux.*geom.dphidx(:,a));
    end
end
R=accumarray(elem(:),Re(:),[geom.N,1],@sum,0);
end


function [lambda,w] = tri_quad_3()
lambda=[1/6,1/6,2/3;1/6,2/3,1/6;2/3,1/6,1/6];
w=[1/3;1/3;1/3];
end


function ref = solve_burgers_fourier_reference(u0fun,cfg)
% Periodic Fourier pseudo-spectral ETDRK4, conservative nonlinearity.
N=cfg.NxRefSpectral;
if mod(N,2)~=0, error('NxRefSpectral must be even.'); end
Lx=cfg.xmax-cfg.xmin; x=cfg.xmin+Lx*(0:N-1)'/N;
u0=u0fun(x); forcingHat=fft(cfg.forcingFun(x));
modes=[0:N/2,-N/2+1:-1].'; kLap=(2*pi/Lx)*modes; kDer=kLap;
kDer(N/2+1)=0;
cutoff=floor(cfg.dealiasFraction*(N/2)); dealiasMask=abs(modes)<=cutoff;
Lop=-cfg.nu*(kLap.^2); dt=cfg.refDt;
totalStepsReal=(cfg.tmax-cfg.tmin)/dt; totalSteps=round(totalStepsReal);
if abs(totalSteps-totalStepsReal)>1e-10
    error('refDt must divide the time interval exactly.');
end
nSaveIntervals=cfg.NtRefSpectral-1;
saveEveryReal=totalSteps/nSaveIntervals; saveEvery=round(saveEveryReal);
if abs(saveEvery-saveEveryReal)>1e-10
    error('ETDRK4 step count must be divisible by NtRefSpectral-1.');
end
E=exp(dt*Lop); E2=exp(dt*Lop/2); M=32;
r=exp(1i*pi*((1:M)-0.5)/M);
LR=dt*Lop(:,ones(M,1))+r(ones(N,1),:);
Q=dt*real(mean((exp(LR/2)-1)./LR,2));
f1=dt*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3,2));
f2=dt*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3,2));
f3=dt*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3,2));
v=fft(u0); U=zeros(cfg.NtRefSpectral,N+1);
tSave=linspace(cfg.tmin,cfg.tmax,cfg.NtRefSpectral).';
U(1,:)=[u0(:).',u0(1)]; saveId=2;
for step=1:totalSteps
    Nv=burgers_fourier_nonlinear(v,kDer,dealiasMask)+forcingHat;
    a=E2.*v+Q.*Nv;
    Na=burgers_fourier_nonlinear(a,kDer,dealiasMask)+forcingHat;
    b=E2.*v+Q.*Na;
    Nb=burgers_fourier_nonlinear(b,kDer,dealiasMask)+forcingHat;
    c=E2.*a+Q.*(2*Nb-Nv);
    Nc=burgers_fourier_nonlinear(c,kDer,dealiasMask)+forcingHat;
    v=E.*v+f1.*Nv+2*f2.*(Na+Nb)+f3.*Nc;
    if mod(step,saveEvery)==0
        uNow=real(ifft(v)); U(saveId,:)=[uNow(:).',uNow(1)];
        saveId=saveId+1;
    end
end
if saveId~=cfg.NtRefSpectral+1, error('Spectral snapshot count mismatch.'); end
ref=struct('x',[x;cfg.xmax],'t',tSave,'U',U,'N',N,'dt',dt, ...
    'method','Periodic Fourier pseudo-spectral ETDRK4 with 2/3 dealiasing');
end


function Nv = burgers_fourier_nonlinear(v,kDer,dealiasMask)
u=real(ifft(v)); Nv=-0.5i*kDer.*fft(u.^2); Nv(~dealiasMask)=0;
end


function Fref = make_ref_interpolant(ref)
try
    Fref=griddedInterpolant({ref.t(:),ref.x(:)},ref.U,'spline','nearest');
catch
    Fref=griddedInterpolant({ref.t(:),ref.x(:)},ref.U,'linear','nearest');
end
end


function err = compute_fem_vs_reference_error(node,elem,u,ref,cfg,Fref)
if nargin<6 || isempty(Fref), Fref=make_ref_interpolant(ref); end
[lambda,w]=tri_quad_3(); NT=size(elem,1); chunkSize=cfg.errorChunkElements;
e2=0; r2=0;
for first=1:chunkSize:NT
    last=min(first+chunkSize-1,NT); E=elem(first:last,:);
    p1=node(E(:,1),:); p2=node(E(:,2),:); p3=node(E(:,3),:);
    area=0.5*abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2))- ...
        (p2(:,2)-p1(:,2)).*(p3(:,1)-p1(:,1)));
    X=[p1(:,1),p2(:,1),p3(:,1)]; T=[p1(:,2),p2(:,2),p3(:,2)];
    Ue=u(E); Xq=X*lambda.'; Tq=T*lambda.'; Uhq=Ue*lambda.';
    Urq=Fref(Tq(:),Xq(:)); Urq=reshape(Urq,size(Xq));
    Urq(~isfinite(Urq))=0;
    e2=e2+sum(area.*sum((Uhq-Urq).^2.*w.',2));
    r2=r2+sum(area.*sum(Urq.^2.*w.',2));
end
uRefNode=Fref(node(:,2),node(:,1)); uRefNode(~isfinite(uRefNode))=0;
nodalErr=u(:)-uRefNode(:);
xLine=linspace(cfg.xmin,cfg.xmax,max(1001,cfg.compareGridX)).';
tLine=(cfg.tmax-1e-12)*ones(size(xLine));
uFemLine=eval_p1_at_points(node,elem,u,[xLine,tLine]);
uRefLine=Fref(tLine,xLine); bad=~isfinite(uFemLine)|~isfinite(uRefLine);
uFemLine(bad)=0; uRefLine(bad)=0;
eFinal2=trapz(xLine,(uFemLine-uRefLine).^2);
rFinal2=trapz(xLine,uRefLine.^2);
err=struct('spaceTimeL2Abs',sqrt(max(e2,0)), ...
    'spaceTimeL2Rel',sqrt(max(e2,0)/max(r2,1e-30)), ...
    'referenceSpaceTimeL2',sqrt(max(r2,0)), ...
    'nodalLinf',max(abs(nodalErr)), ...
    'nodalRMS',sqrt(mean(nodalErr.^2)), ...
    'finalTimeL2Abs',sqrt(max(eFinal2,0)), ...
    'finalTimeL2Rel',sqrt(max(eFinal2,0)/max(rFinal2,1e-30)));
end


function vals = eval_p1_at_points(node,elem,u,pts)
vals=NaN(size(pts,1),1);
try
    TR=triangulation(elem,node); ti=pointLocation(TR,pts); inside=~isnan(ti);
    if any(inside)
        bc=cartesianToBarycentric(TR,ti(inside),pts(inside,:));
        tri=elem(ti(inside),:); vals(inside)=sum(bc.*u(tri),2);
    end
    if any(~inside)
        F=scatteredInterpolant(node(:,1),node(:,2),u(:),'linear','nearest');
        vals(~inside)=F(pts(~inside,1),pts(~inside,2));
    end
catch
    F=scatteredInterpolant(node(:,1),node(:,2),u(:),'linear','nearest');
    vals=F(pts(:,1),pts(:,2));
end
end


function uNew = interpolate_to_new_nodes(oldNode,oldElem,oldU,newNode)
nOld=size(oldNode,1); nNew=size(newNode,1);
prefixMatch=nNew>=nOld && isequal(newNode(1:nOld,:),oldNode);
uNew=zeros(nNew,1);
if prefixMatch
    uNew(1:nOld)=oldU(:); queryIds=(nOld+1:nNew).';
    if isempty(queryIds), return; end
else
    queryIds=(1:nNew).';
end
try
    TR=triangulation(oldElem,oldNode); queryPoints=newNode(queryIds,:);
    ti=pointLocation(TR,queryPoints); inside=~isnan(ti);
    if any(inside)
        bc=cartesianToBarycentric(TR,ti(inside),queryPoints(inside,:));
        tri=oldElem(ti(inside),:);
        uNew(queryIds(inside))=sum(bc.*oldU(tri),2);
    end
    if any(~inside)
        F=scatteredInterpolant(oldNode(:,1),oldNode(:,2),oldU,'linear','nearest');
        uNew(queryIds(~inside))=F(queryPoints(~inside,1),queryPoints(~inside,2));
    end
catch
    F=scatteredInterpolant(oldNode(:,1),oldNode(:,2),oldU,'linear','nearest');
    uNew(queryIds)=F(newNode(queryIds,1),newNode(queryIds,2));
end
if any(~isfinite(uNew)), error('Coarse-to-Gmsh interpolation produced NaN/Inf.'); end
end
