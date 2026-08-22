function generate_burgers_mesh_size_data(varargin)
%GENERATE_BURGERS_MESH_SIZE_DATA Convert MeshOp labels and build test references.
%
% Copy burgers_5100.mat into this directory, then run this function. It writes
% burgers_mesh_size_3100.h5 with
%   rows 1:3000    <- source samples 1:3000 (training),
%   rows 3001:3100 <- source samples 5001:5100 (testing).
%
% target_score is the exact AFEM leaf generation sampled on the 101x101
% query grid. For binary NVB, A_K=A0/2^g, hence the equivalent scalar size is
%
%   h = sqrt(2*A0)*2^(-g/2),  A0=((2/32)*(1/32))/2=1/1024.
%
% Only the 100 test samples receive Fourier ETDRK4 reference solutions.
% Completed reference batches are resumable. Examples:
%
%   generate_burgers_mesh_size_data
%   generate_burgers_mesh_size_data('UseParallel',false)
%   generate_burgers_mesh_size_data('BatchSize',10,'NumWorkers',4)

baseDir = fileparts(mfilename('fullpath'));
parser = inputParser;
addParameter(parser,'SourceFile',fullfile(baseDir,'burgers_5100.mat'), ...
    @(x)ischar(x)||isstring(x));
addParameter(parser,'OutputFile',fullfile(baseDir,'burgers_mesh_size_3100.h5'), ...
    @(x)ischar(x)||isstring(x));
addParameter(parser,'BatchSize',10,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(parser,'UseParallel',true,@(x)islogical(x)||isnumeric(x));
addParameter(parser,'NumWorkers',0,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(parser,'Overwrite',false,@(x)islogical(x)||isnumeric(x));
addParameter(parser,'SpectralNx',512,@(x)isnumeric(x)&&isscalar(x)&&x>=4);
addParameter(parser,'Dt',2.5e-4,@(x)isnumeric(x)&&isscalar(x)&&x>0);
parse(parser,varargin{:});
cfg = parser.Results;
cfg.SourceFile = char(cfg.SourceFile);
cfg.OutputFile = char(cfg.OutputFile);
cfg.BatchSize = round(cfg.BatchSize);
cfg.NumWorkers = round(cfg.NumWorkers);
cfg.UseParallel = logical(cfg.UseParallel);
cfg.Overwrite = logical(cfg.Overwrite);
cfg.SpectralNx = round(cfg.SpectralNx);
cfg.Dt = double(cfg.Dt);
cfg.SourceIndices = [1:3000,5001:5100];
cfg.TrainSamples = 3000;
cfg.TestSamples = 100;
cfg.NumSamples = 3100;
cfg.TestRows = 3001:3100;
cfg.Nu = 5e-3;
cfg.Xmin = -1;
cfg.Xmax = 1;
cfg.Tmin = 0;
cfg.Tmax = 1;
cfg.DealiasFraction = 2/3;
cfg.InitialGrid = 32;
cfg.MaxGeneration = 12;

if ~isfile(cfg.SourceFile)
    error(['Missing %s. Copy the original burgers_5100.mat into this ', ...
        'directory first.'],cfg.SourceFile);
end
if mod(cfg.SpectralNx,2)~=0
    error('SpectralNx must be even.');
end
if cfg.Overwrite && isfile(cfg.OutputFile)
    delete(cfg.OutputFile);
end

fprintf('Source : %s\n',cfg.SourceFile);
fprintf('Output : %s\n',cfg.OutputFile);
fprintf('Split  : source 1:3000 train, source 5001:5100 test\n');
fprintf('Target : equivalent continuous log mesh size from AFEM generation\n');
fprintf('Solver : test-only Fourier ETDRK4, N=%d, dt=%.8g, nu=%.8g\n', ...
    cfg.SpectralNx,cfg.Dt,cfg.Nu);

source = load_source(cfg);
if ~isfile(cfg.OutputFile)
    initialize_output(source,cfg);
    write_converted_data(source,cfg);
elseif double(h5readatt(cfg.OutputFile,'/','conversion_complete'))~=1
    error(['Existing output has an incomplete conversion. Delete it or ', ...
        'rerun with Overwrite=true.']);
else
    validate_existing_output(source,cfg);
end

completed = double(h5readatt(cfg.OutputFile,'/','completed_reference_samples'));
if completed>=cfg.TestSamples
    fprintf('All %d spectral test references already exist.\n',cfg.TestSamples);
    return;
end

solver = build_solver(source.query_x,source.query_t,cfg);
referenceSource = rmfield(source,{'target_score','forcing_query'});
useParallel = cfg.UseParallel && license('test','Distrib_Computing_Toolbox');
if useParallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            if cfg.NumWorkers>0
                pool = parpool('local',cfg.NumWorkers);
            else
                pool = parpool('local');
            end
        end
        fprintf('Parallel pool: %d workers\n',pool.NumWorkers);
    catch poolError
        warning('NOEM:MeshSizeBaseline:ParallelPoolFailed', ...
            'Parallel pool failed (%s); using serial mode.',poolError.message);
        useParallel = false;
    end
end
if ~useParallel
    fprintf('Serial reference generation\n');
end

wallStart = tic;
for first = completed+1:cfg.BatchSize:cfg.TestSamples
    last = min(first+cfg.BatchSize-1,cfg.TestSamples);
    count = last-first+1;
    batch = zeros(numel(source.query_x),numel(source.query_t),count,'single');
    if useParallel
        parfor j=1:count
            row = 3000+first+j-1;
            batch(:,:,j) = solve_reference(row,referenceSource,solver);
        end
    else
        for j=1:count
            row = 3000+first+j-1;
            batch(:,:,j) = solve_reference(row,referenceSource,solver);
        end
    end
    require_finite(batch,'spectral test-reference batch');
    % MATLAB [x,t,sample] is h5py [sample,t,x].
    h5write(cfg.OutputFile,'/reference_solution_test',batch, ...
        [1,1,first],[size(batch,1),size(batch,2),count]);
    h5writeatt(cfg.OutputFile,'/','completed_reference_samples',uint32(last));
    fprintf('[reference] %3d/%d | batch=%d | elapsed=%.1f s\n', ...
        last,cfg.TestSamples,count,toc(wallStart));
end
fprintf('Completed: %s\n',cfg.OutputFile);
end


function source = load_source(cfg)
variables = who('-file',cfg.SourceFile);
required = {'target_score','ic_coefficients','grf_modes', ...
    'grf_spectral_std','forcing_xi_cos','forcing_xi_sin', ...
    'forcing_modes','forcing_spectral_std','forcing_input', ...
    'query_x','query_t'};
missing = setdiff(required,variables);
if ~isempty(missing)
    error('Source MAT is missing: %s',strjoin(missing,', '));
end
M = matfile(cfg.SourceFile);
if size(M,'target_score',3)<5100
    error('target_score must contain at least 5100 samples.');
end
source.query_x = double(M.query_x(:,1));
source.query_t = double(M.query_t(:,1));
if numel(source.query_x)~=101 || numel(source.query_t)~=101
    error('Expected the original 101x101 query grid.');
end
source.grf_modes = double(M.grf_modes(:,1));
source.grf_std = double(M.grf_spectral_std(:,1));
source.forcing_modes = double(M.forcing_modes(:,1));
source.forcing_std = double(M.forcing_spectral_std(:,1));

allIc = M.ic_coefficients;
allForcingCos = M.forcing_xi_cos;
allForcingSin = M.forcing_xi_sin;
allForcingInput = M.forcing_input;
source.ic = double(allIc(:,cfg.SourceIndices));
source.forcing_cos = double(allForcingCos(:,cfg.SourceIndices));
source.forcing_sin = double(allForcingSin(:,cfg.SourceIndices));
source.forcing_query = single(allForcingInput(:,cfg.SourceIndices));
source.target_score = cat(3, ...
    single(M.target_score(:,:,1:3000)), ...
    single(M.target_score(:,:,5001:5100)));
source.source_indices = uint32(cfg.SourceIndices(:));
if ismember('source_attempt_id',variables)
    allAttempt = M.source_attempt_id;
    source.source_attempt_ids = uint32(allAttempt(cfg.SourceIndices));
    source.source_attempt_ids = source.source_attempt_ids(:);
else
    source.source_attempt_ids = source.source_indices;
end
require_finite(source.ic,'initial-condition coefficients');
require_finite(source.forcing_cos,'forcing cosine coefficients');
require_finite(source.forcing_sin,'forcing sine coefficients');
require_finite(source.target_score,'target score');
if any(source.target_score(:)<0) || any(source.target_score(:)>cfg.MaxGeneration)
    error('target_score lies outside [0,%d].',cfg.MaxGeneration);
end
end


function initialize_output(source,cfg)
nx = numel(source.query_x);
nt = numel(source.query_t);
n = cfg.NumSamples;
chunk = min(16,n);
h5create(cfg.OutputFile,'/query_x',[nx,1],'Datatype','double');
h5create(cfg.OutputFile,'/query_t',[nt,1],'Datatype','double');
h5create(cfg.OutputFile,'/source_dataset_index',[n,1],'Datatype','uint32');
h5create(cfg.OutputFile,'/source_attempt_id',[n,1],'Datatype','uint32');
h5create(cfg.OutputFile,'/input',[nx,nt,4,n],'Datatype','single', ...
    'ChunkSize',[nx,nt,4,1],'Deflate',1,'Shuffle',true);
h5create(cfg.OutputFile,'/target_generation',[nx,nt,n],'Datatype','single', ...
    'ChunkSize',[nx,nt,1],'Deflate',1,'Shuffle',true);
h5create(cfg.OutputFile,'/target_log_h',[nx,nt,n],'Datatype','single', ...
    'ChunkSize',[nx,nt,1],'Deflate',1,'Shuffle',true);
h5create(cfg.OutputFile,'/target_normalized_log_h',[nx,nt,n], ...
    'Datatype','single','ChunkSize',[nx,nt,1],'Deflate',1,'Shuffle',true);
h5create(cfg.OutputFile,'/ic_coefficients',[size(source.ic,1),n], ...
    'Datatype','double','ChunkSize',[size(source.ic,1),chunk]);
h5create(cfg.OutputFile,'/forcing_xi_cos',[size(source.forcing_cos,1),n], ...
    'Datatype','double','ChunkSize',[size(source.forcing_cos,1),chunk]);
h5create(cfg.OutputFile,'/forcing_xi_sin',[size(source.forcing_sin,1),n], ...
    'Datatype','double','ChunkSize',[size(source.forcing_sin,1),chunk]);
h5create(cfg.OutputFile,'/grf_modes',[numel(source.grf_modes),1],'Datatype','double');
h5create(cfg.OutputFile,'/grf_spectral_std',[numel(source.grf_std),1],'Datatype','double');
h5create(cfg.OutputFile,'/forcing_modes',[numel(source.forcing_modes),1],'Datatype','double');
h5create(cfg.OutputFile,'/forcing_spectral_std',[numel(source.forcing_std),1],'Datatype','double');
h5create(cfg.OutputFile,'/input_channel_mean',[4,1],'Datatype','double');
h5create(cfg.OutputFile,'/input_channel_std',[4,1],'Datatype','double');
h5create(cfg.OutputFile,'/reference_solution_test',[nx,nt,cfg.TestSamples], ...
    'Datatype','single','ChunkSize',[nx,nt,1],'Deflate',1,'Shuffle',true);

h5write(cfg.OutputFile,'/query_x',source.query_x);
h5write(cfg.OutputFile,'/query_t',source.query_t);
h5write(cfg.OutputFile,'/source_dataset_index',source.source_indices);
h5write(cfg.OutputFile,'/source_attempt_id',source.source_attempt_ids);
h5write(cfg.OutputFile,'/ic_coefficients',source.ic);
h5write(cfg.OutputFile,'/forcing_xi_cos',source.forcing_cos);
h5write(cfg.OutputFile,'/forcing_xi_sin',source.forcing_sin);
h5write(cfg.OutputFile,'/grf_modes',source.grf_modes);
h5write(cfg.OutputFile,'/grf_spectral_std',source.grf_std);
h5write(cfg.OutputFile,'/forcing_modes',source.forcing_modes);
h5write(cfg.OutputFile,'/forcing_spectral_std',source.forcing_std);
h5writeatt(cfg.OutputFile,'/','conversion_complete',uint8(0));
h5writeatt(cfg.OutputFile,'/','completed_reference_samples',uint32(0));
h5writeatt(cfg.OutputFile,'/','total_samples',uint32(n));
h5writeatt(cfg.OutputFile,'/','train_samples',uint32(cfg.TrainSamples));
h5writeatt(cfg.OutputFile,'/','test_samples',uint32(cfg.TestSamples));
h5writeatt(cfg.OutputFile,'/','initial_grid',uint32(cfg.InitialGrid));
h5writeatt(cfg.OutputFile,'/','maximum_generation',uint32(cfg.MaxGeneration));
h5writeatt(cfg.OutputFile,'/','spectral_nx',uint32(cfg.SpectralNx));
h5writeatt(cfg.OutputFile,'/','etdrk4_dt',cfg.Dt);
h5writeatt(cfg.OutputFile,'/','viscosity',cfg.Nu);
h5writeatt(cfg.OutputFile,'/','target_definition', ...
    'h=sqrt(2*A0)*2^(-generation/2), A0=1/1024');
h5writeatt(cfg.OutputFile,'/','axis_convention_h5py', ...
    'input[sample,channel,t,x], target[sample,t,x]');
end


function write_converted_data(source,cfg)
nx = numel(source.query_x);
nt = numel(source.query_t);
n = cfg.NumSamples;
A0 = ((cfg.Xmax-cfg.Xmin)/cfg.InitialGrid)* ...
    ((cfg.Tmax-cfg.Tmin)/cfg.InitialGrid)/2;
h0 = sqrt(2*A0);
logHMax = log(h0);
logHMin = logHMax-0.5*cfg.MaxGeneration*log(2);

[X,T] = ndgrid(source.query_x,source.query_t);
channelSum = zeros(4,1);
channelSquareSum = zeros(4,1);
channelCount = zeros(4,1);
conversionBatch = 16;
for first=1:conversionBatch:n
    last = min(first+conversionBatch-1,n);
    count = last-first+1;
    inputBatch = zeros(nx,nt,4,count,'single');
    for localId=1:count
        sample = first+localId-1;
        inputBatch(:,:,1,localId) = repmat(single(evaluate_ic( ...
            source.query_x,source.ic(:,sample),source)),1,nt);
        inputBatch(:,:,2,localId) = repmat( ...
            source.forcing_query(:,sample),1,nt);
        inputBatch(:,:,3,localId) = single(X);
        inputBatch(:,:,4,localId) = single(T);
    end
    generationBatch = source.target_score(:,:,first:last);
    logHBatch = single(logHMax)-0.5*single(log(2))*generationBatch;
    normalizedBatch = (logHBatch-single(logHMin))/ ...
        single(logHMax-logHMin);
    normalizedBatch = min(max(normalizedBatch,0),1);
    h5write(cfg.OutputFile,'/input',inputBatch,[1,1,1,first], ...
        [nx,nt,4,count]);
    h5write(cfg.OutputFile,'/target_generation',generationBatch, ...
        [1,1,first],[nx,nt,count]);
    h5write(cfg.OutputFile,'/target_log_h',logHBatch, ...
        [1,1,first],[nx,nt,count]);
    h5write(cfg.OutputFile,'/target_normalized_log_h',normalizedBatch, ...
        [1,1,first],[nx,nt,count]);
    trainLast = min(last,cfg.TrainSamples);
    if first<=trainLast
        localTrainCount = trainLast-first+1;
        for channel=1:4
            values = double(inputBatch(:,:,channel,1:localTrainCount));
            channelSum(channel) = channelSum(channel)+sum(values(:));
            channelSquareSum(channel) = channelSquareSum(channel)+ ...
                sum(values(:).^2);
            channelCount(channel) = channelCount(channel)+numel(values);
        end
    end
    if mod(last,320)==0 || last==n
        fprintf('[conversion] %4d/%d\n',last,n);
    end
end
channelMean = channelSum./channelCount;
channelVariance = max(channelSquareSum./channelCount-channelMean.^2,0);
channelStd = sqrt(channelVariance);
if any(channelStd<=0) || any(~isfinite(channelStd))
    error('Invalid training input standard deviation.');
end
h5write(cfg.OutputFile,'/input_channel_mean',channelMean);
h5write(cfg.OutputFile,'/input_channel_std',channelStd);
h5writeatt(cfg.OutputFile,'/','initial_triangle_area',A0);
h5writeatt(cfg.OutputFile,'/','h0',h0);
h5writeatt(cfg.OutputFile,'/','log_h_min',logHMin);
h5writeatt(cfg.OutputFile,'/','log_h_max',logHMax);
h5writeatt(cfg.OutputFile,'/','conversion_complete',uint8(1));
fprintf('Converted %d target-generation fields to log mesh size.\n',n);
end


function validate_existing_output(source,cfg)
stored = double(h5read(cfg.OutputFile,'/source_dataset_index'));
if ~isequal(stored(:),double(source.source_indices(:)))
    error('Existing output has a different source-index mapping.');
end
if double(h5readatt(cfg.OutputFile,'/','spectral_nx'))~=cfg.SpectralNx || ...
        abs(double(h5readatt(cfg.OutputFile,'/','etdrk4_dt'))-cfg.Dt)>1e-15
    error('Existing reference settings differ; use Overwrite=true.');
end
end


function value = evaluate_ic(x,coeff,source)
nGrf = numel(source.grf_modes);
value = -coeff(1)*sin(pi*(x(:)+coeff(2))) + evaluate_series( ...
    x,coeff(3:2+nGrf),coeff(3+nGrf:2+2*nGrf), ...
    source.grf_modes,source.grf_std);
end


function solver = build_solver(query_x,query_t,cfg)
N = cfg.SpectralNx;
Lx = cfg.Xmax-cfg.Xmin;
solver.x = cfg.Xmin+Lx*(0:N-1)'/N;
solver.query_x = query_x(:);
solver.query_t = query_t(:);
stepsReal = (cfg.Tmax-cfg.Tmin)/cfg.Dt;
solver.totalSteps = round(stepsReal);
if abs(solver.totalSteps-stepsReal)>1e-10
    error('Dt must divide [tmin,tmax] exactly.');
end
saveSteps = round((solver.query_t-cfg.Tmin)/cfg.Dt);
if max(abs(solver.query_t-(cfg.Tmin+saveSteps*cfg.Dt)))>5e-7
    error('Every query_t value must lie on an ETDRK4 step.');
end
solver.saveId = zeros(solver.totalSteps+1,1,'uint16');
solver.saveId(saveSteps+1) = uint16(1:numel(saveSteps));
mode = [0:N/2,-N/2+1:-1].';
k = (2*pi/Lx)*mode;
solver.kDer = k;
solver.kDer(N/2+1)=0;
solver.dealiasMask = abs(mode)<=floor(cfg.DealiasFraction*(N/2));
linear = -cfg.Nu*k.^2;
solver.E = exp(cfg.Dt*linear);
solver.E2 = exp(cfg.Dt*linear/2);
r = exp(1i*pi*((1:32)-0.5)/32);
LR = cfg.Dt*linear(:,ones(32,1))+r(ones(N,1),:);
solver.Q = cfg.Dt*real(mean((exp(LR/2)-1)./LR,2));
solver.f1 = cfg.Dt*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3,2));
solver.f2 = cfg.Dt*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3,2));
solver.f3 = cfg.Dt*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3,2));
normalized = (solver.query_x-cfg.Xmin)/Lx;
solver.interpolation = exp(2i*pi*mode*normalized.')/N;
end


function solution = solve_reference(row,source,solver)
coeff = source.ic(:,row);
nGrf = numel(source.grf_modes);
u0 = -coeff(1)*sin(pi*(solver.x+coeff(2))) + evaluate_series( ...
    solver.x,coeff(3:2+nGrf),coeff(3+nGrf:2+2*nGrf), ...
    source.grf_modes,source.grf_std);
forcing = evaluate_series(solver.x,source.forcing_cos(:,row), ...
    source.forcing_sin(:,row),source.forcing_modes,source.forcing_std);
v = fft(u0);
forcingHat = fft(forcing);
solution = zeros(numel(solver.query_x),numel(solver.query_t));
solution(:,1) = interpolate_fourier(v,solver.interpolation);
for step=1:solver.totalSteps
    Nv = burgers_nonlinear(v,solver)+forcingHat;
    a = solver.E2.*v+solver.Q.*Nv;
    Na = burgers_nonlinear(a,solver)+forcingHat;
    b = solver.E2.*v+solver.Q.*Na;
    Nb = burgers_nonlinear(b,solver)+forcingHat;
    c = solver.E2.*a+solver.Q.*(2*Nb-Nv);
    Nc = burgers_nonlinear(c,solver)+forcingHat;
    v = solver.E.*v+solver.f1.*Nv+2*solver.f2.*(Na+Nb)+solver.f3.*Nc;
    saveId = double(solver.saveId(step+1));
    if saveId>0
        solution(:,saveId) = interpolate_fourier(v,solver.interpolation);
    end
end
require_finite(solution,'one spectral solution');
solution = single(solution);
end


function value = evaluate_series(x,xiCos,xiSin,modes,spectralStd)
x = double(x(:));
value = cos(pi*x*double(modes(:)).')* ...
    (double(spectralStd(:)).*double(xiCos(:))) + ...
    sin(pi*x*double(modes(:)).')* ...
    (double(spectralStd(:)).*double(xiSin(:)));
end


function value = burgers_nonlinear(v,solver)
u = real(ifft(v));
value = -0.5i*solver.kDer.*fft(u.^2);
value(~solver.dealiasMask)=0;
end


function value = interpolate_fourier(v,interpolation)
value = real((v.'*interpolation).');
end


function require_finite(value,name)
if any(~isfinite(double(value(:))))
    error('%s contains NaN or Inf.',name);
end
end
