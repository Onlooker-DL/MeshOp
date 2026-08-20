function generate_burgers_spectral_3100(varargin)
%GENERATE_BURGERS_SPECTRAL_3100 Build direct-solution labels for samples 1:3100.
%
% Put burgers_5100.mat in the same folder as this file, then run:
%
%   cd D:\NOEM\neural_operator_solution\data\burgers
%   generate_burgers_spectral_3100
%
% Optional examples:
%
%   generate_burgers_spectral_3100('BatchSize',16,'NumWorkers',8)
%   generate_burgers_spectral_3100('UseParallel',false,'BatchSize',4)
%
% The method matches the AFEM data generator reference:
%   - periodic Fourier pseudo-spectral discretization in x;
%   - conservative nonlinearity -0.5*d_x(u^2);
%   - 2/3 de-aliasing;
%   - ETDRK4 time integration, dt=2.5e-4;
%   - N=512 spectral points and viscosity nu=5e-3.
%
% Output is an ordinary HDF5 file named burgers_spectral_3100.h5. Its
% on-disk dimension order is chosen so h5py reads solution as
% [sample,x,t], exactly as expected by src/training/train_burgers.py.
%
% The file is updated only after a complete finite batch. Rerunning this
% function resumes from the root attribute completed_samples.

baseDir = fileparts(mfilename('fullpath'));

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser,'SourceFile',fullfile(baseDir,'burgers_5100.mat'), ...
    @(x)ischar(x) || isstring(x));
addParameter(parser,'OutputFile',fullfile(baseDir,'burgers_spectral_3100.h5'), ...
    @(x)ischar(x) || isstring(x));
addParameter(parser,'BatchSize',16,@(x)isnumeric(x) && isscalar(x) && x>=1);
addParameter(parser,'UseParallel',true,@(x)islogical(x) || isnumeric(x));
addParameter(parser,'NumWorkers',0,@(x)isnumeric(x) && isscalar(x) && x>=0);
addParameter(parser,'MaxBatches',inf,@(x)isnumeric(x) && isscalar(x) && x>=1);
addParameter(parser,'Overwrite',false,@(x)islogical(x) || isnumeric(x));
addParameter(parser,'SpectralNx',512,@(x)isnumeric(x) && isscalar(x) && x>=4);
addParameter(parser,'SensorNx',256,@(x)isnumeric(x) && isscalar(x) && x>=4);
addParameter(parser,'Dt',2.5e-4,@(x)isnumeric(x) && isscalar(x) && x>0);
parse(parser,varargin{:});
cfg = parser.Results;

cfg.SourceFile = char(cfg.SourceFile);
cfg.OutputFile = char(cfg.OutputFile);
cfg.BatchSize = round(double(cfg.BatchSize));
cfg.NumWorkers = round(double(cfg.NumWorkers));
cfg.MaxBatches = double(cfg.MaxBatches);
cfg.SpectralNx = round(double(cfg.SpectralNx));
cfg.SensorNx = round(double(cfg.SensorNx));
cfg.Dt = double(cfg.Dt);
cfg.UseParallel = logical(cfg.UseParallel);
cfg.Overwrite = logical(cfg.Overwrite);
cfg.NumSamples = 3100;
cfg.TrainSamples = 3000;
cfg.TestSamples = 100;
cfg.Nu = 5.0e-3;
cfg.Xmin = -1.0;
cfg.Xmax = 1.0;
cfg.Tmin = 0.0;
cfg.Tmax = 1.0;
cfg.DealiasFraction = 2/3;

if mod(cfg.SpectralNx,2)~=0
    error('SpectralNx must be even.');
end
if ~isfile(cfg.SourceFile)
    error(['Source MAT file not found:\n  %s\nPlace burgers_5100.mat ', ...
        'in the same folder as this generator.'],cfg.SourceFile);
end
if cfg.Overwrite && isfile(cfg.OutputFile)
    delete(cfg.OutputFile);
end

fprintf('Source : %s\n',cfg.SourceFile);
fprintf('Output : %s\n',cfg.OutputFile);
fprintf('Split  : samples 1:3000 train, 3001:3100 test\n');
fprintf('Solver : Fourier ETDRK4, N=%d, dt=%.8g, nu=%.8g\n', ...
    cfg.SpectralNx,cfg.Dt,cfg.Nu);

source = load_source_inputs(cfg.SourceFile,cfg);
solver = build_solver(source.query_x,source.query_t,cfg);
completed = initialize_or_validate_output(cfg.OutputFile,source,cfg);

if completed>=cfg.NumSamples
    fprintf('Output already contains all %d samples. Nothing to do.\n', ...
        cfg.NumSamples);
    return;
end

useParallel = cfg.UseParallel && license('test','Distrib_Computing_Toolbox');
if cfg.UseParallel && ~useParallel
    warning('Parallel Computing Toolbox unavailable; using a serial loop.');
end
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
        warning('Parallel pool failed (%s). Falling back to serial generation.', ...
            poolError.message);
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
    count = last-first+1;
    solutionBatch = zeros(numel(source.query_x),numel(source.query_t),count,'single');
    u0QueryBatch = zeros(numel(source.query_x),count,'single');
    forcingQueryBatch = zeros(numel(source.query_x),count,'single');
    u0SensorBatch = zeros(cfg.SensorNx,count,'single');
    forcingSensorBatch = zeros(cfg.SensorNx,count,'single');
    icErrors = zeros(count,1);

    if useParallel
        parfor localId = 1:count
            sampleId = first+localId-1;
            [solutionBatch(:,:,localId),u0QueryBatch(:,localId), ...
                forcingQueryBatch(:,localId),u0SensorBatch(:,localId), ...
                forcingSensorBatch(:,localId),icErrors(localId)] = ...
                solve_one_sample(sampleId,source,solver);
        end
    else
        for localId = 1:count
            sampleId = first+localId-1;
            [solutionBatch(:,:,localId),u0QueryBatch(:,localId), ...
                forcingQueryBatch(:,localId),u0SensorBatch(:,localId), ...
                forcingSensorBatch(:,localId),icErrors(localId)] = ...
                solve_one_sample(sampleId,source,solver);
        end
    end

    require_finite(solutionBatch,'solution batch');
    require_finite(u0QueryBatch,'initial-condition query batch');
    require_finite(forcingQueryBatch,'forcing query batch');
    require_finite(u0SensorBatch,'initial-condition sensor batch');
    require_finite(forcingSensorBatch,'forcing sensor batch');
    if max(icErrors)>1e-8
        error('Maximum initial spectral interpolation error is %.3e.', ...
            max(icErrors));
    end

    % MATLAB HDF5 dimensions are the reverse of h5py dimensions. The
    % solution dataset is [t,x,sample] here and [sample,x,t] in h5py.
    h5write(cfg.OutputFile,'/initial_condition',u0QueryBatch, ...
        [1,first],[size(u0QueryBatch,1),count]);
    h5write(cfg.OutputFile,'/forcing',forcingQueryBatch, ...
        [1,first],[size(forcingQueryBatch,1),count]);
    h5write(cfg.OutputFile,'/initial_condition_sensors',u0SensorBatch, ...
        [1,first],[size(u0SensorBatch,1),count]);
    h5write(cfg.OutputFile,'/forcing_sensors',forcingSensorBatch, ...
        [1,first],[size(forcingSensorBatch,1),count]);
    h5write(cfg.OutputFile,'/solution',permute(solutionBatch,[2,1,3]), ...
        [1,1,first],[size(solutionBatch,2),size(solutionBatch,1),count]);
    h5writeatt(cfg.OutputFile,'/','completed_samples',uint32(last));

    fprintf(['[spectral] %4d/%d | batch=%d | max IC error=%.2e | ', ...
        'elapsed=%.1f s\n'],last,cfg.NumSamples,count,max(icErrors),toc(wallStart));
    batchCounter = batchCounter+1;
    if batchCounter>=cfg.MaxBatches && last<cfg.NumSamples
        fprintf('Stopped intentionally after %d new batch(es); resume is safe.\n', ...
            batchCounter);
        return;
    end
end

fprintf('Completed spectral data: %s\n',cfg.OutputFile);
end


function source = load_source_inputs(fileName,cfg)
variables = who('-file',fileName);
required = {'ic_coefficients','grf_modes','grf_spectral_std', ...
    'forcing_xi_cos','forcing_xi_sin','forcing_modes', ...
    'forcing_spectral_std','query_x','query_t'};
missing = setdiff(required,variables);
if ~isempty(missing)
    error('Source MAT is missing: %s',strjoin(missing,', '));
end

M = matfile(fileName);
icSize = size(M,'ic_coefficients');
if icSize(2)<cfg.NumSamples
    error('ic_coefficients has only %d samples.',icSize(2));
end

source.grf_modes = double(M.grf_modes(:,1));
source.grf_std = double(M.grf_spectral_std(:,1));
source.forcing_modes = double(M.forcing_modes(:,1));
source.forcing_std = double(M.forcing_spectral_std(:,1));
source.query_x = double(M.query_x(:,1));
source.query_t = double(M.query_t(:,1));
source.ic = double(M.ic_coefficients(:,1:cfg.NumSamples));
source.forcing_cos = double(M.forcing_xi_cos(:,1:cfg.NumSamples));
source.forcing_sin = double(M.forcing_xi_sin(:,1:cfg.NumSamples));

expectedIc = 2+2*numel(source.grf_modes);
if size(source.ic,1)~=expectedIc
    error('ic_coefficients has %d rows; expected %d.', ...
        size(source.ic,1),expectedIc);
end
if size(source.forcing_cos,1)~=numel(source.forcing_modes) || ...
        ~isequal(size(source.forcing_cos),size(source.forcing_sin))
    error('Forcing random coefficients are inconsistent.');
end
if numel(source.query_x)~=101 || numel(source.query_t)~=101
    error('Expected the original 101 x 101 query grid.');
end

if ismember('source_attempt_id',variables)
    allIds = M.source_attempt_id;
    allIds = allIds(:);
    source.source_ids = uint32(allIds(1:cfg.NumSamples));
else
    source.source_ids = uint32((1:cfg.NumSamples).');
end

fields = {'grf_modes','grf_std','forcing_modes','forcing_std', ...
    'query_x','query_t','ic','forcing_cos','forcing_sin'};
for k = 1:numel(fields)
    require_finite(source.(fields{k}),fields{k});
end
end


function solver = build_solver(query_x,query_t,cfg)
N = cfg.SpectralNx;
Lx = cfg.Xmax-cfg.Xmin;
solver.x = cfg.Xmin+Lx*(0:N-1)'/N;
solver.sensor_x = cfg.Xmin+Lx*(0:cfg.SensorNx-1)'/cfg.SensorNx;
solver.query_x = double(query_x(:));
solver.query_t = double(query_t(:));

totalStepsReal = (cfg.Tmax-cfg.Tmin)/cfg.Dt;
solver.totalSteps = round(totalStepsReal);
if abs(solver.totalSteps-totalStepsReal)>1e-10
    error('Dt must divide the full time interval exactly.');
end
solver.saveSteps = round((solver.query_t-cfg.Tmin)/cfg.Dt);
if max(abs(solver.query_t-(cfg.Tmin+solver.saveSteps*cfg.Dt)))>5e-7
    error('Every query_t value must lie on an ETDRK4 step.');
end
solver.saveId = zeros(solver.totalSteps+1,1,'uint16');
solver.saveId(solver.saveSteps+1) = uint16(1:numel(solver.saveSteps));

modeNumbers = [0:N/2, -N/2+1:-1].';
kLap = (2*pi/Lx)*modeNumbers;
solver.kDer = kLap;
solver.kDer(N/2+1) = 0;
cutoff = floor(cfg.DealiasFraction*(N/2));
solver.dealiasMask = abs(modeNumbers)<=cutoff;

linear = -cfg.Nu*(kLap.^2);
solver.E = exp(cfg.Dt*linear);
solver.E2 = exp(cfg.Dt*linear/2);
contourCount = 32;
r = exp(1i*pi*((1:contourCount)-0.5)/contourCount);
LR = cfg.Dt*linear(:,ones(contourCount,1))+r(ones(N,1),:);
solver.Q = cfg.Dt*real(mean((exp(LR/2)-1)./LR,2));
solver.f1 = cfg.Dt*real(mean( ...
    (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3,2));
solver.f2 = cfg.Dt*real(mean( ...
    (2+LR+exp(LR).*(-2+LR))./LR.^3,2));
solver.f3 = cfg.Dt*real(mean( ...
    (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3,2));

normalizedQuery = (solver.query_x-cfg.Xmin)/Lx;
solver.interpolation = exp(2i*pi*modeNumbers*normalizedQuery.')/N;
end


function completed = initialize_or_validate_output(fileName,source,cfg)
nx = numel(source.query_x);
nt = numel(source.query_t);
n = cfg.NumSamples;

if isfile(fileName)
    expected = [nt,nx,n];
    info = h5info(fileName,'/solution');
    if ~isequal(double(info.Dataspace.Size),expected)
        error('Existing /solution size is %s; expected %s.', ...
            mat2str(info.Dataspace.Size),mat2str(expected));
    end
    check_attribute(fileName,'spectral_nx',cfg.SpectralNx);
    check_attribute(fileName,'sensor_nx',cfg.SensorNx);
    check_attribute(fileName,'etdrk4_dt',cfg.Dt);
    check_attribute(fileName,'viscosity',cfg.Nu);
    completed = double(h5readatt(fileName,'/','completed_samples'));
    if completed<0 || completed>n || completed~=round(completed)
        error('Invalid completed_samples attribute: %.16g.',completed);
    end
    fprintf('Resuming after %d completed samples.\n',completed);
    return;
end

chunkSamples = min(cfg.BatchSize,n);
h5create(fileName,'/query_x',[nx,1],'Datatype','double');
h5create(fileName,'/query_t',[nt,1],'Datatype','double');
h5create(fileName,'/sensor_x',[cfg.SensorNx,1],'Datatype','double');
h5create(fileName,'/source_attempt_id',[n,1],'Datatype','uint32');
h5create(fileName,'/initial_condition',[nx,n], ...
    'Datatype','single','ChunkSize',[nx,chunkSamples]);
h5create(fileName,'/forcing',[nx,n], ...
    'Datatype','single','ChunkSize',[nx,chunkSamples]);
h5create(fileName,'/initial_condition_sensors',[cfg.SensorNx,n], ...
    'Datatype','single','ChunkSize',[cfg.SensorNx,chunkSamples]);
h5create(fileName,'/forcing_sensors',[cfg.SensorNx,n], ...
    'Datatype','single','ChunkSize',[cfg.SensorNx,chunkSamples]);
h5create(fileName,'/solution',[nt,nx,n], ...
    'Datatype','single','ChunkSize',[nt,nx,1], ...
    'Deflate',1,'Shuffle',true);

sensor_x = cfg.Xmin+(cfg.Xmax-cfg.Xmin)*(0:cfg.SensorNx-1)'/cfg.SensorNx;
h5write(fileName,'/query_x',source.query_x);
h5write(fileName,'/query_t',source.query_t);
h5write(fileName,'/sensor_x',sensor_x);
h5write(fileName,'/source_attempt_id',source.source_ids);
h5writeatt(fileName,'/','completed_samples',uint32(0));
h5writeatt(fileName,'/','total_samples',uint32(n));
h5writeatt(fileName,'/','train_samples',uint32(cfg.TrainSamples));
h5writeatt(fileName,'/','test_samples',uint32(cfg.TestSamples));
h5writeatt(fileName,'/','spectral_nx',uint32(cfg.SpectralNx));
h5writeatt(fileName,'/','sensor_nx',uint32(cfg.SensorNx));
h5writeatt(fileName,'/','etdrk4_dt',cfg.Dt);
h5writeatt(fileName,'/','viscosity',cfg.Nu);
h5writeatt(fileName,'/','dealias_fraction',cfg.DealiasFraction);
h5writeatt(fileName,'/','train_index_range_zero_based','[0,3000)');
h5writeatt(fileName,'/','test_index_range_zero_based','[3000,3100)');
h5writeatt(fileName,'/','pde', ...
    'u_t + u*u_x - nu*u_xx = f(x), periodic x in [-1,1)');
h5writeatt(fileName,'/','reference_method', ...
    ['Periodic Fourier pseudo-spectral ETDRK4 with conservative ', ...
     'nonlinearity and 2/3 de-aliasing']);
h5writeatt(fileName,'/','reference_note', ...
    'High-resolution numerical reference, not closed-form exact');
completed = 0;
end


function check_attribute(fileName,name,expected)
actual = double(h5readatt(fileName,'/',name));
if ~isscalar(actual) || abs(actual-double(expected))> ...
        1e-12*max(1,abs(double(expected)))
    error('Existing attribute %s=%.16g; expected %.16g.', ...
        name,actual,double(expected));
end
end


function [solution,u0Query,forcingQuery,u0Sensors,forcingSensors,icError] = ...
        solve_one_sample(sampleId,source,solver)
nGrf = numel(source.grf_modes);
coeff = source.ic(:,sampleId);
amplitude = coeff(1);
shift = coeff(2);
grfCos = coeff(3:2+nGrf);
grfSin = coeff(3+nGrf:2+2*nGrf);
forcingCos = source.forcing_cos(:,sampleId);
forcingSin = source.forcing_sin(:,sampleId);

u0Spectral = -amplitude*sin(pi*(solver.x+shift)) + ...
    evaluate_series(solver.x,grfCos,grfSin, ...
    source.grf_modes,source.grf_std);
forcingSpectral = evaluate_series(solver.x,forcingCos,forcingSin, ...
    source.forcing_modes,source.forcing_std);
u0Query = -amplitude*sin(pi*(solver.query_x+shift)) + ...
    evaluate_series(solver.query_x,grfCos,grfSin, ...
    source.grf_modes,source.grf_std);
forcingQuery = evaluate_series(solver.query_x,forcingCos,forcingSin, ...
    source.forcing_modes,source.forcing_std);
u0Sensors = -amplitude*sin(pi*(solver.sensor_x+shift)) + ...
    evaluate_series(solver.sensor_x,grfCos,grfSin, ...
    source.grf_modes,source.grf_std);
forcingSensors = evaluate_series(solver.sensor_x,forcingCos,forcingSin, ...
    source.forcing_modes,source.forcing_std);

v = fft(u0Spectral);
forcingHat = fft(forcingSpectral);
solution = zeros(numel(solver.query_x),numel(solver.query_t));
solution(:,1) = interpolate_fourier(v,solver.interpolation);

for step = 1:solver.totalSteps
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

require_finite(solution,sprintf('spectral solution sample %d',sampleId));
icError = max(abs(solution(:,1)-u0Query));
solution = single(solution);
u0Query = single(u0Query);
forcingQuery = single(forcingQuery);
u0Sensors = single(u0Sensors);
forcingSensors = single(forcingSensors);
end


function value = evaluate_series(x,xiCos,xiSin,modes,spectralStd)
x = double(x(:));
modes = double(modes(:));
spectralStd = double(spectralStd(:));
value = cos(pi*x*modes.')*(spectralStd.*double(xiCos(:))) + ...
    sin(pi*x*modes.')*(spectralStd.*double(xiSin(:)));
end


function value = burgers_nonlinear(v,solver)
u = real(ifft(v));
value = -0.5i*solver.kDer.*fft(u.^2);
value(~solver.dealiasMask) = 0;
end


function value = interpolate_fourier(v,interpolation)
value = real((v.'*interpolation).');
end


function require_finite(value,name)
if any(~isfinite(double(value(:))))
    error('%s contains NaN or Inf.',name);
end
end
