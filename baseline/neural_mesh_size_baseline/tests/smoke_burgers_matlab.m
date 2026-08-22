function smoke_burgers_matlab()
%SMOKE_BURGERS_MATLAB Small synthetic end-to-end interface test.
projectRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot,'src','fem','matlab'));
tmpRoot=tempname(fullfile(projectRoot,'tests'));
mkdir(tmpRoot); cleanup=onCleanup(@()rmdir(tmpRoot,'s')); %#ok<NASGU>
meshDir=fullfile(tmpRoot,'meshes'); outDir=fullfile(tmpRoot,'out');
mkdir(meshDir); mkdir(outDir);
dataFile=fullfile(tmpRoot,'tiny.h5');
h5create(dataFile,'/source_dataset_index',[1,1],'Datatype','uint32');
h5write(dataFile,'/source_dataset_index',uint32(5001));
h5create(dataFile,'/ic_coefficients',[4,1]);
h5write(dataFile,'/ic_coefficients',[1;0;0;0]);
h5create(dataFile,'/forcing_xi_cos',[1,1]); h5write(dataFile,'/forcing_xi_cos',0);
h5create(dataFile,'/forcing_xi_sin',[1,1]); h5write(dataFile,'/forcing_xi_sin',0);
h5create(dataFile,'/grf_modes',[1,1]); h5write(dataFile,'/grf_modes',1);
h5create(dataFile,'/grf_spectral_std',[1,1]); h5write(dataFile,'/grf_spectral_std',0);
h5create(dataFile,'/forcing_modes',[1,1]); h5write(dataFile,'/forcing_modes',1);
h5create(dataFile,'/forcing_spectral_std',[1,1]); h5write(dataFile,'/forcing_spectral_std',0);
h5writeatt(dataFile,'/','train_samples',uint32(0));
h5writeatt(dataFile,'/','test_samples',uint32(1));
h5writeatt(dataFile,'/','total_samples',uint32(1));

[nodes,triangles]=uniform_mesh(8);
test_id=int32(1); source_dataset_index=int32(5001); %#ok<NASGU>
inference_time_sec=1e-3; mesh_time_sec=2e-3; %#ok<NASGU>
mesh_background_field_time_sec=5e-4; %#ok<NASGU>
mesh_gmsh_generate_time_sec=1e-3; mesh_extraction_time_sec=5e-4; %#ok<NASGU>
query_x=[-1;0]; query_t=[0;1]; predicted_log_h=zeros(2); %#ok<NASGU>
coordinate_mode='normalized_xt'; %#ok<NASGU>
save(fullfile(meshDir,'test_001_source_5001.mat'),'nodes','triangles', ...
    'test_id','source_dataset_index','inference_time_sec','mesh_time_sec', ...
    'mesh_background_field_time_sec','mesh_gmsh_generate_time_sec', ...
    'mesh_extraction_time_sec','query_x','query_t','predicted_log_h', ...
    'coordinate_mode');

cfg=struct('dataFile',dataFile,'meshDirectory',meshDir, ...
    'outputDirectory',outDir,'meshCoordinateMode','normalized_xt', ...
    'testIds',1,'saveSampleIds',[], ...
    'makePlots',false,'resume',false,'xmin',-1,'xmax',1,'tmin',0,'tmax',1, ...
    'nu',5e-3,'initialGrid',4,'newtonMaxIt',12,'newtonTol',1e-6, ...
    'newtonRelTol',1e-6,'newtonVerbose',false,'newtonLineSearch',true, ...
    'newtonLineSearchMax',8,'newtonRegularizationTrials',3, ...
    'newtonRegularizationBase',1e-12,'newtonMaxUpdateFactor',2, ...
    'NxRefSpectral',32,'NtRefSpectral',5,'refDt',1e-2, ...
    'dealiasFraction',2/3,'errorChunkElements',1000,'compareGridX',101, ...
    'maxElements',10000);
results=evaluate_burgers_matlab(cfg);
assert(height(results.table)==1);
assert(isfinite(results.table.relative_l2));
assert(isfile(fullfile(outDir,'per_sample_matlab_fem_metrics.csv')));
fprintf('MATLAB Burgers hybrid smoke test passed.\n');
end


function [node,elem] = uniform_mesh(n)
xg=linspace(-1,1,n+1); tg=linspace(0,1,n+1);
[XX,TT]=meshgrid(xg,tg); node=[XX(:),TT(:)];
id=@(i,j)j*(n+1)+i+1; elem=zeros(2*n*n,3); c=0;
for j=0:n-1
    for i=0:n-1
        n00=id(i,j); n10=id(i+1,j); n01=id(i,j+1); n11=id(i+1,j+1);
        c=c+1; elem(c,:)=[n00,n10,n11];
        c=c+1; elem(c,:)=[n00,n11,n01];
    end
end
end
