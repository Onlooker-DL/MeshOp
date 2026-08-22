function cfg = burgers_matlab_config()
%BURGERS_MATLAB_CONFIG Configuration for the U-Net/Gmsh/MATLAB baseline.

here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(here)));
experiment = 'b3000_continuous_mesh_size_mse';

cfg.projectRoot = projectRoot;
cfg.dataFile = fullfile(projectRoot,'data','burgers', ...
    'burgers_mesh_size_3100.h5');
cfg.meshDirectory = fullfile(projectRoot,'results','burgers','unet', ...
    experiment,'gmsh_meshes_matlab_normalized_xt');
cfg.outputDirectory = fullfile(projectRoot,'results','burgers','unet', ...
    experiment,'matlab_fem_normalized_xt');
cfg.meshCoordinateMode = 'normalized_xt';

% Empty means all 100 test samples. MATLAB test IDs are one-based.
cfg.testIds = [];
cfg.saveSampleIds = [4,69];
cfg.makePlots = true;
cfg.resume = true;

% Match MeshOp's Burgers FEM and spectral-reference settings.
cfg.xmin = -1;
cfg.xmax = 1;
cfg.tmin = 0;
cfg.tmax = 1;
cfg.nu = 5e-3;
cfg.initialGrid = 32;
cfg.newtonMaxIt = 40;
cfg.newtonTol = 1e-6;
cfg.newtonRelTol = 1e-6;
cfg.newtonVerbose = false;
cfg.newtonLineSearch = true;
cfg.newtonLineSearchMax = 16;
cfg.newtonRegularizationTrials = 5;
cfg.newtonRegularizationBase = 1e-12;
cfg.newtonMaxUpdateFactor = 2;

cfg.NxRefSpectral = 512;
cfg.NtRefSpectral = 201;
cfg.refDt = 2.5e-4;
cfg.dealiasFraction = 2/3;
cfg.errorChunkElements = 300000;
cfg.compareGridX = 2001;
cfg.maxElements = 3000000;
end
