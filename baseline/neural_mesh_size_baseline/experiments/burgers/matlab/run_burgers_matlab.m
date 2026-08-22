function results = run_burgers_matlab(varargin)
%RUN_BURGERS_MATLAB Run periodic P1 space-time FEM on exported Gmsh meshes.
%
% From the project root:
%   addpath('experiments/burgers/matlab');
%   run_burgers_matlab
%   run_burgers_matlab('TestIds',[4 69])

here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(here)));
addpath(fullfile(projectRoot,'src','fem','matlab'));

cfg = burgers_matlab_config();
parser = inputParser;
addParameter(parser,'TestIds',cfg.testIds,@(x)isnumeric(x)&&isvector(x));
addParameter(parser,'SaveSampleIds',cfg.saveSampleIds,@(x)isnumeric(x)&&isvector(x));
addParameter(parser,'MakePlots',cfg.makePlots,@(x)islogical(x)||isnumeric(x));
addParameter(parser,'Resume',cfg.resume,@(x)islogical(x)||isnumeric(x));
parse(parser,varargin{:});
cfg.testIds = double(parser.Results.TestIds(:).');
cfg.saveSampleIds = double(parser.Results.SaveSampleIds(:).');
cfg.makePlots = logical(parser.Results.MakePlots);
cfg.resume = logical(parser.Results.Resume);

results = evaluate_burgers_matlab(cfg);
end
