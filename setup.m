function projectRoot = setup(cleanPath)
    %SETUP Initialize the Learning-Mesh-Operators-for-Adaptive-PDE-Solvers MATLAB environment.
    %
    % Usage:
    %   setup
    %   projectRoot = setup
    %   setup(false)
    %
    % Inputs:
    %   cleanPath
    %       true  - reset MATLAB search path before adding this project
    %               (default, recommended to avoid conflicts with old projects)
    %       false - preserve the current MATLAB search path
    %
    % Outputs:
    %   projectRoot
    %       Absolute path of the Learning-Mesh-Operators-for-Adaptive-PDE-Solvers project root.
    %
    % This setup:
    %   1. Detects the project root from this setup.m file.
    %   2. Optionally restores the default MATLAB path.
    %   3. Adds the project root and src directory.
    %   4. Adds experiment entry-point directories non-recursively.
    %   5. Creates runtime/output directories when missing.
    %   6. Sets TMPDIR for processes launched from MATLAB.
    %   7. Checks whether MATLAB's run function is shadowed.
    
        if nargin < 1 || isempty(cleanPath)
            cleanPath = true;
        end
    
        validateattributes( ...
            cleanPath, ...
            {'logical', 'numeric'}, ...
            {'scalar'}, ...
            mfilename, ...
            'cleanPath');
    
        cleanPath = logical(cleanPath);
    
        % Project root: directory containing this setup.m file.
        setupFile = mfilename('fullpath');
    
        if isempty(setupFile)
            error( ...
                'setup:UnableToLocateFile', ...
                'MATLAB cannot determine the location of setup.m.');
        end

        % mfilename('fullpath') omits the .m extension on some MATLAB
        % releases; normalize it so the shadowing check below compares it
        % with which('setup') on equal terms.
        if numel(setupFile) < 2 || ~strcmp(setupFile(end-1:end), '.m')
            setupFile = [setupFile '.m'];
        end
    
        projectRoot = fileparts(setupFile);
    
        % Remove stale paths from other MATLAB projects.
        if cleanPath
            restoredefaultpath;
            rehash toolboxcache;
        end
    
        % ---------------------------------------------------------------------
        % Required project directories
        % ---------------------------------------------------------------------
        srcRoot = fullfile(projectRoot, 'src');
    
        burgersExperimentRoot = fullfile( ...
            projectRoot, ...
            'experiments', ...
            'burgers');

        reactionDiffusionAccuExperimentRoot = fullfile( ...
            projectRoot, ...
            'experiments', ...
            'reaction_diffusion_accu');
    
        % ---------------------------------------------------------------------
        % Add MATLAB code paths
        % ---------------------------------------------------------------------
    
        % Put the current project before other user paths.
        addpath(projectRoot, '-begin');
    
        % src contains reusable FEM and utility implementations.
        if isfolder(srcRoot)
            addpath(genpath(srcRoot), '-begin');
        else
            warning( ...
                'setup:MissingSourceDirectory', ...
                'Source directory does not exist:\n%s', ...
                srcRoot);
        end
    
        % Add experiment entry points only.
        %
        % Do not use genpath(experiments), because recursively adding arbitrary
        % experiment folders can introduce generic filenames such as run.m,
        % setup.m, or config.m and shadow MATLAB functions.
        if isfolder(burgersExperimentRoot)
            if isfolder(fullfile(burgersExperimentRoot, 'fem'))
                addpath(fullfile(burgersExperimentRoot, 'fem'), '-begin');
            end
            if isfolder(fullfile(burgersExperimentRoot, 'data_gen'))
                addpath(fullfile(burgersExperimentRoot, 'data_gen'), '-begin');
            end
        end

        if isfolder(reactionDiffusionAccuExperimentRoot)
            if isfolder(fullfile(reactionDiffusionAccuExperimentRoot, 'fem'))
                addpath(fullfile(reactionDiffusionAccuExperimentRoot, 'fem'), '-begin');
            end
            if isfolder(fullfile(reactionDiffusionAccuExperimentRoot, 'data_gen'))
                addpath(fullfile(reactionDiffusionAccuExperimentRoot, 'data_gen'), '-begin');
            end
        end

        % Plotting utilities.
        toolsBurgersRoot = fullfile(projectRoot, 'tools', 'burgers');
        toolsRDRoot = fullfile(projectRoot, 'tools', 'reaction_diffusion');
        if isfolder(toolsBurgersRoot)
            addpath(toolsBurgersRoot, '-begin');
        end
        if isfolder(toolsRDRoot)
            addpath(toolsRDRoot, '-begin');
        end
    
        % ---------------------------------------------------------------------
        % Create runtime and output directories
        % ---------------------------------------------------------------------
        directoriesToCreate = {
            fullfile(projectRoot, '.runtime')
            fullfile(projectRoot, '.runtime', 'tmp')
            fullfile(projectRoot, 'logs')
            fullfile(projectRoot, 'result')
            fullfile(projectRoot, 'figures')
        };
    
        for directoryId = 1:numel(directoriesToCreate)
            currentDirectory = directoriesToCreate{directoryId};
    
            if ~isfolder(currentDirectory)
                [success, message, messageId] = mkdir(currentDirectory);
    
                if ~success
                    error( ...
                        'setup:DirectoryCreationFailed', ...
                        ['Unable to create directory:\n%s\n' ...
                         'MATLAB message (%s): %s'], ...
                        currentDirectory, ...
                        messageId, ...
                        message);
                end
            end
        end
    
        % Used by Python/multiprocessing processes launched from MATLAB.
        runtimeTmp = fullfile(projectRoot, '.runtime', 'tmp');
        setenv('TMPDIR', runtimeTmp);
        setenv('TEMP', runtimeTmp);
        setenv('TMP', runtimeTmp);
    
        % Refresh MATLAB's function cache after changing the path.
        rehash;
    
        % ---------------------------------------------------------------------
        % Shadowing checks
        % ---------------------------------------------------------------------
        runLocation = which('run');
    
        if isempty(runLocation)
            warning( ...
                'setup:RunFunctionNotFound', ...
                'MATLAB could not locate the run function.');
        elseif startsWith(runLocation, projectRoot)
            warning( ...
                'setup:RunFunctionShadowed', ...
                ['The MATLAB run function is shadowed by a project file:\n%s\n' ...
                 'Rename the conflicting run.m file.'], ...
                runLocation);
        end
    
        % Check the setup function currently being used.
        setupLocation = which('setup');
    
        if ~strcmp(setupLocation, setupFile)
            warning( ...
                'setup:SetupFunctionShadowed', ...
                ['The active setup function is not this project''s setup.m.\n' ...
                 'Expected: %s\n' ...
                 'Active:   %s'], ...
                setupFile, ...
                setupLocation);
        end
    
        % ---------------------------------------------------------------------
        % Display summary
        % ---------------------------------------------------------------------
        fprintf('\n');
        fprintf('============================================================\n');
        fprintf('Neural-operator AFEM MATLAB setup\n');
        fprintf('============================================================\n');
        fprintf('Project root : %s\n', projectRoot);
        fprintf('Source root  : %s\n', srcRoot);
        fprintf('Temporary dir: %s\n', runtimeTmp);
        fprintf('Clean path   : %s\n', string(cleanPath));
        fprintf('MATLAB run   : %s\n', runLocation);
        fprintf('============================================================\n');
        fprintf('\n');
    end
