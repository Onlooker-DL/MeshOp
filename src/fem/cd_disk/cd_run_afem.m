function [node,elem,generation,u,hist,snapshots] = cd_run_afem( ...
        cfg,grf,beta,node,elem,generation,nCycles)
%CD_RUN_AFEM The ONE shared AFEM loop: SOLVE -> ESTIMATE -> MARK -> NVB
% REFINE.  Used verbatim by data generation and by every AFEM row of the
% FEM comparison, guaranteeing identical logic (same initial mesh, same
% theta=0.8, same NVB, same SUPG solve).
%
%   snapshots{1}      = initial (solved at cycle 0)
%   snapshots{r+1}    = mesh after r complete cycles
%   hist fields       = cumulative timing breakdown
%       assemblyTimeSec / solveTimeSec / refineTimeSec / otherTimeSec /
%       timeSec (total)
    if nargin < 7 || isempty(nCycles)
        nCycles = cfg.mainAFEMCycles;
    end
    if nargin < 6 || isempty(generation)
        generation = zeros(size(elem,1),1);
    end
    beta = double(beta(:).');

    hist = repmat(struct('cycle',0,'nodes',0,'elems',0,'eta',NaN, ...
        'marked',0,'maxGen',0,'timeSec',0, ...
        'assemblyTimeSec',0,'solveTimeSec',0, ...
        'refineTimeSec',0,'otherTimeSec',0),nCycles+1,1);
    snapshots = cell(nCycles+1,1);
    cumulativeTime = 0;
    cumAsm = 0; cumSol = 0; cumRef = 0; cumOther = 0;

    for ell = 0:nCycles
        cycleTimer = tic;
        [u,sinfo] = cd_solve_p1_supg(node,elem,cfg,grf,beta);
        tAsm = sinfo.assemblyTimeSec;
        tSol = sinfo.solveTimeSec;

        eta2 = cd_estimator('estimate',node,elem,u,cfg,grf,beta);
        eta2(~isfinite(eta2) | eta2<0) = 0;

        hist(ell+1).cycle = ell;
        hist(ell+1).nodes = size(node,1);
        hist(ell+1).elems = size(elem,1);
        hist(ell+1).eta = sqrt(sum(eta2));
        hist(ell+1).maxGen = max(generation);
        snapshots{ell+1} = struct('node',node,'elem',elem,'gen',generation,'u',u);

        tRef = 0;
        if ell == nCycles
            totalCycle = toc(cycleTimer);
            tOther = totalCycle - tAsm - tSol;
            cumAsm = cumAsm + tAsm;
            cumSol = cumSol + tSol;
            cumOther = cumOther + max(tOther,0);
            cumulativeTime = cumulativeTime + totalCycle;
        else
            marked = cd_estimator('mark',eta2,cfg.theta);
            marked = find(marked);
            hist(ell+1).marked = numel(marked);

            if size(elem,1) >= cfg.maxElements
                error('cd_run_afem:MaxElements', ...
                    'Element count reached cfg.maxElements=%d.',cfg.maxElements);
            end

            tRefTimer = tic;
            [node,elem,generation,~] = cd_nvb_refine_conforming_local( ...
                node,elem,generation,marked,cfg);
            tRef = toc(tRefTimer);

            totalCycle = toc(cycleTimer);
            tOther = totalCycle - tAsm - tSol - tRef;
            cumAsm = cumAsm + tAsm;
            cumSol = cumSol + tSol;
            cumRef = cumRef + tRef;
            cumOther = cumOther + max(tOther,0);
            cumulativeTime = cumulativeTime + totalCycle;
        end

        hist(ell+1).timeSec = cumulativeTime;
        hist(ell+1).assemblyTimeSec = cumAsm;
        hist(ell+1).solveTimeSec = cumSol;
        hist(ell+1).refineTimeSec = cumRef;
        hist(ell+1).otherTimeSec = cumOther;
    end
end