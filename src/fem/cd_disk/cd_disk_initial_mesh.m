function [node,elem,A0,bnd] = cd_disk_initial_mesh(cfg)
%CD_DISK_INITIAL_MESH Initial disk mesh, EXACTLY as the one-sample code:
%
%   [p,t] = initmesh(@circleg, 'Hmax', cfg.HmaxInitial, 'Jiggle','minimum')
%   elem = t(1:3,:).'  then the shared NVB initial labeling.
%
% Requires MATLAB PDE Toolbox (same requirement as
% one_sample_cd_disk_random_localized_grf_K16.m).
    if ~isfield(cfg,'HmaxInitial') || isempty(cfg.HmaxInitial)
        cfg.HmaxInitial = 0.1;   % same default as the one-sample code
    end
    if exist('initmesh','file') ~= 2
        error('cd_disk_initial_mesh:NoPDEToolbox', ...
            'initmesh requires the MATLAB PDE Toolbox.');
    end

    % initmesh with Jiggle uses the global RNG; fix the seed so every run
    % (data generation, FEM comparison) produces the IDENTICAL mesh.
    rngState = rng;
    rng(20240815,'twister');
    [pLegacy,~,tLegacy] = initmesh(@circleg, ...
        'Hmax', cfg.HmaxInitial, ...
        'Jiggle','minimum');
    rng(rngState);

    node = double(pLegacy.');
    elem = double(tLegacy(1:3,:).');
    elem = cd_nvb_label_initial_mesh(node,elem);

    p1=node(elem(:,1),:); p2=node(elem(:,2),:); p3=node(elem(:,3),:);
    area = 0.5*abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
                   (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2)));
    A0 = median(area);
    bnd = cd_mesh_boundary_nodes(elem);
end

function bnd = cd_mesh_boundary_nodes(elem)
    n = max(elem(:));
    edges = sort([elem(:,[1,2]);elem(:,[2,3]);elem(:,[3,1])],2);
    [uE,~,ic] = unique(edges,'rows');
    mult = accumarray(ic,1);
    bndEdges = uE(mult==1,:);
    bnd = unique(bndEdges(:));
    bnd = bnd(bnd>=1 & bnd<=n);
end