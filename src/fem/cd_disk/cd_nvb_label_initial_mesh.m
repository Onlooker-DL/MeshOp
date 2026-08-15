%CD_NVB_LABEL_INITIAL_MESH Shared disk NVB labeling.
% Verbatim from one_sample_cd_disk_random_localized_grf_K16.m.
function elem=cd_nvb_label_initial_mesh(node,elem)
% Store positive triangle as [z0,z1,z2], with reference edge [z1,z2].
% The initial reference edge is chosen as the triangle's longest edge.

elem=double(elem);

for K=1:size(elem,1)

    v=elem(K,:);
    p=node(v,:);

    if det([p(2,:)-p(1,:);p(3,:)-p(1,:)])<0
        v=v([1,3,2]);
        p=node(v,:);
    end

    l23=sum((p(2,:)-p(3,:)).^2);
    l31=sum((p(3,:)-p(1,:)).^2);
    l12=sum((p(1,:)-p(2,:)).^2);

    [~,id]=max([l23,l31,l12]);

    if id==1
        elem(K,:)=[v(1),v(2),v(3)];
    elseif id==2
        elem(K,:)=[v(2),v(3),v(1)];
    else
        elem(K,:)=[v(3),v(1),v(2)];
    end
end

end

%% =====================================================================
