# Burgers data

Copy the original MeshOp file into this directory without committing it:

```text
data/burgers/burgers_5100.mat
```

Then run `generate_burgers_mesh_size_data` in MATLAB. The generator converts
the stored `target_score` generation labels into equivalent scalar mesh sizes
and computes Fourier ETDRK4 references only for test source samples
5001--5100. It writes `burgers_mesh_size_3100.h5`; completed reference batches
are resumable.

The stored normalized learning target is `1-generation/12`; it is unchanged
by the later `xi=(x+1)/2` meshing-coordinate normalization. Existing HDF5
data and trained checkpoints therefore do not need regeneration.
