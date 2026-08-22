# Burgers continuous mesh-size baseline

This experiment adapts the image-to-mesh idea of Chan et al. to the
space--time Burgers problem. A U-Net predicts a continuous scalar size field
on the fixed 101x101 query grid. Space and time are first normalized by
`xi=(x+1)/2, tau=t`; Gmsh performs isotropic periodic triangular meshing on
the unit square, and the nodes are mapped back to `[-1,1] x [0,1]`. MATLAB
first solves a common 32x32 space--time
mesh, interpolates that coarse solution onto the generated mesh, and uses it
as the Newton warm start. The periodic reduction, P1 weak form, quadrature,
Newton safeguards, spectral reference and physical relative L2 evaluator
match MeshOp v1.1.1.

The U-Net channel widths are 3--6--12--24--27: 48,064 trainable real
parameters, within 0.3% of the 48,190-real-parameter Burgers FNO.

The split is source samples 1--3000 for training and 5001--5100 for testing.
The test reference is a 512-point periodic Fourier pseudo-spectral ETDRK4
solve with `dt=2.5e-4`, viscosity `5e-3`, conservative nonlinearity, and 2/3
de-aliasing.

`evaluation.size_scale` is a single global multiplier for the predicted Gmsh
size field. If it is calibrated for a DOF-matched table, select it using only
training/validation samples and keep it fixed for all 100 test samples.

Run `src/meshing/export_burgers_meshes.py` after training, then execute
`run_burgers_matlab` from `experiments/burgers/matlab`. MATLAB reports both
the MeshOp-matched online time and a strict time that additionally includes
coarse-to-generated-mesh interpolation.

The normalized size convention is
`h_hat=(1/32)*2^(-generation/2)`. Thus the original 32x32 grid consists of
right-isosceles triangles in normalized coordinates, without treating space
and time as quantities with the same physical units.
