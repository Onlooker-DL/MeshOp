# CD-Disk experiment (steady convection-diffusion on the unit disk)

    -eps*Delta u + beta.grad u = f   in Omega,   u = 0 on dOmega,
    Omega = {(x,y): x^2+y^2<1}, eps = 1e-3, |beta|=1,

with a constant convection direction beta (random per sample) and a
transversely localized K=16 Fourier-GRF forcing.

## Layout

    experiments/cd_disk/
      operator/                     Python operator training + export
        run_operator.py             runner (fno/cno/deeponet/pod_deeponet)
        configs/operators/*.yaml    b3000 configs (+ fno b1000/b2000)
      fem/
        cd_disk_fem_config.m        method switches (mirrors burgers)
        run_cd_disk_fem.m           FEM comparison entry point

    src/fem/cd_disk/                SHARED AFEM machinery (single source,
                                    auto-loaded by setup.m via genpath(src))
      cd_config.m                   all PDE/score/grid parameters
      cd_disk_initial_mesh.m        deterministic disk initial mesh (initmesh)
      cd_nvb_label_initial_mesh.m   NVB labeling
      cd_nvb_refine_conforming_local.m  NVB conforming refinement
      cd_grf.m                      localized-GRF sampling/reconstruction/eval
      cd_solve_p1_supg.m            P1 SUPG solver (direct `\`)
      cd_estimator.m                residual estimator + Dorfler (theta=0.8)
      cd_run_afem.m                 THE shared SOLVE-ESTIMATE-MARK-REFINE loop
      cd_score_eval_grid.m          generation field on the polar query grid
      cd_score_to_mesh.m            score-driven mesh realization
      cd_reference.m                Fourier-Chebyshev disk reference + error

    src/fem/cd_disk_fem_comparison.m  FEM comparison implementation

## Consistency guarantee

Data generation and the FEM comparison use the SAME core (src/fem/cd_disk):
same initial mesh, same theta=0.8, same NVB, same SUPG solve, same polar
query grid (nr=64 clustered toward r=1, ntheta=128 = 8192 points), same
score cap 12 and generation threshold 0.5. Feeding the stored target score
back through cd_score_to_mesh reproduces the score-driven mesh with the
same rules used to create the data.

## Pipeline

1. Download the dataset from the GitHub Release:
       data/cd_disk/cd_disk_3100.mat

2. Train/export the operator (server with torch):
       python experiments/cd_disk/operator/run_operator.py \
         --config experiments/cd_disk/operator/configs/operators/fno_b3000.yaml
   -> result/operators/cd_disk/fno/cd_b3000_mse/predictions.mat
   predictions.mat contains only the 100 held-out test samples (source ids
   3001-3100), matching the reported test metrics.

3. FEM comparison (method switches in cd_disk_fem_config.m):
       run_cd_disk_fem()
   -> result/fem/cd_disk/<op>/<exp>/base|hybrid/... (summary.mat/csv + samples)

## Notes

- The initial disk mesh uses MATLAB PDE Toolbox `initmesh(@circleg,...)`
  with a fixed seed; it is identical to the mesh used for data generation.
- mainAFEMCycles=12; NVB closure can locally exceed 12 generations; the score
  field is clipped to [0,12] everywhere (labels, realization, training).
- The P1 SUPG solve is a direct sparse mldivide; no iterative solver and no
  coarse-start warm-up is needed for this linear problem.
- Spectral reference: coarse (127,64) / fine (255,128); coarse/fine rel RMS
  ~3e-4 for the one-sample test.