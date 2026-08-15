# Result files

`result/` contains a small set of trained operator exports so that the paper's
FEM stage can be reproduced without re-training. Everything else under
`result/` is regenerable output and is not committed (see `.gitignore`).

## What is included

### Operator predictions - `result/operators/`

Trained 3000-sample operator exports for the three experiments:

| Problem | Model | Experiment |
| --- | --- | --- |
| burgers | FNO / CNO / DeepONet / DeepONet-multi / POD-DeepONet | `b3000_mse` |
| reaction_diffusion (accu) | FNO | `rd_accu_b3000_mse` |
| cd_disk | FNO | `cd_b3000_mse` |

> Note: All neural-operator configurations and the included trained operator
> exports use seed 42. Each operator folder contains some of the following:

- `predictions.mat` - canonical test-set score predictions (100 test samples;
  for cd_disk these are source ids 3001-3100); consumed directly by the
  MATLAB FEM stage.
- `final_model.pt` (burgers / reaction-diffusion) or `checkpoint_last.pt`
  (cd_disk, shipped via Release) - trained weights.
- `final_metrics.json`, `per_sample_test_metrics.csv`,
  `training_history.csv` / `history.json`, etc. - evaluation numbers.
- `resolved_config.json` / `run_config.json` - exact configuration used.
- `split_indices.npz` - deterministic data split (stored inside the FNO
  experiment folders).

## Data splits

Deterministic `prefix_train_tail_test` split: the first N samples are used for
training and the final 100 samples for testing. Split indices are stored as
`split_indices.npz` inside the FNO operator folders and regenerated
automatically if missing.

## How to use directly

- **FEM reproduction:** set `operatorName` and `operatorExperiment` in
  `experiments/<problem>/fem/*_fem_config.m` (e.g. `fno` / `cd_b3000_mse`) and
  run the FEM entry point; it resolves `result/operators/.../predictions.mat`
  automatically. See the one-sample example in
  `examples/burgers_fem_single_sample/`.
- **Plotting:** the tools under `tools/` and the examples read the same
  `predictions.mat`.

## What is NOT included (run it yourself)

- `result/fem/...` - FEM comparison outputs (meshes, errors, timings).
  Regenerate with the FEM scripts (see the root `README.md`).
- Operator exports for other training sizes (b1000, b2000) or other CD models.
  Train them with the commands in `experiments/README.md`.
- `last_*` checkpoints and POD basis files (`pod_basis.npz`) - regenerable
  and not shipped.
- Data sets. They are provided as GitHub Release assets; see the root
  `README.md` for download instructions.

## Note on version control

The shipped operator files (`predictions.mat`, metrics/configs, final
checkpoints, `split_indices.npz`) are explicitly whitelisted in `.gitignore` and
are tracked with a normal `git add`; no force-add is needed. Regenerable
artifacts (data sets, FEM results, `last_*` checkpoints, POD basis files,
logs, figures) remain ignored.