# Result files

`result/` contains a small set of trained operator exports so that the paper's
FEM stage can be reproduced without re-training. Everything else under
`result/` is regenerable output and is not committed (see `.gitignore`).

## What is included

### Operator predictions — `result/operators/`

Trained 3000-sample operator exports for the two experiments:

| Problem | Model | Experiment |
| --- | --- | --- |
| burgers | FNO / CNO / DeepONet / DeepONet-multi / POD-DeepONet | `b3000_mse` |
| reaction_diffusion (accu) | FNO | `rd_accu_b3000_mse` |


> Note: All neural-operator configurations and the included trained operator exports use seed 42.
Each operator folder contains:

- `predictions.mat` — canonical test-set score predictions (100 test samples);
  consumed directly by the MATLAB FEM stage.
- `final_<model>_score_model.pt` (or `final_model.pt`) — trained weights.
- `final_metrics.json`, `per_sample_test_metrics.csv`,
  `training_history.csv` / `history.json`, etc. — evaluation numbers.
- `resolved_config.json` / `run_config.json` — exact configuration used.
- `split_indices.npz` — deterministic data split (FNO folders).

### Data splits — `result/splits/`

Deterministic `prefix_train_tail_test` split indices for the shipped
configurations. They are regenerated automatically if missing.

## How to use directly

- **FEM reproduction:** set `operatorName` and `operatorExperiment` in
  `experiments/<problem>/fem/*_fem_config.m` (e.g. `fno` / `b3000_mse`) and run
  the FEM entry point; it resolves `result/operators/.../predictions.mat`
  automatically. See the one-sample example in
  `examples/burgers_fem_single_sample/`.
- **Plotting:** the tools under `tools/` and the examples read the same
  `predictions.mat`.

## What is NOT included (run it yourself)

- `result/fem/...` — FEM comparison outputs (meshes, errors, timings).
  Regenerate with the FEM scripts (see the root `README.md`).
- Operator exports for other training sizes (b1000, b2000) or other models.
  Train them with the commands in `experiments/README.md`.
- `last_*` checkpoints and POD basis files (`pod_basis.npz`) — regenerable
  and not shipped.
- Data sets. See the root `README.md` for generation instructions.

## Note on version control

The shipped operator files (`predictions.mat`, metrics/configs, final
checkpoints, `split_indices.npz`) are explicitly whitelisted in `.gitignore` and
are tracked with a normal `git add`; no force-add is needed. Regenerable
artifacts (data sets, FEM results, `last_*` checkpoints, POD basis files,
logs, figures) remain ignored.
