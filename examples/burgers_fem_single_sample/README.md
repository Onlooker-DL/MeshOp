# Example: reproduce one Burgers sample with MATLAB FEM

This example reproduces the paper's finite-element stage for ONE Burgers test
instance (Test 001, dataset id 5001). It builds two meshes from refinement
fields — the FNO-predicted field and the AFEM-derived target field — solves
the space-time Burgers problem on each, and saves mesh/error outputs.

It requires a trained operator export (`predictions.mat`). The example
defaults to `result/operators/burgers/fno/b3000_mse/predictions.mat`: either
train the operator first (full configs under
`experiments/burgers/operator/configs/operators/`, or the tiny
`examples/burgers_single_sample` Python example), or use a released pretrained
export.

## Files

| File | Purpose |
| --- | --- |
| `burgers_fem_example_config.m` | one-sample FEM configuration |
| `run_burgers_fem_example.m` | MATLAB entry point |

## Requirements

- MATLAB (a recent release); no extra toolboxes are needed for this example.
- `data/burgers/burgers_5100.mat` (generate with
  `experiments/burgers/data_gen/generate_data.m`).
- `result/operators/burgers/fno/b3000_mse/predictions.mat`.

## Run

```matlab
cd examples/burgers_fem_single_sample
run_burgers_fem_example
```

The example calls `setup.m` itself; no manual path setup is needed.

## Outputs

- `result/fem/burgers/fno/b3000_mse/base/example_1sample/` — numerical
  results (errors, DOFs, timings) and saved meshes.
- `figures/fem/burgers/fno/b3000_mse/base/example_1sample/` — mesh and error
  figures.

The `outputTag = 'example_1sample'` keeps the example results in a separate
folder so the paper's full 100-sample results are never overwritten.

## What to expect

- Two solved cases for Test 001: FNO-seeded AFEM and target-seeded AFEM,
  plus the high-resolution reference.
- Runtime: seconds to a few minutes on a desktop.