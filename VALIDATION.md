# Validation status

Completed in the delivery environment:

- parsed every Python source file with the standard Python AST parser;
- compiled every Python source file with `compileall`;
- parsed and checked all eight YAML experiment configurations;
- checked that every top-level MATLAB function name matches its filename;
- checked that both FEM runners invoke the same PDE-specific FEM program with
  zero correction cycles for base and the configured number for hybrid;
- loaded both `run_operator.py --help` entry points successfully.
- verified by unit test that every Burgers operator configuration reads
  `burgers_5100.mat`;
- verified that 2000-sample training uses zero-based indices 0,...,1999,
  3000-sample training uses 0,...,2999, and both use 5000,...,5099 as the
  identical test set;
- verified that 2000- and 3000-sample result, figure, and split paths are
  distinct.

Not executed in the delivery environment:

- PyTorch model-forward or training tests, because PyTorch is not installed;
- MATLAB data generation and FEM solves, because MATLAB/Octave is unavailable;
- end-to-end numerical reproduction, because the large `.mat` data sets and a
  CUDA runtime are not part of this workspace.

Before a production run, install `requirements.txt`, place the checkpoint data
under `data/`, run one short-epoch operator smoke test, and set
`cfg.numSamples = 5` for the first MATLAB FEM smoke comparison.
