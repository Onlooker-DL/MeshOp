# Data

The full data sets used in the paper are not stored directly in this Git
repository because of their file sizes. They are provided as assets in the
GitHub Release.

## Download the data

### Option 1: Download from GitHub Releases

1. Open the repository page on GitHub.
2. Click **Releases** on the right-hand side of the repository page.
3. Open the latest release.
4. Under **Assets**, download:
   - `burgers_5100.mat`
   - `reaction_diffusion_accu_3100.mat`
5. Create the following directories in the repository and place the downloaded
   files in the corresponding locations:

```text
data/
├── burgers/
│   └── burgers_5100.mat
└── reaction_diffusion_accu/
    └── reaction_diffusion_accu_3100.mat
