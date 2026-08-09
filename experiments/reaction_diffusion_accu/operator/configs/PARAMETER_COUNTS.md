# Parameter-matched reaction_diffusion_accu operator configurations

Target scale: approximately 250,000 trainable parameter elements, using the
same `sum(p.numel() for p in model.parameters() if p.requires_grad)` convention
as the training code.

| Configuration | Architecture | Trainable parameters |
|---|---|---:|
| fno_b1000.yaml | width=11, modes=5^3, layers=4 | 243,515 |
| fno_b2000.yaml | width=11, modes=5^3, layers=4 | 243,515 |
| fno_b3000.yaml | width=11, modes=5^3, layers=4 | 243,515 |
| deeponet_b3000.yaml | hidden=54, latent=88, depths=4/4 | 250,035 |
| transolver_b3000.yaml | width=26, layers=3, heads=2, slices=32 | 249,957 |
| pod_deeponet_b3000.yaml | hidden=56, depth=4, rank r<=128 | 243,097--250,336 |

POD-DeepONet cannot have one fixed parameter count before POD construction,
because the retained rank is selected from the training data by the 0.99 energy
criterion. With this configuration, every possible retained rank from 1 to 128
still keeps the trainable parameter count within the stated interval.

All files use:

    data/reaction_diffusion_accu/reaction_diffusion_accu_3100.mat

The test set is always the final 100 samples (3001--3100). FNO training sizes
are 1000, 2000, and 3000; the other backbones use 3000 training samples.

Copy the `operators` directory to:

    experiments/reaction_diffusion_accu/operator/configs/operators/

If your launch script uses `configs/operators` instead, either change that
path in the script or place the directory there. The path spelling must match.
