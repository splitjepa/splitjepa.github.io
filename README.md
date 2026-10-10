<div align="center">

# SplitJEPA

[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://splitjepa.github.io/)

</div>

Code and Lean formalization for the paper [*SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction*](https://arxiv.org/abs/2610.12349).

SplitJEPA learns a latent representation organized into **invariant** and **variant** blocks directly from observations, without an observation decoder or reconstruction loss. Predictive learning recovers the complete latent state, while agreement across matched observations separates the factors they share from those that change.

Under stationary Gaussian predictive dynamics, the two latent subspaces are identifiable up to independent within-block orthogonal transformations. This repository contains a synthetic block-recovery experiment, RGB-based robotic manipulation experiments, and a Lean 4 formalization of the algebraic block-identifiability argument.

## Quick start: synthetic experiment

From the repository root, install PyTorch and run one seed:

```bash
cd simulation
python -m pip install -r requirements.txt
python run.py --seed 0 --device cpu
```

For an available NVIDIA GPU, use `--device cuda`. The run writes `simulation/results/seed=0.json` when launched as above. All experiment settings are collected in `CONFIG` in [`simulation/run.py`](simulation/run.py).

The experiment trains one LeJEPA encoder on predictive pairs from a Gaussian latent world observed through a nonlinear map, then freezes it. Both methods use the same centered, ZCA-whitened representation. SplitJEPA additionally estimates an orthogonal rotation from matched-pair differences: directions with the smallest difference covariance form the invariant block, and the remaining directions form the variant block.

The default configuration uses four invariant and four variant dimensions. Ground-truth latent values are used to generate the synthetic world and evaluate recovery, but are not targets for encoder training or block fitting.

| Output metric | What it measures |
| --- | --- |
| `global_r2` | Held-out linear fit of the complete representation from the true latents |
| `invariant_r2`, `variant_r2` | Held-out recovery of each true block from its corresponding learned block |
| `a12_fro`, `a21_fro` | Cross-block leakage in the fitted linear map; lower is better |
| `block_purity` | Fraction of fitted linear-map energy within the diagonal blocks; higher is better |

## Robotic manipulation

The robot experiments compare SplitJEPA with a single-latent LeJEPA baseline on **PickCube** and **PushCube** in ManiSkill. SplitJEPA uses a shared ResNet image trunk with separate invariant and variant projection heads. Its objective combines action-conditioned latent prediction, matched-pair agreement on the invariant block, and variance/covariance regularization on the complete representation.

With Python 3.10 or later, install the robot package from the repository root:

```bash
cd robot
python -m pip install -e '.[robot]'
```

**Prepared demonstration data and matched-pair files are required before training.** They are not included in this checkout. Place each task's data under `robot/data/<Task>_hshared100/`, where `<Task>` is `PickCube-v1` or `PushCube-v1`:

```text
episodes/traj_000000.npz    # image_base, image_side, action
index.json                 # Episode ids and lengths
split.json                 # Train/validation episode ids
pairs.h5                   # Matched cross-trajectory and viewpoint pairs
```

Then, from `robot/`, run the full pipeline:

```bash
python run.py --config configs/pickcube.yaml --seed 0 --stage all
python run.py --config configs/pushcube.yaml --seed 0 --stage all
```

The pipeline trains representations, freezes the encoders for behavior cloning, and evaluates the resulting policies. The policy comparisons are:

| Method | Frozen representation supplied to the policy |
| --- | --- |
| `ours_zh` | Invariant block with zero padding: `[r_H, 0]` |
| `ours_zhzl` | Complete SplitJEPA representation: `[r_H, r_L]` |
| `lejepa` | Single latent with the same total dimension |

All methods use the same policy input width and MLP architecture.

## Formal verification in Lean 4

[`lean/`](lean/) pins Lean and Mathlib to **v4.28.0**. With Lean's `elan` toolchain manager installed, run from the repository root:

```bash
cd lean
lake update
lake exe cache get
lake build
lake env lean SplitJEPA/Sanity.lean
```

The formalization covers observational equivalence under triangular latent mixing, elimination of both cross-block matrices, orthogonality of the recovered diagonal blocks, and finite variant isolation for invariant-only downstream computations. `Sanity.lean` prints the axiom dependencies of the public results.

See [`lean/README.md`](lean/README.md) for the paper-to-Lean mapping.

## Citation

If you find this work useful, please cite:

```bibtex
@article{hua2026splitjepa,
  title   = {SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction},
  author  = {Hua, Ruijin and Liu, Zichuan and Zhao, Zhuokai and Zheng, Yujia},
  journal = {arXiv preprint arXiv:2610.12349},
  year    = {2026}
}
```
