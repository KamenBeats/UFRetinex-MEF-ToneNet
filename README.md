# UFRetinex-MEF-ToneNet

**Unsupervised Flexible Retinex Multi-Exposure Fusion with Transformer-Guided Reflectance Estimation and Adaptive Tone Correction**

---

## Overview

This repository contains the official PyTorch implementation of **UFRetinex-MEF-ToneNet**, a two-phase pipeline for multi-exposure image fusion (MEF):

- **Phase 1 — UFRetinex-MEF**: An *unsupervised* Retinex-based fusion framework that accepts a *flexible* number of input exposures (N ≥ 2). It decomposes each exposure into an illumination map and a shared reflectance using a Transformer-based architecture, then fuses them via Laplacian Pyramid blending to avoid halo artifacts.
- **Phase 2 — ToneNet**: A lightweight supervised tone-correction network trained on top of the frozen Phase 1 outputs. It applies a per-image brightness adjustment, per-channel curve correction, and color correction matrix (CCM) to match a reference ground-truth distribution.

### Key Features

- **Flexible N-input fusion** — works on 2, 3, 4, 5, … exposure stacks without retraining.
- **Unsupervised Phase 1** — no ground-truth fused images required; self-supervised via leave-one-out cross-evaluation.
- **Laplacian Pyramid MEF** — artifact-free illumination blending at multiple frequency bands.
- **Attention-based aggregation** — permutation-invariant cross-exposure feature aggregation via learned softmax weights.
- **Retinex Prior Gate** — blends the learned reflectance with a quality-weighted mean image for robustness.
- **Cross-attention in decoder** — illumination guides reflectance refinement without directly corrupting texture.
- **ToneNet** — global tone/color correction via iterative curve + CCM; initialized to identity for stable training.
- **WandB integration** — optional experiment tracking with metric logging and image visualization.

---

### Module Summary

| Module | Role |
|---|---|
| `L_net` | Per-image illumination estimation |
| `SRE` | Shared Reflectance Estimator (Transformer) |
| `AttentionAggregation` | Permutation-invariant softmax pooling |
| `RetinexPriorGate` | Blend SRE output with quality-prior |
| `PyramidLFusion` | Laplacian Pyramid MEF for illumination |
| `FusionNet` | Full Phase 1 model (orchestrator) |
| `ToneNet` | Phase 2 tone/color correction |

---

## 📁 Repository Structure

```
UFRetinex-MEF-ToneNet-Final/
├── nets/
│   ├── net.py              # All model definitions (L_net, SRE, FusionNet, ToneNet, etc.)
│   └── restormer.py        # Transformer building blocks (MDTA attention, GDFN, LayerNorm)
├── data/
│   └── dataset.py          # Dataset classes and collate functions
├── configs/
│   └── train_config.yaml   # Training hyperparameters for Phase 1 & 2
├── model/
│   └── ckpt.pth            # Pretrained Phase 1 checkpoint
├── train.py                # Phase 1 training script (unsupervised Retinex-MEF)
├── train_tonenet.py        # Phase 2 training script (ToneNet, supervised)
├── test.py                 # Inference / evaluation script
├── utils.py                # Loss functions and utility helpers
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Environment Setup

### Prerequisites

- OS: Linux / Windows / macOS
- Python 3.10
- CUDA 12.8+ (recommended) or CPU

### 1. Create Conda Environment

```bash
conda create -n ufretinex python=3.10 -y
conda activate ufretinex
```

### 2. Install PyTorch

Choose the command matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

**CUDA 12.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install WandB for Experiment Tracking

```bash
pip install wandb
wandb login
```

---

## 📂 Dataset Preparation

### Phase 1 (Unsupervised — no GT required)

Organize your dataset as follows. Each scene is a folder containing N exposure images:

```
data_set/
├── train/
│   ├── scene_001/
│   │   ├── 1.jpg        # under-exposed
│   │   ├── 2.jpg        # normal
│   │   └── 3.jpg        # over-exposed
│   ├── scene_002/
│   │   ├── 1.png
│   │   └── 2.png
│   └── ...
└── val/
    ├── scene_A/
    └── ...
```

- Images inside each scene folder are sorted **numerically** by filename.
- Any number of exposures per scene (N ≥ 2) is supported.
- Images must share the same spatial resolution within a scene (mismatched sizes are auto-resized).

### Phase 2 (Supervised — GT required)

For ToneNet training, each scene additionally needs a ground-truth fused image.
```
data_set/train/scene_001/
├── 1.jpg
├── 2.jpg
├── 3.jpg
└── gt.png     ← ground truth (any extension)
```

---

## 🔧 Configuration

All hyperparameters are managed in `configs/train_config.yaml`.

```yaml
model:
  sre_dim: 32              # SRE Transformer feature dimension
  decoder_dim: 64          # Decoder Transformer feature dimension
  num_blocks: 3            # Number of cross-attention Transformer blocks in decoder
  num_heads: 8             # Multi-head attention heads
  ffn_expansion_factor: 2  # Feed-forward network expansion ratio
  use_retinex_prior: true  # Enable Retinex Prior Gate
  l_fusion_hidden: 16      # (unused in PyramidLFusion, kept for API compatibility)

data:
  train_dir: "data_set/train"
  val_dir: "data_set/val"
  gt_dir: null             # Set path for Phase 2 GT
  patch_size: 128          # Random crop size (0 = full image)
  n_max: 5                 # Maximum exposures per batch
  num_workers: 0           # DataLoader workers
  augment: true            # Random flip/rotation augmentation

training:                  # Phase 1 settings
  epochs: 50
  batch_size: 2
  lr: 1.0e-4
  step_size: 10            # LR scheduler step (StepLR)
  gamma: 0.1               # LR decay factor
  cross_eval_mode: "rotate" # "rotate" (all targets) or "random" (one target/iter)
  save_every: 5
  val_every: 5

loss:                      # Phase 1 loss weights
  w_recon: 1.0
  w_smooth: 0.1
  w_init: 0.1
  w_suppress: 0.3
  w_consist: 0.1
  w_fusion: 0.5
  w_ssim: 0.2
  w_color_angle: 1.0
  w_chroma: 2.5

tonenet:                   # Phase 2 settings
  base_dim: 32
  n_curve_iters: 10        # Iterations of the tone curve x ← x + A·x·(1−x)
  epochs: 50
  lr: 3.0e-4
  w_histogram: 2.0         # Wasserstein histogram matching
  w_style: 0.5             # Gram-matrix style loss
  w_structure: 1.0         # Gradient structure preservation
  w_global_hue: 1.0        # Conditional hue loss
  w_lum_mean: 12.0         # Global luminance mean matching
  w_lum_hist: 8.0          # Luminance histogram matching
```

---

## 🚀 Training

### Phase 1 — Unsupervised Retinex-MEF

```bash
python train.py --config configs/train_config.yaml --gpu 0
```

**Resume from checkpoint:**
```bash
python train.py --config configs/train_config.yaml \
                --resume exp/MM_DD_HH_MM/model/ckpt_10.pth \
                --gpu 0
```

**Initialize from pretrained weights:**
```bash
python train.py --config configs/train_config.yaml \
                --pretrained model/phase_1/ckpt.pth \
                --gpu 0
```

**Disable WandB logging:**
```bash
python train.py --config configs/train_config.yaml --no_wandb
```

---

### Phase 2 — ToneNet (Supervised Tone Correction)

Requires a trained Phase 1 checkpoint and a dataset with GT images.

```bash
python train_tonenet.py \
    --config configs/train_config.yaml \
    --phase1_ckpt exp/MM_DD_HH_MM/model/best.pth \
    --gpu 0
```

**Resume ToneNet training:**
```bash
python train_tonenet.py \
    --config configs/train_config.yaml \
    --phase1_ckpt exp/MM_DD_HH_MM/model/best.pth \
    --resume exp/tonenet_MM_DD_HH_MM/model/tonenet_10.pth \
    --gpu 0
```

---

## Inference / Testing

### Phase 1 only (Retinex-MEF fusion)

```bash
# Test an entire directory of scene folders
python test.py \
    --test_dir dataset/test \
    --model_phase_1 model/ckpt.pth \
    --output results/ \
    --gpu 0

# Test specific scene folders
python test.py \
    --input path/to/scene1 path/to/scene2 \
    --model_phase_1 model/ckpt.pth \
    --output results/
```

### Phase 1 + Phase 2 (with ToneNet)

```bash
python test.py \
    --test_dir dataset/test \
    --model_phase_1 model/ckpt.pth \
    --model_phase_2 exp/tonenet_MM_DD_HH_MM/model/best_tonenet.pth \
    --output results/ \
    --gpu 0
```

### Additional Options

| Argument | Description | Default |
|---|---|---|
| `--resize N` | Resize longer side to N pixels before inference | `0` (no resize) |
| `--save_intermediates` | Also save reflectance map (R), illumination (L) | `False` |
| `--gpu` | CUDA device ID | `"0"` |

---

## Loss Functions

### Phase 1 Losses

| Loss | Symbol | Description |
|---|---|---|
| Reconstruction | `L_recon` | L1 loss between reconstructed target and original target image |
| Illumination Smooth | `L_smooth` | Structure-guided TV loss: suppresses illumination gradients proportional to image gradients |
| Illumination Init | `L_init` | L1 loss between L and max channel of image (Retinex prior) |
| Suppress | `L_suppress` | ReLU loss preventing R×L from exceeding any input exposure |
| Consistency | `L_consist` | L1 between Rhat and R_refined (encourages decoder to not diverge) |
| Color Angle | `L_color` | Cosine angle loss between RGB vectors to suppress color cast |
| Chrominance | `L_chroma` | Chrominance consistency of R across all exposures (masked to well-exposed regions) |
| Fusion | `L_fusion` | L1 + SSIM + L_smooth between full fusion output and pseudo-GT |

**Total Phase 1 loss:**
```
L = w_recon·L_recon + w_smooth·L_smooth + w_init·L_init
  + w_suppress·L_suppress + w_consist·L_consist
  + w_color_angle·L_color + w_chroma·L_chroma
  + w_fusion·L_fusion
```

### Phase 2 Losses (ToneNet)

| Loss | Symbol | Description |
|---|---|---|
| Histogram | `L_hist` | Wasserstein-1 (sort-based) per-channel histogram matching |
| Style | `L_style` | Multi-scale Gram-matrix style loss (1×, 2×, 4× downsampling) |
| Structure | `L_struct` | Gradient-based structure preservation vs. Retinex output |
| Conditional Hue | `L_hue` | Penalizes global hue error only when input has color cast |
| Luminance Mean | `L_lum_mean` | Matches global mean luminance per image |
| Luminance Histogram | `L_lum_hist` | Wasserstein-1 on luminance distribution |

**Total Phase 2 loss:**
```
L = w_hist·L_hist + w_style·L_style + w_struct·L_struct
  + w_hue·L_hue + w_lum_mean·L_lum_mean + w_lum_hist·L_lum_hist
```

---

## Technical Details

### Leave-One-Out Cross-Evaluation (Phase 1)

The model is trained without ground-truth fused images. For each training step:
1. Given N exposures from a scene, one image is designated the **target** and the remaining N-1 are the **inputs**.
2. The model fuses the N-1 inputs using the **target's illumination map** (L_override) as the output illumination.
3. The fused result is compared against the target image via reconstruction and color losses.
4. In `rotate` mode, all N images take turns being the target; in `random` mode, one is chosen at random.

This forces the model to learn robust reflectance estimation and scene-consistent fusion without ever seeing the "true" fused result.

### Laplacian Pyramid Fusion

The `PyramidLFusion` module fuses N illumination maps by:
1. Computing per-pixel MEF quality weights (3-tier: EV bell-curve × saturation penalty × bloom proximity).
2. Building separate Laplacian pyramids for each illumination map.
3. Blending each pyramid level using Gaussian-smoothed quality weights.
4. Collapsing the blended pyramid back to full resolution.

This eliminates the transition halos caused by direct weighted-sum blending, especially near blown-out regions.

### MEF Quality Weights

The `compute_mef_quality` function computes a three-tier quality weight for each pixel of each exposure:

1. **EV-aware bell-curve**: Normalizes each image by its mean luminance before scoring, so extreme dark/bright pairs are handled correctly via `base_q = 1 - (2·L_ev - 1)²`.
2. **Saturation penalty**: Sigmoid penalty for pixels above `sat_thresh=0.88`, detecting blown-out regions.
3. **Bloom proximity penalty**: Spatially suppresses regions *near* blown areas using adaptive Gaussian average pooling.
4. **Dark noise penalty**: Suppresses very dark pixels (L < 0.04) with low SNR via a sigmoid gate.

### ToneNet Curve

ToneNet applies an **iterative Michaelis-Menten-style tone curve**:
```
for _ in range(n_curve_iters):
    x = x + A * x * (1 - x)
```
where `A ∈ (-1, 1)` is learned per-channel from the image statistics. This curve is smooth, differentiable, and bounded in `[0, 1]`. Positive A brightens the image (especially midtones); negative A darkens. The final step applies a 3×3 color correction matrix (CCM) initialized to identity.

---

## Requirements

```
python >= 3.10
torch >= 2.0.1
torchvision
einops == 0.8.1
opencv-python == 4.8.1.78
numpy == 1.24.4
scipy == 1.8.0
matplotlib == 3.7.2
pyyaml >= 6.0
tqdm == 4.66.1
wandb (optional)
```

---

## Acknowledgements

- The Transformer building blocks (`restormer.py`) are adapted from [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://github.com/swz30/Restormer) (CVPR 2022).
- The MEF quality metric is inspired by [Exposure Fusion](https://mericam.github.io/exposure_fusion/) (Mertens et al., 2007) and [MEF-SSIM](https://ece.uwaterloo.ca/~z70wang/research/MEF/).
- Laplacian Pyramid blending follows the classical multi-resolution spline approach (Burt & Adelson, 1983).
- The Retinex-based multi-exposure fusion framework is inspired by [Retinex-MEF](https://github.com/HaowenBai/Retinex-MEF) (HaowenBai).

---
