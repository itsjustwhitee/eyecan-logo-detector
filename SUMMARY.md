# Full Documentation — Eyecan Logo Detection Challenge

## Summary of Changes and Updates

This document summarizes all changes made to the project, explains the technical choices, and provides a guide for using the training, inference, and test scripts.

---

## 1. New Repository Assets

Two new assets are now included directly in the repository:

### Pre-trained Checkpoint: `checkpoints/best_20k.pt`

A checkpoint trained on **20k synthetic images** using the `sample_background/` pool. Ready for immediate use:

```bash
python src/test_inference.py --checkpoint checkpoints/best_20k.pt
```

This allows you to test the detector on real images without training, and to use it as a baseline for comparison when training your own model.

### Background Pool: `sample_background/` (~1000 images)

A curated subset of the MIT Indoor Scenes dataset included for reproducibility and quick experiments. Using it avoids the need to download the full 15k-image dataset for initial tests:

```bash
python src/dataset_generator.py --bg_dir sample_background --num_samples 5000 --seed 42
```

---

## 2. Module B — Current Augmentation Parameters

The parameters below reflect the **actual values in `dataset_generator.py`**.

### Spatial Transforms

```python
A.Affine(
    scale=(0.025, 0.85),
    rotate=(-180, 180),
    translate_percent=(-0.25, 0.25),
    shear=(-10, 10),
    p=0.99,
),
A.Perspective(scale=(0.05, 0.10), p=0.5),
```

| Parameter | Value | Rationale |
|---|---|---|
| Scale | 0.025–0.85 | Wide range: from very small to large logos. |
| Rotate | ±180° | Full rotation diversity. |
| Translate | ±25% | Keeps logo mostly within frame. |
| Shear | ±10° | Simulates mild surface curvature. |
| Perspective | 5–10%, p=0.5 | Moderate projective warp applied half the time. |

### Logo Color Augmentation

```python
A.HueSaturationValue(
    hue_shift_limit=8,
    sat_shift_limit=20,
    val_shift_limit=15,
    p=0.65,
),
A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.30),
```

Applied **before compositing**, only to logo pixels (`alpha > 0`). This avoids hue casts on the background.

### Scene-Level Photometric Augmentation

```python
A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.25, p=0.70),
A.GaussianBlur(blur_limit=(3, 8), p=0.15),
A.GaussNoise(var_limit=(1.0, 7.0), p=0.10),
A.ImageCompression(quality_lower=70, quality_upper=95, p=0.20),
```

Applied **after compositing** to the full scene. `GaussNoise` is new — it simulates sensor noise and low-light grain.

### Logo Max Ratio

```python
max_logo_ratio = 0.5  # default parameter in _generate_sample()
```

The logo's longest side is scaled to at most 50% of the shorter canvas dimension.

---

## 3. Module A Explained (Logo Detector)

### Model Architecture

```
MobileNetV3-Small backbone (ImageNet pretrained)
    ↓ Global Average Pooling
    ↓ 576-d feature vector
576 → Linear(128) → Hardswish → Dropout(0.3) → Linear(2) → Sigmoid
    ↓
(x, y) ∈ (0, 1)
```

### Technical Choices

#### Loss: Smooth L1 (Huber, beta=0.01)

```python
criterion = nn.SmoothL1Loss(beta=0.01)
```

- Quadratic (MSE-like) for `|error| < 0.01` ⇒ precise convergence.
- Linear (MAE-like) for larger errors ⇒ robust to outliers.
- At 640×480: transition at ~6 px in pixel space.

#### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

Decoupled weight decay prevents weight explosion and acts as implicit L2 regularization.

#### LR Scheduler: CosineAnnealing

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 0.01
)
```

Smoothly decays from `lr` to `lr/100` over the full training budget.

#### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents gradient explosion on batches with outlier samples (very small or blurry logos).

#### Early Stopping

```python
patience = 10  # epochs without val_loss improvement
```

Typical stopping point: epoch 30–45 on 10k samples, epoch 50–60 on 20k samples.

### Evaluation Metrics

**Validation Loss (SmoothL1):** Typical value 0.003–0.005 ≈ 1.5–2.4 px average error @ 640×480.

**Validation Accuracy:** Fraction of predictions with Euclidean pixel error < 10% of the short side (challenge requirement). Threshold at 640×480: 48 px. Typical value: 70–85%.

---

## 4. `test_inference.py` — Updated CLI and Behaviour

The script now exposes a full CLI:

```bash
python src/test_inference.py \
    --checkpoint checkpoints/best_20k.pt \
    --input      test_images \
    --output     test_results
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Model weights to load. |
| `--input` | `test_images` | Input directory (recursive image search). |
| `--output` | `test_results` | Output directory for annotated images. |

**Changes from the previous version:**
- Added `--checkpoint`, `--input`, `--output` CLI arguments (previously hardcoded).
- GUI is now **auto-disabled** in headless environments (Colab, servers without `$DISPLAY`), no crash.
- Inference is forced to **CPU** for maximum cross-environment compatibility.
- Shadow offset is **dynamic** (`max(1, thickness / 2)`) instead of a fixed `+1 px`.
- Window size is resized to 800×600 for better display on high-resolution monitors.

**Crosshair rendering:**

```python
r         = max(10, int(min(H, W) * 0.03))       # 3% of short side
thickness = max(2, int(min(H, W) * 0.004))       # 0.4% of short side
offset    = max(1, int(thickness / 2))            # dynamic shadow offset
```

---

## 5. Notebook (`eyecal_logo_detector.ipynb`)

An end-to-end Jupyter/Colab notebook is included. It covers:

1. **Setup** — dependency installation and path configuration.
2. **Dataset generation** — calls `generate_dataset()` using `sample_background/`.
3. **Training** — calls `train()` and plots the training log.
4. **Inference** — loads `best_20k.pt` or the freshly trained checkpoint and runs prediction on sample images.
5. **Visualization** — displays results with Matplotlib inline.

The notebook is designed to run without modification on Google Colab (headless, CPU or GPU).

---

## 6. Essential Commands

```bash
# Inference with pre-trained checkpoint (no setup needed)
python src/test_inference.py --checkpoint checkpoints/best_20k.pt

# Generate dataset (bundled backgrounds, quick)
python src/dataset_generator.py --bg_dir sample_background --num_samples 5000 --seed 42

# Generate dataset (full MIT Indoor Scenes)
python src/dataset_generator.py --bg_dir backgrounds/Images --num_samples 20000 --seed 42

# Train
python src/logo_detector.py --train \
    --dataset_dir generated_dataset \
    --epochs 100 \
    --lr 1e-3 \
    --seed 42

# Single image inference
python src/logo_detector.py --infer \
    --checkpoint checkpoints/best_20k.pt \
    --image test_images/photo.jpg

# Batch test with custom paths
python src/test_inference.py \
    --checkpoint checkpoints/best.pt \
    --input my_photos \
    --output my_results
```

---

## 7. Conclusions

The project is complete and fully documented:

✅ **`checkpoints/best_20k.pt`**: Pre-trained checkpoint (20k samples) included for immediate use and as a baseline.
✅ **`sample_background/`**: ~1000 bundled background images for quick generation without external downloads.
✅ **Module B**: Dataset generator with documented augmentation parameters matching the actual code.
✅ **Module A**: Logo detector (MobileNetV3-Small) with robust training pipeline and fast inference.
✅ **`test_inference.py`**: Updated with CLI arguments, headless auto-detection, dynamic crosshair shadow, and CPU-forced device.
✅ **Notebook**: End-to-end `eyecal_logo_detector.ipynb` covering generation, training, inference, and visualization.
✅ **Documentation**: README, MODULE-A.md, MODULE-B.md, and SUMMARY.md all aligned with the actual code.