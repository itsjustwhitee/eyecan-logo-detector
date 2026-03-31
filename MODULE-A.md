# Eyecan Challenge 02 — Logo Detector (Module A)

Deep-learning-based detector that predicts the **normalised (x, y) centroid**
of the Eyecan.ai logo inside an input image, compatible with the
[pipelime Underfolder](https://github.com/eyecan-ai/pipelime-python) dataset
produced by Module B.

---

## Quick start

```bash
# Training (all defaults)
python src/logo_detector.py --train

# Custom training run
python src/logo_detector.py --train \
    --dataset_dir generated_dataset \
    --output_dir  checkpoints \
    --epochs      30 \
    --lr          1e-3 \
    --batch_size  32 \
    --val_split   0.15 \
    --dropout     0.3 \
    --seed        0

# Single-image inference
python src/logo_detector.py --infer \
    --checkpoint checkpoints/best.pt \
    --image      my_photo.jpg

# Full help
python src/logo_detector.py --help
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--train` / `--infer` | *(required)* | Operating mode |
| `--dataset_dir` | `generated_dataset` | Root of the pipelime Underfolder dataset |
| `--output_dir` | `checkpoints` | Where to save checkpoints and training log |
| `--epochs` | `30` | Maximum training epochs |
| `--lr` | `1e-3` | Peak learning rate (AdamW) |
| `--batch_size` | `32` | Samples per mini-batch |
| `--val_split` | `0.15` | Fraction of data held out for validation |
| `--dropout` | `0.3` | Dropout probability in the regression head |
| `--seed` | `0` | RNG seed for train/val split reproducibility |
| `--checkpoint` | *(none)* | Path to `.pt` file — required for `--infer` |
| `--image` | *(none)* | Input image path — required for `--infer` |

---

## Output files

```
checkpoints/
  best.pt             ← weights at the epoch with lowest validation loss
  last.pt             ← weights at the final epoch
  training_log.json   ← per-epoch metrics (train/val loss, val accuracy %)
```

---

## Architecture — `LogoDetector`

```
Input image (B, 3, H, W)
        │
        ▼
  _preprocess()                   ToTensor → Resize 224×224 → ImageNet normalise
        │
        ▼
  MobileNetV3-Small backbone       ImageNet-pretrained, fully fine-tuned
  (features + AdaptiveAvgPool)     Output: (B, 576)
        │
        ▼
  Regression head
    Linear(576 → 128)
    Hardswish
    Dropout(p=0.3)
    Linear(128 → 2)
    Sigmoid                        → (B, 2) ∈ (0, 1)
        │
        ▼
  (x̂, ŷ) normalised centroid
```

### Design rationale

**Backbone — MobileNetV3-Small**

MobileNetV3-Small was chosen for its balance between accuracy and
efficiency. Its ImageNet weights provide immediately useful low-level features
(edges, colours, textures) that transfer well to logo localisation.
The full backbone is fine-tuned (not frozen), giving the model freedom to
specialise its filters to the Eyecan logo's distinctive hexagonal shape and
colour palette.

**Regression head — two-layer MLP + Sigmoid**

The head is intentionally minimal:

- `Linear(576 → 128)` + `Hardswish` + `Dropout` — compress features and
  regularise.
- `Linear(128 → 2)` + `Sigmoid` — project to (x, y) in (0, 1) without any
  external clamping.  Sigmoid is differentiable everywhere and naturally
  enforces the output range.

**Loss — SmoothL1 (Huber, β = 0.01)**

SmoothL1 behaves like L1 for large residuals (robust to centroid-placement
outliers produced by extreme augmentation) and like L2 for small residuals
(precise near convergence).  β = 0.01 sets the transition point at 1 % of
the image dimension — small enough that the model is penalised
quadratically in the "good detection" zone.

---

## Training schedule

| Component | Choice | Rationale |
|---|---|---|
| Optimiser | AdamW, weight_decay=1e-4 | Adaptive LR + L2 on weights; prevents overfit |
| Scheduler | CosineAnnealingLR, η_min=1%·lr | Smooth LR decay; no manual milestones needed |
| Grad clipping | max_norm=1.0 | Prevents gradient explosion when the backbone unfreezes |
| Early stopping | patience=10 on val_loss | Stops wasted epochs if the model has converged |

### Expected results (640×480, 5000 samples, 30 epochs, RTX 3080)

| Metric | Value |
|---|---|
| Best val loss (SmoothL1) | ~0.003 |
| Val accuracy (≤10% error) | ≥92% |
| Training time | ~20 min |

---

## Accuracy metric — `pixel_accuracy`

The challenge requires the predicted centroid to be within **10 % of the
shortest image dimension** of the ground-truth centroid.  For a 640×480
image this is 48 px.

```
error_px = ||pred_px − gt_px||₂
accurate  = error_px < 0.10 × min(W, H)
```

`pixel_accuracy` reports the fraction of predictions satisfying this
constraint in a given batch / epoch.

---

## Notebook usage

### Training

```python
import sys
sys.path.insert(0, "src")
from logo_detector import train
from pathlib import Path

detector = train(
    dataset_dir = Path("generated_dataset"),
    output_dir  = Path("checkpoints"),
    epochs      = 30,
    lr          = 1e-3,
    batch_size  = 32,
)
```

### Loading a pre-trained checkpoint

```python
from logo_detector import LogoDetector

detector = LogoDetector.load("checkpoints/best.pt")
```

### Inference on a custom image

```python
import cv2
from logo_detector import LogoDetector

detector = LogoDetector.load("checkpoints/best.pt")

bgr = cv2.imread("my_photo.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

x, y = detector.predict(rgb)          # normalised coords ∈ [0, 1]
H, W = bgr.shape[:2]
print(f"Logo centroid → ({int(x*W)} px, {int(y*H)} px)")
```

### Visualisation

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(rgb)
ax.plot(int(x * W), int(y * H), "r+", markersize=18, markeredgewidth=3,
        label="predicted")
ax.set_title(f"Logo centroid: ({int(x*W)}, {int(y*H)})")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Requirements

```
python >= 3.10
torch >= 2.0
torchvision >= 0.15
opencv-python
numpy
tqdm
```

Install:

```bash
pip install torch torchvision opencv-python numpy tqdm
```

No additional packages are required.  Module A is self-contained and does
**not** depend on pipelime at runtime (pipelime is used only by Module B's
generator).

---

## Notes

**Why not an out-of-the-box detector?**  The challenge explicitly forbids
frameworks such as Detectron2.  `LogoDetector` is an original, single-class
regression model with no external detection dependency.

**Why regression and not heatmap?**  For single-object localisation,
coordinate regression with a Sigmoid head is simpler, faster to train, and
equally accurate as heatmap approaches.  A full heatmap model would be
significantly harder to train and would not improve accuracy meaningfully
for a single logo.

**Resolution independence.**  Both the dataset generator and the detector
work in normalised coordinates.  You can generate data at 640×480 and run
inference on 1920×1080 images — the (x, y) output is always in [0, 1].
Convert to pixels with `x_px = x * W`, `y_px = y * H`.