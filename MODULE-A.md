# Module A: Logo Detector

Deep learning-based centroid regression model for detecting the Eyecan logo.
Predicts the normalized coordinates (x, y) ∈ [0, 1] of the logo center in an image.

---

## Table of Contents

- [Architecture and Technical Choices](#architecture-and-technical-choices)
- [Usage — Training](#usage--training)
- [Usage — Inference](#usage--inference)
- [Dataset Format](#dataset-format)
- [Training Pipeline](#training-pipeline)
- [Notebook Scripts](#notebook-scripts)
- [Test & Visualization](#test--visualization)
- [Current Limitations](#current-limitations)

---

## Architecture and Technical Choices

### Backbone: MobileNetV3-Small (ImageNet-pretrained)

**Motivation:**

1. **Computational efficiency**: MobileNetV3-Small is optimized for resource-constrained devices.

2. **Transfer Learning**: ImageNet pre-trained weights provide low-level features (edges, colors, textures)
   already effective for logo localization, drastically reducing training time.

3. **Mobile-first architecture**: Inverted residual blocks with depthwise separable convolutions.
   - Maximizes accuracy while reducing parameters and FLOPs.
   - Ideal for edge deployment (browser, mobile, embedded).

**Backbone structure:**
```
Input RGB (3 × H × W)
    ↓
MobileNetV3 Feature Extractor (features)
    ↓
Global Average Pooling (avgpool)
    ↓
Feature Vector (576-dimensional)
```

### Regression Head: 2-layer MLP

**Architecture:**
```
576 → Linear(576, 128) → Hardswish → Dropout(0.3) → Linear(128, 2) → Sigmoid → (x, y)
```

**Technical choices:**

1. **Hardswish activation**: Efficient activation function used in MobileNetV3
   - Maintains consistency with the backbone
   - Better gradient flow compared to ReLU for regression

2. **Dropout(0.3)**: Regularization to prevent overfitting
   - Essential on synthetic datasets (10k–20k samples)
   - Automatically disabled in `.eval()` mode

3. **Final Sigmoid**: Guarantees output ∈ (0, 1) **without post-processing**
   - Resolution-agnostic normalized coordinates
   - No manual clamping required

4. **Xavier initialization**: Small initial weights for stability
   - Prevents Sigmoid saturation at the start of training

### Loss Function: Smooth L1 (Huber Loss)

**Formula:**
```
SmoothL1(pred, target) = {
    0.5 * (pred - target)² / beta,    if |pred - target| < beta
    |pred - target| - 0.5 * beta,     otherwise
}
```

**Parameters:** `beta = 0.01`

**Motivation:**
- **Precision**: Quadratic behavior (MSE) near zero ⇒ precise convergence on the centroid.
- **Robustness**: Linear behavior (MAE) for large errors ⇒ less sensitive to occasional outliers.
- **Stability**: Avoids exploding gradients during early training compared to pure MSE.

**Discarded alternative:** Pure MSE
- Sensitive to outliers: a single sample with error >0.5 dominates the gradient.
- On synthetic datasets with random backgrounds, some samples can confuse the model.

---

## Usage — Training

### CLI (command line)

```bash
python src/logo_detector.py --train \
    --dataset_dir generated_dataset \
    --output_dir checkpoints \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 32 \
    --val_split 0.15 \
    --dropout 0.3 \
    --seed 42
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | `generated_dataset` | Root of the Underfolder directory. |
| `--output_dir` | `checkpoints` | Where to save `best.pt`, `last.pt`, `training_log.json`. |
| `--epochs` | `30` | Maximum number of epochs. |
| `--lr` | `1e-3` | Initial learning rate (AdamW). |
| `--batch_size` | `32` | Samples per mini-batch |
| `--val_split` | `0.15` | Fraction of the dataset used for validation. |
| `--dropout` | `0.3` | Dropout probability in the head. |
| `--seed` | `0` | Seed for reproducibility. |

### Output

The following files are saved in `--output_dir`:

```
checkpoints/
  ├── best.pt              ← checkpoint with the best validation loss
  ├── last.pt              ← checkpoint from the last epoch
  └── training_log.json    ← training history (for plotting)
```

**`training_log.json` format:**
```json
[
  {
    "epoch": 1,
    "train_loss": 0.01234,
    "val_loss": 0.01456,
    "val_acc_pct": 67.3
  },
  ...
]
```

**`val_acc_pct` metric** reppresents the percentage of predictions with Euclidean error < 10% of the short side (challenge requirement).
At 640×480, the short side is 480 px ⇒ threshold = 48 px.

---

## Usage — Inference

### CLI (single image)

```bash
python src/logo_detector.py --infer \
    --checkpoint checkpoints/best.pt \
    --image test_images/my_photo.jpg
```

**Output:**
```
  Predicted centroid (normalised) : x=0.4231  y=0.6180
  Predicted centroid (pixels)     : x=270     y=296
```

### Python API (programmatic)

```python
import cv2
from pathlib import Path
from logo_detector import LogoDetector

# Load model
detector = LogoDetector.load("checkpoints/best.pt")

# Read image (OpenCV BGR → RGB)
bgr = cv2.imread("test_images/photo.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Predict (auto-detect GPU/CPU)
x_norm, y_norm = detector.predict(rgb)

# Convert to pixels
H, W = bgr.shape[:2]
px, py = int(x_norm * W), int(y_norm * H)

print(f"Logo centroid → ({px} px, {py} px)")
```

**Technical notes:**
- `predict()` automatically sets the model to `.eval()` and disables dropout.
- Input: numpy array RGB uint8, **any resolution** (internally resized to 224×224).
- Output: normalized coordinates ∈ [0, 1] (multiply by W, H for absolute pixels).

---

## Dataset Format

Module A requires a dataset in **pipelime Underfolder** format generated by Module B:

```
generated_dataset/
  data/
    00000_image.jpg        ← RGB composited scene
    00000_metadata.json    ← {"x": 0.4231, "y": 0.6180}
    00001_image.jpeg
    00001_metadata.json
    ...
```

**Extended format support (robustness):**

`LogoDataset` automatically searches for image files matching the pattern `{stem}_image.*`:
- **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.JPG`, `.JPEG`, `.PNG` (case-insensitive).
- **Metadata:** Currently `.json` only.

**Example metadata JSON:**
```json
{
  "x": 0.4231,
  "y": 0.6180
}
```

Normalized coordinates: `x = centroid_px / image_width`, `y = centroid_px / image_height`

### Programmatic loading

```python
from pathlib import Path
from logo_detector import LogoDataset

dataset = LogoDataset(Path("generated_dataset"))
print(f"Loaded {len(dataset)} samples")

# Access a sample
image_tensor, label_tensor = dataset[0]
# image_tensor: torch.Tensor (3, 224, 224), ImageNet-normalized
# label_tensor: torch.Tensor (2,), values [x_norm, y_norm]
```

---

## Training Pipeline

### Preprocessing (applied to each image)

```python
_preprocess = transforms.Compose([
    transforms.ToTensor(),                           # HWC uint8 → CHW float [0,1]
    transforms.Resize((224, 224), antialias=True),   # canonical MobileNet size
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),                  # ImageNet mean
        std=(0.224, 0.224, 0.224)                     # ImageNet std
    ),
])
```

> **Important:** Augmentation (brightness, rotation, warp) is applied **upstream** by the dataset generator.
The training loop does **not** re-apply augmentation to avoid distributional mismatch.

### Training Schedule

| Component | Configuration | Rationale |
|---|---|---|
| **Optimizer** | AdamW, `lr=1e-3`, `weight_decay=1e-4` | AdamW prevents weight explosion, implicit L2 decay |
| **LR Scheduler** | CosineAnnealingLR, `T_max=epochs`, `eta_min=lr*0.01` | Smooth decay from `lr` to `lr/100`, no manual step |
| **Gradient Clipping** | `max_norm=1.0` | Prevents explosion on batches with outliers |
| **Early Stopping** | patience=10 epochs on `val_loss` | Avoids overfitting, typically stops at epoch 30–45 |
| **Train/Val Split** | 85% / 15% random split (seeded) | Small but sufficient validation set (750 samples @ 5k) |

### Typical behavior

**On 10k samples, epochs=100:**
- Training stops between epoch 30–45 (early stopping)
- Best validation loss: ~0.003–0.005 (SmoothL1)
- Validation accuracy (10% threshold): 70–85%

> **Note:** Accuracy never reaches 100% because:
> 1. Synthetic dataset with extreme augmentation (rotation ±180°, perspective warp)
> 2. Some very complex backgrounds confuse the model
> 3. Very small logos or logos in extreme corners are inherently ambiguous

---

## Notebook Scripts

At the bottom of `logo_detector.py` (line ~545+) there are **3 commented blocks** that can be copied directly into Jupyter/Colab:

### 1. Training from Notebook

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

**Output:** Returns the best.pt model already loaded and ready for inference.

### 2. Inference from Notebook

```python
import cv2, sys
sys.path.insert(0, "src")
from logo_detector import LogoDetector
from pathlib import Path

detector = LogoDetector.load("checkpoints/best.pt")

bgr = cv2.imread("my_photo.jpeg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
x, y = detector.predict(rgb)
H, W = bgr.shape[:2]
print(f"Logo centroid → ({int(x*W)} px, {int(y*H)} px)")
```

**Usage:** Quick test on individual images without launching the CLI script.

### 3. Matplotlib Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(rgb)
px, py = int(x * W), int(y * H)
ax.plot(px, py, "r+", markersize=18, markeredgewidth=3, label="predicted")
ax.set_title(f"Logo centroid: ({px}, {py})")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Test & Visualization

### Script: `test_inference.py`

Batch inference on a directory of real images with an interactive GUI:

```bash
python src/test_inference.py
```

**How it works:**

1. Reads all images in `test_images/` (recursive)
2. For each image:
   - Loads with OpenCV (BGR)
   - Runs forward pass: `detector.predict(rgb)`
   - Denormalizes coordinates: `px = x_norm * W`, `py = y_norm * H`
   - Draws a **red crosshair** (circle + cross) with black shadow
   - Saves the annotated image to `test_results/pred_{filename}`
   - Shows a GUI window (OpenCV): **PRESS ANY KEY TO CONTINUE**

**Crosshair parameters:**

| Parameter | Formula | Description |
|---|---|---|
| Radius | `max(10, min(H,W) * 0.03)` | Dynamic scale: 3% of the short side |
| Thickness | `max(2, min(H,W) * 0.004)` | Dynamic scale: 0.4% of the short side |
| Color | `(0, 0, 255)` BGR | Bright red |
| Shadow | Offset `(+1, +1)` px, black | Improves visibility on light backgrounds |

**Console output:**
```
[INFO] Running inference on 12 images...

-> photo_001.jpg            | Predicted centroid: ( 342 px,  256 px)
-> photo_002.JPG            | Predicted centroid: ( 128 px,  401 px)
...

[INFO] Inference complete. Visual results saved in 'test_results'.
```

**Known limitations:**

On real images (logo printed/projected in environment), performance varies:
- ✅ **Accurate** when the logo is visible, well-lit, angle <30°.
- ⚠️ **Approximate** with very small logos, extreme perspective, or blur.
- ❌ **Off-target** with partial occlusions, reflections, or very complex backgrounds.

**Possible improvements:**
1. Increase dataset size (20k → 50k+ samples).
2. Less extreme augmentation (rotation ±90° instead of ±180°).
3. Collect photos and fine-tune.

---

## Current limitations

1. **Synthetic ⇒ real domain gap:**
   - The model is trained on perfect compositing with soft shadows
   - Real images: reflections, optical distortions, aggressive JPEG compression

3. **Background diversity:**
   - The dataset generator uses random ImageNet backgrounds.
   - Real scenes: specific contexts (offices, trade show booths) with recurring patterns

4. **Single-scale training:**
   - Fixed resize to 224×224 loses information for very small logos (<5% of the image)
---

## Troubleshooting

### ImportError: cannot import Sample from pipelime

**Solution:** Module A does **not** depend on pipelime — only the dataset generator (Module B) does.

```bash
# Required for Module A
pip install torch torchvision opencv-python numpy tqdm
```
> *Remeber to install torch\* according to your GPU (if any/wanted).*

### CUDA out of memory during training

**Solution 1:** Reduce batch size
```bash
python src/logo_detector.py --train --batch_size 16
```

**Solution 2:** Train on CPU (much slower)
```python
from logo_detector import train
train(..., device=torch.device("cpu"))
```

### Validation accuracy stuck at ~50%

**Possible causes:**
1. Dataset too small (<5k samples)
2. Augmentation misconfigured in Module B
3. Learning rate too high or too low

**Debugging:**
```python
# Check coordinate distribution in the dataset
import json
from pathlib import Path

coords = []
for f in Path("generated_dataset/data").glob("*_metadata.json"):
    meta = json.loads(f.read_text())
    coords.append((meta["x"], meta["y"]))

import numpy as np
print("X range:", np.min([c[0] for c in coords]), np.max([c[0] for c in coords]))
print("Y range:", np.min([c[1] for c in coords]), np.max([c[1] for c in coords]))
# Both should cover [0, 1] uniformly
```

---

## Technical Notes

### Reproducibility

For identical results across runs:
```bash
python src/logo_detector.py --train --seed 42
```

**Seed controls:**
- Random train/val split
- Weight initialization (Xavier is already deterministic)
- Dropout mask during training

**Does not control:**
- CUDA non-deterministic ops (conv backward)
- DataLoader workers (if `num_workers > 0`)

For full reproducibility:
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```
(⚠️ Slows down training)

---

## References

- [MobileNetV3 Paper (Howard et al. 2019)](https://arxiv.org/abs/1905.02244)
- [Smooth L1 Loss (Fast R-CNN)](https://arxiv.org/abs/1504.08083)
- [AdamW Optimizer (Loshchilov & Hutter 2019)](https://arxiv.org/abs/1711.05101)
- [Pipelime Documentation](https://github.com/eyecan-ai/pipelime-python)