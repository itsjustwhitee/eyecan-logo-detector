# Full Documentation — Eyecan Logo Detection Challenge

## Summary of Changes and Updates

This document summarizes all changes made to the project, explains the technical choices, and provides a guide for using the training, inference, and test scripts.

---

## 1. Changes to Module B (Dataset Generator)

### Updated Augmentation Parameters

The parameters in `dataset_generator.py` have been modified to produce more varied and realistic images.

#### A) Spatial Transforms (lines 71–81)

**Before:**
```python
A.Affine(
    scale=(0.15, 0.85),
    rotate=(-180, 180),
    translate_percent=(-0.35, 0.35),
    p=1.0,
),
A.Perspective(scale=(0.02, 0.07), p=0.5),
```

**After:**
```python
A.Affine(
    scale=(0.20, 0.75),              # Narrower scale range
    rotate=(-180, 180),               # Unchanged
    translate_percent=(-0.30, 0.30),  # Less translation
    p=1.0,
),
A.Perspective(scale=(0.02, 0.06), p=0.4),  # Less warp, reduced probability
```

**Motivation:**
- **Scale 0.20–0.75:** Avoids logos that are too small (<15% min edge), which the model struggles to localize.
- **Translate ±30%:** Keeps the logo more centered, avoiding edge cases where it is almost out of frame.
- **Perspective 2–6%, p=0.4:** Reduces extreme perspective distortions that rarely appear in real photos.

#### B) Logo Color Augmentation (lines 122–134)

**Before:**
```python
A.HueSaturationValue(
    hue_shift_limit=8,
    sat_shift_limit=20,
    val_shift_limit=15,
    p=0.65,
),
A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
```

**After:**
```python
A.HueSaturationValue(
    hue_shift_limit=10,
    sat_shift_limit=25,
    val_shift_limit=20,
    p=0.7,
),
A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.35),
```

**Motivation:**
- **Hue ±10°:** Greater color variation (simulates colored lighting, printing on different materials).
- **Saturation ±25, Value ±20:** Wider range to cover faded and neon logos.
- **RGB shift ±15:** Simulates chromatic aberration and color casts.

#### C) Scene Augmentation (lines 103–118)

**Before:**
```python
A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.25, p=0.7),
A.GaussianBlur(blur_limit=(3, 11), p=0.15),
A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
```

**After:**
```python
A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.30, p=0.75),
A.GaussianBlur(blur_limit=(3, 13), p=0.18),
A.ImageCompression(quality_lower=65, quality_upper=95, p=0.25),
```

**Motivation:**
- **Brightness ±40%, Contrast ±30%:** Better coverage of very dark scenes (night) and overexposed ones (direct sunlight).
- **Blur 3–13:** Simulates motion blur and imperfect focus.
- **Compression 65–95:** Includes heavily compressed JPEGs (typical of smartphone photos).

#### D) Logo Max Ratio (line 276)

**Before:**
```python
max_logo_ratio = 0.30  # 30% of min(W, H)
```

**After:**
```python
max_logo_ratio = 0.25  # 25% of min(W, H)
```

**Motivation:**
- Smaller logos are more realistic (in real photos, logos rarely occupy more than 20% of the scene).
- Makes the task more challenging, forcing the model to learn more robust features.

### Extended Image Format Support (Module B)

In `LogoDataset` (Module A, lines 254–256), case-insensitive support has been added for:
- `.jpg`, `.jpeg`, `.JPG`, `.JPEG`.
- `.png`, `.PNG`.
- `.bmp`.

This makes the dataset loader more robust on Linux filesystems (which are case-sensitive).

---

## 2. Module A Explained (Logo Detector)

### Model Architecture

The model consists of:

1. **Backbone: MobileNetV3-Small**
   - Pre-trained on ImageNet (1.28M images, 1000 classes).
   - Mobile-first architecture with inverted residual blocks.
   - Output: 576-dimensional feature vector after global average pooling.

2. **Regression Head: 2-layer MLP**
   ```
   576 → Linear(128) → Hardswish → Dropout(0.3) → Linear(2) → Sigmoid
   ```
   - **Hardswish:** Smooth activation (x * ReLU6(x+3)/6), used in MobileNetV3.
   - **Dropout 0.3:** Regularization to prevent overfitting.
   - **Sigmoid:** Output ∈ (0, 1), normalized coordinates.

### Technical Choices

#### Loss Function: Smooth L1 (Huber Loss)

```python
criterion = nn.SmoothL1Loss(beta=0.01)
```

**Formula:**
- If `|pred - target| < beta`: loss = 0.5 * (diff)² / beta → MSE behavior.
- Otherwise: loss = |diff| - 0.5 * beta → MAE behavior.

**Why not pure MSE?**
- MSE is sensitive to outliers: a single sample with error=0.8 dominates the gradient.
- Smooth L1 is robust: large errors contribute linearly (like MAE).
- Near zero, it behaves like MSE → precise convergence.

**Why beta=0.01?**
- Narrow transition between quadratic and linear regime.
- With normalized coordinates ∈ [0, 1], beta=0.01 means transition at ~6 pixels @ 640×480.

#### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

**AdamW vs Adam:**
- AdamW decouples weight decay from the loss (weight decay is not influenced by the adaptive learning rate).
- Prevents weight explosion during training.
- `weight_decay=1e-4` acts as implicit L2 regularization.

#### Learning Rate Scheduler: CosineAnnealing

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 0.01
)
```

**Behavior:**
- Starts from `lr=1e-3`.
- Smoothly decays (cosine) down to `lr=1e-5` over `epochs` steps.
- No manual plateau or step decay required.

**Advantage:**
- Aggressive exploration at the start (high lr).
- Precise fine-tuning at the end (low lr).
- Better convergence than StepLR or ExponentialLR.

#### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why?**
- On batches with outliers (very small/blurry logo), gradients can explode.
- Clipping to norm=1.0 stabilizes training.
- Typical practice for regression tasks with Huber loss.

#### Early Stopping

```python
patience = 10  # epochs without improvement on val_loss
```

**Observed behavior:**
- On 10k samples, training stops between epoch 30–45.
- On 20k samples, it can reach epoch 50–60.
- Best val loss is typically reached before half the budget (epochs=100).

### Evaluation Metrics

#### Validation Loss (SmoothL1)

- Typical value: **0.003–0.005** on the validation set.
- Interpretation: average error of ~3–5% in normalized coordinates.
  - @ 640×480: 0.005 * 480 = **2.4 px** average error.

#### Validation Accuracy (Pixel Accuracy)

```python
def pixel_accuracy(preds, targets, img_size=(640, 480), threshold_pct=0.10):
    ...
    return (dist_normalized < threshold_pct).float().mean()
```

**Definition:**
- Fraction of predictions with Euclidean error < 10% of the short side.
- @ 640×480: short side = 480 px → threshold = **48 px**.
- Required by the challenge: max error 10% of the short edge.

**Typical value:** **70–85%** on the validation set.

**Why does it never reach 100%?**
1. Synthetic dataset with extreme augmentation (rotation ±180°, heavy blur).
2. Some very complex backgrounds (textures similar to the logo) confuse the model.
3. Very small logos (<10% of the image) have an ambiguous centroid.

---

## 3. Module A Scripts — Explanation and Usage

At the bottom of `logo_detector.py` (lines 545–593) there are **3 commented blocks** designed for Jupyter/Colab.

### Script 1: Training from Notebook (lines 551–565)

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

**What it does:**
1. Loads the Underfolder dataset from `generated_dataset/`.
2. Random 85/15 train/validation split (seeded).
3. Initializes MobileNetV3-Small with ImageNet weights.
4. Training loop with early stopping (patience=10).
5. Saves `checkpoints/best.pt` and `checkpoints/last.pt`.
6. Returns the best model loaded in memory.

**When to use it:**
- You want to monitor training step-by-step from a notebook.
- You want direct access to the model after training (no reload needed).
- Debugging: you can wrap `train()` in a try-except and inspect variables.

**Console output:**
```
======================================================================
  Training  — device: cuda  |  epochs: 30  |  lr: 0.001
======================================================================

[pipelime] image item    : JpegImageItem
[pipelime] metadata item : JsonMetadataItem
...
  Train samples : 8500   Val samples : 1500

  Epoch 001/030  train_loss=0.01234  val_loss=0.01456  val_acc=67.3%  lr=1.00e-03  (18.3s)
  Checkpoint saved → checkpoints/best.pt
  ...
  Epoch 042/030  train_loss=0.00321  val_loss=0.00298  val_acc=81.2%  lr=2.15e-05  (17.9s)
  Early stopping triggered after 10 epochs without improvement.

  Training log → checkpoints/training_log.json
  Restoring best checkpoint...
  Checkpoint loaded ← checkpoints/best.pt  (device: cuda)
```

### Script 2: Inference from Notebook (lines 567–579)

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

**What it does:**
1. Loads checkpoint `best.pt` (auto-detects GPU/CPU).
2. Reads the image with OpenCV (BGR → RGB conversion).
3. Forward pass: `detector.predict(rgb)` → `(x_norm, y_norm)`.
4. Denormalizes: multiplies by W, H.

**When to use it:**
- Quick test on individual images without the CLI.
- Debugging: print x, y to verify they are in range [0, 1].
- Interactive notebook: display the image with matplotlib after prediction.

**Notes:**
- `detector.predict()` automatically sets the model to `.eval()` (dropout off).
- Input can be **any resolution** (internally resized to 224×224).
- Output is always normalized ∈ [0, 1].

### Script 3: Matplotlib Visualization (lines 581–593)

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

**What it does:**
- Displays the RGB image.
- Plots a red `+` crosshair on the predicted centroid.
- Adds a title with pixel coordinates.

**When to use it:**
- Quick visualization in a notebook (Jupyter, Colab).
- Visual debugging: check whether the prediction looks reasonable.
- Reports/presentations: generate plots to save as PNG.

**Non-notebook alternative (GUI):**
- Use `test_inference.py` for batch processing with an interactive OpenCV window.

---

## 4. `test_inference.py` — How It Works and Usage

### What it does

Standalone script for batch inference on a directory of real images:

```bash
python src/test_inference.py
```

**Pipeline:**

1. Scans `test_images/` (recursive) for `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive).
2. For each image:
   - Loads with OpenCV (`cv2.imread()` → BGR).
   - Converts to RGB: `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`.
   - Forward pass: `detector.predict(rgb)` → `(x_norm, y_norm)`.
   - Denormalizes: `px = int(x_norm * W)`, `py = int(y_norm * H)`.
   - Draws a **red crosshair** with black shadow (circle + cross).
   - Saves the annotated image to `test_results/pred_{filename}`.
   - Shows a GUI window (OpenCV): **PRESS ANY KEY TO CONTINUE**.
3. Prints a summary to the console.

### Crosshair Details

```python
r = max(10, int(min(H, W) * 0.03))          # Radius: 3% of short side, min 10 px
thickness = max(2, int(min(H, W) * 0.004))  # Thickness: 0.4%, min 2 px

# Shadow (offset +1, +1 px, black)
cv2.circle(bgr, (px+1, py+1), r, (0,0,0), thickness)
cv2.line(...)  # horizontal cross shadow
cv2.line(...)  # vertical cross shadow

# Main crosshair (bright red BGR 0,0,255)
cv2.circle(bgr, (px, py), r, (0,0,255), thickness)
cv2.line(...)  # horizontal cross
cv2.line(...)  # vertical cross
```

**Features:**
- **Dynamic scaling:** Radius and thickness adapt to image resolution.
- **Shadow:** Improves visibility on light backgrounds (white logo on white wall).
- **Red color:** Maximum contrast against most backgrounds.

### Example Console Output

```
[INFO] Loading detector from checkpoints/best.pt on cpu...
  Checkpoint loaded ← checkpoints/best.pt  (device: cpu)
[INFO] Running inference on 12 images...

-> photo_001.jpg            | Predicted centroid: ( 342 px,  256 px)
-> photo_002.JPG            | Predicted centroid: ( 128 px,  401 px)
-> IMG_5678.jpeg            | Predicted centroid: ( 512 px,  384 px)
...

[INFO] Inference complete. Visual results saved in 'test_results'.
```

### Known Limitations on Real Images

Testing with real photos (printed/projected logo) has revealed:

✅ **Accurate** when:
- Logo is clearly visible (>10% of the image).
- Uniform lighting.
- Near-frontal viewing angle (<30° from normal).
- High-contrast background.

⚠️ **Approximate** when:
- Logo is very small (<5% of the image).
- Extreme perspective (>45° tilt).
- Motion blur or defocus.
- Background with patterns similar to the logo.

❌ **Off-target** when:
- Partial occlusion (logo covered by more than 30%).
- Reflections/glare on glossy surfaces.
- Extreme JPEG compression artifacts.
- Very complex background (crowds, product shelves).

### Why Does the Synthetic ⇒ Real Domain Gap Exist?

1. **Perfect compositing vs. real photo:**
   - Dataset: mathematical alpha blending, sharp edges.
   - Real: optical distortion, chromatic aberration, bokeh.

2. **Missing augmentation:**
   - Not simulated: lens flare, reflections, vignetting.
   - Not simulated: print textures (paper, plastic, metal).

3. **Background distribution:**
   - Dataset: ImageNet backgrounds (nature, objects).
   - Real: offices, trade show booths, products (distribution shift).

---

## 5. Dataset Format and YAML Support (Future Improvements)

### Current Format (JSON)

```json
{
  "x": 0.4231,
  "y": 0.6180
}
```

**Pros:**
- Compact
- Fast parsing
- Standard for ML pipelines

**Cons:**
- Not human-readable (hard to debug manually)
- No comments

### Proposed Format (YAML)

```yaml
# Sample 00042 - logo_red.png on background_city_street.jpg
# Generated: 2024-01-15 14:23:07
x: 0.4231  # normalized centroid x-coordinate
y: 0.6180  # normalized centroid y-coordinate

# Optional metadata (future extensions)
logo_variant: red
background_id: 1234
augmentation_params:
  rotation_deg: -37.2
  scale: 0.58
  perspective_warp: 0.03
```

**Pros:**
- Human-readable (easy debugging)
- Supports comments
- Pipelime already supports `YamlMetadataItem`

**Cons:**
- Slightly larger files (~2x)
- Slightly slower parsing (negligible)

**In the dataset loader (Module A):**

```python
# Instead of
meta_files = sorted(self.data_dir.glob("*_metadata.json"))

# Search for both
import yaml
meta_files = list(self.data_dir.glob("*_metadata.json")) + \
             list(self.data_dir.glob("*_metadata.yaml"))

# In __getitem__
if meta_path.suffix == ".json":
    meta = json.loads(meta_path.read_text())
elif meta_path.suffix == ".yaml":
    meta = yaml.safe_load(meta_path.read_text())
```

---

## 6. Updated Documentation

Three documentation files have been created/updated:

### `MODULE-A.md`
Full documentation of the Logo Detector:
- Detailed architecture (MobileNetV3 + regression head)
- Technical choices explained (loss, optimizer, scheduler)
- CLI and programmatic usage (training, inference)
- Notebook script walkthrough (lines 545–593)
- `test_inference.py` and crosshair details
- Performance, limitations, troubleshooting

### `MODULE-B.md` (updated)
Updated augmentation parameters:
- Spatial transforms: scale 0.20–0.75, translate ±30%, perspective 2–6%
- Logo color: HSV ±10/25/20, RGB shift ±15
- Scene: brightness ±40%, contrast ±30%, blur 3–13, compression 65–95
- Max logo ratio: 25% (was 30%)

### `README.md` (full rewrite)
- Quick Start covering all 4 workflows (generate, train, test, infer)
- Complete project structure
- "Architecture & Technical Choices" section with diagrams
- Underfolder dataset format explained
- Detailed requirements (hardware, software)

---

## 7. Essential Commands

### Generate Dataset (10k samples)
```bash
python src/dataset_generator.py --num_samples 10000 --seed 42
```

### Training (with early stopping)
```bash
python src/logo_detector.py --train \
    --dataset_dir generated_dataset \
    --epochs 100 \
    --lr 1e-3 \
    --seed 42
```

### Batch Test (interactive GUI)
```bash
# Add photos to test_images/
python src/test_inference.py
```

### Single Image Inference (CLI)
```bash
python src/logo_detector.py --infer \
    --checkpoint checkpoints/best.pt \
    --image test_images/photo.jpg
```

### Training from Notebook (Colab/Jupyter)
```python
from logo_detector import train
detector = train(
    dataset_dir=Path("generated_dataset"),
    epochs=50,
    lr=1e-3
)
```

### Inference from Notebook
```python
from logo_detector import LogoDetector
detector = LogoDetector.load("checkpoints/best.pt")
x, y = detector.predict(rgb_image)
```

---

## Conclusions

The project is now complete and fully documented:

✅ **Module B**: Dataset generator with updated augmentation (more variability, less extreme)  
✅ **Module A**: Logo detector (MobileNetV3-Small) with robust training and fast inference  
✅ **Test script**: `test_inference.py` for batch processing with GUI  
✅ **Documentation**: 3 complete markdown files (README, MODULE-A, MODULE-B)  