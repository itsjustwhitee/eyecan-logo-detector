# Eyecan AI - Logo Detection Challenge

This repository contains the full implementation for the [eyecan.ai](eyecan.ai) Logo Detector challenge accomplished by [me](https://justwhitee.org/).
The project is divided into two main components:

- **Module A**: Deep Learning-based Detector (Training and Inference).
- **Module B**: Synthetic Dataset Generator using Pipelime and Albumentations.

A **pre-trained checkpoint** (`checkpoints/best_20k.pt`, trained on 20k images) and a **sample background pool** (`sample_background/`, ~1000 images) are included so you can run inference or quick experiments immediately without downloading external datasets or training from scratch.

---

## Quick Start

### 0. Use the pre-trained checkpoint (no training needed)

```bash
# Put your photos in test_images/ and run directly
python src/test_inference.py --checkpoint checkpoints/best_20k.pt
```

### 1. Generate Training Dataset (Module B)

```bash
# Quick test with bundled background pool (~1000 images)
python src/dataset_generator.py \
    --bg_dir sample_background \
    --num_samples 5000 \
    --seed 42

# Full run with the MIT Indoor Scenes dataset
python src/dataset_generator.py \
    --bg_dir backgrounds/Images \
    --num_samples 20000 \
    --seed 42
```

Output: `generated_dataset/` with synthetic images + metadata.

### 2. Train Logo Detector (Module A)

```bash
python src/logo_detector.py --train \
    --dataset_dir generated_dataset \
    --epochs 100 \
    --seed 42
```

Output: `checkpoints/best.pt` (typically converges in 30–45 epochs).

### 3. Test on Real Images

```bash
# Default: reads test_images/, writes to test_results/, uses checkpoints/best.pt
python src/test_inference.py

# Explicit arguments
python src/test_inference.py \
    --checkpoint checkpoints/best_20k.pt \
    --input      test_images \
    --output     test_results
```

Output: Annotated images in `test_results/` + interactive GUI (when a display is available).

### 4. Single-Image Inference (CLI)

```bash
python src/logo_detector.py --infer \
    --checkpoint checkpoints/best_20k.pt \
    --image test_images/photo.jpg
```

For detailed instructions:
- 📘 [**Module A Documentation (Logo Detector)**](MODULE-A.md)
- 📗 [**Module B Documentation (Dataset Generator)**](MODULE-B.md)

---

## Tech Stack

- **Language**: Python 3.12
- **Dataset Framework**: [pipelime-python 2.2.0](https://github.com/eyecan-ai/pipelime-python)
- **Augmentation**: Albumentations 1.4+
- **Vision**: OpenCV 4.8+
- **Deep Learning**: PyTorch 2.0+, TorchVision 0.15+
- **Model Architecture**: MobileNetV3-Small (ImageNet-pretrained)
- **Hardware**: NVIDIA GPU (recommended), CPU fallback, AMD ROCm (limited support)

---

## Architecture & Technical Choices

### Module A: Logo Detector

**Model:** MobileNetV3-Small + Custom Regression Head

```
Input RGB (H×W×3)
    ↓ Resize to 224×224
    ↓ ImageNet normalization
MobileNetV3-Small Backbone (ImageNet pretrained)
    ↓ Global Average Pooling
    ↓ Feature vector (576-d)
Regression Head:
    576 → Linear(128) → Hardswish → Dropout(0.3) → Linear(2) → Sigmoid
    ↓
Output: (x, y) normalized coordinates ∈ [0, 1]
```

**Why MobileNetV3-Small?**

1. **Efficiency**: Fast training, fast inference (~10 ms/image on CPU), small checkpoint (<20 MB).
2. **Transfer Learning**: ImageNet weights provide excellent low-level features (edges, textures, colors), drastically reducing training time.
3. **Deployment-Ready**: Depthwise separable convolutions ⇒ low FLOPs, suitable for edge devices.

**Loss Function: Smooth L1 (Huber)**

```python
criterion = nn.SmoothL1Loss(beta=0.01)
```

Combines MSE precision near zero with MAE robustness to outliers. `beta=0.01` places the transition at ~6 px @ 640×480.

**Training Schedule:**

| Component | Configuration | Rationale |
|---|---|---|
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 | Prevents weight explosion. |
| LR Scheduler | CosineAnnealing, T_max=epochs, eta_min=lr*0.01 | Smooth decay, no manual tuning. |
| Gradient Clipping | max_norm=1.0 | Stabilizes training on outlier batches. |
| Early Stopping | patience=10 on val_loss | Prevents overfitting, typical stop @ 30–45 epochs. |
| Train/Val Split | 85% / 15% | Validation set: ~1.5k samples on 10k dataset. |

**Performance (Validation Set, 10k training samples):**
- Best val loss (SmoothL1): 0.003–0.005
- Accuracy (±10% of short edge): 70–85%
- Training time (Colab T4): 15–25 min

### Module B: Dataset Generator

**Pipeline:** Background + Logo Compositing with Tracked Centroid

```
1. Load random background → resize to (W, H)
2. Load random logo → scale to 50% of min(W, H)
3. Center logo on transparent canvas (H×W×4 RGBA)
4. Apply spatial transforms (Affine + Perspective) while tracking centroid keypoint
   └─ If centroid exits canvas ⇒ reject & retry
5. Apply color augmentation ONLY to logo pixels (HSV + RGB shift)
6. Generate soft shadow beneath logo on background
7. Alpha-composite logo onto background
8. Apply scene-level photometric augmentation (brightness, blur, noise, compression)
9. Save as JPEG + JSON metadata with normalized (x, y)
```

**Key Augmentation Parameters (actual values in code):**

| Stage | Transform | Parameters |
|---|---|---|
| Spatial | Affine | scale 0.025–0.85, rotate ±180°, translate ±25%, shear ±10°, p=0.99 |
| Spatial | Perspective | warp 5–10%, p=0.5 |
| Logo color | HueSaturationValue | hue ±8, sat ±20, val ±15, p=0.65 |
| Logo color | RGBShift | ±10 per channel, p=0.30 |
| Scene | RandomBrightnessContrast | ±35% brightness, ±25% contrast, p=0.70 |
| Scene | GaussianBlur | kernel 3–8, p=0.15 |
| Scene | GaussNoise | var 1.0–7.0, p=0.10 |
| Scene | ImageCompression | quality 70–95, p=0.20 |

**Key Design Decisions:**

1. **Logo-only color augmentation** (before compositing): Prevents hue shifts on background, simulates different print materials and lighting.
2. **Soft shadow** (simulates physical depth): Blurred, offset alpha mask darkens background. Parameters: `strength=0.12, blur_ksize=25, shift=(3,4)`.
3. **Scene-level augmentation** (after compositing): Global brightness/contrast, blur, sensor noise, and JPEG compression applied together to background and logo.
4. **Memory batching**: Generates in chunks of `batch_size=200` (default). Peak RAM: ~175 MB @ 640×480.

### Data Format: Pipelime Underfolder

```
generated_dataset/
  data/
    00000_image.jpg       # RGB composited scene (640×480 default)
    00000_metadata.json   # {"x": 0.4231, "y": 0.6180}
    00001_image.jpeg      # Supports .jpg, .jpeg, .png, .bmp (case-insensitive)
    00001_metadata.json
```

Normalized coordinates: `x = logo_centroid_px / image_width`, `y = logo_centroid_px / image_height` ∈ [0, 1].

---

## Project Structure

```
eyecan_challenge/
├── src/
│   ├── dataset_generator.py    # Module B: Synthetic data generation
│   ├── logo_detector.py        # Module A: CNN training & inference
│   └── test_inference.py       # Batch testing with GUI visualization
├── logos/                      # PNG logo variants (RGBA or RGB)
│   ├── logo_red.png
│   ├── logo_black.png
│   └── logo_white.png
├── sample_background/          # Bundled pool of ~1000 background images
├── backgrounds/
│   └── Images/                 # Full MIT Indoor Scenes dataset (not included)
├── generated_dataset/          # Output from Module B (Underfolder format)
│   └── data/
│       ├── 00000_image.jpg
│       ├── 00000_metadata.json
│       └── ...
├── checkpoints/                # Checkpoints
│   ├── best_20k.pt             # ← Pre-trained checkpoint (20k samples, included)
│   ├── best.pt                 # Best checkpoint from your own training run
│   ├── last.pt                 # Last epoch checkpoint from your own training run
│   └── training_log.json       # Training history (JSON, for plotting)
├── test_images/                # Real-world test photos (your own)
├── test_results/               # Annotated predictions from test_inference.py
├── eyecal_logo_detector.ipynb  # End-to-end Jupyter/Colab notebook
├── MODULE-A.md                 # Complete Module A documentation
├── MODULE-B.md                 # Complete Module B documentation
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Requirements

### Core Dependencies

```
# Python version
python >= 3.10

# Module B (Dataset Generation)
pipelime-python == 2.2.0
albumentations >= 1.4
opencv-python >= 4.8
numpy >= 1.24
tqdm >= 4.65

# Module A (Training & Inference)
torch >= 2.0
torchvision >= 0.15
opencv-python >= 4.8
numpy >= 1.24
tqdm >= 4.65
```

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install all dependencies
pip install pipelime-python==2.2.0 albumentations opencv-python numpy tqdm torch torchvision

# Or use requirements.txt
pip install -r requirements.txt
```

> If you want to use a GPU, install the PyTorch version distributed by your manufacturer (NVIDIA/AMD/Intel).

### Hardware Requirements

**Training (Module A):**
- Recommended: NVIDIA/AMD GPU with 4GB+ VRAM.
- Fallback: CPU (slower, ~10× training time).
> AMD GPUs (ROCm) can have compatibility issues on WSL2 ⇒ force CPU mode if needed.

**Dataset Generation (Module B):**
- CPU-only (no GPU needed).
- RAM: ~200 MB per batch (at default `batch_size=200`).

**Inference:**
- CPU sufficient (~10 ms/image).
- GPU optional for batch processing.