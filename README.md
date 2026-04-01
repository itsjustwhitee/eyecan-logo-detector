# Eyecan AI - Logo Detection Challenge

This repository contains the full implementation for the [eyecan.ai](eyecan.ai) Logo Detector challenge accomplished by [me](https://justwhitee.org/).
The project is divided into two main components:

- **Module A**: Deep Learning-based Detector (Training and Inference).
- **Module B**: Synthetic Dataset Generator using Pipelime and Albumentations.

## Quick Start

> Here some examples follows.
### 1. Generate Training Dataset (Module B)
```bash
python src/dataset_generator.py \
    --bg_dir backgrounds/Images \
    --logos_dir logos \
    --num_samples 20000 \
    --seed 42
```
Output: `generated_dataset/` with 20k synthetic images + metadata.
 
### 2. Train Logo Detector (Module A)
```bash
python src/logo_detector.py --train \
    --dataset_dir generated_dataset \
    --epochs 100 \
    --seed 42
```
Output: `checkpoints/best.pt` (typically converges in 30-45 epochs).
 
### 3. Test on Real Images
```bash
# Put your photos in test_images/
python src/test_inference.py
```
Output: Annotated images in `test_results/` + interactive GUI
 
### 4. Single-Image Inference (CLI)
```bash
python src/logo_detector.py --infer \
    --checkpoint checkpoints/best.pt \
    --image test_images/photo.jpg
```
 
For detailed instructions:
- 📘 [**Module A Documentation (Logo Detector)**](MODULE-A.md)
- 📗 [**Module B Documentation (Dataset Generator)**](MODULE-B.md)

## Tech Stack
- **Language**: Python 3.12
- **Core Library**: [pipelime-python 2.2.0](https://github.com/eyecan-ai/pipelime-python)
- **Language**: Python 3.12
- **Dataset Framework**: [pipelime-python 2.2.0](https://github.com/eyecan-ai/pipelime-python)
- **Augmentation**: Albumentations 1.4+
- **Vision**: OpenCV 4.8+
- **Deep Learning**: PyTorch 2.0+, TorchVision 0.15+
- **Model Architecture**: MobileNetV3-Small (ImageNet-pretrained)
- **Hardware**: NVIDIA GPU (recommended), CPU fallback, AMD ROCm (limited support)

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
 
1. **Efficiency**: 
   - Fast training
   - Fast inference (~10 ms/image on CPU)
   - Small model size (<20 MB checkpoint)
 
2. **Transfer Learning**: 
   - ImageNet weights provide excellent low-level features (edges, textures, colors)
   - Drastically reduces training time vs. random init
   - Proven backbone used in production mobile apps
 
3. **Deployment-Ready**:
   - Depthwise separable convolutions ⇒ low FLOPs
   - Suitable for edge devices (browser, mobile, embedded)
 
**Loss Function: Smooth L1 (Huber)**
 
```python
criterion = nn.SmoothL1Loss(beta=0.01)
```
 
**Rationale:**
- Combines MSE precision near zero with MAE robustness to outliers
- `beta=0.01`: Tight transition ⇒ accurate centroid convergence
- More stable than pure MSE on synthetic datasets with occasional hard negatives
 
**Training Schedule:**
 
| Component | Configuration | Rationale |
|---|---|---|
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 | Prevents weight explosion |
| LR Scheduler | CosineAnnealing, T_max=epochs, eta_min=1e-5 | Smooth decay, no manual tuning |
| Gradient Clipping | max_norm=1.0 | Stabilizes training on outlier batches |
| Early Stopping | patience=10 on val_loss | Prevents overfitting, typical stop @ 30-45 epochs |
| Train/Val Split | 85% / 15% | Validation set: ~1,5k samples on 10k dataset |
 
**Performance (Validation Set, 10k training samples):**
- Best val loss (SmoothL1): 0.003-0.005
- Accuracy (±10% of short edge): 70-85%
- Training time (Colab T4): 15-25 min
 
### Module B: Dataset Generator
 
**Pipeline:** Background + Logo Compositing with Tracked Centroid
 
```
1. Load random background (among the backgrounds' dataset) → resize to (W, H)
2. Load random logo → scale to 25% of min(W, H)
3. Center logo on transparent canvas (H×W×4 RGBA)
4. Apply spatial transforms (Affine + Perspective) while tracking centroid keypoint
   └─ If centroid exits canvas ⇒ reject & retry
5. Apply color augmentation ONLY to logo pixels (HSV + RGB shift)
6. Generate soft shadow beneath logo on background
7. Alpha-composite logo onto background
8. Apply scene-level photometric augmentation (brightness, blur, compression)
9. Save as JPEG + JSON metadata with normalized (x, y)
```
 
**Key Design Decisions:**
 
1. **Logo-only color augmentation** (before compositing):
   - Prevents hue shifts on background (unnatura/unreal)
   - Simulates different print materials and lighting
   - Parameters: HSV (hue ±10°, sat ±25, val ±20), RGBShift (±15 per channel)
 
2. **Soft shadow** (simulates physical depth):
   - Blurred, offset alpha mask → darken background
   - Makes logo look anchored, not pasted
   - Parameters: strength=0.12, blur_ksize=25, shift=(3,4)
 
3. **Scene-level augmentation** (after compositing):
   - Global brightness/contrast: ±40%/±30% (simulates day/night, indoor/outdoor)
   - Gaussian blur: kernel 3-13 (defocus, motion blur)
   - JPEG compression: quality 65-95 (real-world artefacts)
 
4. **Spatial augmentation parameters** (updated for better variability):
   - **Scale:** 0.20–0.75 (was 0.15–0.85) → tighter range, less tiny logos
   - **Rotation:** ±180° (full circle)
   - **Translation:** ±30% (was ±35%) → keeps logo more centered
   - **Perspective:** 2-6%, p=0.4 (was 2-7%, p=0.5) → less extreme warping
 
5. **Memory batching:**
   - Generates in chunks of `batch_size=200` (default)
   - Peak RAM: ~175 MB @ 640×480 (vs. 10+ GB if all in memory)
 
**Augmentation Philosophy:**
 
- **Diversity over realism**: Training on hard synthetic examples (extreme rotations, heavy blur) 
  makes the model robust to easier real-world cases
- **Synthetic → Real domain gap**: Model may struggle on real photos due to:
  - Lens distortion, chromatic aberration (not simulated)
  - Reflection, glare, occlusions (partially simulated via brightness/blur)
  - Print quality variations (partially covered by color augmentation)
 
### Data Format: Pipelime Underfolder
 
**Structure:**
```
generated_dataset/
  data/
    00000_image.jpg       # RGB composited scene (640×480 default)
    00000_metadata.json   # {"x": 0.4231, "y": 0.6180}
    00001_image.jpeg      # Supports .jpg, .jpeg, .png, .bmp (case-insensitive)
    00001_metadata.json
```
 
**Metadata format:**
```json
{
  "x": 0.4231,  // normalized x: logo_centroid_px / image_width
  "y": 0.6180   // normalized y: logo_centroid_px / image_height
}
```
> x, y ∈ [0, 1]
 
**Normalization rationale:**
- Resolution-agnostic: Same metadata works for 640×480, 1920×1080, etc.
- Model predicts normalized coords → multiply by W, H at inference
- Consistent with many object detection frameworks (YOLO, Faster R-CNN)

## Project Structure
 
```
eyecan_challenge/
├── src/
│   ├── dataset_generator.py    # Module B: Synthetic data generation
│   ├── logo_detector.py         # Module A: CNN training & inference
│   └── test_inference.py        # Batch testing with GUI visualization
├── logos/                       # PNG logo variants (RGBA or RGB)
│   ├── logo_red.png
│   ├── logo_black.png
│   └── logo_white.png
├── backgrounds/
│   └── Images/                  # Background images (JPEG/PNG/BMP, recursive)
├── generated_dataset/           # Output from Module B (Underfolder format)
│   └── data/
│       ├── 00000_image.jpg
│       ├── 00000_metadata.json
│       └── ...
├── checkpoints/                 # Output from Module A
│   ├── best.pt                  # Best validation loss checkpoint
│   ├── last.pt                  # Last epoch checkpoint
│   └── training_log.json        # Training history
├── test_images/                 # Real-world test photos (your own)
├── test_results/                # Annotated predictions from test_inference.py
├── MODULE-A.md                  # Complete Module A documentation
├── MODULE-B.md         # Complete Module B documentation
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```
 
---
 
## Requirements
 
### Core Dependencies
 
```bash
# Python version
python >= 3.10
 
# Module B (Dataset Generation)
pipelime-python == 2.2.0
albumentations >= 1.3
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
> Please, note that if you want to use a GPU, you need to install the pytorch version distributed by your manufacturer (NVIDIA/AMD/Intel).
 
### Hardware Requirements
 
**Training (Module A):**
- Recommended: NVIDIA/AMD GPU with 4GB+ VRAM
- Fallback: CPU (slower)
> Note that AMD GPUs (ROCm) can have compatibility issues on WSL2 (⇒ force CPU mode).
 
**Dataset Generation (Module B):**
- CPU-only (no GPU needed)
- RAM: ~200 MB per batch (at default `batch_size=200`)
 
**Inference:**
- CPU sufficient (~10 ms/image on modern laptop)
- GPU optional for batch processing
 
---