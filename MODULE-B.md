# Module B: Dataset Generator

Synthetic training-data generator.
Composites the (eyecan) logo onto random background images, applies geometric
and photometric augmentation, and saves the result as a
[pipelime Underfolder](https://github.com/eyecan-ai/pipelime-python) dataset.

For the backgrounds database, the [MIT Indoor Scenes](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019?resource=download) was used.
>The database contains 67 Indoor categories, and a total of 15620 images. The number of images varies across categories, but there are at least 100 images per category. All images are in jpg format. The images provided here are for research purposes only.
>For reproducibility and fast testing, a small pool of this database of images is present in `sample_background/` (~1000 images). Otherwhise the full dataset is available on the page above.
---
 
## Usage
 
```bash
# All defaults (5000 samples, 640×480)
python src/dataset_generator.py
 
# Custom run
python src/dataset_generator.py \
    --bg_dir      backgrounds/Images \
    --logos_dir   logos \
    --num_samples 5000 \
    --img_size    640 480 \
    --output      generated_dataset \
    --batch_size  200 \
    --num_workers 0 \
    --seed        42
 
# Full help
python src/dataset_generator.py --help
```
 
### Arguments
 
| Argument | Default | Description |
|---|---|---|
| `--bg_dir` | `backgrounds/Images` | Background images directory (recursive). |
| `--logos_dir` | `logos` | Logo PNGs — any colour variant, RGBA or RGB. |
| `--num_samples` | `5000` | Total number of images to generate. |
| `--img_size W H` | `640 480` | Output resolution in pixels. |
| `--output` | `generated_dataset` | Destination Underfolder directory. |
| `--batch_size` | `200` | Samples per write batch — controls peak RAM. |
| `--num_workers` | `0` | Parallel writer processes (`0` = single-threaded). |
| `--seed` | *(none)* | Integer seed for reproducibility. |
 
---
 
## Output format — pipelime Underfolder
 
```
generated_dataset/
  data/
    00000_image.jpg        ← composited RGB scene
    00000_metadata.json    ← {"x": 0.4231, "y": 0.6180}
    00001_image.jpg
    ...
```
 
`x` and `y` are the **normalised coordinates** of the logo centroid —
divided by image width and height — so they are always in `[0.0, 1.0]`
regardless of resolution.
 
Loading the dataset in Python:
 
```python
from pipelime.sequences import SamplesSequence
 
dataset  = SamplesSequence.from_underfolder("generated_dataset")
sample   = dataset[0]
image    = sample["image"]()      # numpy uint8 RGB (H×W×3)
metadata = sample["metadata"]()   # {"x": float, "y": float}
```
 
---
 
## Generation pipeline
 
```
background image ──┐
                    ├──► alpha composite ──► scene photometric ──► save
logo (RGBA)        │
  │                │
  ▼                │
spatial aug        │
  │ (keypoint      │
  │  tracked)      │
  ▼                │
logo color aug ────┘
  │
soft shadow ──► applied to background before composite
```
 
### Step 1 — Background
A random image from `--bg_dir` is loaded with OpenCV (BGR) and immediately
converted to RGB. It is resized to the target resolution.
 
### Step 2 — Logo sizing
A random PNG from `--logos_dir` is scaled so its longest side is at most
**25 % of the shorter canvas dimension**, keeping the logo visible without
dominating the scene.
 
### Step 3 — Spatial augmentation
The logo is centred on a blank RGBA canvas the same size as the output.
Two transforms are applied to the canvas while **tracking a keypoint at the
logo centroid**:
 
| Transform | Parameters | Effect |
|---|---|---|
| `Affine` | scale 0.20–0.75, rotate ±180°, translate ±30 % | Size, orientation, position diversity. |
| `Perspective` | warp 2–6 %, p=0.4 | Logo seen from an angle. |
 
If the centroid exits the canvas the sample is discarded and regenerated
(rejection sampling). The max attempt budget is `num_samples × 10`.
 
### Step 4 — Logo-only colour augmentation
Applied to the warped logo **before compositing**, only on pixels where
`alpha > 0`. The background is not touched.
 
| Transform | Parameters | Rationale |
|---|---|---|
| `HueSaturationValue` | hue ±10, sat ±25, val ±20, p=0.7 | Simulates different print materials, coloured lighting on the logo surface. |
| `RGBShift` | ±15 per channel, p=0.35 | Subtle independent per-channel offset for further colour diversity. |
 
**Why only on the logo?** Applying colour transforms to the full scene after
compositing shifts the hue of the background in ways that vary unpredictably
across albumentations versions. Operating on the logo alone — before blending —
avoids this issue entirely and produces clean, realistic results.
 
### Step 5 — Soft shadow
Before compositing, the background is darkened beneath the logo using a
blurred, offset copy of the logo's alpha mask. This gives the logo a subtle
physical anchor, making it look less like a sticker pasted on top.
 
| Parameter | Value | Effect |
|---|---|---|
| `strength` | 0.12 | Maximum 12 % darkening. |
| `blur_ksize` | 25 | Soft, diffuse penumbra. |
| `shift` | (3, 4) px | Simulates light from top-left. |
 
### Step 6 — Alpha compositing
The warped, colour-augmented RGBA logo is blended onto the shadow-modified
background in `float32` to avoid integer rounding artefacts:
 
```
output = background × (1 − α) + logo_rgb × α
```
 
### Step 7 — Scene-level photometric augmentation
Applied to the **final composited image** so background and logo are
perturbed together, simulating global lighting conditions.
 
| Transform | Parameters | Effect |
|---|---|---|
| `RandomBrightnessContrast` | ±40 % brightness, ±30 % contrast, p=0.75 | Dark scenes, overexposed scenes, midtone variation. |
| `GaussianBlur` | kernel 3–13, p=0.18 | Defocus, motion blur, surveillance-camera quality. |
| `ImageCompression` | quality 65–95, p=0.25 | JPEG artefacts common in real photos and video frames. |
 
The wide brightness range (`±0.35`) is intentional: it produces realistically
dark (poorly lit rooms) and overexposed (outdoor sun) scenes without distorting
colour relationships between logo and background.
 
**Why `RandomBrightnessContrast` and not `ColorJitter`?**
`RandomBrightnessContrast` applies the same scalar to all three channels, so
it cannot shift hue. `ColorJitter` and `RandomGamma` modify R, G, B
independently with behaviour that differs across albumentations versions,
producing strong green or magenta casts on the whole scene.
 
---
 
## Memory management
 
Generating 10 000 samples at 640×480 and holding them all in RAM before
writing would require 10+ GB — enough to trigger an OOM kill.
pipelime `Item` objects keep an uncompressed copy of the pixel data until
they are serialised to disk.
 
The generator writes in small batches:
1. Generate `--batch_size` samples in memory.
2. Write the batch to a temporary sub-directory.
3. Rename the files into the main `data/` folder with the correct global
   index offset.
4. Drop the Python list (`batch = []`) — all `Sample` objects are garbage-
   collected and RAM is freed before the next batch begins.
 
Peak RAM at 640×480:
 
| `--batch_size` | Peak RAM |
|---|---|
| 50 | ~44 MB |
| **200** *(default)* | **~175 MB** |
| 500 | ~440 MB |
| 1 000 | ~880 MB |
 
---
 
## Importing from a notebook
 
```python
from pathlib import Path
import sys
sys.path.insert(0, "src")
from dataset_generator import generate_dataset
 
generate_dataset(
    bg_dir      = Path("backgrounds/Images"),
    logos_dir   = Path("logos"),
    num_samples = 2000,
    img_size    = (640, 480),
    output      = Path("generated_dataset"),
    batch_size  = 200,
    num_workers = 0,
    seed        = 0,
)
```
 
---
 
## Notes on `--num_workers`
 
pipelime's Underfolder writer uses Python multiprocessing. Each worker forks
the process and duplicates the full batch in RAM. Keep `num_workers=0`
(single-threaded) unless you have several GB of free RAM per batch.
 
---
 
## Notes on pipelime 2.2.0 / Python 3.12 compatibility
 
`PipelimeCommand` with the choixe/typer CLI was the original design.
On Python 3.12 with pipelime 2.2.0, unset fields are returned as raw
`FieldInfo` objects rather than their declared default values, making the
command unusable without passing every argument. The CLI is therefore
implemented with plain `argparse`; pipelime is used only for
`Sample`, `SamplesSequence`, and `to_underfolder()`, where it works correctly.