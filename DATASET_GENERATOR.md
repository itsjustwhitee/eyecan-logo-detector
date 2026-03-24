# Eyecan Challenge 02 — Dataset Generator

Synthetic training-data generator for the **Eyecan Logo Detector** challenge.
It composites the Eyecan logo onto random background images, applies geometric
and photometric augmentation, and saves the result as a
[pipelime Underfolder](https://github.com/eyecan-ai/pipelime-python) dataset.

---

## Project structure

```
eyecan_challenge/
├── src/
│   └── dataset_generator.py   # this module
├── logos/                     # PNG logo variants (RGBA or RGB)
│   ├── logo_red.png
│   ├── logo_black.png
│   └── logo_white.png
├── backgrounds/
│   └── Images/                # any JPEG/PNG/BMP background images
└── generated_dataset/         # output (created automatically)
    └── data/
        ├── 00000_image.jpg
        ├── 00000_metadata.json
        └── ...
```

---

## Requirements

```
python >= 3.10
pipelime-python == 2.2.0
albumentations >= 1.3
opencv-python
numpy
tqdm
```

Install everything into your virtual environment:

```bash
pip install pipelime-python==2.2.0 albumentations opencv-python numpy tqdm
```

---

## Usage

```bash
# All defaults (5000 samples, 640×480, backgrounds/Images, logos/)
python src/dataset_generator.py

# Custom run
python src/dataset_generator.py \
    --bg_dir      backgrounds/Images \
    --logos_dir   logos \
    --num_samples 1000 \
    --img_size    640 480 \
    --output      generated_dataset \
    --num_workers 0 \
    --seed        42

# Full help
python src/dataset_generator.py --help
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--bg_dir` | `backgrounds/Images` | Directory of background images (searched recursively) |
| `--logos_dir` | `logos` | Directory of logo PNGs (any colour variant, RGBA or RGB) |
| `--num_samples` | `5000` | Number of images to generate |
| `--img_size W H` | `640 480` | Output resolution in pixels |
| `--output` | `generated_dataset` | Destination Underfolder directory |
| `--num_workers` | `0` | Parallel writer processes (`0` = single-threaded) |
| `--seed` | *(none)* | Integer seed for reproducibility |

---

## Output format — pipelime Underfolder

Each sample is stored as two files sharing the same numeric prefix:

```
00042_image.jpg       — composited RGB scene
00042_metadata.json   — {"x": 0.4231, "y": 0.6180}
```

`x` and `y` are the **normalised coordinates** of the logo centroid
(divided by image width and height respectively), so they are always in
`[0.0, 1.0]` regardless of resolution.

To load the dataset with pipelime:

```python
from pipelime.sequences import SamplesSequence

dataset = SamplesSequence.from_underfolder("generated_dataset")
sample  = dataset[0]

image    = sample["image"]()      # numpy uint8 RGB array
metadata = sample["metadata"]()   # dict: {"x": float, "y": float}
```

---

## How a sample is generated

```
background image  ──┐
                     ├─► composite ──► photometric aug ──► save
logo PNG (RGBA)  ──► spatial aug ──┘
                        │
                    keypoint tracked ──► normalised (x, y) ──► metadata
```

### 1 — Background
A random image from `--bg_dir` is loaded and resized to the target resolution.

### 2 — Logo sizing
A random PNG from `--logos_dir` is scaled so its longest side is at most
**30 % of the shorter canvas dimension**.  This keeps the logo clearly visible
without dominating the scene.

### 3 — Spatial augmentation (on the RGBA canvas)
The logo is first placed centred on a transparent canvas the same size as the
output image.  Two transforms are then applied:

| Transform | Parameters | Effect |
|---|---|---|
| `Affine` | scale 0.15–0.85, rotate ±180°, translate ±35 % | Size, orientation, position diversity |
| `Perspective` | warp 2–7 %, p=0.5 | Simulates logo seen from an angle |

A **keypoint at the logo centroid** travels through both transforms.
If the keypoint exits the canvas after warping the sample is discarded and
regenerated — this is the rejection-sampling loop in `_generate_sample`.

### 4 — Alpha compositing
The warped RGBA logo is blended onto the RGB background using its alpha channel:

```
output = background × (1 − α) + logo_rgb × α
```

Blending is done in `float32` to avoid integer rounding artefacts,
then cast back to `uint8`.

### 5 — Photometric augmentation (on the composited scene)

| Transform | Parameters | Effect |
|---|---|---|
| `RandomBrightnessContrast` | ±15 %, p=0.6 | Lighting variation |
| `GaussianBlur` | kernel 3–5, p=0.1 | Defocus / motion blur |

**Why not ColorJitter or RandomGamma?**
Both transforms modify R, G and B channels with independent values, causing
visible hue shifts (strong green or magenta tints) across the entire scene.
`RandomBrightnessContrast` applies the same scalar to all three channels,
so it cannot alter the hue.

---

## Importing `generate_dataset` from a notebook

The generator is exposed as a plain Python function, so it can be called
directly from a Jupyter / Colab notebook without going through the CLI:

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
    num_workers = 0,
    seed        = 0,
)
```

---

## Notes on `--num_workers`

pipelime's Underfolder writer supports parallel writing via Python
multiprocessing.  Each worker forks the process, duplicating the entire in-memory
sample list.  On machines with limited RAM this causes the OS to kill the process
(OOM).  The safe default is `0` (single-threaded).  Increase to `1` or `2` only
if you have several GB of free RAM.

---

## Notes on pipelime 2.2.0 compatibility

The original design used `PipelimeCommand` with the choixe/typer CLI layer.
On Python 3.12 with pipelime 2.2.0, unset fields in a `PipelimeCommand` are
returned as raw `FieldInfo` objects instead of their declared default values,
making the command unusable without explicitly passing every argument.

The CLI is therefore implemented with plain `argparse`.
pipelime is used only for `Sample`, `SamplesSequence`, and `to_underfolder()`,
where it works correctly.
