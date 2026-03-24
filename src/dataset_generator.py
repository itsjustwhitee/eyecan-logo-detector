"""
Synthetic dataset generator for the Eyecan Logo Detection challenge.

Each sample is a background image with one logo composited onto it.
The ground-truth label is the normalised (x, y) centroid of the logo,
tracked through every geometric transformation so it is always pixel-accurate.

Output: pipelime Underfolder directory
    <output>/
      data/
        00000_image.jpg       # RGB, composited scene
        00000_metadata.json   # {"x": 0.42, "y": 0.31}  — values in [0, 1]
        ...

Usage:
    python src/dataset_generator.py --help
    python src/dataset_generator.py                          # all defaults
    python src/dataset_generator.py --num_samples 1000 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

import pipelime.items as pli
from pipelime.sequences import SamplesSequence

# pipelime 2.x keeps Sample in .sequences.sample but also re-exports it from
# .sequences in some builds — we try the canonical path first.
try:
    from pipelime.sequences.sample import Sample  # type: ignore[import]
except ImportError:
    from pipelime.sequences import Sample  # type: ignore[import]


# ---------------------------------------------------------------------------
# pipelime item class resolution
# ---------------------------------------------------------------------------
# Item class names vary slightly across pipelime minor versions.
# _get_cls picks the first name that actually exists in the module so the
# generator works without pinning a specific sub-version.

def _get_cls(module, candidates: List[str]) -> type:
    for name in candidates:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        f"None of {candidates} found in {module.__name__}. "
        "Check your pipelime-python version."
    )


_ImageItemCls = _get_cls(pli, ["JpegImageItem", "PngImageItem", "NumpyImageItem", "ImageItem"])
_MetaItemCls  = _get_cls(pli, ["JsonMetadataItem", "YamlMetadataItem", "MetadataItem"])


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def _build_spatial_transform() -> A.Compose:
    """Geometric pipeline applied to the RGBA logo canvas before compositing.

    Affine handles scale, rotation and translation in one pass (always on).
    Perspective adds a mild projective warp 50 % of the time, simulating
    the logo seen from an angle rather than head-on.

    KeypointParams propagates the logo centroid through every transform so
    the saved (x, y) label always matches the warped position.
    Keypoints that land outside the canvas are dropped (remove_invisible=True);
    _generate_sample treats that as a failed attempt and retries.
    """
    return A.Compose(
        [
            A.Affine(
                scale=(0.15, 0.85),          # logo occupies 15–85 % of the short edge
                rotate=(-180, 180),          # full rotation range
                translate_percent=(-0.35, 0.35),
                p=1.0,
            ),
            A.Perspective(scale=(0.02, 0.07), p=0.5),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def _build_photometric_transform() -> A.Compose:
    """Photometric pipeline applied to the final composited RGB image.

    Only brightness/contrast and blur are used — deliberately no hue or
    saturation transforms.  ColorJitter and RandomGamma modify channels
    independently in ways that vary across albumentations versions, causing
    strong green/magenta casts on the whole scene.
    RandomBrightnessContrast applies the same scalar to all three channels,
    so it can never shift the hue.
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.10),
        ]
    )


# ---------------------------------------------------------------------------
# Asset loaders
# ---------------------------------------------------------------------------

def _load_logos(logos_dir: Path) -> List[np.ndarray]:
    """Return all PNG logos in *logos_dir* as RGBA uint8 arrays (RGB order)."""
    logos = []
    for path in logos_dir.glob("*.png"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            logos.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logos.append(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]))
    if not logos:
        raise FileNotFoundError(f"No PNG logos found in {logos_dir}")
    return logos


def _load_bg_paths(bg_dir: Path) -> List[Path]:
    """Recursively collect every JPEG/PNG/BMP under *bg_dir*."""
    exts  = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.bmp")
    paths = [p for ext in exts for p in bg_dir.rglob(ext)]
    if not paths:
        raise FileNotFoundError(f"No background images found in {bg_dir}")
    return paths


# ---------------------------------------------------------------------------
# Single-sample generation
# ---------------------------------------------------------------------------

def _generate_sample(
    logos: List[np.ndarray],
    bg_paths: List[Path],
    img_size: Tuple[int, int],
    spatial_tf: A.Compose,
    photo_tf: A.Compose,
    max_logo_ratio: float = 0.30,
) -> Optional[Sample]:
    """Build one composited sample.  Returns None on rejection (caller retries).

    Pipeline:
      1. Load and resize a random background to img_size.
      2. Pick a random logo; scale it so its longest side = max_logo_ratio * min(W,H).
      3. Centre the logo on a blank RGBA canvas the size of the output image.
      4. Apply the spatial transform to the canvas while tracking the centroid keypoint.
         If the keypoint exits the canvas → return None.
      5. Alpha-blend the warped logo onto the background (float32 for precision).
      6. Apply photometric augmentation to the composited RGB image.
      7. Wrap image + normalised centroid into a pipelime Sample.

    All images are kept in RGB throughout.  pipelime's JpegImageItem uses PIL
    internally and expects RGB — do not convert to BGR before handing off.
    """
    W, H = img_size

    # Background
    bg_bgr = cv2.imread(str(random.choice(bg_paths)))
    if bg_bgr is None:
        return None
    bg_rgb = cv2.cvtColor(cv2.resize(bg_bgr, (W, H)), cv2.COLOR_BGR2RGB)

    # Logo — fit inside max_logo_ratio * shortest canvas edge
    logo_rgba = random.choice(logos).copy()
    lh, lw    = logo_rgba.shape[:2]
    max_px    = int(min(W, H) * max_logo_ratio)
    scale     = max_px / max(lh, lw)
    logo_rgba = cv2.resize(logo_rgba,
                           (max(1, int(lw * scale)), max(1, int(lh * scale))))
    lh, lw = logo_rgba.shape[:2]

    # Place logo centred on a full-size transparent canvas
    canvas      = np.zeros((H, W, 4), dtype=np.uint8)
    y0, x0      = (H - lh) // 2, (W - lw) // 2
    canvas[y0:y0+lh, x0:x0+lw] = logo_rgba

    # Spatial transform — centroid keypoint travels with the image
    cx, cy = float(x0 + lw / 2), float(y0 + lh / 2)
    result = spatial_tf(image=canvas, keypoints=[(cx, cy)])
    if not result["keypoints"]:
        return None
    fx, fy = result["keypoints"][0]
    if not (0 <= fx < W and 0 <= fy < H):
        return None

    # Alpha-blend warped logo over background
    warped     = result["image"]
    alpha      = (warped[:, :, 3] / 255.0)[..., np.newaxis]
    composited = (bg_rgb.astype(np.float32) * (1.0 - alpha)
                  + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    # Photometric augmentation on the final scene
    final_img = photo_tf(image=composited)["image"]

    return Sample({
        "image":    _ImageItemCls(final_img),
        "metadata": _MetaItemCls({"x": float(fx / W), "y": float(fy / H)}),
    })


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(
    bg_dir: Path,
    logos_dir: Path,
    num_samples: int,
    img_size: Tuple[int, int],
    output: Path,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> None:
    """Generate *num_samples* composited images and write a pipelime Underfolder."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"[pipelime] image item    : {_ImageItemCls.__name__}")
    print(f"[pipelime] metadata item : {_MetaItemCls.__name__}")
    print(f"[config]   bg_dir        : {bg_dir}")
    print(f"[config]   logos_dir     : {logos_dir}")
    print(f"[config]   num_samples   : {num_samples}")
    print(f"[config]   img_size      : {img_size}  (W x H)")
    print(f"[config]   output        : {output}")
    print(f"[config]   num_workers   : {num_workers}")
    print(f"[config]   seed          : {seed}")

    print("\nLoading logos …")
    logos = _load_logos(logos_dir)
    print(f"  -> {len(logos)} variant(s) loaded.")

    print("Indexing backgrounds …")
    bg_paths = _load_bg_paths(bg_dir)
    print(f"  -> {len(bg_paths)} image(s) found.")

    spatial_tf = _build_spatial_transform()
    photo_tf   = _build_photometric_transform()

    samples: List[Sample] = []
    max_attempts = num_samples * 10  # generous budget for rejection-sampling

    print(f"\nGenerating {num_samples} samples …")
    with tqdm(total=num_samples, unit="sample") as pbar:
        attempts = 0
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            s = _generate_sample(logos, bg_paths, img_size, spatial_tf, photo_tf)
            if s is not None:
                samples.append(s)
                pbar.update(1)

    if len(samples) < num_samples:
        print(f"\n[WARNING] Only {len(samples)}/{num_samples} samples generated "
              f"after {attempts} attempts — check your assets.")

    print(f"\nWriting Underfolder to: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # SamplesSequence.from_list is standard in 2.x; bare constructor as fallback.
    try:
        seq = SamplesSequence.from_list(samples)
    except AttributeError:
        seq = SamplesSequence(samples)  # type: ignore[arg-type]

    seq.to_underfolder(str(output), exists_ok=True).run(num_workers=num_workers)
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# We use plain argparse rather than pipelime's PipelimeCommand/choixe CLI.
# In pipelime 2.2.0 on Python 3.12, unset PipelimeCommand fields are returned
# as raw FieldInfo objects instead of their declared defaults, making the
# command unusable without passing every argument explicitly.

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic Underfolder dataset for Eyecan logo detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bg_dir",      type=Path, default=Path("backgrounds/Images"),
                   help="Directory with background JPEG/PNG images (searched recursively).")
    p.add_argument("--logos_dir",   type=Path, default=Path("logos"),
                   help="Directory with logo PNGs (RGBA or RGB, any colour variant).")
    p.add_argument("--num_samples", type=int,  default=5000,
                   help="Number of samples to generate.")
    p.add_argument("--img_size",    type=int,  nargs=2, default=[640, 480],
                   metavar=("W", "H"), help="Output resolution.")
    p.add_argument("--output",      type=Path, default=Path("generated_dataset"),
                   help="Destination Underfolder directory.")
    p.add_argument("--num_workers", type=int,  default=0,
                   help="Parallel writer processes. 0 = single-threaded (safe default).")
    p.add_argument("--seed",        type=int,  default=None,
                   help="RNG seed for reproducibility.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        bg_dir      = args.bg_dir,
        logos_dir   = args.logos_dir,
        num_samples = args.num_samples,
        img_size    = tuple(args.img_size),
        output      = args.output,
        num_workers = args.num_workers,
        seed        = args.seed,
    )