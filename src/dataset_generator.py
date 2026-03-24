"""
EyecanDatasetGenerator
======================
Synthetic dataset generator for the Eyecan Logo Detection challenge.
Tested with pipelime-python 2.2.0 / Python 3.12.

Usage
-----
    # Run with all defaults:
    python src/dataset_generator.py

    # Override individual fields (standard argparse syntax):
    python src/dataset_generator.py \\
        --bg_dir    backgrounds/Images \\
        --logos_dir logos \\
        --num_samples 5000 \\
        --output    generated_dataset \\
        --img_size  640 480 \\
        --num_workers 4 \\
        --seed 42

    # Show help:
    python src/dataset_generator.py --help

NOTE on pipelime 2.2.0
-----------------------
PipelimeCommand's CLI layer (choixe/typer) does not correctly resolve field
defaults in pipelime 2.2.0 on Python 3.12: unset fields are returned as raw
FieldInfo objects instead of their default values.  We therefore use plain
argparse for argument parsing and reserve pipelime exclusively for what it
does reliably: Sample construction, SamplesSequence, and to_underfolder().

Domain Randomisation
--------------------
  Geometric  : Affine (scale / rotate / translate) + Perspective warp.
               A keypoint at the logo centroid is tracked through every
               spatial transformation, so the ground-truth is pixel-accurate.
  Photometric: ColorJitter · GaussianBlur · GaussNoise · RandomGamma applied
               to the final composited image (background + logo together).

Output Underfolder structure
-----------------------------
    generated_dataset/
      data/
        00000_image.jpg
        00000_metadata.json
        ...

Metadata format (normalised coordinates in [0.0, 1.0]):
    {"x": <float>, "y": <float>}
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

# ---------------------------------------------------------------------------
# Resolve Sample class (pipelime 2.x re-exports from .sequences.sample)
# ---------------------------------------------------------------------------
try:
    from pipelime.sequences.sample import Sample  # type: ignore[import]
except ImportError:
    from pipelime.sequences import Sample  # type: ignore[import]

# ---------------------------------------------------------------------------
# Resolve Item classes — tolerant of minor pipelime version differences
# ---------------------------------------------------------------------------

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
# Augmentation pipelines — built once at module level
# ---------------------------------------------------------------------------

def _build_spatial_transform() -> A.Compose:
    """Geometric augmentations with keypoint tracking."""
    return A.Compose(
        [
            A.Affine(
                scale=(0.15, 0.85),
                rotate=(-180, 180),
                translate_percent=(-0.35, 0.35),
                p=1.0,
            ),
            A.Perspective(scale=(0.02, 0.07), p=0.5),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def _build_photometric_transform() -> A.Compose:
    """Photometric augmentations applied to the composited image.

    We intentionally avoid any transform that touches hue or saturation:
      - ColorJitter and RandomGamma both shift per-channel in unpredictable
        ways across albumentations versions, producing strong green/magenta
        tints on the whole scene.
      - RandomBrightnessContrast applies the same scalar to all channels so
        it cannot introduce colour casts.
      - GaussianBlur simulates defocus / motion blur.
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.10),
        ]
    )


# ---------------------------------------------------------------------------
# Asset loading helpers
# ---------------------------------------------------------------------------

def _load_logos(logos_dir: Path) -> List[np.ndarray]:
    """Load all PNG logos from *logos_dir* as RGBA arrays."""
    logos = []
    for path in logos_dir.glob("*.png"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            logos.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            logos.append(np.dstack([rgb, alpha]))
    if not logos:
        raise FileNotFoundError(f"No PNG logos found in {logos_dir}")
    return logos


def _load_bg_paths(bg_dir: Path) -> List[Path]:
    """Recursively collect all background image paths under *bg_dir*."""
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
    """
    Composite one logo onto one background and return a pipelime Sample.

    Returns None when the logo keypoint lands outside the canvas after the
    spatial transform; the caller retries in that case.
    """
    W, H = img_size

    # ── Background ─────────────────────────────────────────────────────────
    bg_bgr = cv2.imread(str(random.choice(bg_paths)))
    if bg_bgr is None:
        return None
    bg_rgb = cv2.cvtColor(cv2.resize(bg_bgr, (W, H)), cv2.COLOR_BGR2RGB)

    # ── Logo: pick & resize to at most max_logo_ratio of the shortest edge ─
    logo_rgba = random.choice(logos).copy()
    lh, lw    = logo_rgba.shape[:2]
    max_px    = int(min(W, H) * max_logo_ratio)
    scale     = max_px / max(lh, lw)
    logo_rgba = cv2.resize(
        logo_rgba,
        (max(1, int(lw * scale)), max(1, int(lh * scale))),
    )
    lh, lw = logo_rgba.shape[:2]

    # ── Centre logo on a blank RGBA canvas of output size ──────────────────
    canvas     = np.zeros((H, W, 4), dtype=np.uint8)
    y0, x0     = (H - lh) // 2, (W - lw) // 2
    canvas[y0 : y0 + lh, x0 : x0 + lw] = logo_rgba

    # ── Spatial transform + keypoint tracking ──────────────────────────────
    cx, cy = float(x0 + lw / 2), float(y0 + lh / 2)
    result = spatial_tf(image=canvas, keypoints=[(cx, cy)])
    if not result["keypoints"]:
        return None
    fx, fy = result["keypoints"][0]
    if not (0 <= fx < W and 0 <= fy < H):
        return None

    # ── Alpha-blend transformed logo onto background ────────────────────────
    warped     = result["image"]
    alpha      = (warped[:, :, 3] / 255.0)[..., np.newaxis]
    composited = (
        bg_rgb.astype(np.float32) * (1.0 - alpha)
        + warped[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)

    # ── Photometric augmentation ───────────────────────────────────────────
    final_img = photo_tf(image=composited)["image"]

    # ── Wrap into a pipelime Sample ────────────────────────────────────────
    # Pass RGB directly: pipelime's JpegImageItem uses PIL internally and
    # expects RGB channel order.  Do NOT convert to BGR here.
    return Sample(
        {
            "image":    _ImageItemCls(final_img),
            "metadata": _MetaItemCls({"x": float(fx / W), "y": float(fy / H)}),
        }
    )


# ---------------------------------------------------------------------------
# Dataset generation — plain function, no pipelime CLI magic needed
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
    """Generate *num_samples* composited images and write an Underfolder dataset."""

    # ── Optional seed ──────────────────────────────────────────────────────
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ── Diagnostics ────────────────────────────────────────────────────────
    print(f"[pipelime] Image item   : {_ImageItemCls.__name__}")
    print(f"[pipelime] Metadata item: {_MetaItemCls.__name__}")
    print(f"[config]   bg_dir       : {bg_dir}")
    print(f"[config]   logos_dir    : {logos_dir}")
    print(f"[config]   num_samples  : {num_samples}")
    print(f"[config]   img_size     : {img_size}")
    print(f"[config]   output       : {output}")
    print(f"[config]   num_workers  : {num_workers}")
    print(f"[config]   seed         : {seed}")

    # ── Load assets ────────────────────────────────────────────────────────
    print("\nLoading logos …")
    logos = _load_logos(logos_dir)
    print(f"  -> {len(logos)} logo variant(s) loaded.")

    print("Indexing backgrounds …")
    bg_paths = _load_bg_paths(bg_dir)
    print(f"  -> {len(bg_paths)} background(s) found.")

    # ── Build augmentation pipelines ───────────────────────────────────────
    spatial_tf = _build_spatial_transform()
    photo_tf   = _build_photometric_transform()

    # ── Generate samples ───────────────────────────────────────────────────
    samples: List[Sample] = []
    max_attempts = num_samples * 10

    print(f"\nGenerating {num_samples} samples …")
    with tqdm(total=num_samples, unit="sample") as pbar:
        attempts = 0
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            s = _generate_sample(
                logos=logos,
                bg_paths=bg_paths,
                img_size=img_size,
                spatial_tf=spatial_tf,
                photo_tf=photo_tf,
            )
            if s is not None:
                samples.append(s)
                pbar.update(1)

    if len(samples) < num_samples:
        print(
            f"\n[WARNING] Only {len(samples)}/{num_samples} samples "
            f"generated after {attempts} attempts — check your assets."
        )

    # ── Write Underfolder dataset ──────────────────────────────────────────
    print(f"\nWriting Underfolder dataset to: {output}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        seq = SamplesSequence.from_list(samples)
    except AttributeError:
        seq = SamplesSequence(samples)  # type: ignore[arg-type]

    seq.to_underfolder(str(output), exists_ok=True).run(num_workers=num_workers)
    print("Done!")


# ---------------------------------------------------------------------------
# CLI — plain argparse (avoids pipelime 2.2.0 FieldInfo resolution bug)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic Underfolder dataset for Eyecan logo detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bg_dir",      type=Path, default=Path("backgrounds/Images"),
                   help="Directory containing background JPEG/PNG images.")
    p.add_argument("--logos_dir",   type=Path, default=Path("logos"),
                   help="Directory containing logo RGBA PNGs.")
    p.add_argument("--num_samples", type=int,  default=5000,
                   help="Number of composited images to generate.")
    p.add_argument("--img_size",    type=int,  nargs=2, default=[640, 480],
                   metavar=("W", "H"), help="Output resolution: width height.")
    p.add_argument("--output",      type=Path, default=Path("generated_dataset"),
                   help="Destination Underfolder directory.")
    p.add_argument("--num_workers", type=int,  default=0,
                   help="Writer processes (0 = single-threaded, safest; increase only with ample RAM).")
    p.add_argument("--seed",        type=int,  default=None,
                   help="RNG seed for reproducibility (omit for random).")
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