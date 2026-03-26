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
    python src/dataset_generator.py --num_samples 5000 --seed 42

Memory strategy
---------------
Samples are generated and flushed to disk in batches (--batch_size, default 200).
Peak RAM is proportional to batch_size, not num_samples.
At 640x480 RGB, 200 images ≈ 175 MB.
"""

from __future__ import annotations

import argparse
import random
import shutil
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
    from pipelime.sequences.sample import Sample    # type: ignore[import]
except ImportError:
    from pipelime.sequences import Sample           # type: ignore[import]


# ---------------------------------------------------------------------------
# pipelime item class resolution
# ---------------------------------------------------------------------------
# Item class names vary across pipelime minor versions; pick the first that exists.

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

    Affine handles scale/rotation/translation in one pass (always applied).
    Perspective adds a mild projective warp 50 % of the time.

    KeypointParams propagates the logo centroid through every transform.
    If the centroid exits the canvas, _generate_sample discards the sample
    and retries (rejection sampling).
    """
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


def _build_scene_transform() -> A.Compose:
    """Photometric pipeline applied to the final composited RGB image.

    Applied after compositing so background and logo are perturbed together,
    simulating global lighting conditions.

    Design rules:
      - RandomBrightnessContrast applies the same scalar to all 3 channels,
        so it cannot shift hue.  ColorJitter and RandomGamma were dropped
        because they modify channels independently and produce green/magenta
        casts across albumentations versions.
      - Wide brightness range (±0.35) is intentional: it produces dark scenes
        (night, poorly lit rooms) and overexposed ones (outdoor sun), both
        realistic conditions for a logo detector.
      - GaussianBlur simulates defocus or motion blur on the whole scene.
    """
    return A.Compose(
        [
            # Wide range to cover dark scenes, overexposed scenes, and midtones
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.25, p=0.7),
            # Occasional heavy blur to simulate out-of-focus or moving camera
            A.GaussianBlur(blur_limit=(3, 11), p=0.15),
            # JPEG compression artefacts — common in real photos and video frames
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
        ]
    )


# ---------------------------------------------------------------------------
# Logo-only colour augmentation
# ---------------------------------------------------------------------------

_LOGO_COLOR_AUG = A.Compose([
    # HueSaturationValue operates in HSV and is safe for per-logo use.
    # Narrow hue range keeps the logo recognisable (red stays red-ish);
    # wider saturation/value range covers faded print and bright neon.
    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=20,
        val_shift_limit=15,
        p=0.65,
    ),
    # RGBShift adds small independent per-channel offsets to simulate
    # coloured lighting hitting the logo surface.
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
])


def _augment_logo_colors(rgba: np.ndarray) -> np.ndarray:
    """Apply colour augmentation only to logo pixels (alpha > 0).

    Operating before compositing and only on the logo avoids the hue-cast
    problem that arises when colour transforms are applied to the full scene.
    """
    rgb  = rgba[..., :3].copy()
    a    = rgba[..., 3]
    mask = a > 0

    rgb_aug = _LOGO_COLOR_AUG(image=rgb)["image"]
    # Write augmented values only where the logo is opaque
    rgb[mask] = rgb_aug[mask]
    return np.dstack([rgb, a])


# ---------------------------------------------------------------------------
# Soft shadow
# ---------------------------------------------------------------------------

def _add_soft_shadow(
    bg_rgb: np.ndarray,
    logo_alpha: np.ndarray,
    strength: float = 0.12,
    blur_ksize: int = 25,
    shift: Tuple[int, int] = (3, 4),
) -> np.ndarray:
    """Darken the background where the logo's shadow would fall.

    Generates a blurred, offset version of the logo's alpha mask and uses it
    to darken the background before compositing.  This gives the logo a subtle
    visual anchor that makes it look less like a sticker pasted on top.

    Args:
        strength:   Maximum darkening factor (0 = no shadow, 1 = black).
        blur_ksize: Gaussian kernel size; larger = softer / more diffuse shadow.
        shift:      (dx, dy) offset in pixels — direction of the light source.
    """
    H, W = logo_alpha.shape
    shadow = (logo_alpha > 0).astype(np.uint8) * 255

    # Translate shadow mask
    M      = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    shadow = cv2.warpAffine(shadow, M, (W, H),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    shadow = cv2.GaussianBlur(shadow, (ksize, ksize), 0)

    shadow_f = (shadow.astype(np.float32) / 255.0) * strength
    bg_f     = bg_rgb.astype(np.float32) * (1.0 - shadow_f[..., None])
    return np.clip(bg_f, 0, 255).astype(np.uint8)


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
    scene_tf: A.Compose,
    max_logo_ratio: float = 0.30,
) -> Optional[Sample]:
    """Build one composited sample.  Returns None on rejection (caller retries).

    Full pipeline:
      1.  Load and resize a random background → RGB.
      2.  Pick a random logo; scale to max_logo_ratio * min(W, H).
      3.  Centre the logo on a blank RGBA canvas.
      4.  Apply spatial transform (Affine + Perspective) tracking the centroid.
          If the centroid exits the canvas → return None.
      5.  Colour-augment logo pixels only (HueSaturationValue + RGBShift).
      6.  Add a soft shadow on the background beneath the logo.
      7.  Alpha-blend warped logo over background (float32 for precision).
      8.  Apply scene-level photometric augmentation (brightness / blur / compression).
      9.  Wrap image + normalised centroid into a pipelime Sample.
    """
    W, H = img_size

    # 1. Background
    bg_bgr = cv2.imread(str(random.choice(bg_paths)))
    if bg_bgr is None:
        return None
    bg_rgb = cv2.cvtColor(cv2.resize(bg_bgr, (W, H)), cv2.COLOR_BGR2RGB)

    # 2. Logo — fit inside max_logo_ratio * shortest canvas edge
    logo_rgba = random.choice(logos).copy()
    lh, lw    = logo_rgba.shape[:2]
    max_px    = int(min(W, H) * max_logo_ratio)
    scale     = max_px / max(lh, lw)
    logo_rgba = cv2.resize(logo_rgba,
                           (max(1, int(lw * scale)), max(1, int(lh * scale))))
    lh, lw = logo_rgba.shape[:2]

    # 3. Centre logo on a full-size transparent canvas
    canvas      = np.zeros((H, W, 4), dtype=np.uint8)
    y0, x0      = (H - lh) // 2, (W - lw) // 2
    canvas[y0:y0+lh, x0:x0+lw] = logo_rgba

    # 4. Spatial transform — centroid keypoint travels with the image
    cx, cy = float(x0 + lw / 2), float(y0 + lh / 2)
    result = spatial_tf(image=canvas, keypoints=[(cx, cy)])
    if not result["keypoints"]:
        return None
    fx, fy = result["keypoints"][0]
    if not (0 <= fx < W and 0 <= fy < H):
        return None

    warped = result["image"]  # RGBA after spatial transform

    # 5. Logo-only colour augmentation (before compositing, does not touch BG)
    warped = _augment_logo_colors(warped)

    # 6. Soft shadow on the background beneath the logo
    bg_rgb = _add_soft_shadow(bg_rgb, warped[:, :, 3], strength=0.12,
                               blur_ksize=25, shift=(3, 4))

    # 7. Alpha-blend warped logo over background
    alpha      = (warped[:, :, 3] / 255.0)[..., np.newaxis]
    composited = (bg_rgb.astype(np.float32) * (1.0 - alpha)
                  + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    # 8. Scene-level photometric augmentation (brightness / blur / compression)
    final_img = scene_tf(image=composited)["image"]

    # 9. pipelime expects RGB — do NOT convert to BGR here
    return Sample({
        "image":    _ImageItemCls(final_img),
        "metadata": _MetaItemCls({"x": float(fx / W), "y": float(fy / H)}),
    })


# ---------------------------------------------------------------------------
# Batch writer
# ---------------------------------------------------------------------------

def _write_batch(
    batch: List[Sample],
    output: Path,
    start_index: int,
    num_workers: int,
) -> None:
    """Write *batch* to the Underfolder at *output*, numbering from *start_index*.

    pipelime's to_underfolder() always starts indices from 0, so we write to a
    temporary directory and rename the files with the correct global offset
    before moving them to the final data/ folder.
    """
    tmp_dir = output / "_batch_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        seq = SamplesSequence.from_list(batch)
    except AttributeError:
        seq = SamplesSequence(batch)  # type: ignore[arg-type]
    seq.to_underfolder(str(tmp_dir), exists_ok=True).run(num_workers=num_workers)

    tmp_data  = tmp_dir / "data"
    real_data = output / "data"
    real_data.mkdir(exist_ok=True)

    for src_path in sorted(tmp_data.iterdir()):
        local_idx  = int(src_path.name.split("_", 1)[0])
        suffix     = src_path.name.split("_", 1)[1]   # e.g. "image.jpg"
        global_idx = start_index + local_idx
        shutil.move(str(src_path), str(real_data / f"{global_idx:05d}_{suffix}"))

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(
    bg_dir: Path,
    logos_dir: Path,
    num_samples: int,
    img_size: Tuple[int, int],
    output: Path,
    batch_size: int = 200,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> None:
    """Generate *num_samples* composited images and write a pipelime Underfolder.

    Samples are produced and flushed to disk in chunks of *batch_size* so that
    peak RAM is proportional to batch_size rather than num_samples.
    At 640×480 the default batch of 200 images uses ~175 MB of RAM.
    """
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
    print(f"[config]   batch_size    : {batch_size}")
    print(f"[config]   num_workers   : {num_workers}")
    print(f"[config]   seed          : {seed}")

    print("\nLoading logos...")
    logos = _load_logos(logos_dir)
    print(f"  -> {len(logos)} variant(s) loaded.")

    print("Indexing backgrounds...")
    bg_paths = _load_bg_paths(bg_dir)
    print(f"  -> {len(bg_paths)} image(s) found.")

    spatial_tf = _build_spatial_transform()
    scene_tf   = _build_scene_transform()

    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)

    total_written = 0
    max_attempts  = num_samples * 10

    print(f"\nGenerating {num_samples} samples in batches of {batch_size}...")
    with tqdm(total=num_samples, unit="sample") as pbar:
        batch: List[Sample] = []
        attempts = 0

        while total_written < num_samples and attempts < max_attempts:
            attempts += 1
            s = _generate_sample(logos, bg_paths, img_size, spatial_tf, scene_tf)
            if s is None:
                continue

            batch.append(s)
            pbar.update(1)

            if len(batch) == batch_size or (total_written + len(batch)) == num_samples:
                _write_batch(batch, output, start_index=total_written,
                             num_workers=num_workers)
                total_written += len(batch)
                batch = []  # release Sample objects → RAM freed

    if total_written < num_samples:
        print(f"\n[WARNING] Only {total_written}/{num_samples} samples written "
              f"after {attempts} attempts — check your assets.")

    print(f"\nDone! {total_written} samples written to: {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Plain argparse instead of pipelime's PipelimeCommand/choixe:
# in pipelime 2.2.0 on Python 3.12, unset PipelimeCommand fields are returned
# as raw FieldInfo objects, making the command unusable without passing every
# argument explicitly from the CLI.

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic Underfolder dataset for Eyecan logo detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bg_dir",      type=Path, default=Path("backgrounds/Images"),
                   help="Directory with background images (searched recursively).")
    p.add_argument("--logos_dir",   type=Path, default=Path("logos"),
                   help="Directory with logo PNGs (RGBA or RGB, any colour variant).")
    p.add_argument("--num_samples", type=int,  default=5000,
                   help="Total number of samples to generate.")
    p.add_argument("--img_size",    type=int,  nargs=2, default=[640, 480],
                   metavar=("W", "H"), help="Output resolution.")
    p.add_argument("--output",      type=Path, default=Path("generated_dataset"),
                   help="Destination Underfolder directory.")
    p.add_argument("--batch_size",  type=int,  default=200,
                   help="Samples per write batch. Lower = less RAM. "
                        "At 640x480 the default (200) uses ~175 MB peak RAM.")
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
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )