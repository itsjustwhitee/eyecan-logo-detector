"""
Module A — Eyecan Logo Detector
================================
A lightweight, single-class CNN regressor that predicts the (x, y) centroid
of the Eyecan logo inside an input image.

Output coordinates are **normalised** to [0, 1] (same convention as the
dataset generator), making the model resolution-agnostic at both train
and inference time.

Architecture overview
---------------------
LogoDetector wraps a MobileNetV3-Small backbone (ImageNet-pretrained) with a
custom regression head. The backbone is a proven, efficient feature extractor;
the head is deliberately minimal — two linear layers with a Sigmoid output to
keep predictions in [0, 1] without any external clamping.

Why MobileNetV3-Small?
  - Fast enough to train in <30 min on a free Colab/Kaggle GPU.
  - Small enough to ship as a single checkpoint (<20 MB).
  - ImageNet weights give immediately useful low-level features (edges,
    colours, textures) that transfer well to logo localisation.
  - Acceptable on CPU at inference time (~10 ms per image on a modern laptop).

Loss: SmoothL1 (Huber) — more robust to occasional outliers than plain MSE
while still converging to a precise centroid at low residuals.

Usage — training
----------------
    python logo_detector.py --train \\
        --dataset_dir generated_dataset \\
        --epochs 30 \\
        --lr 1e-3 \\
        --output_dir checkpoints

Usage — inference
-----------------
    python logo_detector.py --infer \\
        --checkpoint checkpoints/best.pt \\
        --image path/to/image.jpeg

Both modes are also importable from a notebook; see the bottom of this file
for a worked example.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LogoDetector(nn.Module):
    """Single-class CNN regressor — predicts normalised (x, y) logo centroid.

    The backbone is MobileNetV3-Small (ImageNet-pretrained by default).
    It is kept fully trainable so that the fine-tuning can adapt low-level
    features to the logo's specific visual signature.

    The regression head maps the 576-d pooled backbone features to (x, y)
    via a two-layer MLP:
        576 → 128 (ReLU + Dropout) → 2 (Sigmoid)

    Sigmoid guarantees predictions ∈ (0, 1) without any post-processing.

    Args:
        pretrained: Load ImageNet weights for the backbone (recommended).
        dropout:    Dropout probability before the final layer (regularisation).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()

        # ---------- backbone ----------
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Remove the built-in classifier — we only need the feature extractor.
        # After adaptive avg-pooling the output is (B, 576, 1, 1).
        self.features  = backbone.features       # conv stages
        self.pool      = backbone.avgpool         # AdaptiveAvgPool2d → (B, 576, 1, 1)

        # ---------- regression head ----------
        self.head = nn.Sequential(
            nn.Flatten(),                         # (B, 576)
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
            nn.Sigmoid(),                         # => (B, 2) ∈ (0, 1)
        )

        # Initialise the head with small weights for a stable start
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        nn.init.xavier_uniform_(self.head[4].weight)
        nn.init.zeros_(self.head[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (B, 3, H, W), values in [0, 1]
               (standard torchvision normalisation applied externally).

        Returns:
            Tensor of shape (B, 2): predicted (x, y) normalised centroid.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        image: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> Tuple[float, float]:
        """Run inference on a single RGB uint8 image (H × W × 3, numpy array).

        This is the primary entry-point for notebook / real-world usage.

        Args:
            image:  RGB uint8 numpy array, any resolution.
            device: Torch device.  Defaults to CUDA if available, else CPU.

        Returns:
            (x, y) normalised centroid in [0, 1].
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)
        tensor = _preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = self(tensor)
        x, y = pred[0].cpu().tolist()
        return float(x), float(y)

    def save(self, path: Path | str) -> None:
        """Save model weights to *path* (a .pt file)."""
        torch.save(self.state_dict(), str(path))
        print(f"  Checkpoint saved → {path}")

    @classmethod
    def load(
        cls,
        path: Path | str,
        device: Optional[torch.device] = None,
        pretrained: bool = False,
    ) -> "LogoDetector":
        """Load a pre-trained LogoDetector from a checkpoint file.

        Args:
            path:      Path to the .pt checkpoint produced by ``save()``.
            device:    Target device.  Auto-detected when None.
            pretrained: Ignored (weights come from checkpoint), kept for
                        signature clarity.

        Returns:
            LogoDetector instance ready for inference.

        Example::

            detector = LogoDetector.load("checkpoints/best.pt")
            x, y = detector.predict(image)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(pretrained=False)
        state = torch.load(str(path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"  Checkpoint loaded ← {path}  (device: {device})")
        return model


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

# The same transform is used at both train and inference time for the
# *input* image.  Augmentation (brightness, flips, etc.) is applied upstream
# by the dataset generator and is NOT repeated here.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.224, 0.224, 0.224)

_preprocess = transforms.Compose([
    transforms.ToTensor(),                           # HWC uint8 -> CHW float [0,1]
    transforms.Resize((224, 224), antialias=True),   # MobileNet canonical size
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LogoDataset(Dataset):
    """Reads a pipelime Underfolder dataset produced by dataset_generator.py.

    Expects the following layout::

        dataset_dir/
          data/
            00000_image.jpeg
            00000_metadata.json   ← {"x": float, "y": float}
            00001_image.jpeg
            ...

    Args:
        dataset_dir: Root of the Underfolder (contains a ``data/`` sub-folder).
        transform:   Optional torchvision transform applied to the image tensor
                     *after* ``_preprocess``.  Useful for extra augmentation
                     at train time (e.g. RandomErasing).
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_dir  = Path(dataset_dir) / "data"
        self.transform = transform

        # Collect all (image_path, metadata_path) pairs
        meta_files = sorted(self.data_dir.glob("*_metadata.json"))
        if not meta_files:
            raise FileNotFoundError(
                f"No *_metadata.json files found in {self.data_dir}. "
                "Check that dataset_generator.py ran successfully."
            )
        self.samples = []
        for meta_path in meta_files:
            stem   = meta_path.stem.replace("_metadata", "")
            img_candidates = list(self.data_dir.glob(f"{stem}_image.*"))
            if img_candidates:
                self.samples.append((img_candidates[0], meta_path))

        print(f"  LogoDataset: {len(self.samples)} samples from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, meta_path = self.samples[idx]

        # --- image ---
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"Cannot read {img_path}")
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = _preprocess(rgb)
        if self.transform is not None:
            image = self.transform(image)

        # --- label ---
        meta  = json.loads(meta_path.read_text())
        label = torch.tensor([meta["x"], meta["y"]], dtype=torch.float32)

        return image, label


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------

def pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    img_size: Tuple[int, int] = (640, 480),
    threshold_pct: float = 0.10,
) -> float:
    """Fraction of predictions within *threshold_pct* of the shortest edge.

    The challenge requires max error 10 % of the shorter dimension
    (e.g. 40 px for 640×480).  This metric reports how often the model
    satisfies that constraint over the given batch / epoch.

    Args:
        preds:          (N, 2) predicted normalised coords.
        targets:        (N, 2) ground-truth normalised coords.
        img_size:       (W, H) used to convert normalised error to pixels.
        threshold_pct:  Fraction of shortest edge (default 0.10 = 10 %).

    Returns:
        Fraction in [0, 1].
    """
    W, H        = img_size
    short_side  = min(W, H)
    threshold_n = threshold_pct  # normalised threshold

    # Scale differences to pixel space, compute Euclidean error
    diff    = preds - targets
    diff_px = diff * torch.tensor([W, H], device=preds.device, dtype=preds.dtype)
    dist_n  = (diff_px.norm(dim=1) / short_side)  # normalised pixel error

    return (dist_n < threshold_pct).float().mean().item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    img_size: Tuple[int, int] = (640, 480),
    dropout: float = 0.3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> LogoDetector:
    """Full training loop with validation and early stopping.

    Saves ``best.pt`` (lowest validation loss) and ``last.pt`` to *output_dir*.
    Returns the best model.

    Training schedule
    -----------------
    - Optimiser: AdamW with weight decay 1e-4 (prevents weight explosion).
    - LR scheduler: CosineAnnealingLR — starts at *lr*, decays smoothly to
      1 % of *lr* over *epochs*.  No manual step-decay needed.
    - Loss: SmoothL1 (Huber, beta=0.01) — combines precision at low residuals
      with robustness to occasional outliers.
    - Early stopping: patience of 10 epochs on validation loss.

    Args:
        dataset_dir:  Root of the Underfolder dataset.
        output_dir:   Where to write checkpoints and training_log.json.
        epochs:       Maximum training epochs.
        lr:           Peak learning rate (AdamW).
        batch_size:   Samples per mini-batch.
        val_split:    Fraction of data reserved for validation.
        img_size:     (W, H) used only by the accuracy metric.
        dropout:      Dropout prob in the regression head.
        seed:         RNG seed for the train/val split.
        device:       Torch device. Auto-detected when None.

    Returns:
        Best LogoDetector (weights restored from ``best.pt``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training  — device: {device}  |  epochs: {epochs}  |  lr: {lr}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    # ---- dataset & split ----
    full_ds  = LogoDataset(dataset_dir)
    n_val    = max(1, int(len(full_ds) * val_split))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"  Train samples : {n_train}   Val samples : {n_val}\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---- model / optimiser / scheduler ----
    model     = LogoDetector(pretrained=True, dropout=dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.SmoothL1Loss(beta=0.01)

    best_val_loss = float("inf")
    patience      = 10
    no_improve    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch:03d}/{epochs} [train]",
                                   leave=False):
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            preds = model(images)
            loss  = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(images)

        train_loss /= n_train
        scheduler.step()

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds     = model(images)
                val_loss += criterion(preds, labels).item() * len(images)
                val_acc  += pixel_accuracy(preds, labels, img_size) * len(images)

        val_loss /= n_val
        val_acc  /= n_val
        elapsed   = time.time() - t0

        print(
            f"  Epoch {epoch:03d}/{epochs}"
            f"  train_loss={train_loss:.5f}"
            f"  val_loss={val_loss:.5f}"
            f"  val_acc={val_acc*100:.1f}%"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"  ({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc_pct": val_acc * 100,
        })

        # ---- checkpointing ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            model.save(output_dir / "best.pt")
        else:
            no_improve += 1

        model.save(output_dir / "last.pt")

        if no_improve >= patience:
            print(f"\n  Early stopping triggered after {patience} epochs without improvement.")
            break

    # ---- save training log ----
    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(history, indent=2))
    print(f"\n  Training log → {log_path}")

    # ---- restore best ----
    print("\n  Restoring best checkpoint...")
    best = LogoDetector.load(output_dir / "best.pt", device=device)
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eyecan Logo Detector — train or infer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Run training.")
    mode.add_argument("--infer", action="store_true", help="Run single-image inference.")

    # training args
    p.add_argument("--dataset_dir", type=Path, default=Path("generated_dataset"))
    p.add_argument("--output_dir",  type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs",      type=int,  default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,  default=32)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--seed",        type=int,  default=0)

    # inference args
    p.add_argument("--checkpoint",  type=Path, default=None,
                   help="Path to best.pt checkpoint (required for --infer).")
    p.add_argument("--image",       type=Path, default=None,
                   help="Input image path (required for --infer).")

    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    if args.train:
        train(
            dataset_dir = args.dataset_dir,
            output_dir  = args.output_dir,
            epochs      = args.epochs,
            lr          = args.lr,
            batch_size  = args.batch_size,
            val_split   = args.val_split,
            dropout     = args.dropout,
            seed        = args.seed,
        )

    elif args.infer:
        if args.checkpoint is None or args.image is None:
            raise SystemExit("--infer requires both --checkpoint and --image.")

        bgr   = cv2.imread(str(args.image))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read {args.image}")
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detector = LogoDetector.load(args.checkpoint)
        x, y     = detector.predict(rgb)
        H, W     = bgr.shape[:2]
        print(f"\n  Predicted centroid (normalised) : x={x:.4f}  y={y:.4f}")
        print(f"  Predicted centroid (pixels)     : x={int(x*W)}  y={int(y*H)}")


if __name__ == "__main__":
    _main()