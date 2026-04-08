"""
demo_webcam.py - Live Logo Detector Demo
=========================================
Reads the webcam feed in real-time, runs the LogoDetector on every frame,
and draws:
  - A red crosshair on the predicted logo centroid.
  - A fading trail of the last N positions.
  - Pixel coordinates (x, y) and normalised coordinates in the top-left corner.
  - Current FPS in the top-right corner.

Usage:
    python src/demo_webcam.py
    python src/demo_webcam.py --checkpoint checkpoints/best.pt
    python src/demo_webcam.py --checkpoint checkpoints/best_20k.pt --camera 1 --trail 40

Controls:
    Q  or  ESC  →  quit
    S           →  save a screenshot to demo_screenshots/

Requirements (same as Module A):
    pip install torch torchvision opencv-python numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from logo_detector import LogoDetector


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_crosshair(
    frame: np.ndarray,
    px: int,
    py: int,
    radius: int,
    thickness: int,
    color: tuple,
    shadow_offset: int = 2,
) -> None:
    """Draw a circle + cross crosshair with a dark shadow for visibility."""
    arm = int(radius * 1.6)
    so  = shadow_offset
    shadow = (0, 0, 0)

    # Shadow
    cv2.circle(frame, (px + so, py + so), radius, shadow, thickness)
    cv2.line(frame, (px - arm + so, py + so), (px + arm + so, py + so), shadow, thickness)
    cv2.line(frame, (px + so, py - arm + so), (px + so, py + arm + so), shadow, thickness)

    # Main marker
    cv2.circle(frame, (px, py), radius, color, thickness)
    cv2.line(frame, (px - arm, py), (px + arm, py), color, thickness)
    cv2.line(frame, (px, py - arm), (px, py + arm), color, thickness)


def _draw_trail(
    frame: np.ndarray,
    trail: deque,
    color_tip: tuple = (0, 0, 255),
    color_tail: tuple = (0, 80, 180),
) -> None:
    """Draw a fading trail of past positions.

    Each point blends from color_tail (oldest) to color_tip (newest)
    and shrinks in radius toward the tail.
    """
    n = len(trail)
    if n < 2:
        return
    for i, (tx, ty) in enumerate(trail):
        alpha   = (i + 1) / n                              # 0 → 1 from tail to tip
        color   = tuple(int(color_tail[c] + alpha * (color_tip[c] - color_tail[c]))
                        for c in range(3))
        radius  = max(2, int(alpha * 8))
        cv2.circle(frame, (tx, ty), radius, color, -1)     # filled dot


def _put_text_shadowed(
    frame: np.ndarray,
    text: str,
    org: tuple,
    font_scale: float,
    thickness: int,
    color: tuple = (255, 255, 255),
    shadow: tuple = (0, 0, 0),
) -> None:
    """Render text with a 1-pixel dark shadow for legibility on any background."""
    cv2.putText(frame, text, (org[0] + 1, org[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_demo(
    checkpoint: Path,
    camera_index: int | str,
    trail_length: int,
    screenshot_dir: Path,
) -> None:
    # ---- load model ----
    print(f"[INFO] Loading model from {checkpoint} ...")
    detector = LogoDetector.load(checkpoint)
    print("[INFO] Model ready. Press Q or ESC to quit, S to save a screenshot.\n")

    # ---- open camera ----
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Try a different --camera value (0, 1, 2 ...)."
        )

    trail: deque[tuple[int, int]] = deque(maxlen=trail_length)
    fps_timer = time.perf_counter()
    fps_display = 0.0
    frame_count = 0

    screenshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[WARN] Frame capture failed - retrying ...")
                continue

            H, W = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ---- inference ----
            x_norm, y_norm = detector.predict(frame_rgb)
            px = int(x_norm * W)
            py = int(y_norm * H)

            # ---- FPS (rolling average over 10 frames) ----
            frame_count += 1
            if frame_count % 10 == 0:
                now          = time.perf_counter()
                fps_display  = 10.0 / (now - fps_timer)
                fps_timer    = now

            # ---- trail ----
            trail.append((px, py))

            # ---- draw ----
            short_side = min(H, W)
            radius     = max(10, int(short_side * 0.03))
            thickness  = max(2,  int(short_side * 0.004))

            _draw_trail(frame_bgr, trail)
            _draw_crosshair(frame_bgr, px, py, radius, thickness, color=(0, 0, 255))

            # ---- overlays ----
            font_scale = max(0.5, short_side / 900)
            txt_thick  = max(1, int(font_scale * 2))

            # Top-left: coordinates
            _put_text_shadowed(
                frame_bgr,
                f"px  ({px:4d}, {py:4d})",
                (12, 28),
                font_scale, txt_thick,
            )
            _put_text_shadowed(
                frame_bgr,
                f"norm ({x_norm:.3f}, {y_norm:.3f})",
                (12, 28 + int(26 * font_scale / 0.6)),
                font_scale, txt_thick,
            )

            # Top-right: FPS
            fps_str   = f"{fps_display:.1f} FPS"
            (tw, th), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, txt_thick)
            _put_text_shadowed(
                frame_bgr,
                fps_str,
                (W - tw - 12, 28),
                font_scale, txt_thick,
                color=(0, 255, 120),   # green tint for FPS
            )

            cv2.imshow("Eyecan Logo Detector - Live Demo  [Q / ESC = quit | S = screenshot]", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
            elif key in (ord("s"), ord("S")):
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = screenshot_dir / f"screenshot_{ts}.jpg"
                cv2.imwrite(str(path), frame_bgr)
                print(f"[INFO] Screenshot saved → {path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Demo closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live webcam demo for the Eyecan Logo Detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/best.pt"),
        help="Path to the model checkpoint (.pt file).",
    )
    p.add_argument(
        "--camera", type=str, default="0",
        help="OpenCV camera index (e.g. 0) OR stream URL (e.g. http://192.168.1.50:8080/video).",
    )
    p.add_argument(
        "--trail", type=int, default=30,
        help="Number of past positions to keep in the fading trail.",
    )
    p.add_argument(
        "--screenshots", type=Path, default=Path("demo_screenshots"),
        help="Directory where screenshots (S key) are saved.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    cam_source = int(args.camera) if args.camera.isdigit() else args.camera
    
    run_demo(
        checkpoint     = args.checkpoint,
        camera_index   = cam_source,
        trail_length   = args.trail,
        screenshot_dir = args.screenshots,
    )
