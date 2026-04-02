"""
Batch Inference & Visualisation Script
======================================
Runs the trained LogoDetector on a directory of real-world test images.
Displays each prediction in an interactive GUI window (if available),
and saves the annotated results to an output directory.
"""

import os
import sys
from pathlib import Path
from typing import List

import cv2
import torch

# Ensure local modules can be imported
sys.path.insert(0, "src")
from logo_detector import LogoDetector

def _get_image_paths(directory: Path) -> List[Path]:
    """Recursively find all images in the directory, handling case sensitivity on Linux."""
    extensions =["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    return[p for ext in extensions for p in directory.rglob(ext)]

def main() -> None:
    input_dir = Path("test_images")
    output_dir = Path("test_results")
    checkpoint_path = Path("checkpoints/best.pt")

    # Ensure directories exist
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    image_paths = _get_image_paths(input_dir)
    if not image_paths:
        print(f"[INFO] No images found in '{input_dir}'. Please add real-world test photos.")
        return

    # Check if we are running in a headless environment (like Google Colab)
    # 'DISPLAY' is present on Linux GUI sessions; win32 implies Windows (usually has GUI)
    has_gui = os.environ.get("DISPLAY") is not None or sys.platform == "win32"
    if not has_gui:
        print("[INFO] Headless environment detected. GUI windows will be disabled.")

    # Force CPU device for robust compatibility
    device = torch.device("cpu")
    print(f"[INFO] Loading detector from {checkpoint_path} on {device}...")
    detector = LogoDetector.load(checkpoint_path, device=device)

    print(f"[INFO] Running inference on {len(image_paths)} images...\n")
    
    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Failed to load {img_path.name}. Skipping.")
            continue

        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Forward pass: predict normalized coordinates [0.0, 1.0]
        x_norm, y_norm = detector.predict(rgb, device=device)
        
        # Denormalize to absolute pixel coordinates
        px, py = int(x_norm * W), int(y_norm * H)

        # Draw crosshair marker with shadow
        r = max(10, int(min(H, W) * 0.03))
        thickness = max(2, int(min(H, W) * 0.004))
        main_color = (0, 0, 255)
        shadow_color = (0, 0, 0)
        offset = max(1, int(thickness / 2))

        # Shadow
        s_px, s_py = px + offset, py + offset
        cv2.circle(bgr, (s_px, s_py), radius=r, color=shadow_color, thickness=thickness)
        cv2.line(bgr, (s_px - int(r * 1.5), s_py), (s_px + int(r * 1.5), s_py), shadow_color, thickness)
        cv2.line(bgr, (s_px, s_py - int(r * 1.5)), (s_px, s_py + int(r * 1.5)), shadow_color, thickness)

        # Main marker
        cv2.circle(bgr, (px, py), radius=r, color=main_color, thickness=thickness)
        cv2.line(bgr, (px - int(r * 1.5), py), (px + int(r * 1.5), py), main_color, thickness)
        cv2.line(bgr, (px, py - int(r * 1.5)), (px, py + int(r * 1.5)), main_color, thickness)

        # Save annotated image
        out_file = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_file), bgr)
        
        print(f"-> {img_path.name:<25} | Predicted centroid: ({px:>4} px, {py:>4} px)")

        # Display only if GUI is available
        if has_gui:
            window_name = f"Prediction: {img_path.name} - PRESS ANY KEY"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.imshow(window_name, bgr)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    if has_gui:
        cv2.destroyAllWindows()
        
    print(f"\n[INFO] Inference complete. Visual results saved in '{output_dir}'.")

if __name__ == "__main__":
    main()