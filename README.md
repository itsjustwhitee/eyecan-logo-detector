# Eyecan AI - Logo Detection Challenge

This repository contains the full implementation for the Eyecan Logo Detector challenge. 
The project is divided into two main components:

- **Module A**: Deep Learning-based Detector (Training and Inference).
- **Module B**: Synthetic Dataset Generator using Pipelime and Albumentations.

## Status
- [x] Module B: Dataset Generation
- [ ] Module A: Model Architecture & Training (In Progress)

## Quick Start
For detailed instructions on how to generate the training data, please refer to:
- 👉 [**Module B Documentation (Dataset Generator)**](DATASET_GENERATOR.md)

## Tech Stack
- **Language**: Python 3.12
- **Core Library**: [pipelime-python 2.2.0](https://github.com/eyecan-ai/pipelime-python)
- **Vision**: OpenCV, Albumentations
- **Deep Learning**: PyTorch
- **Hardware Acceleration**: AMD ROCm (WSL2) / NVIDIA (Cloud)

## Project structure

```
eyecan_challenge/
├── src/
│   └── dataset_generator.py   # Module B
├── logos/                     # PNG logo variants (RGBA or RGB)
│   ├── logo_red.png
│   ├── logo_black.png
│   └── logo_white.png
├── backgrounds/
│   └── Images/                # any JPEG/PNG/BMP background images
└── generated_dataset/         # output (created by Module B)
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
