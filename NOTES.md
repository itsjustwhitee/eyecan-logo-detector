## Dataset
Dataset immagini background: [MIT Indoor Scenes](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019?resource=download)
The database contains 67 Indoor categories, and a total of 15620 images. The number of images varies across categories, but there are at least 100 images per category. All images are in jpg format. The images provided here are for research purposes only.

## Ambiente di test
WSL2 (ubuntu 24) on windows machine.
Advanced troubleshooting to test models locally on WSL2 with ROCm support to GPU passthrough (AMD 9070 XT). [DOC Reference](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html)

Recommend to use a virtual python ambient (to avoid breaking system dependencies and such).
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Dependencies 
```bash
pip install opencv-python albumentations pipelime matplotlib jupyter pandas tqdm
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```