# YOLO Safety Helmet Detection

Professional camera-based detection for safety helmets, heads, and people using YOLO.

## Overview

This repository contains scripts for real-time camera detection and tracking using a YOLO model.

- `1_camera_detection.py` — real-time camera detection
- `2_tracking.py` — object tracking with YOLO
- `3_halmet_chek.py` — helmet check
- `4_danger_zone.py` — danger zone detection
- `5_fall_detection.py` — fall detection
- `split_yolo_classes.py` — class splitting utility

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics YOLO

Install dependencies with:

```bash
pip install opencv-python-headless numpy ultralytics
```

## Usage

Run camera detection with:

```bash
python 1_camera_detection.py --model Safety-Helmet-Detection-main/output/best.pt --conf 0.25 --camera 0
```

Press `q` to quit.

## Notes

- Keep large model files outside the repository or use Git LFS.
- `.gitignore` excludes temporary and environment files.
