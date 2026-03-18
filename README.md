# ISBI 2026 CSV Challenge 4th Place Solution

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow?style=for-the-badge)](https://huggingface.co/yws0322/csv2026)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/yws0322/csv2026-lpsib-solution)

This repository contains the 4th place solution for the [CSV 2026 Challenge](http://119.29.231.17/index.html) (Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound).

## Challenge Overview

The CSV 2026 challenge requires jointly solving two tasks on carotid plaque ultrasound images with a single unified model:

1. **Segmentation**: Segment carotid plaque and vascular structures from two-view (longitudinal and transverse) US images.
2. **Classification**: Classify plaque vulnerability (low-risk: RADS 2 vs. high-risk: RADS 3–4) following the International Plaque-RADS Guidelines.

## Methods
![Fig. 1](assets/fig1.png)

## Environments and Requirements

> The following specs describe the environment used for training. Other configurations may work as long as the requirements are satisfied.

- **OS**: Ubuntu 22.04.5 LTS
- **CPU**: Intel Core i7-7700 @ 3.60GHz (8 cores)
- **RAM**: 62GB
- **GPU**: NVIDIA GeForce RTX 2080 Ti (11GB)
- **CUDA**: 13.0 (Driver: 580.126.09)
- **Python**: 3.10.12

To install requirements:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is a large-scale carotid ultrasound collection containing 1,500 paired cases, each consisting of longitudinal and transverse B-mode images.

**Split**

| Split | Cases | Labeled |
|-------|:------:|:--------:|
| Train | 1,000 | 200 (20%) |
| Validation | 200 | - |
| Test | 300 | - |

**Folder Structure**

```
train/
├── images/               # 001.h5, 002.h5, ...
│   └── *.h5
│       ├── long_image   # [512, 512, 1] — longitudinal B-mode
│       └── trans_image  # [512, 512, 1] — transverse B-mode
├── labels/               # 001_label.h5, 002_label.h5, ...
│   └── *_label.h5
│       ├── long_mask    # [512, 512] — longitudinal segmentation mask
│       ├── trans_mask   # [512, 512] — transverse segmentation mask
│       └── label        # 0: low-risk (RADS 2), 1: high-risk (RADS 3–4)
└── train.txt            # list of training file names
```

## Configuration
The global settings are managed within the `DEFAULT_CFG` dictionary in `train.py`. Below are the key categories:

**Data & Paths**
- `data_root`: Path to your training dataset (default: ./CSV2026_Dataset_Train).
- `split_json`: Path to the 4-fold cross-validation split file (default: split4.json).
- `n_folds`: Total number of folds for cross-validation.
- `checkpoint_dir_prefix`: Prefix for the directories where model weights and logs are saved.

**Architecture & Optimization**
- `convnext_model`: Backbone variant (default: convnext_nano).
- `img_size`: Input image dimensions.
- `encoder_freeze_epochs`: Duration of initial encoder freeze.
- `early_stopping_patience`: Threshold for early termination.

**Mean Teacher (Semi-Supervised Learning)**
- `use_mean_teacher`: Enables the semi-supervised consistency
- `ema_alpha`: The decay rate for the Teacher model update.
- `consistency_weight`: Scale for the consistency loss.
- `consistency_rampup_epochs`: Period to scale up consistency loss.
- `consistency_confidence_threshold`: Minimum confidence for segmentation consistency.
- `consistency_cls_confidence_threshold`: Minimum confidence for classification consistency.

## Training

Before training, check `DEFAULT_CFG` in `train.py` and set paths/options for your environment:

- `data_root` (default: `./CSV2026_Dataset_Train`)
- `split_json` / `n_folds`
- `device` (`cuda`, `mps`, or fallback to available device)

### Run Commands
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Training
```bash
# 1) Train a single fold (e.g. fold 0)
python train.py --fold 0

# 2) Train a fold range (e.g. fold 1, 2, 3)
# end_fold is exclusive
python train.py --start_fold 1 --end_fold 4

# 3) Train with given 4 fold cv split
python train.py
```

### Output Checkpoints

- Single fold: saved under `checkpoint_dir_prefix/fold_{k}_...`
- Range/default k-fold: saved under `checkpoint_dir_prefix/kfold_summary/fold_{k}`
- K-fold summary statistics: `checkpoint_dir_prefix/kfold_summary/statistics.txt`

## Inference

`inference.py` runs submission-style prediction on HDF5 files.

### Input / Output Format

- Input directory: `DATA_ROOT/images/*.h5`
- Required input keys in each file: `long_img`, `trans_img`
- Output file: `OUTPUT_DIR/{case_name}_pred.h5`
- Output keys: `long_mask`, `trans_mask`, `cls`

### Run With CLI Arguments

```bash
# Example
python inference.py \
	--data-root ./CSV2026_Dataset_Val \
	--model-path ./checkpoints_convnext_unet/kfold_summary/fold_0/best_st_fold0_model.pth \
	--output-dir ./preds \
	--resize-target 512 \
	--gpu 0 \
	--cls-threshold 0.8
```

### Run With Environment Variables

```bash
export DATA_ROOT=./CSV2026_Dataset_Val
export MODEL_PATH=./checkpoints_convnext_unet/kfold_summary/fold_0/best_st_fold0_model.pth
export OUTPUT_DIR=./preds
export RESIZE_TARGET=512
export GPU=0
export CLS_THRESHOLD=0.8

python inference.py
```

### Arguments

- `--data-root`: Dataset root path (must contain `images/`)
- `--model-path`: Checkpoint path
- `--output-dir`: Directory to save `*_pred.h5`
- `--resize-target`: Inference resize size (default: `512`)
- `--gpu`: Value for `CUDA_VISIBLE_DEVICES` (default: `0`)
- `--cls-threshold`: Classification threshold (default: `0.8`)

If CLI args are omitted, `inference.py` falls back to environment variables.

### Run With Docker
1. Pull the image

```bash
docker pull yws0322/csv2026-lpsib-solution
```

2. Run inference

```bash
docker run --rm --gpus all \
  -v /path/to/dataset:/data \
  -v /path/to/weights:/weights \
  -v /path/to/output:/output/preds \
  -e DATA_ROOT=/data \
  -e MODEL_PATH=/weights/best.pth \
  -e OUTPUT_DIR=/output/preds \
  -e RESIZE_TARGET=512 \
  -e GPU=0 \
  -e CLS_THRESHOLD=0.8 \
  yws0322/csv2026-lpsib-solution
```

Mount your local paths to `/data`, `/weights`, `/output/preds` and set the corresponding environment variables. The container reads `DATA_ROOT/images/*.h5` as input and writes `OUTPUT_DIR/{case_name}_pred.h5` as output.

## Results
Our method achieved the following performance on [CSV 2026: Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound](http://119.29.231.17/test.html)

|    Model   |   F1   | Segmentation | Time(ms) |
|   :----:   | :----: |    :----:    |   :----: |
|   MAGNET   |  68.03 |     60.25    |  98.52    |