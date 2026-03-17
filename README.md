# ISBI 2026 CSV Challenge 4th Place Solution

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow?style=for-the-badge)](https://huggingface.co/yws0322/csv2026)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/yws0322/csv2026-lpsib-solution)

This repository contains the 4th place solution for the [CSV 2026 Challenge](http://119.29.231.17/index.html) (Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound).

## Challenge Overview

The CSV 2026 challenge requires jointly solving two tasks on carotid plaque ultrasound images with a single unified model:

1. **Segmentation**: Segment carotid plaque and vascular structures from two-view (longitudinal and transverse) US images.
2. **Classification**: Classify plaque vulnerability (low-risk: RADS 2 vs. high-risk: RADS 3–4) following the International Plaque-RADS Guidelines.

## Methods

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

| Split | Cases | Labeled |
|-------|:------:|:--------:|
| Train | 1,000 | 200 (20%) |
| Validation | 200 | - |
| Test | 300 | - |

### Data Structure

```
train/
├── image/               # 001.h5, 002.h5, ...
│   └── *.h5
│       ├── long_image   # [512, 512, 1] — longitudinal B-mode
│       └── trans_image  # [512, 512, 1] — transverse B-mode
├── label/               # 001_label.h5, 002_label.h5, ...
│   └── *_label.h5
│       ├── long_mask    # [512, 512] — longitudinal segmentation mask
│       ├── trans_mask   # [512, 512] — transverse segmentation mask
│       └── label        # 0: low-risk (RADS 2), 1: high-risk (RADS 3–4)
└── train.txt            # list of training file names
```

## Training

## Inference

## Results