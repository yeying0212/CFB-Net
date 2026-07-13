# <p align="center">CFB-Net: Cross-Level Frequency-Domain Boundary-Aware Lightweight Network for SAR Flood Detection</p>

> **Authors:**
> Xingye Yang, Tianyu Wei, Lyuzhou Gao, Wenchao Liu, Jue Wang (Corresponding Author), and Liang Chen
>
> **Code Repository:** [https://github.com/yeying0212/CFB_Net](https://github.com/yeying0212/CFB_Net)

---

## 1. Overview

SAR imagery is a critical tool for flood disaster monitoring due to its all-weather, day-and-night imaging capability. However, existing bi-temporal SAR flood detection methods primarily rely on spatial-domain feature fusion, which often results in missed detections and false alarms when encountering irregular-boundary floods.


## 2. Supported Methods

This repository provides unified training and testing pipelines for the following methods:

| Method | Venue | Code Source |
|:--------|:------|:------------|
| **CFB-Net** (Ours) | — | [`models/cfb_net.py`](models/cfb_net.py) |
| [BIT](https://github.com/justchenhao/BIT_CD) | TGRS 2021 | [`models/BITNet.py`](models/BITNet.py) |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) | GRSL 2021 | [`models/SNUNet.py`](models/SNUNet.py) |
| [AFCFNet](https://github.com/wm-Githuber/AFCF3D-Net) | TGRS 2023 | [`models/AFCFNet.py`](models/AFCFNet.py) |
| [ELW_CDNet](https://github.com/dyl96/ELW_CDNet) | GRSL 2023 | [`models/ELWCDNet.py`](models/ELWCDNet.py) |
| [ACAHNet](https://github.com/CCRG-XJU/ChangeDetection_ACAHNet_TGRS2023) | TGRS 2023 | [`models/ACAHNet.py`](models/ACAHNet.py) |
| [LiST-Net](https://github.com/Tamer-Saleh) | TGRS 2024 | [`models/LiSTNet.py`](models/LiSTNet.py) |
| [EGENet](https://github.com/Jnmz/EGENet-IG24) | IGARSS 2024 | [`models/EGENet.py`](models/EGENet.py) |
| [SEIFNet](https://github.com/lixinghua5540/SEIFNet) | TGRS 2024 | [`models/SEIFNet.py`](models/SEIFNet.py) |
| [RFANet](https://github.com/Youzhihui/RFANet) | ISPRS 2024 | [`models/RFANet.py`](models/RFANet.py) |
| [LCD-Net](https://github.com/WenyuLiu6/LCD-Net) | JSTARS 2025 | [`models/LCDNet.py`](models/LCDNet.py) |
| [SFEARNet](https://github.com/miao-0417/SFEARNet) | TGRS 2025 | [`models/SFEARNet.py`](models/SFEARNet.py) |
| [LWGANet](https://github.com/AeroVILab-AHU/LWGANet) | AAAI 2026 | [`models/LWGANet_CD.py`](models/LWGANet_CD.py) |

---

## 3. Usage

### 3.1 Datasets

The experiments are conducted on two publicly available SAR flood detection datasets:

- **S1GFloods:** 5,360 pairs of pre-flood and post-flood Sentinel-1 SAR images (256×256), covering 46 flood events across 6 continents from 2015 to 2022. Split into train/val/test at 8:1:1 ratio.

- **ETCI-2021:** Released for the NASA IMPACT ETCI 2021 Competition on Flood Detection. After data cleaning, 20,758 image pairs (256×256) spanning five geographic regions. Split into train/val/test at 8:1:1 ratio.



#### Dataset Structure

Crop all datasets into 256×256 patches and organize as:

```
datasets/
├── A/              # Pre-event images
├── B/              # Post-event images
├── label/          # Ground truth masks
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Generate list files by running:
```shell
ls -R ./label/* > test.txt
```

> **Quick Start:** A small sample dataset is provided in [`./samples/`](samples/) for verifying the pipeline.

### 3.2 Environment Setup

```shell
conda create -n CFBNet python=3.8
conda activate CFBNet
pip install -r requirements.txt
```

**Core dependencies:** PyTorch 1.7.1+, torchvision 0.8.2+, NumPy, OpenCV, Pillow, SciPy, Matplotlib, tqdm, einops, thop.

### 3.3 Training

```shell
sh ./train_test_tools/train.sh
```

Or directly:

```shell
python ./train_test_tools/train.py \
    --file_root <dataset_name> \
    --lr 5e-4 \
    --max_steps 100ep \
    --batch_size 16 \
    --inWidth 256 \
    --inHeight 256
```

**Key hyperparameters:**

| Parameter | Default | Description |
| `--file_root` | — | Dataset name (`S1G`, `etci`, `URBAN`, `quick_start`) |
| `--lr` | `5e-4` | Initial learning rate |
| `--max_steps` | `40000` | Total training iterations |
| `--batch_size` | `16` | Batch size per GPU |
| `--lr_mode` | `poly` | Learning rate schedule (`step` or `poly`) |
| `--inWidth` / `--inHeight` | `256` | Input image resolution |

The training script uses polynomial learning rate decay with warm-up, random scaling, cropping, flipping, and channel exchange for data augmentation. After each epoch, validation F1 is evaluated and the best checkpoint is saved.

### 3.4 Testing

```shell
sh ./train_test_tools/test.sh
```

Or:

```shell
python ./train_test_tools/test.py \
    --file_root <dataset_name> \
    --batch_size 1 \
    --lr 5e-4 \
    --max_steps 40000
```

The test script outputs:
- **Per-image prediction maps** (TP/FP/TN/FN color-coded) in `./Predict/<dataset_name>/`
- **Per-image metrics** (Boundary F1, Precision, Recall) saved to `test_cd_metrics.xlsx`
- **Overall metrics** saved as `.mat` file

---


## 4. Code Structure

```
CFB-Net/
├── models/
│   ├── cfb_net.py              # CFB-Net (MSAB + CSFB + TFF + Decoder)
│   ├── MobileNetV2.py          # Lightweight backbone
│   ├── RFANet.py               # RFANet (ISPRS 2024)
│   ├── BITNet.py               # BIT (TGRS 2021)
│   ├── SNUNet.py               # SNUNet (GRSL 2021)
│   ├── AFCFNet.py              # AFCFNet (TGRS 2023)
│   ├── ELWCDNet.py             # ELW_CDNet (GRSL 2023)
│   ├── ACAHNet.py              # ACAHNet (TGRS 2023)
│   ├── LiSTNet.py              # LiST-Net (TGRS 2024)
│   ├── EGENet.py               # EGENet (IGARSS 2024)
│   ├── SEIFNet.py              # SEIFNet (TGRS 2024)
│   ├── LCDNet.py               # LCD-Net (JSTARS 2025)
│   ├── SFEARNet.py             # SFEARNet (TGRS 2025)
│   ├── LWGANet_CD.py           # LWGANet (AAAI 2026)
│   ├── LWGANet_backbone.py     # LWGANet backbone components
│   ├── ShuffleNetV2.py         # ShuffleNetV2 backbone
│   ├── ViTAEv2.py              # ViTAEv2 backbone
│   ├── ResNet.py               # ResNet backbone
│   └── __init__.py             # Model registry
├── train_test_tools/
│   ├── train.py                # Training script
│   ├── test.py                 # Testing & evaluation script
│   ├── train.sh                # Training shell launcher
│   ├── test.sh                 # Testing shell launcher
│   └── torchutils.py           # Torch utilities
├── dataset.py                  # Bi-temporal CD dataset loader
├── Transforms.py               # Data augmentation transforms
├── metric_tool.py              # Evaluation metrics (IoU, F1, Boundary-F1)
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── samples/                    # Quick-start sample data
├── assets/                     # Architecture and result figures
└── README.md
```

---

## 5. Acknowledgement

This repository is built with reference to the following open-source projects:

| Method | Repository |
|:--------|:-----------|
| BIT | [https://github.com/justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD) |
| SNUNet | [https://github.com/likyoo/Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet) |
| AFCFNet | [https://github.com/wm-Githuber/AFCF3D-Net](https://github.com/wm-Githuber/AFCF3D-Net) |
| ELW_CDNet | [https://github.com/dyl96/ELW_CDNet](https://github.com/dyl96/ELW_CDNet) |
| ACAHNet | [https://github.com/CCRG-XJU/ChangeDetection_ACAHNet_TGRS2023](https://github.com/CCRG-XJU/ChangeDetection_ACAHNet_TGRS2023) |
| LiST-Net | [https://github.com/Tamer-Saleh](https://github.com/Tamer-Saleh) |
| EGENet | [https://github.com/Jnmz/EGENet-IG24](https://github.com/Jnmz/EGENet-IG24) |
| SEIFNet | [https://github.com/lixinghua5540/SEIFNet](https://github.com/lixinghua5540/SEIFNet) |
| RFANet | [https://github.com/Youzhihui/RFANet](https://github.com/Youzhihui/RFANet) |
| LCD-Net | [https://github.com/WenyuLiu6/LCD-Net](https://github.com/WenyuLiu6/LCD-Net) |
| SFEARNet | [https://github.com/miao-0417/SFEARNet](https://github.com/miao-0417/SFEARNet) |
| LWGANet | [https://github.com/AeroVILab-AHU/LWGANet](https://github.com/AeroVILab-AHU/LWGANet) |
| A2Net | [https://github.com/guanyuezhen/A2Net](https://github.com/guanyuezhen/A2Net) |
| CDLab | [https://github.com/Bobholamovic/CDLab](https://github.com/Bobholamovic/CDLab) |
| MobileSal | [https://github.com/yuhuan-wu/MobileSal](https://github.com/yuhuan-wu/MobileSal) |

All code is provided for academic use only.

---


