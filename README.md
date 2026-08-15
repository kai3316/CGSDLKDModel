# CGSDLKDModel

**Dual-Strategy Knowledge Distillation-Based Lightweight Detector for Navel Orange in Complex Scenarios**

Official implementation of **CGSDLKDModel**, a compact navel-orange detector co-designed through targeted architectural modules and a dual-strategy knowledge-distillation scheme for edge deployment in complex orchard environments.

> **Manuscript status:** under review at *Computers and Electronics in Agriculture* (Manuscript No. COMPAG-D-26-00963). The citation below will be finalized upon acceptance.

---

## Overview

Orchard fruit detection is challenged by cluttered foliage, severe occlusion, frequent fruit overlap, large scale variation, and limited on-device compute. This work derives a lightweight detector from YOLOv11n and recovers the accuracy lost during compression with a co-designed distillation scheme.

Four architectural modules, each targeting a specific orchard difficulty, are introduced:

| Module | Purpose |
|---|---|
| **Gr-CSP** | Grouped residual feature extraction for cluttered foliage |
| **Dy-Sample** | Content-adaptive upsampling for overlapping fruit boundaries |
| **Sf-Conv** | Shared-weight multi-scale fusion for extreme scale variation |
| **Led-Head** | Instance-mask-augmented detection for dense clusters |

A **dual-strategy distillation** then transfers knowledge from an improved YOLOv11m teacher (GSDL-YOLOv11m):

- **Feature-level:** multi-layer channel-wise feature distillation (CWD) across backbone, neck, and head;
- **Output-level:** confidence-weighted prediction distillation that stabilizes classification under illumination variation and dense overlap.

## Main Results

Relative to the original YOLOv11m baseline, CGSDLKDModel retains **90.8% mAP50** while reducing parameters by **97.1%** (0.578 M) and FLOPs by **97.9%**, at a model size of **1.4 MB**.

| Metric | Value |
|---|---|
| mAP50 | 90.8% |
| Precision / Recall | 93.0% / 85.3% |
| mAP50–95 | 74.1% |
| Parameters | 0.578 M |
| Model size | 1.4 MB |
| FPS (RTX 2080 Ti) | 78.4 |
| FPS (Raspberry Pi 5) | 5.75 (173.8 ms) |
| Power (Raspberry Pi 5, compute unit only) | 4.32 W |

### Cross-dataset applicability

CGSDLKDModel is retrained from scratch on each public benchmark (the experiments test architectural applicability across crops, not zero-shot generalization):

| Dataset | mAP50 |
|---|---|
| MinneApple | 83.2% |
| Mango dataset-1 | 98.2% |

## Repository Structure

```
CGSDLKDModel/
├── README.md
├── models/            # module definitions (Gr-CSP, Dy-Sample, Sf-Conv, Led-Head)  [to be uploaded]
├── distillation/      # dual-strategy distillation losses                          [to be uploaded]
├── configs/           # model & training configuration files                       [to be uploaded]
├── tools/             # train / eval / export scripts                              [to be uploaded]
├── weights/           # trained weights (released separately)                      [to be uploaded]
└── data/              # dataset preparation scripts                                [to be uploaded]
```

> Code, configuration files, and trained weights are being prepared for release and will be uploaded shortly.

## Requirements

- Python 3.10+
- PyTorch 2.2
- NVIDIA GPU with 12 GB+ VRAM for training (tested on an RTX 2080 Ti)

## Training

All experiments follow a fixed protocol (single training run):

- Input resolution: 640×640
- Optimizer: SGD, learning rate 0.01
- Batch size: 16
- Epochs: 200
- Workers: 8
- Random initialization

## Edge Deployment

Benchmarked on a Raspberry Pi 5 (Broadcom BCM2712, quad-core Cortex-A76, 2.4 GHz, 8 GB RAM) with FP32 inference at 640×640, batch size 1, 4 CPU threads (averaged over 300 runs). The reported power consumption refers to the Raspberry Pi computing unit only and excludes camera, display, wireless, and power-conversion losses.

The 5.75 FPS throughput supports fixed or low-frequency field monitoring; higher frame rates for mobile operation can be pursued via INT8 quantization or dedicated inference accelerators.

## Citation

If you use this work, please cite the paper (final citation to be completed upon acceptance):

```bibtex
@article{hua2025cgsdlkd,
  title   = {Dual-Strategy Knowledge Distillation-Based Lightweight Detector
             for Navel Orange in Complex Scenarios},
  author  = {Jing Hua and Jize Deng and Binfeng Tang and Jingqiu Zhang and Hua Yin and Kai Su},
  journal = {Computers and Electronics in Agriculture},
  note    = {under review},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
