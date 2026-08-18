# CGSDLKDModel

Code for the paper "Dual-Strategy Knowledge Distillation-Based Lightweight Detector for Navel Orange in Complex Scenarios" (Computers and Electronics in Agriculture, under review).

This code is a modification of Ultralytics YOLO11 (version 8.3.9). On top of the upstream code we added four modules and a knowledge-distillation trainer. The upstream project is copyright Ultralytics and distributed under AGPL-3.0; this repository keeps the same license. See MODULES.md for what was changed.

## Directory layout

```
CGSDLKDModel/
├── ultralytics/                        modified ultralytics package
│   ├── nn/extra_modules/block.py       Gr-CSP, Dy-Sample, Sf-Conv
│   ├── nn/extra_modules/head.py        Led-Head
│   ├── nn/tasks.py                     model parser
│   ├── models/yolo/detect/distill.py   distillation trainer
│   └── utils/distill_loss.py           distillation losses
├── configs/                            model yaml configs
├── weights/                            trained checkpoints
├── datasets/                           dataset config template
├── train.py                            train teacher or student
├── distill.py                          distillation training
├── val.py                              validation
├── detect.py                           inference
├── MODULES.md
├── requirements.txt
└── LICENSE
```

## Module names in the paper vs in the code

| Paper name | Code class | File |
|------------|------------|------|
| Gr-CSP | RGCSPELAN | ultralytics/nn/extra_modules/block.py |
| Dy-Sample | DySample | ultralytics/nn/extra_modules/block.py |
| Sf-Conv | FeaturePyramidSharedConv | ultralytics/nn/extra_modules/block.py |
| Led-Head | Detect_Efficient | ultralytics/nn/extra_modules/head.py |

The distillation losses (CWDLoss, LogicalLoss/OutputLoss) are in ultralytics/utils/distill_loss.py. The training loop is DetectionDistiller in ultralytics/models/yolo/detect/distill.py.

## Install

```
conda create -n cgsdl python=3.10 -y
conda activate cgsdl
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu122
pip install -r requirements.txt
pip install -e .
```

The four modules are pure PyTorch and need no CUDA compilation. The repository also contains some optional third-party operators (Mamba, DCNv3/DCNv4, and others) from the upstream fork; they are imported with try/except and are not needed to run this model.

## Dataset

The navel orange dataset has 3015 images, one class, split 7:2:1 (2106 train / 603 val / 306 test). Put the images and labels in YOLO format and set the path in datasets/navel_orange.yaml:

```
path: /path/to/navel_orange_dataset
train: images/train
val: images/val
test: images/test
names:
  0: navel-orange
```

The image and label files are not included in this release. See datasets/README.md.

## Reproduce

1. Train the teacher (GSDL-YOLOv11m). Set MODEL = "configs/yolo11-techer.yaml" in train.py, then run `python train.py`.

2. Train the student. Set MODEL = "configs/yolo11-student.yaml" (2.132M) or "configs/yolo11-studentCCC.yaml" (0.578M), then run `python train.py`.

3. Distill. Check the paths in distill.py, then run `python distill.py`. This produces the final CGSDLKDModel.

4. Evaluate and detect with `python val.py` and `python detect.py`.

## Weights

| File | Model | Params |
|------|-------|--------|
| teacher_gsdl_yolo11m_best.pt | GSDL-YOLOv11m | 17.185M |
| student_gsdl_yolo11n_best.pt | GSDL-YOLOv11n | 2.132M |
| student_compressed_cgsdl_yolo11n_best.pt | CGSDL-YOLOv11n | 0.578M |
| distilled_cgsdlkdmodel_best.pt | CGSDLKDModel | 0.578M |

See weights/README.md for the mapping to the paper tables.

## License

This code is based on the Ultralytics YOLO11 project (https://github.com/ultralytics/ultralytics), and is released under the same AGPL-3.0 license.
