# Weights

Four checkpoints are provided. Each is the best.pt from the corresponding training run, copied from the original runs/ directory (source path kept below).

| File | Model | Params | Source path |
|------|-------|--------|-------------|
| teacher_gsdl_yolo11m_best.pt | GSDL-YOLOv11m (teacher) | 17.185M | runs/train/techer-/weights/best.pt |
| student_gsdl_yolo11n_best.pt | GSDL-YOLOv11n | 2.132M | runs/train/student-/weights/best.pt |
| student_compressed_cgsdl_yolo11n_best.pt | CGSDL-YOLOv11n (compressed) | 0.578M | runs/train/yolo11-studentCCC.yaml-/weights/best.pt |
| distilled_cgsdlkdmodel_best.pt | CGSDLKDModel (distilled) | 0.578M | runs/distill/KD/weights/best.pt |

Correspondence to the paper tables:

- teacher_gsdl_yolo11m_best.pt is GSDL-YOLOv11m (92.5% mAP50).
- student_gsdl_yolo11n_best.pt is GSDL-YOLOv11n (90.8% mAP50, 2.132M).
- student_compressed_cgsdl_yolo11n_best.pt is CGSDL-YOLOv11n (89.4% mAP50, 0.578M), the "Compress" row in the student ablation.
- distilled_cgsdlkdmodel_best.pt is the final CGSDLKDModel (90.8% mAP50, 0.578M, 1.4 MB). This run used four-layer feature alignment (12,15,18,21) with CWD and L2 output distillation.

Note: the val mAP50 logged in the source run's results.csv (about 89.9%) is the val-split metric, while the paper reports the test-split mAP50 (90.8%). Please check this against your own records.

Other runs (not shipped):

The original runs/ directory also has the layer-ablation and cross-dataset runs. Copy a specific checkpoint from the source tree if needed:

- runs/distillXR/KD12, KD15, KD18, KD21: single-layer feature distillation (configs 1-4).
- runs/distillXR/KD12,15, KD12,15,18, KD15,21, KD18,21: multi-layer combinations (configs 5-9).
- runs/distillXR/mango1: Mango dataset-1 distilled model.
- runs/trainM/yolo11-techer.yaml5: teacher trained on Mango dataset-1.
