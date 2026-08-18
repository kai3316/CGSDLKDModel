"""
Dual-strategy knowledge distillation for CGSDLKDModel.

Teacher = GSDL-YOLOv11m;  student = compressed CGSDL-YOLOv11n.
    - feature-level  : channel-wise distillation (CWD), layers 12,15,18,21
    - output-level   : confidence-weighted L2 distillation

Edit the paths below (data / teacher_weights), then run:
    python distill.py
"""
import warnings

warnings.filterwarnings("ignore")

from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == "__main__":
    param_dict = {
        # ---- student & data ----
        "model": "configs/yolo11-studentCCC.yaml",   # compressed student (0.578M)
        "data": "datasets/navel_orange.yaml",        # your dataset config
        "imgsz": 640,
        "epochs": 200,
        "batch": 16,
        "workers": 8,
        "cache": True,
        "optimizer": "SGD",
        "device": "0",
        "close_mosaic": 20,
        "project": "runs/distill",
        "name": "KD",

        # ---- teacher ----
        # NOTE: the teacher architecture is read from the .pt checkpoint (its embedded
        # yaml), so no separate config path is needed here.
        "prune_model": False,
        "teacher_weights": "weights/teacher_gsdl_yolo11m_best.pt",

        # ---- distillation signals ----
        "kd_loss_type": "all",             # 'feature' | 'logical' | 'all'
        "kd_loss_decay": "constant",

        "logical_loss_type": "l2",         # output distillation (confidence-weighted L2)
        "logical_loss_ratio": 1.0,

        # Feature-layer ablation (paper Table "Feature-layer distillation ablation"):
        #   config 10 (full):  '12,15,18,21'   (used below)
        #   single layers:    '12' | '15' | '18' | '21'
        "teacher_kd_layers": "12,15,18,21",
        "student_kd_layers": "12,15,18,21",
        "feature_loss_type": "cwd",         # channel-wise distillation (also: 'mgd', 'mimic')
        "feature_loss_ratio": 1.0,
    }

    # If the distillation loss becomes NaN on some GPUs, disable AMP:
    #   param_dict["amp"] = False

    model = DetectionDistiller(overrides=param_dict)
    model.distill()
