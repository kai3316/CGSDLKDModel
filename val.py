"""
Validate a trained model (mAP / Precision / Recall).

Edit MODEL and DATA, then run:
    python val.py
"""
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("weights/distilled_cgsdlkdmodel_best.pt")  # trained weight path

    model.val(
        data="datasets/navel_orange.yaml",
        split="test",          # 'train' | 'val' | 'test'
        imgsz=640,
        batch=16,
        # iou=0.7,             # IoU threshold (default 0.7)
        project="runs/val",
        name="exp",
    )
