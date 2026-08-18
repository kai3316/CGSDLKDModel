"""
Train the teacher (GSDL-YOLOv11m) or a student (GSDL-YOLOv11n / compressed CGSDL-YOLOv11n).

Edit MODEL / DATA / NAME below, then run:
    python train.py
"""
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

# ---- edit these ----
MODEL = "configs/yolo11-techer.yaml"      # teacher:  yolo11-techer.yaml
                                          # student:  yolo11-student.yaml (2.132M)
                                          #           yolo11-studentCCC.yaml (0.578M, compressed)
DATA = "datasets/navel_orange.yaml"       # your dataset config
NAME = "exp"                              # run name (saved under runs/train/<name>)

if __name__ == "__main__":
    model = YOLO(MODEL)
    # model.load("yolo11n.pt")  # optional: initialize from pretrained weights

    model.train(
        data=DATA,
        imgsz=640,               # input resolution
        epochs=200,              # training epochs
        batch=16,                # batch size
        workers=8,               # dataloader workers (set 0 on Windows if it hangs)
        cache=True,              # cache images to RAM
        close_mosaic=20,         # disable mosaic in the last 20 epochs
        optimizer="SGD",         # optimizer
        lr0=0.01,                # initial learning rate
        device="0",              # GPU id ('' for CPU)
        project="runs/train",
        name=NAME,
    )
