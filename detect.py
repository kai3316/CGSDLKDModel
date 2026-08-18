"""
Run inference / prediction with a trained model.

Edit SOURCE, then run:
    python detect.py
"""
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("weights/distilled_cgsdlkdmodel_best.pt")  # trained weight path

    model.predict(
        source="datasets/images/test",   # image / folder / video / webcam
        imgsz=640,
        project="runs/detect",
        name="exp",
        save=True,
        conf=0.25,             # confidence threshold
        # iou=0.7,
        # line_width=2,
        # show_labels=False,
        # save_txt=True,       # save YOLO-format .txt predictions
    )
