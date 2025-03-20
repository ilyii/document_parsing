import os
from ultralytics import YOLO


MODEL_WEIGHTS = "yolo11s.pt"
# DATA_YAML = r"C:\Users\ihett\Downloads\signatures_dataset_padded\data.yaml"
# DATA_YAML = r"C:\Users\ihett\Downloads\stampdet.v18i.yolov11\data.yaml"
# DATA_YAML = r"C:\Users\ihett\Downloads\Detector firma.v1-version-1.yolov11\data.yaml"
# DATA_YAML = r"C:\Users\ihett\Workspace\document_parsing\src\dla\yolo\signature.yaml"
DATA_YAML = r"C:\Users\ihett\Downloads\stampdet.v18i.yolov11\data.yaml"
CFG_YAML = r"src\dla\yolo\cfg.yaml"

if __name__ == "__main__":
    model = YOLO("yolov8m.pt").to("cuda")

    args = {
        "data": DATA_YAML,
        # "cfg": CFG_YAML,
        "epochs": 250,
        "imgsz": 480,
        "batch": -1,
        "project": "runs/dla",
        "name": "yolo_signature",
        "workers": 8,
        # "fraction": 0.3
    }

    results = model.train(**args)
