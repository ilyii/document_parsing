import os
from ultralytics import RTDETR


MODEL_WEIGHTS = "rtdetr-l.pt"
# DATA_YAML = r"C:\Users\ihett\Downloads\signatures_dataset_padded\data.yaml"
DATA_YAML = r"C:\Users\ihett\Downloads\Fin.v1i.yolov11\data.yaml"
CFG_YAML = r"src\dla\yolo\cfg.yaml"

if __name__ == "__main__":
    model = RTDETR(MODEL_WEIGHTS).to("cuda")

    args = {
        "data": DATA_YAML,
        # "cfg": CFG_YAML,
        "epochs": 500,
        "imgsz": 480,
        "batch": -1,
        "project": "runs/dla",
        "name": "rtdetr",
        "workers": 8,
        # "fraction": 0.3
    }

    results = model.train(**args)
