import os

import torch
from ultralytics import YOLO
import argparse


# MODEL_WEIGHTS = "yolov8s.pt"
# DATA_YAML = r"X:\doc_layout_analysis\bddla\data.yaml"
# CFG_YAML = r"src\dla\yolo\cfg.yaml"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str)
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--project", type=str, default="runs/")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--resume", type=bool, default=False)

    return parser.parse_args()


def transfer_weights(weights, model):
    weights_dict = torch.load(weights)
    model_dict = model.state_dict()
    # overwrite the model dict with all weights that are in the weights dict
    model_dict.update({k: v for k, v in weights_dict.items() if k in model_dict})
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    opt = get_args()
    model = YOLO(opt.weights).to("cuda")
    base_model = YOLO("yolo11x.pt")
    model = transfer_weights(opt.weights, base_model).to("cuda")

    args = {
        "data": opt.data,
        # "cfg": opt.cfg,
        "epochs": opt.epochs,
        "imgsz": opt.imgsz,
        "batch": opt.batch,
        "project": opt.project,
        "name": opt.name,
        "workers": 8,
        "resume": opt.resume
    }

    results = model.train(**args)


