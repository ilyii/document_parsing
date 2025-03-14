from ultralytics import YOLO

MODEL_CFG_PATH = r"D:\workspace\doc-understanding\weights\yolov10m.yaml"
MODEL_PT_WEIGHTS_PATH = r"C:\Users\ihett\Downloads\yolov10x_layout.pt"
DATA_CFG_PATH = r"X:\tsr\data.yaml"

if __name__ == "__main__":
    model = YOLO(
        r"D:\workspace\doc-understanding\tsr\cisol_plus_tucd\weights\last.pt"
    ).to("cuda")
    results = model.train(
        data=DATA_CFG_PATH,
        epochs=100,
        imgsz=640,
        project="tsr",
        name="cisol_plus_tucd",
        resume=True,
        workers=8,
        batch=8,
    )
