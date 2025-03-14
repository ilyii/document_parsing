from ultralytics import YOLO

DATA_CFG_PATH = r"X:\doc_layout_analysis\BDDLA\data.yaml"

if __name__ == "__main__":
    model = YOLO("yolo11s.pt").to("cuda")
    results = model.train(
        data=DATA_CFG_PATH,
        epochs=100,
        imgsz=480,
        project="runs/dla",
        name="yolov11s_v1",
        # resume=True,
        workers=8,
        batch=40,
    )
