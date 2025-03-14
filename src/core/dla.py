from ultralytics import YOLO

ID2LABEL = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


def run_dla(imgp):
    model = YOLO(r"D:\workspace\doc-understanding\weights\yolov10b_dla.pt")
    extracts = []
    labels = []
    results = model(imgp)
    for result in results:
        for e_idx, bbox in enumerate(result.boxes):
            x1, y1, x2, y2 = list(map(int, bbox.xyxy[0].tolist()))
            crop = result.orig_img[y1:y2, x1:x2]
            lbl = int(bbox.cls[0].item())
            extracts.append(crop)
            labels.append(ID2LABEL[lbl])
    return extracts, labels


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    import utils

    SRC = sys.argv[1]
    DST_ROOT = sys.argv[2]
    WEIGHTS_DIR = sys.argv[3]
    WEIGHTS = [os.path.join(WEIGHTS_DIR, f) for f in os.listdir(WEIGHTS_DIR)]

    imagepaths = utils.find_images(SRC)
    
    for weights in WEIGHTS:
        model_name = Path(weights).stem
        # dst_path = os.path.join(DST_ROOT, model_name)
        # os.makedirs(dst_path, exist_ok=True)
        try:
            model = YOLO(weights)
            res = model(imagepaths, save=True, project="dla", name=model_name)
        except Exception as e:
            import traceback
            print(e)
            print(traceback.format_exc())
            continue

    
