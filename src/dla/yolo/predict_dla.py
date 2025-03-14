import os
import sys

import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO


ID2LABEL = {
    0: "text",
    1: "title",
    2: "figure",
    3: "table",
    4: "list",
    5: "form_field",
    6: "checkbox",
    7: "logo",
    8: "signature",
}
IMAGES_DIR = sys.argv[1]
MODEL_WEIGHTS_PATH = (
    r"D:\workspace\doc-understanding\runs\dla\yolov11s_v13\weights\best.pt"
)


def find_images(root) -> list:
    if os.path.isfile(root):
        return [root]
    elif os.path.isdir(root):
        return [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        exit("Could not find images.")


if __name__ == "__main__":
    model = YOLO(MODEL_WEIGHTS_PATH)
    images = find_images(IMAGES_DIR)
    results = model.predict(images)

    for result in results:
        boxes = result.boxes
        xyxys = boxes.xyxy
        classes = boxes.cls
        img = result.orig_img
        for xyxy, cl in zip(xyxys, classes):
            xyxy = xyxy.int().tolist()
            cl = cl.int().item()
            cv2.rectangle(
                img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
            )
            cv2.putText(
                img,
                ID2LABEL[cl],
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
            
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        plt.show()
