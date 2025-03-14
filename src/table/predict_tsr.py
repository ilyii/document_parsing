import os
import sys

import cv2
from ultralytics import YOLO

IMAGES_DIR = sys.argv[1]
MODEL_WEIGHTS_PATH = (
    r"D:\workspace\doc-understanding\tsr\cisol_plus_tucd\weights\best.pt"
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
        print(boxes)
        xyxys = boxes.xyxy
        classes = boxes.cls
        img = result.orig_img
        for xyxy, cl in zip(xyxys, classes):
            xyxy = xyxy.int().tolist()
            cl = cl.int().item()
            if int(cl) == 2:
                img = cv2.rectangle(
                    img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
                )
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
