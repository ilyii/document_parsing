import json
import os
import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path


def draw_bboxes(img:np.ndarray, bboxes:list, clss:int):
    for bbox, cls in zip(bboxes, clss):
        xcenter, ycenter, w, h = bbox
        xcenter = xcenter * img.shape[1]
        ycenter = ycenter * img.shape[0]
        w = w * img.shape[1]
        h = h * img.shape[0]

        x1 = int(xcenter - w/2)
        y1 = int(ycenter - h/2)
        x2 = int(xcenter + w/2)
        y2 = int(ycenter + h/2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, cls, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0,0), 2)
    return img

def create_collage(imgs):
    # Create nearly square collage
    n = len(imgs)
    rows = int(np.sqrt(n))
    cols = int(n / rows)
    if rows * cols < n:
        cols += 1

    # Resize images to same size
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]

    # Create collage
    collage = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        collage[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return collage


if __name__ == "__main__":

    N = 25
    SRC = sys.argv[1]
    image_dir = os.path.join(SRC, "images")
    label_dir = os.path.join(SRC, "labels")
    ID2LABEL = json.load(open(os.path.join(SRC, "id2label.json")))

    imagepaths = random.sample(os.listdir(image_dir), N)
    imagepaths = [os.path.join(image_dir, p) for p in imagepaths]
    labelpaths = [os.path.join(label_dir, Path(p).stem+ ".txt") for p in imagepaths]

    imgs = []
    for imagepath, labelpath in zip(imagepaths, labelpaths):
        img = cv2.imread(imagepath)
        with open(labelpath, "r") as f:
            lines = f.readlines()
            bboxes = []
            clss = []
            for line in lines:
                cls, xcenter, ycenter, w, h = list(map(float, line.split()))
                bboxes.append([xcenter, ycenter, w, h])
                clss.append(ID2LABEL[str(int(cls))])
            img = draw_bboxes(img, bboxes, clss)
        imgs.append(img)

    collage = create_collage(imgs)
    plt.imshow(collage)
    plt.show()





    