import copy
import random
from PIL import Image
import os
import argparse

import cv2
from matplotlib import pyplot as plt
import numpy as np

def resize_with_aspect_ratio(img, size):
    width, height = img.size
    aspect_ratio = width / height
    new_width, new_height = size
    if new_width / new_height > aspect_ratio:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)
    return img.resize((new_width, new_height))
     

def find_images(root) -> list:
    if os.path.isfile(root):
        return [root]
    elif os.path.isdir(root):
        imgs = []
        for root, _, files in os.walk(root):
            imgs.extend(
                [
                    os.path.join(root, f)
                    for f in files
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
            )
        return imgs
    else:
        exit("Could not find images.")


def visualize_annotations(img, boxes):
    img = np.array(img)

    for box in boxes:
        xc, yc, w, h = box
        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2
        cv2.rectangle(img, (int(round(xmin*img.shape[1])), int(round(ymin*img.shape[0]))),
                        (int(round(xmax*img.shape[1])), int(round(ymax*img.shape[0]))), (0, 255, 0), 2) 
    plt.imshow(img)
    plt.show()
    


def main(input_dir, output_dir, height, width):
    annotations = []
    os.makedirs(output_dir, exist_ok=True)

    images = find_images(input_dir)
    for img_path in images:
            lbl_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
            
            img = Image.open(img_path)
            background = Image.new('RGB', (width, height), (255, 255, 255))

            posx = random.randint(0, width - img.width)
            posy = random.randint(0, height - img.height)

            background.paste(img, (posx, posy))
            annotations.append((posx,posy, posx + img.width, posy + img.height))

            new_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                # Align labels to the new image size and position
                for line in f.readlines():
                    cls, xc, yc, w, h = map(float, line.strip().split())
                    xc = (xc * img.width + posx) / width
                    yc = (yc * img.height + posy) / height
                    w = w * img.width / width
                    h = h * img.height / height
                    line = f"{cls} {xc} {yc} {w} {h}"
                    new_lines.append(line)

            
            # Save
            out_img_path = os.path.join(output_dir, os.path.dirname(img_path)[len(input_dir):].lstrip(os.path.sep), os.path.basename(img_path))
            background.save(out_img_path)

            out_lbl_path = os.path.join(output_dir, os.path.dirname(lbl_path)[len(input_dir):].lstrip(os.path.sep), os.path.basename(lbl_path))
            os.makedirs(os.path.dirname(out_lbl_path), exist_ok=True)
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

            # Visualize
            # boxes = [list(map(float, line.strip().split()[1:])) for line in new_lines]
            # visualize_annotations(background, boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and paste them on a blank white background of a given size.")
    parser.add_argument("--imgs", help="Path to the input directory of images")
    parser.add_argument("--dst", help="Path to save the processed images")
    parser.add_argument("--imgsz", type=int, nargs=2, help="HEIGHT WIDTH")
    
    args = parser.parse_args()
    main(args.imgs, args.dst, *args.imgsz)
