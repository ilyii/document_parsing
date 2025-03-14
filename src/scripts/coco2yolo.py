from collections import defaultdict
import json
import os
import sys
from tqdm import tqdm


def coco2yolo(json_path: str, save_path: str = None):
    """COCO json to YOLO formatter.
    
    Args:
    -----
    json_path : str
        Path to COCO json file.
    save_path : str
        Path to save YOLO formatted annotations.
    """
    def _convert(annotation, image_width, image_height):
        x, y, w, h = annotation["bbox"]

        # Normalize values
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        w /= image_width
        h /= image_height
        return f"{annotation['category_id']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
    
    annots = json.load(open(json_path, "r", encoding="utf-8"))
    
    # Group annotations by image_id
    grouped_annots = defaultdict(list)
    for a in annots["annotations"]:
        grouped_annots[a["image_id"]].append(a)

    # Convert annotations
    for image_info in tqdm(annots["images"]):
        image_id = image_info["id"]
        image_width, image_height = image_info["width"], image_info["height"]
        yolo_annots = [
            _convert(a, image_width, image_height) for a in grouped_annots[image_id]
        ]
    
        if save_path:
            output_filename = os.path.join(save_path, os.path.splitext(image_info["file_name"])[0] + ".txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_annots))


if __name__ == "__main__":
    # Usage: python coco2yolo.py path_to_coco_json save_dir
    jsonp = sys.argv[1]
    savedir = None
    if len(sys.argv) > 2:
        savedir = sys.argv[2]

    os.makedirs(savedir, exist_ok=True)
    
    coco2yolo(jsonp, savedir)