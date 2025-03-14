from collections import defaultdict
import os


def find_dirs(root):
    for path, dirs, files in os.walk(root):
        if "images" in dirs and "labels" in dirs:
            yield path

        # for dir in dirs:
        #     if "images" in os.listdir(os.path.join(path, dir)) and "labels" in os.listdir(os.path.join(path, dir)):
        #         yield os.path.join(path, dir)


def validate(images, labels):
    if len(images) != len(labels):
        print(f"Number of images and labels do not match: {len(images)} images, {len(labels)} labels")
        return
    
    images = sorted(images)
    labels = sorted(labels)

    correct, incorrect = [], []
    for img, lbl in zip(images, labels):
        img_name = ".".join(os.path.basename(img).split(".")[:-1])
        lbl_name = ".".join(os.path.basename(lbl).split(".")[:-1])
        if img_name != lbl_name:
            incorrect.append((img, lbl))
        else:
            correct.append((img, lbl))

    if len(incorrect) > 0:
        print(f"Correct: {len(correct)}")
        print(f"Incorrect: {len(incorrect)}")
        return False
    
    return True


def get_class_distribution(labels):
    class_dist = defaultdict(int)
    for lbl in labels:
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f.readlines():
                class_dist[line.split()[0]] += 1
    return class_dist
        

def main():
    imagelist = list()
    label_distribution = defaultdict(dict)
    for root in ROOTS:
        src_dirs = find_dirs(root)
        images, labels = [], []
        for src_dir in src_dirs:
            images.extend([os.path.join(src_dir, "images", img) for img in os.listdir(os.path.join(src_dir, "images"))])
            labels.extend([os.path.join(src_dir, "labels", lbl) for lbl in os.listdir(os.path.join(src_dir, "labels"))])

        print(f"----- {root} -----")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")
        if validate(images, labels):
            print("> Validation passed")
        else:
            print("> Validation failed!")
            continue

        imagelist.extend(images)

        class_dist = get_class_distribution(labels)
        label_distribution[root] = class_dist


    print(f"Total number of images: {len(imagelist)}")
    for root, class_dist in label_distribution.items():
        print(f"----- {root} -----")
        
        for cls, count in class_dist.items():
            print(f"{cls}: {count}")

    
  

if __name__ == "__main__":    
    TRAIN_N = 2000
    VAL_N = 75
    ROOTS = [
        r"X:\doc_layout_analysis\publaynet\train-0",
        r"X:\doc_layout_analysis\logo",
        r"X:\doc_layout_analysis\signature",
        r"X:\doc_layout_analysis\signatures.v2-release.yolov11"
        

        ]
    main()







