import os
import cv2
import numpy as np
import json
import imagehash
from PIL import Image

# ---- Configuration ---- #
TARGET_RESOLUTION = (1024, 768)  # Expected resolution (width, height)
ASPECT_RATIO = 4 / 3  # Expected aspect ratio
ANNOTATION_THRESHOLD = 1  # Minimum annotations required
BLUR_THRESHOLD = 100  # Variance of Laplacian threshold for sharpness detection
HASH_DIFFERENCE_THRESHOLD = 5  # Maximum Hamming distance for duplicate detection
OCR_CONFIDENCE_THRESHOLD = 0.7  # If using OCR confidence

# ---- Utility Functions ---- #
def load_yolo_annotations(annotation_dir):
    """Loads YOLO annotations from a folder containing .txt files."""
    annotations = {}
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            image_name = filename.replace(".txt", ".jpg")
            with open(os.path.join(annotation_dir, filename), 'r') as f:
                lines = f.readlines()
            annotations[image_name] = len(lines)  # Count of bounding boxes
    return annotations

def check_annotations(image_name, annotations):
    """Checks if an image has sufficient YOLO annotations."""
    return annotations.get(image_name, 0) >= ANNOTATION_THRESHOLD

def check_resolution(image_path):
    """Checks if an image matches the target resolution."""
    image = cv2.imread(image_path)
    if image is None:
        return False
    h, w = image.shape[:2]
    return (w, h) == TARGET_RESOLUTION

def check_aspect_ratio(image_path):
    """Ensures image aspect ratio matches expected value."""
    image = cv2.imread(image_path)
    if image is None:
        return False
    h, w = image.shape[:2]
    return abs((w / h) - ASPECT_RATIO) < 0.05  # 5% tolerance

def detect_blurriness(image_path):
    """Detects blurriness using variance of Laplacian."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > BLUR_THRESHOLD

def perceptual_hash(image_path):
    """Computes perceptual hash for image similarity detection."""
    image = Image.open(image_path)
    return imagehash.phash(image)

def filter_duplicates(image_paths):
    """Filters near-duplicate images using perceptual hashing."""
    unique_images = []
    hashes = {}
    for img_path in image_paths:
        img_hash = perceptual_hash(img_path)
        if any(img_hash - h < HASH_DIFFERENCE_THRESHOLD for h in hashes.values()):
            continue
        hashes[img_path] = img_hash
        unique_images.append(img_path)
    return unique_images

def dataset_statistics(image_dir, annotation_dir):
    """Computes dataset statistics before curation for YOLO format."""
    annotations = load_yolo_annotations(annotation_dir) if annotation_dir else {}
    total_images = len([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    annotated_images = sum(1 for img in os.listdir(image_dir) if check_annotations(img, annotations)) if annotation_dir else 0
    correct_resolution = sum(1 for img in os.listdir(image_dir) if check_resolution(os.path.join(image_dir, img)))
    correct_aspect_ratio = sum(1 for img in os.listdir(image_dir) if check_aspect_ratio(os.path.join(image_dir, img)))
    sharp_images = sum(1 for img in os.listdir(image_dir) if detect_blurriness(os.path.join(image_dir, img)))
    
    stats = {
        "Total Images": total_images,
        "Annotated Images": annotated_images,
        "Correct Resolution": correct_resolution,
        "Correct Aspect Ratio": correct_aspect_ratio,
        "Sharp Images": sharp_images
    }
    return stats

def curate_images(image_dir, annotation_dir=None):
    """Main function to curate document images in YOLO format."""
    annotations = load_yolo_annotations(annotation_dir) if annotation_dir else {}
    curated_images = []
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        
        if annotation_dir and not check_annotations(image_name, annotations):
            continue  # Skip unannotated images
        if not check_resolution(image_path):
            continue  # Skip images with incorrect resolution
        if not check_aspect_ratio(image_path):
            continue  # Skip images with incorrect aspect ratio
        if not detect_blurriness(image_path):
            continue  # Skip blurry images
        
        curated_images.append(image_path)
    
    curated_images = filter_duplicates(curated_images)  # Remove near-duplicates
    return curated_images

# ---- Run Curation ---- #
if __name__ == "__main__":
    image_directory = "path/to/image/folder"
    annotation_directory = "path/to/annotation/folder"  # YOLO format annotations
    
    stats = dataset_statistics(image_directory, annotation_directory)
    print("Dataset Statistics Before Curation:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    curated_list = curate_images(image_directory, annotation_directory)
    print(f"Curated {len(curated_list)} images out of {len(os.listdir(image_directory))}")
