import sys

import cv2
import torch
from doctr.models import detection_predictor
import numpy as np

import copy
import os
import sys

import cv2
from ultralytics import YOLO

def find_separators(boxes, axis='x', threshold_multiplier=2):
    edges = []
    for (x1, y1, x2, y2) in boxes:
        if axis == 'x':
            edges.extend([x1, x2])
        else:
            edges.extend([y1, y2])
    edges.sort()
    gaps = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
    if not gaps:
        return []
    median_gap = np.median(gaps)
    threshold = median_gap * threshold_multiplier
    separators = []
    for i, gap in enumerate(gaps):
        if gap > threshold:
            sep = (edges[i] + edges[i+1]) / 2
            separators.append(sep)
    return separators


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


import numpy as np

def cluster_coordinates(coords, tol=5):
    """
    Cluster a sorted list of coordinates using a tolerance value.
    Returns the average value for each cluster.
    """
    coords = sorted(coords)
    clusters = []
    current_cluster = [coords[0]]
    for c in coords[1:]:
        if abs(c - current_cluster[-1]) <= tol:
            current_cluster.append(c)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [c]
    clusters.append(np.mean(current_cluster))
    return clusters

def get_table_structure(bboxes, tol=5):
    """
    Given a list of bounding boxes (xyxy format), recover the grid structure.
    
    Parameters:
      bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
      tol: Tolerance in pixels to cluster similar coordinate values.
    
    Returns:
      x_boundaries: Sorted list of x boundaries (columns)
      y_boundaries: Sorted list of y boundaries (rows)
      grid: A dictionary mapping (row, col) cell indices to the bboxes that cover that cell.
    """
    # Extract all x and y coordinates (edges)
    x_coords = []
    y_coords = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    # Cluster nearby coordinates to get candidate boundaries
    x_boundaries = cluster_coordinates(x_coords, tol=tol)
    y_boundaries = cluster_coordinates(y_coords, tol=tol)
    
    # Sort boundaries (though cluster_coordinates already returns sorted averages)
    x_boundaries = sorted(x_boundaries)
    y_boundaries = sorted(y_boundaries)
    
    # Build a grid mapping: for each cell defined by adjacent boundaries, assign bboxes that span it.
    grid = {}
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
        # Find the grid indices where the bbox starts and ends.
        # Using a simple linear scan: the first boundary that is "close" to or greater than the coordinate.
        start_col = next((i for i, x in enumerate(x_boundaries) if x >= x1 - tol), None)
        end_col   = next((i for i, x in enumerate(x_boundaries) if x >= x2 - tol), None)
        start_row = next((i for i, y in enumerate(y_boundaries) if y >= y1 - tol), None)
        end_row   = next((i for i, y in enumerate(y_boundaries) if y >= y2 - tol), None)
        
        # Skip if any boundary index is not found.
        if None in (start_col, end_col, start_row, end_row):
            continue
        
        # Each bbox might span multiple cells if it covers more than one grid interval.
        # We assign the bbox to every cell that it spans.
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                grid.setdefault((r, c), []).append(bbox)
    
    return x_boundaries, y_boundaries, grid


def compute_dynamic_tolerance(bboxes, factor=0.1):
    """
    Compute a dynamic tolerance based on the text height estimated from bounding boxes.
    
    Parameters:
      bboxes: list of bounding boxes in [x1, y1, x2, y2] format.
      factor: multiplier to scale the median height to compute tolerance.
      
    Returns:
      tolerance: a dynamic tolerance value (in pixels).
    """
    # Compute height for each bounding box (assumed text height)
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    
    # Use median height to reduce outlier influence
    median_height = np.median(heights)
    
    # Dynamic tolerance is a fraction of the median text height
    tolerance = factor * median_height
    return tolerance



def snap_bboxes_to_grid(bboxes, factor=0.1):
    """
    Snap bounding boxes to a coherent grid by clustering and snapping
    their edges, using a dynamic tolerance derived from the median text height.
    
    Args:
        bboxes (list of [x1, y1, x2, y2]): 
            The input bounding boxes in xyxy format.
        factor (float):
            Fraction of the median text height used as tolerance.
    
    Returns:
        snapped_bboxes (list of [x1', y1', x2', y2']):
            The new bounding boxes after snapping edges to cluster lines.
    """
    if not bboxes:
        return []

    # 1) Compute median text height
    heights = [(y2 - y1) for (x1, y1, x2, y2) in bboxes]
    median_height = np.median(heights) if heights else 0.0
    # If everything is zero or we have an empty set, fallback to some default
    if median_height <= 0:
        median_height = 10.0

    # 2) Define dynamic tolerance
    tolerance = factor * median_height

    # 3) Gather all x and y coordinates
    x_coords = []
    y_coords = []
    for (x1, y1, x2, y2) in bboxes:
        x_coords.append(x1)
        x_coords.append(x2)
        y_coords.append(y1)
        y_coords.append(y2)

    # 4) Cluster function
    def cluster_values(vals, tol):
        """
        Merge close values in sorted list into clusters. 
        Return the mean for each cluster.
        """
        vals = sorted(vals)
        clusters = []
        current = [vals[0]]
        for v in vals[1:]:
            if abs(v - current[-1]) <= tol:
                current.append(v)
            else:
                clusters.append(np.mean(current))
                current = [v]
        clusters.append(np.mean(current))
        return clusters

    # 5) Perform clustering
    x_clusters = cluster_values(x_coords, tolerance)
    y_clusters = cluster_values(y_coords, tolerance)

    # 6) Snap a single coordinate to the nearest cluster line if within tolerance
    def snap_value_to_clusters(val, clusters, tol):
        # e.g. find the cluster c if |val - c| <= tol, else keep val
        best_c = val
        best_dist = float('inf')
        for c in clusters:
            dist = abs(val - c)
            if dist < best_dist:
                best_dist = dist
                best_c = c
        # Snap only if best_dist <= tol
        return best_c if best_dist <= tol else val

    # 7) Snap bounding boxes
    snapped_bboxes = []
    for (x1, y1, x2, y2) in bboxes:
        # Snap each edge to the nearest cluster line
        new_x1 = snap_value_to_clusters(x1, x_clusters, tolerance)
        new_x2 = snap_value_to_clusters(x2, x_clusters, tolerance)
        new_y1 = snap_value_to_clusters(y1, y_clusters, tolerance)
        new_y2 = snap_value_to_clusters(y2, y_clusters, tolerance)

        # Ensure x1 < x2, y1 < y2 after snapping
        if new_x2 < new_x1:
            new_x1, new_x2 = new_x2, new_x1
        if new_y2 < new_y1:
            new_y1, new_y2 = new_y2, new_y1

        snapped_bboxes.append([new_x1, new_y1, new_x2, new_y2])

    return snapped_bboxes


def detect_table_separation_lines(image, hksize=15, vksize=15):
    """
    Detect horizontal and vertical separation lines in a table image.
    
    This function works dynamically by setting kernel sizes relative 
    to the image dimensions. It first binarizes the image (using Otsu's method)
    then extracts horizontal and vertical lines with morphological operations.
    
    Parameters:
        image (np.array): Input image in BGR or grayscale.
        scale (int): Scale factor to determine the kernel size 
                     relative to image width/height (default=15).
    
    Returns:
        horizontal_lines (list): List of tuples (x1, y, x2, y) for horizontal lines.
        vertical_lines (list): List of tuples (x, y1, x, y2) for vertical lines.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply thresholding (Otsu's method) and invert so that lines are white
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define dynamic kernel sizes based on image dimensions.
    # For horizontal lines, the kernel width is a fraction of image width.
    horiz_kernel_size = (hksize, 1)
    # For vertical lines, the kernel height is a fraction of image height.
    vert_kernel_size = (1, vksize)
    print(horiz_kernel_size, vert_kernel_size)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, horiz_kernel_size)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, vert_kernel_size)

    # Use morphological operations to extract horizontal lines:
    horiz_lines_img = cv2.erode(binary, horiz_kernel, iterations=1)
    horiz_lines_img = cv2.dilate(horiz_lines_img, horiz_kernel, iterations=1)

    # And similarly for vertical lines:
    vert_lines_img = cv2.erode(binary, vert_kernel, iterations=1)
    vert_lines_img = cv2.dilate(vert_lines_img, vert_kernel, iterations=1)

    
    # Find contours for horizontal lines
    contours_h, _ = cv2.findContours(horiz_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_lines = []
    for cnt in contours_h:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter: expect long horizontal line (w should be a significant fraction of image width)
        if w > image.shape[1] * 0.3 and h < image.shape[0] * 0.05:
            y_center = y + h // 2
            horizontal_lines.append((x, y_center, x + w, y_center))

    # Find contours for vertical lines
    contours_v, _ = cv2.findContours(vert_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_lines = []
    for cnt in contours_v:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter: expect tall vertical line (h should be a significant fraction of image height)
        if h > image.shape[0] * 0.3 and w < image.shape[1] * 0.05:
            x_center = x + w // 2
            vertical_lines.append((x_center, y, x_center, y + h))

    return horizontal_lines, vertical_lines


if __name__ == "__main__":

    """
    Algorithm
    - Detect texts
    - Predict table cells
    - Postprocess to find lines
    # - Algorithmically use pypdf to find table lines


    """
    image_orig = find_images(IMAGES_DIR)
    # # calculate max width and height to resize and still fit in 1024x1024
    # max_dim = max(image.shape[:2])
    # scale = 1024 / max_dim
    # image = cv2.resize(image, None, fx=scale, fy=scale)
    # # Pad to 1024x1024
    # background_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # background_img[:image.shape[0], :image.shape[1]] = image
    

    model = detection_predictor(
        arch="db_resnet50",
        pretrained=True,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=True,
    )
    image = copy.deepcopy(image_orig)
    image = cv2.cvtColor(cv2.imread(image[0]), cv2.COLOR_BGR2RGB)
    out = model([image])
    # xmin, ymin, xmax, ymax, prob
    # print(out)

    from doctr.utils.geometry import detach_scores

    def _to_absolute(geom, img_shape: tuple[int, int]) -> list[list[int]]:
        h, w = img_shape
        if len(geom) == 2:  # Assume straight pages = True -> [[xmin, ymin], [xmax, ymax]]
            (xmin, ymin), (xmax, ymax) = geom
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        else:  # For polygons, convert each point to absolute coordinates
            return [[int(point[0] * w), int(point[1] * h)] for point in geom]

    
    for doc, res in zip([image], out):
        img_shape = (doc.shape[0], doc.shape[1])
        # Detach the probability scores from the results
        detached_coords, prob_scores = detach_scores([res.get("words")])

        for i, coords in enumerate(detached_coords[0]):
            coords = coords.reshape(2, 2).tolist() if coords.shape == (4, ) else coords.tolist()

            # Convert relative to absolute pixel coordinates
            points = np.array(_to_absolute(coords, img_shape), dtype=np.int32).reshape((-1, 1, 2))

            # Draw the bounding box on the image
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Save the modified image with bounding boxes
    # cv2.imwrite("output.jpg", image)
    cv2.imshow("doctr", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    image = copy.deepcopy(image_orig)
    model = YOLO(MODEL_WEIGHTS_PATH)
    results = model.predict(image)

    for result in results:
        boxes = result.boxes
        xyxys = boxes.xyxy
        classes = boxes.cls
        img = copy.deepcopy(result.orig_img)
        for xyxy, cl in zip(xyxys, classes):
            xyxy = xyxy.int().tolist()
            cl = cl.int().item()
            if int(cl) == 2:
                img = cv2.rectangle(
                    img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
                )
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        detached_xyxy = xyxys.float().tolist()
        # Vksize - min height of text from text detection (detached_coords)
        print(detached_coords[0])
        proposed_vksize = int(round(min([y2 - y1 for _, y1, _, y2 in detached_coords[0]]) * img.shape[0]))
        proposed_hksize = int(round(min([x2 - x1 for x1, _, x2, _ in detached_coords[0]]) * img.shape[1]))
        

        # Find separation lines
        img2 = copy.deepcopy(result.orig_img)
        horizontal, vertical = detect_table_separation_lines(img2, hksize=proposed_hksize, vksize=proposed_vksize)
        # Draw separation lines
        for (x1, y, x2, _) in horizontal:
            img2 = cv2.line(img2, (x1, y), (x2, y), (0, 0, 255), 2)
        for (x, y1, x, y2) in vertical:
            img2 = cv2.line(img2, (x, y1), (x, y2), (255, 0, 0), 2)




        # snapped_xyxy = snap_bboxes_to_grid(detached_xyxy, factor=0.1)
        
        # img2 = copy.deepcopy(result.orig_img)
        # for xyxy in snapped_xyxy:
        #     xyxy = [int(round(x)) for x in xyxy]
        #     img2 = cv2.rectangle(
        #         img2, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
        #     )

        # x_boundaries, y_boundaries, grid = get_table_structure(detached_xyxy, tol=compute_dynamic_tolerance(detached_xyxy))

        # print("X boundaries (columns):", x_boundaries)
        # print("Y boundaries (rows):", y_boundaries)
        # print("Grid mapping (cell index -> bboxes):")
        # for key in sorted(grid.keys()):
        #     print(f"Cell {key}: {grid[key]}")

        # Draw boundary lines
        # for x in x_boundaries:
        #     img2 = cv2.line(img2, (int(x), 0), (int(x), img2.shape[0]), (255, 0, 0), 2)
        # for y in y_boundaries:
        #     img2 = cv2.line(img2, (0, int(y)), (img2.shape[1], int(y)), (255, 0, 0), 2)
        # for cell_id, cell_bboxes in grid.items():
            # for bbox in cell_bboxes:
            #     print(cell_id, bbox)
            #     x1, y1, x2, y2 = bbox
            #     # round mathematically
            #     x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                # img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Comparison image
        img3 = cv2.hconcat([img, img2])
        cv2.imshow("Comparison", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()