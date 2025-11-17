import numpy as np
from json import dump
import re
import cv2
from typing import List, Any, Tuple

def pre(text: str):
    text = text.strip().lower()
    text = "".join(text.split())
    text = re.sub(r'[^a-z]', '', text)  # giữ lại chỉ chữ cái
    return text

def post_process(texts):    
    key = ['date', 'family name', 'first name', 'candidate id', 'date of birth', 'sex (m/f)', 'band', 'date end']
    out = {}
    for k in key:
        for ind in range(len(texts)):
            # print(pre(texts[ind]))
            # print(pre(k))
            key = pre(k)
            text = pre(texts[ind])
            if abs(len(text) - len(key)) >= 10:
                continue
            if key in text or is_equivalent(key, text):
                # print(texts[ind+1])
                out[k] = str(texts[ind+1])
                break
            
    for ind in range(len(texts) - 1, -1, -1):
        if 'date' in pre(texts[ind]):
            out['date end'] = str(texts[ind+1])
            break
    return out

def save_output(out, filename="output.json"):
    with open(filename, "w") as f:
        dump(out, f)
def similarity_ratio(s1: str, s2: str):
    if len(s1) != len(s2):
        return 0.0

    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    return matches / len(s1)


def is_equivalent(s1: str, s2: str, threshold=0.6):
    return similarity_ratio(s1, s2) >= threshold


# ========= OCR drawing & geometry helpers =========
def poly_to_easyocr_box(poly):
    """Convert a 4-point polygon [[x1,y1],...[x4,y4]] to an EasyOCR rect [x_min, x_max, y_min, y_max].

    Args:
        poly: Iterable of 4 points (x, y)

    Returns:
        list[int, int, int, int]
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x_min = int(min(xs))
    x_max = int(max(xs))
    y_min = int(min(ys))
    y_max = int(max(ys))
    return [x_min, x_max, y_min, y_max]


def draw_paddle_poly_with_easy_label(img, poly, text, color=(0, 255, 0)):
    """Draw PaddleOCR polygon and put EasyOCR text label on image (in-place).

    Args:
        img: BGR image (numpy array)
        poly: 4-point polygon [[x1,y1],...[x4,y4]]
        text: label string to draw
        color: BGR color tuple
    """
    poly_np = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [poly_np], True, color, 2)
    x, y = poly_np[0]
    cv2.putText(img, str(text), (int(x) + 5, int(y) - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def draw_bbox_with_label(img, bbox, text, color=(0, 255, 0), fmt: str = "xyxy"):
    """Draw an axis-aligned bbox and put a text label on image (in-place).

    Args:
        img: BGR image (numpy array)
        bbox: 4-element box. Supported formats:
            - "xyxy": [x_min, y_min, x_max, y_max]
            - "xxyy": [x_min, x_max, y_min, y_max]
        text: label string to draw
        color: BGR color tuple
        fmt: format of bbox
    """
    if fmt == "xyxy":
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    elif fmt == "xxyy":
        x_min, x_max, y_min, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        raise ValueError("Unsupported fmt for bbox. Use 'xyxy' or 'xxyy'.")

    x_min_i, y_min_i, x_max_i, y_max_i = int(x_min), int(y_min), int(x_max), int(y_max)

    # draw rectangle
    cv2.rectangle(img, (x_min_i, y_min_i), (x_max_i, y_max_i), color, 2)

    # choose label position: prefer above the box if space, else below
    label_y = y_min_i - 7
    if label_y < 0:
        label_y = y_min_i + 15

    cv2.putText(img, str(text), (x_min_i + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def group_and_flatten_boxes_texts(boxes: List[List[float]], texts: List[str], threshold: float = 9.0, sort_within_line: bool = True) -> Tuple[List[List[float]], List[str]]:
    """Group boxes & texts by center-y (sequential) and return flattened lists.

    Inputs:
      - boxes: list of [xmin, xmax, ymin, ymax]
      - texts: list of strings (same length as boxes)
    Behavior:
      - iterate boxes/texts in order; group sequentially when center-y difference < threshold
      - optionally sort items within each line by center-x
      - finally flatten groups and return (flat_boxes, flat_texts)

    Returns:
      - flat_boxes: List of boxes (same format) in grouped & left-to-right order
      - flat_texts: List of texts aligned to flat_boxes
    """
    if len(boxes) != len(texts):
        raise ValueError("boxes and texts must have the same length")

    grouped_boxes: List[List[List[float]]] = []
    grouped_texts: List[List[str]] = []

    current_boxes: List[List[float]] = []
    current_texts: List[str] = []
    prev_center_y: float | None = None

    for box, txt in zip(boxes, texts):
        if not (isinstance(box, (list, tuple)) and len(box) >= 4 and all(isinstance(x, (int, float)) for x in box[:4])):
            raise ValueError("Each box must be [xmin, xmax, ymin, ymax] with numeric values")

        xmin, xmax, ymin, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

        if not current_boxes:
            current_boxes = [list(box)]
            current_texts = [txt]
            prev_center_y = center_y
            continue
        
        # compare with previous box center-y (not group mean)
        if prev_center_y is not None and abs(center_y - prev_center_y) < threshold:
            current_boxes.append(list(box))
            current_texts.append(txt)
            prev_center_y = center_y
        else:
            # finalize current line (optionally sort by center_x)
            if sort_within_line:
                centers = [((b[0] + b[1]) / 2.0, i) for i, b in enumerate(current_boxes)]
                order = [i for _, i in sorted(centers, key=lambda t: t[0])]
                current_boxes = [current_boxes[i] for i in order]
                current_texts = [current_texts[i] for i in order]

            grouped_boxes.append(current_boxes)
            grouped_texts.append(current_texts)

            current_boxes = [list(box)]
            current_texts = [txt]
            prev_center_y = center_y

    # finalize last group
    if current_boxes:
        if sort_within_line:
            centers = [((b[0] + b[1]) / 2.0, i) for i, b in enumerate(current_boxes)]
            order = [i for _, i in sorted(centers, key=lambda t: t[0])]
            current_boxes = [current_boxes[i] for i in order]
            current_texts = [current_texts[i] for i in order]
        grouped_boxes.append(current_boxes)
        grouped_texts.append(current_texts)

    
    flat_boxes: List[List[float]] = [b for group in grouped_boxes for b in group]
    flat_texts: List[str] = [t for group in grouped_texts for t in group]
    return flat_boxes, flat_texts


def flatten_grouped_boxes(grouped: List[List[Any]]) -> List[Any]:
    """Flatten grouped boxes (list of lists) into a single list preserving group order.

    Args:
        grouped: output from `group_boxes_by_center_y`.
    Returns:
        Flat list with elements in the order of groups and within-group left->right ordering.
    """
    return [item for group in grouped for item in group]