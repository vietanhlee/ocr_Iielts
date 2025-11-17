import numpy as np
from json import dump
import re
import cv2

def pre(text: str):
    text = text.strip().lower()
    text = "".join(text.split())
    text = re.sub(r'[^a-z]', '', text)  # giữ lại chỉ chữ cái
    return text

def post_process(texts):    
    key = ['date', 'family name', 'first name', 'candidate id', 'date of birth', 'sex (m/f)', 'band', 'date end']
    iterater = iter(key)
    k = next(iterater)

    out = {}
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
            k = next(iterater, None)
            if k is None:
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