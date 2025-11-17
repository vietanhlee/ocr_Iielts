"""Simple image post-processing utilities.

This script replaces very-dark pixels with white and saves the result.

Usage examples (PowerShell):
  # process single file and write to output path
  python g:\\OCR_paddle\\image_processing.py --input input/1.jpg --output output/1_proc.jpg --threshold 30

  # process all images in a folder and write to output folder
  python g:\\OCR_paddle\\image_processing.py --input input --output output/processed --threshold 30

  # overwrite files in-place (BE CAREFUL)
  python g:\\OCR_paddle\\image_processing.py --input input --inplace --threshold 20
"""
from __future__ import annotations

import os
from typing import Optional
import cv2
import numpy as np
import argparse


def threshold_dark_to_white(img: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Return a copy of the image where pixels darker than `threshold` are set to white.

    - `img` is expected as a BGR uint8 numpy array (as returned by `cv2.imread`).
    - `threshold` is in [0,255]. Pixels with grayscale intensity < threshold become white.

    The function preserves original colors for pixels that are not considered 'dark'.
    """
    if img is None:
        raise ValueError("img must be a valid numpy array")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < int(threshold)

    out = img.copy()
    # set pixels where mask is True to white (255,255,255)
    out[mask] = (255, 255, 255)
    return out


def process_path(input_path: str, output_path: Optional[str], threshold: int = 30, inplace: bool = False) -> None:
    """Process a single file or a directory.

    - If `input_path` is a file: process and save to `output_path` (or overwrite if inplace True).
    - If `input_path` is a directory: process all image files and save into `output_path` directory
      (or overwrite files in-place when `inplace=True`).
    """
    if os.path.isdir(input_path):
        # directory mode
        out_dir = output_path if output_path and not inplace else input_path
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        files = sorted(os.listdir(input_path))
        for fn in files:
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            src = os.path.join(input_path, fn)
            dst = os.path.join(out_dir, fn) if not inplace else src
            img = cv2.imread(src)
            if img is None:
                print(f"Warning: cannot read {src}, skipping")
                continue
            proc = threshold_dark_to_white(img, threshold=threshold)
            ok = cv2.imwrite(dst, proc)
            if not ok:
                print(f"Failed to write {dst}")
    else:
        # file mode
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        dst = input_path if inplace or output_path is None else output_path
        img = cv2.imread(input_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {input_path}")
        proc = threshold_dark_to_white(img, threshold=threshold)
        ok = cv2.imwrite(dst, proc)
        if not ok:
            raise RuntimeError(f"Failed to write output: {dst}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replace dark pixels with white in images")
    p.add_argument("--input", required=True, help="Input file or directory")
    p.add_argument("--output", required=False, help="Output file or directory (optional)")
    p.add_argument("--threshold", type=int, default=30, help="Intensity threshold (0-255). Pixels with intensity < threshold become white")
    p.add_argument("--inplace", action="store_true", help="Overwrite input files (file or directory)")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    process_path(args.input, args.output, threshold=args.threshold, inplace=args.inplace)
    print("Processing complete")
