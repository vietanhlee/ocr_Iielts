from __future__ import annotations

import cv2
import numpy as np
import easyocr
from paddleocr import TextDetection
from typing import List, Tuple, Optional

from utils import poly_to_easyocr_box, draw_paddle_poly_with_easy_label


class PaddleEasyOCR:
    """
    OOP pipeline: detect text boxes with PaddleOCR, recognize with EasyOCR.

    Attributes (after predict):
    - annotated_image: np.ndarray (BGR) with polygons and labels drawn
    - polys: List[List[List[int]]]: list of 4-point polygons [[x,y],...]
    - easy_boxes: List[List[int]]: list of [x_min, x_max, y_min, y_max]
    - texts: List[Tuple[str, float]]: recognized text and confidence
    """

    def __init__(
        self,
        det_model_dir: str = "PP-OCRv5_server_det",
        *,
        unclip_ratio: float = 1.7,
        thresh: float = 0.1,
        box_thresh: float = 0.3,
        easy_langs: Tuple[str, ...] = ("en",),
        decoder: str = "beamsearch",
        batch_size: int = 16,
        blocklist: str = "~`'!@#$%^&*_+-={}[]|;:\"<>,?\\",
        reader_verbose: bool = False,
    ) -> None:
        # Initialize models
        self.det_model = TextDetection(
            model_dir=det_model_dir,
            unclip_ratio=unclip_ratio,
            thresh=thresh,
            box_thresh=box_thresh,
        )
        self.reader = easyocr.Reader(list(easy_langs), verbose=reader_verbose)

        # Runtime configs
        self.decoder = decoder
        self.batch_size = batch_size
        self.blocklist = blocklist

        # Outputs
        self.image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None
        self.polys: List[List[List[int]]] = []
        self.easy_boxes: List[List[int]] = []
        self.texts: List[Tuple[str, float]] = []

    def predict(self, image_path: str) -> List[Tuple[str, float]]:
        """Run detection (Paddle) then recognition (EasyOCR) on an image path.

        Returns a list of (text, confidence) tuples and populates attributes.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect with PaddleOCR
        det = self.det_model.predict(image_path, batch_size=self.batch_size)
        polys_np = det[0].get("dt_polys", [])  # list of (4,2) arrays
        self.polys = [np.array(p).astype(int).tolist() for p in polys_np]
        self.easy_boxes = [poly_to_easyocr_box(p) for p in self.polys]

        # Recognize with EasyOCR (using rectangles converted from polys)
        results = self.reader.recognize(
            img_cv_grey=img_gray,
            horizontal_list=self.easy_boxes,
            free_list=[],  # must be [] (not None)
            decoder=self.decoder,
            batch_size=self.batch_size,
            blocklist=self.blocklist,
            detail=1,
        )

        # Save texts (align by index with polys)
        self.texts = [(str(t), float(s)) for (_, t, s) in results]

        # Build annotated image
        draw_img = img.copy()
        for i, (_, text, _score) in enumerate(results):
            if i < len(self.polys):
                draw_paddle_poly_with_easy_label(draw_img, self.polys[i], text)
        self.image = img
        self.annotated_image = draw_img

        return self.texts

    def save_annotated(self, output_path: str) -> None:
        """Save the annotated image to a file if available."""
        if self.annotated_image is None:
            raise RuntimeError("No annotated image. Call predict() first.")
        ok = cv2.imwrite(output_path, self.annotated_image)
        if not ok:
            raise IOError(f"Failed to write image to {output_path}")


if __name__ == "__main__":
    # Example usage
    pipeline = PaddleEasyOCR()
    texts = pipeline.predict("input/1.jpg")
    print("Recognized:")
    for i, (t, s) in enumerate(texts):
        print(f"  [{i:02d}] {t} (conf={s:.3f})")
    pipeline.save_annotated("output_poly_label.jpg")
    print("Saved to output_poly_label.jpg")
