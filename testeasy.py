import cv2
import numpy as np
import easyocr
from paddleocr import TextDetection
from typing import List, Tuple, Optional

from utils import *


class PaddleEasyOCR:
    """
    OOP pipeline: detect text boxes with PaddleOCR, recognize with EasyOCR.

    Attributes (after predict):
    - annotated_image: np.ndarray (BGR) with polygons and labels drawn
    - polys: List[List[List[int]]]: list of 4-point polygons [[x,y],...]
    - boxes: List[List[int]]: list of [x_min, x_max, y_min, y_max]
    - texts: List[Tuple[str, float]]: recognized text and confidence
    """

    def __init__(
        self,
        det_model_dir: str = "PP-OCRv5_server_det",
        *,
        unclip_ratio: float = 1.8,
        thresh: float = 0.1,
        box_thresh: float = 0.3,
        easy_langs: Tuple[str, ...] = ("en",),
        decoder: str = "beamsearch",
        batch_size: int = 8,
        blocklist: str = "~`'!@#$%^&*_+-={}[]|;:\"<>,?\\",
        reader_verbose: bool = False,
    ) -> None:
        # Khởi tạo mô hình
        self.det_model = TextDetection(
            model_dir=det_model_dir,
            unclip_ratio=unclip_ratio,
            thresh=thresh,
            box_thresh=box_thresh,
        )
        self.reader = easyocr.Reader(list(easy_langs), verbose=reader_verbose)

        self.decoder = decoder
        self.batch_size = batch_size
        self.blocklist = blocklist

        self.polys: List[List[List[int]]] = []
        self.boxes: List[List[int]] = []
        self.texts: List[Tuple[str, float]] = []

    def predict_single(self, image_path: str) -> List[Tuple[str, float]]:
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
        polys = [np.array(p).astype(int).tolist() for p in polys_np]
        boxes = [poly_to_easyocr_box(p) for p in polys]

        results = self.reader.recognize(
            img_cv_grey=img_gray,
            horizontal_list=boxes,
            free_list=[],  
            decoder=self.decoder,
            batch_size=self.batch_size,
            blocklist=self.blocklist,
            detail=1,
        )
        
        # Save texts (align by index with polys)
        # texts = [(str(t), float(s)) for (_, t, s) in results]
        texts = [str(t) for (_, t, s) in results]
        # Build annotated image
        draw_img = img.copy()
        for i, (_, text, _score) in enumerate(results):
            if i < len(self.polys):
                draw_paddle_poly_with_easy_label(draw_img, self.polys[i], text)
        annotated_image = draw_img

        return texts, annotated_image
    
    def predict_multi(self, image_paths: List[str]) -> List[List[Tuple[str, float]]]:
        """Run detection and recognition on multiple image paths.

        Returns a list of lists of (text, confidence) tuples for each image.
        """
        all_texts = []
        annotated_images = []
        for image_path in image_paths:
            texts, annotated_image = self.predict_single(image_path)
            all_texts.append(texts)
            annotated_images.append(annotated_image)
        return all_texts, annotated_images
    
    def predict_multi_and_extract(self, image_paths: List[str]) -> List[dict]:
        all_texts, annotated_images = self.predict_multi(image_paths)
        out_dict = {}
        for image_path, texts in zip(image_paths, all_texts):
            out_dict[image_path] = post_process(texts)
        return out_dict, annotated_images
    
    def save_annotated(self, output_path: str = 'output.png', annotated_image: np.ndarray = None) -> None:
        """Save the annotated image to a file if available."""
        if annotated_image is None:
            raise ValueError("No annotated image provided.")
        ok = cv2.imwrite(output_path, annotated_image)
        if not ok:
            raise IOError(f"Failed to write image to {output_path}")


if __name__ == "__main__":
    import os
    folder = "input"
    pipeline = PaddleEasyOCR()
    image_paths = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    results = pipeline.predict_multi_and_extract(image_paths)
    print(results)