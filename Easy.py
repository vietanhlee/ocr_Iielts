import easyocr
import numpy as np
import os
from typing import List, Dict, Tuple
from utils import *
import cv2


class Easy():
    def __init__(
        self,
        easy_langs: Tuple[str, ...] = ("en",),
        decoder: str = "beamsearch",
        batch_size: int = 8,
        blocklist: str = "~`'!@#$%^&*_+-={}[]|:\\\"<>,?\\",
        reader_verbose: bool = False,
        low_text: float = 0.3,
        min_size: int = 10,
    ):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(list(easy_langs), verbose=reader_verbose)
        self.decoder = decoder
        self.batch_size = batch_size
        self.blocklist = blocklist
        self.low_text = low_text
        self.min_size = min_size

    def predict_multi_and_extract(self, image_paths: List[str]) -> Tuple[List[np.ndarray], Dict[str, str]]:
        """Process multiple images (one-by-one) and return annotated images + extracted text dict.

        EasyOCR doesn't support multi-image batch OCR directly here, so we run readtext per image.
        """
        dict_extracted: Dict[str, str] = {}
        images_annotated: List[np.ndarray] = []

        for img_path in image_paths:
            # read image
            img = cv2.imread(img_path)
            if img is None:
                # skip unreadable files but keep key as empty
                dict_extracted[img_path] = ""
                continue

            # EasyOCR returns list of [bbox, text, confidence]
            results = self.reader.readtext(
                img_path,
                detail=1,
                paragraph=False,
                decoder=self.decoder,
                batch_size=1,  # process single image at a time
                blocklist=self.blocklist,
                low_text=self.low_text,
                min_size=self.min_size,
            )

            texts = [r[1] for r in results]
            # convert bboxes to int polygons
            polys = [np.array(r[0]).astype(int).tolist() for r in results]

            # post-process and store extracted text (keep same structure as Paddle version)
            dict_extracted[img_path] = str(post_process(np.array(texts)))

            # draw polygons and labels
            for poly, txt in zip(polys, texts):
                draw_paddle_poly_with_easy_label(img, poly, txt)

            images_annotated.append(img)

        return images_annotated, dict_extracted



if __name__ == "__main__":
    easy = Easy()
    input_folder = "input"
    files = os.listdir(input_folder)
    image_paths = [
        os.path.join(input_folder, filename) 
        for filename in files 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    images_annotated, dict_extracted = easy.predict_multi_and_extract(image_paths)
    print(dict_extracted)
    for idx, img in enumerate(images_annotated):
        cv2.imwrite(f"output_easy_annotated_{idx}.png", img)
