import os
import cv2
import numpy as np
import easyocr
from paddleocr import TextDetection
from typing import Dict, List, Tuple
from utils import *


class PaddleEasy:
    def __init__(
        self,
        det_model_dir: str = "PP-OCRv4_server_det",
        *,
        unclip_ratio: float = 2.1, #1.9
        thresh: float = 0.3, #0.7
        box_thresh: float = 0.7, #0.05
        easy_langs: Tuple[str, ...] = ("en",),
        decoder: str = "beamsearch",
        batch_size: int = 8,
        blocklist: str = "~`'!@#$%^&*_+-={}[]|;:\"<>,?\\",
        reader_verbose: bool = False,
    ) -> None:
        # Initialize models
        self.det_model = TextDetection(
            model_dir=det_model_dir,
            model_name=det_model_dir,
            unclip_ratio= unclip_ratio,
            thresh=thresh,
            box_thresh=box_thresh,
        )
        self.reader = easyocr.Reader(list(easy_langs), verbose=reader_verbose)

        # Runtime configs
        self.decoder = decoder
        self.batch_size = batch_size
        self.blocklist = blocklist

    def predict_single(self, image_path: str) -> List[Tuple[str, float]]:
        img = cv2.imread(image_path)

        det = self.det_model.predict(img, batch_size=self.batch_size)
        polys_np = det[0].get("dt_polys", [])  
        polys = [np.array(p).astype(int).tolist() for p in polys_np][::-1]
        easy_boxes = [poly_to_easyocr_box(p) for p in polys]
        
        results = self.reader.recognize(
            img_cv_grey= img,
            horizontal_list=easy_boxes,
            free_list=[],  # must be [] (not None)
            decoder=self.decoder,
            batch_size=self.batch_size,
            blocklist=self.blocklist,
            detail=1,   
        )

        # Save texts (align by index with polys)
        texts = [str(t) for (_, t, s) in results]
        
        easy_boxes, texts = group_and_flatten_boxes_texts(easy_boxes, texts, threshold= 13, sort_within_line= True)
        
        for i, text in enumerate(texts):
            if i < len(easy_boxes):
                draw_bbox_with_label(img, easy_boxes[i], text, fmt="xxyy")
        return texts, img
    def predict_multi_and_extract(self, image_paths: List[str]) -> Tuple[List[np.ndarray], Dict[str, str]]:
        dict_extracted: Dict[str, str] = {}
        images_annotated: List[np.ndarray] = []

        for img_path in image_paths:
            texts, annotated_img = self.predict_single(img_path)
            dict_extracted[img_path] = str(post_process(np.array([t for t in texts])))
            images_annotated.append(annotated_img)

        return images_annotated, dict_extracted

if __name__ == "__main__":
    paddle_easy = PaddleEasy()
    input_folder = "input"
    files = os.listdir(input_folder)
    image_paths = [
        os.path.join(input_folder, filename) 
        for filename in files 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    images_annotated, dict_extracted = paddle_easy.predict_multi_and_extract(image_paths)
    print(dict_extracted)
    for idx, img in enumerate(images_annotated):
        cv2.imwrite(f"output_easy_annotated_{idx}.png", img)

