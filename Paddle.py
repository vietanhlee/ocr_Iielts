from paddleocr import PaddleOCR
import numpy as np
import os
from typing import List, Dict, Tuple
from utils import *
import cv2
class Paddle():
    def __init__(self, text_detection_model_name = "PP-OCRv5_server_det",
        text_recognition_model_name = "PP-OCRv5_mobile_rec",
        text_detection_model_dir = "PP-OCRv5_server_det",
        text_recognition_model_dir = "PP-OCRv5_mobile_rec",
        use_doc_orientation_classify = False,
        use_doc_unwarping = False,
        use_textline_orientation = False,
        text_det_unclip_ratio = 1.2,
        textline_orientation_batch_size = 8,
        text_recognition_batch_size = 8,
        text_det_box_thresh = 0.7,
        text_det_thresh = 0.3):
        self.ocr = PaddleOCR(
            text_detection_model_name = text_detection_model_name,
            text_recognition_model_name = text_recognition_model_name,
            text_detection_model_dir = text_detection_model_dir,
            text_recognition_model_dir = text_recognition_model_dir,
            use_doc_orientation_classify = use_doc_orientation_classify,
            use_doc_unwarping = use_doc_unwarping,
            use_textline_orientation = use_textline_orientation,
            text_det_unclip_ratio = text_det_unclip_ratio,
            textline_orientation_batch_size = textline_orientation_batch_size,
            text_recognition_batch_size = text_recognition_batch_size,
            text_det_box_thresh = text_det_box_thresh,
            text_det_thresh = text_det_thresh)

    def predict_multi_and_extract(self, image_paths: List[str]) -> Tuple[List[np.ndarray], Dict[str, str]]:
        """Xử lý OCR cho nhiều ảnh và trả về kết quả."""
        results = self.ocr.predict(input=image_paths)
        
        dict_extracted = {}
        images_annotated = []
        for img_path, result in zip(image_paths, results):
            polys = result["dt_polys"]
            # ensure each polygon is a list of int points
            polys = [np.array(p).astype(int).tolist() for p in polys]
            img = cv2.imread(img_path)
            texts = result['rec_texts']
            dict_extracted[result["input_path"]] = str(post_process(np.array(texts)))

            # Draw each polygon with its corresponding text label
            for poly, txt in zip(polys, texts):
                draw_paddle_poly_with_easy_label(img, poly, txt)

            images_annotated.append(img)
        
        return images_annotated, dict_extracted


if __name__ == "__main__":
    paddle = Paddle()
    input_folder = "input"
    files = os.listdir(input_folder)
    image_paths = [
        os.path.join(input_folder, filename) 
        for filename in files 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
        
    images_annotated, dict_extracted =  paddle.predict_multi_and_extract(image_paths)
    print(dict_extracted)
    for idx, img in enumerate(images_annotated):
        cv2.imwrite(f"output_annotated_{idx}.png", img)