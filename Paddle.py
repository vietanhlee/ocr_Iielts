from paddleocr import PaddleOCR
import numpy as np
import os
from typing import List, Dict
from utils import *

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

def process_ocr(image_paths: List[str]) -> Dict[str, str]:
    """Xử lý OCR cho nhiều ảnh và trả về kết quả."""
    results = ocr.predict(input=image_paths)
    ocr_results = {}
    for result in results:
        ocr_results[result["input_path"]] = str(post_process(np.array(result['rec_texts'])))
    return ocr_results


def ocr_and_save(input_folder: str, output_filepath: str = "output.json", type: str = "paddle") -> Dict[str, str]:
    """Thực hiện OCR trên tất cả ảnh trong thư mục và lưu kết quả vào file JSON."""
    files = os.listdir(input_folder)
    image_paths = [
        os.path.join(input_folder, filename) 
        for filename in files 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    ocr_results = process_ocr(image_paths)
    save_output(ocr_results, output_filepath)
    return ocr_results

