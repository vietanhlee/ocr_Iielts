from paddleocr import PaddleOCR
import easyocr
import numpy as np
import os
import argparse
from typing import List, Dict
from json import dumps
from utils import *

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

reader = easyocr.Reader(['en'])

def process_ocr(image_paths: List[str]) -> Dict[str, str]:
    """Xử lý OCR cho nhiều ảnh và trả về kết quả."""
    results = ocr.predict(input=image_paths)
    ocr_results = {}
    for result in results:
        ocr_results[result["input_path"]] = str(post_process(np.array(result['rec_texts'])))
    return ocr_results

def process_easyocr(image_paths: List[str]) -> Dict[str, str]:
    """Xử lý OCR cho nhiều ảnh sử dụng EasyOCR và trả về kết quả."""
    ocr_results = {}
    for image_path in image_paths:
        result = reader.readtext(image_path)
        texts = [res[1] for res in result]
        ocr_results[image_path] = str(post_process(texts))
    return ocr_results

def ocr_and_save(input_folder: str, output_filepath: str = "output.json", type: str = "paddle") -> Dict[str, str]:
    """Thực hiện OCR trên tất cả ảnh trong thư mục và lưu kết quả vào file JSON."""
    files = os.listdir(input_folder)
    image_paths = [
        os.path.join(input_folder, filename) 
        for filename in files 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if type == "easyocr":
        ocr_results = process_easyocr(image_paths)
    else:
        ocr_results = process_ocr(image_paths)
    save_output(ocr_results, output_filepath)
    return ocr_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thực hiện OCR trên các ảnh trong thư mục và lưu kết quả vào JSON"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Đường dẫn đến thư mục chứa ảnh đầu vào"
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs='?',
        default="output.json",
        help="Đường dẫn đến file JSON đầu ra (mặc định: output.json)"
    )
    parser.add_argument(
        "type",
        type=str,
        nargs='?',
        default="paddle",
        help="Loại OCR sử dụng: 'paddle' hoặc 'easyocr' (mặc định: paddle)"
    )
    args = parser.parse_args()
    
    ocr_results = ocr_and_save(args.input_folder, args.output_file, args.type)
    print(dumps(ocr_results, indent=4)) 