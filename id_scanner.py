import cv2
import numpy as np
import pytesseract
import re
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_exam_info_from_image(image_path):
    if not os.path.exists(image_path):
        raise Exception(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Image could not be read by OpenCV")

    max_height = 800
    scale_factor = min(1.0, max_height / image.shape[0])
    image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    denoise_val = 19
    thresh_val = 147
    sharpen_val = 0
    bright_val = 50
    contrast_val = 2.0
    resize_val = 2.0
    blur_val = 0
    invert = 1
    edge_detect = 0

    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if denoise_val > 0:
        processed = cv2.fastNlMeansDenoising(processed, None, denoise_val, 7, 21)

    if sharpen_val > 0:
        kernel = np.array([[0, -1, 0], [-1, 5 + sharpen_val, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)

    _, processed = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)
    processed = cv2.convertScaleAbs(processed, alpha=contrast_val, beta=bright_val)

    if blur_val > 0:
        processed = cv2.GaussianBlur(processed, (blur_val * 2 + 1, blur_val * 2 + 1), 0)

    if invert:
        processed = cv2.bitwise_not(processed)

    if resize_val != 1.0:
        processed = cv2.resize(processed, None, fx=resize_val, fy=resize_val, interpolation=cv2.INTER_CUBIC)

    if edge_detect:
        processed = cv2.Canny(processed, 100, 200)

    raw_text = pytesseract.image_to_string(processed)
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

    return parse_exam_info(lines)

def parse_exam_info(text_lines):
    exam_type = ""
    name = ""
    score = ""
    date_taken = ""

    for line in text_lines:
        if "entrance exam" in line.lower():
            exam_type = "Entrance Exam"

        if "name" in line.lower() or "examinee" in line.lower():
            name_match = re.search(r"(?:name|examinee)[:\-]?\s*(.+)", line, re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()

        if "score" in line.lower() and not score:
            score_match = re.search(r"score[:\-]?\s*(\d+)", line, re.IGNORECASE)
            if score_match:
                score = score_match.group(1)

        if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', line) and not date_taken:
            date_match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', line)
            if date_match:
                date_taken = date_match.group(1)

    if not name:
        for line in text_lines:
            if line.isupper() and 2 <= len(line.split()) <= 4 and not any(char.isdigit() for char in line):
                name = line.strip()
                break

    return {
        "exam_type": exam_type,
        "name": name,
        "score": score,
        "date_taken": date_taken
    }
