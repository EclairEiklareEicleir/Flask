import cv2
import numpy as np
import pytesseract
import re
import difflib

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_gwa(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Image not found!"}

    max_height = 800
    scale_factor = min(1.0, max_height / image.shape[0])
    image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    denoise_val = 13
    thresh_val = 222
    sharpen_val = 1
    bright_val = 50
    contrast_val = 100 / 50
    resize_val = 2.0
    blur_val = 0
    invert = 1

    processed = image.copy()
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    if denoise_val > 0:
        processed = cv2.fastNlMeansDenoising(processed, None, denoise_val, 7, 21)

    if sharpen_val > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + sharpen_val, -1],
                           [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)

    _, processed = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)
    processed = cv2.convertScaleAbs(processed, alpha=contrast_val, beta=bright_val)

    if blur_val > 0:
        processed = cv2.GaussianBlur(processed, (blur_val * 2 + 1, blur_val * 2 + 1), 0)

    if invert:
        processed = cv2.bitwise_not(processed)

    if resize_val != 1.0:
        processed = cv2.resize(processed, None, fx=resize_val, fy=resize_val, interpolation=cv2.INTER_CUBIC)

    def normalize_text(text):
        text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        return text

    def clean_text(text):
        return re.sub(r'[^a-zA-Z\s]', '', text)

    raw_text = pytesseract.image_to_string(processed)

    ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    def fuzzy_match(word, target, cutoff=0.7):
        return difflib.SequenceMatcher(None, word.lower(), target.lower()).ratio() >= cutoff

    semester_gwa_pairs = []
    added_pairs = set()
    last_semester = None
    i = 0

    while i < len(ocr_data['text']):
        word = normalize_text(ocr_data['text'][i].strip().lower())
        cleaned_word = clean_text(word)
        try:
            conf = int(ocr_data['conf'][i])
        except ValueError:
            i += 1
            continue

        if conf < 60 or not cleaned_word:
            i += 1
            continue

        if "semester" in cleaned_word:
            prev = normalize_text(ocr_data['text'][i - 1].strip().lower()) if i > 0 else ""
            full = f"{prev} {cleaned_word}".strip()

            if "first" in full:
                last_semester = "First Semester"
            elif "second" in full or fuzzy_match(full, "second semester"):
                last_semester = "Second Semester"

        elif "average" in cleaned_word or "general" in cleaned_word:
            lookahead_text = " ".join([normalize_text(ocr_data['text'][j].strip().lower()) for j in range(i, min(i + 5, len(ocr_data['text'])))] )
            if fuzzy_match(lookahead_text, "general average for the semester") and last_semester:
                for j in range(i + 1, min(i + 10, len(ocr_data['text']))):
                    candidate = normalize_text(ocr_data['text'][j].strip())
                    if re.match(r'^\d+(\.\d+)?$', candidate):
                        pair = (last_semester, candidate)
                        if pair not in added_pairs:
                            semester_gwa_pairs.append(pair)
                            added_pairs.add(pair)
                        break
        i += 1

    gwa_values = []

    for sem, gwa in semester_gwa_pairs:
        try:
            gwa_values.append(float(gwa))
        except ValueError:
            continue

    if len(gwa_values) == 2:
        average_gwa = sum(gwa_values) / 2
        return {"gwa": f"{average_gwa:.2f}"}
    else:
        return {"error": "Unable to calculate average GWA. Ensure there are two GWA values."}
