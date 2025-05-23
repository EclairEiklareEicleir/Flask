import cv2
import numpy as np
import pytesseract
import re
import difflib

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_city_info(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Image not found!"}

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
    edge = 0

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

    if edge:
        processed = cv2.Canny(processed, 100, 200)

    raw_text = pytesseract.image_to_string(processed)
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

    def fuzzy_match(word, target, cutoff=0.6):
        return difflib.SequenceMatcher(None, word.lower(), target.lower()).ratio() >= cutoff

    def find_city(text_lines):
        known_cities = [
            "GENERAL SANTOS CITY", "DAVAO CITY", "CEBU CITY", "QUEZON CITY", "ZAMBOANGA CITY",
            "ILOILO CITY", "BAGUIO CITY", "BACOLOD CITY", "BATANGAS CITY", "TAGUM CITY",
            "CAGAYAN DE ORO CITY", "SAN FERNANDO CITY", "CALOOCAN CITY", "CITY OF PASIG", "MAKATI CITY"
        ]
        for line in text_lines:
            if "city" in line.lower():
                match = difflib.get_close_matches(line.upper(), known_cities, n=1, cutoff=0.5)
                if match:
                    return match[0]
                words = line.upper().split()
                for i in range(len(words) - 1):
                    if words[i+1] == "CITY":
                        return f"{words[i]} CITY"
        return None

    def extract_vin(text):
        vin_patterns = [
            r'\bVIN[:\s\-]*([A-Z0-9\-]{12,})\b',
            r'\bVOTER(?:\'?S)?(?:\s*ID)?[:\s\-]*([A-Z0-9\-]{12,})\b'
        ]
        for pattern in vin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def find_commission(text_lines):
        for line in text_lines:
            if "COMMISSION ON ELECTION" in line.upper():
                return "Commission on Election"
        return None


    def extract_name(text_lines):
        vin_index = dob_index = None
        for i, line in enumerate(text_lines):
            if 'vin' in line.lower():
                vin_index = i
            elif re.search(r'date\s*of\s*birth', line, re.IGNORECASE):
                dob_index = i
                break

        if vin_index is not None and dob_index is not None and dob_index > vin_index:
            name_lines = text_lines[vin_index+1:dob_index]
            return ' '.join(name_lines).strip()
        return "Name not found."

    result = {
        "city": find_city(lines) or "City not found.",
        "vin": extract_vin(raw_text) or "VIN not found.",
        "commission": find_commission(lines) or "Issuing commission not found.",
        "name": extract_name(lines)
    }

    return result
