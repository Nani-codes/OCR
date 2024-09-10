import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(image, output_dir, page_number):
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        lines = cv2.add(horizontal_lines, vertical_lines)
        cleaned_image = cv2.subtract(inverted_image, lines)
        final_image = cv2.bitwise_not(cleaned_image)

        denoised = cv2.fastNlMeansDenoising(final_image, h=30)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        resized = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        os.makedirs(output_dir, exist_ok=True)
        processed_image_path = os.path.join(output_dir, f'processed_page_{page_number}.png')
        cv2.imwrite(processed_image_path, resized)

    except Exception as e:
        print(f"Error during image preprocessing for page {page_number}: {e}")

    return resized

def ocr_process_page(img, lang='eng', output_dir=None, page_number=None):
    preprocessed_img = preprocess_image(img, output_dir, page_number)
    text = pytesseract.image_to_string(preprocessed_img, lang=lang, config='--psm 4 --oem 3')
    return text, preprocessed_img

def ocr_process(pdf_path, lang='eng', output_dir=None):
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        raise Exception(f"Error during PDF conversion: {e}")

    extracted_text = ""
    processed_images = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ocr_process_page, img, lang, output_dir, i + 1) for i, img in enumerate(images)]
        for future in futures:
            text, processed_img = future.result()
            extracted_text += text + "\n\n"
            processed_images.append(processed_img)

    return extracted_text, processed_images
