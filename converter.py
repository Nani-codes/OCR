import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import re

def preprocess_image(image, output_dir, page_number):
    """Preprocess the image for better OCR results and save the processed image."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to remove noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Optional: Resize image for better OCR accuracy
    resized = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Save the processed image
    processed_image_path = os.path.join(output_dir, f'processed_page_{page_number}.png')
    cv2.imwrite(processed_image_path, resized)
    print(f"Processed image saved to {processed_image_path}")

    return resized

def ocr_process(pdf_path, lang='eng', output_dir=None):
    """Perform OCR on the PDF, save processed images, and save or return the extracted text."""
    try:
        images = convert_from_path(pdf_path)
        print(f"Converted {len(images)} pages from PDF.")
    except Exception as e:
        print(f"Error during PDF conversion: {e}")
        return None

    extracted_text = ""
    for i, img in enumerate(images):
        preprocessed_img = preprocess_image(img, output_dir, i + 1)
        text = pytesseract.image_to_string(preprocessed_img, lang=lang, config='--psm 4 --oem 3')
        extracted_text += text + "\n"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(pdf_path).replace('.pdf', '.txt'))
        try:
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(extracted_text)
            print(f"OCR completed. Text saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving text file: {e}")
    else:
        return extracted_text

def clean_extracted_text(text):
    """Clean up the extracted text."""
    # Remove excessive white spaces
    cleaned_text = re.sub(r'\s+', ' ', text)

    # Additional cleaning for common OCR errors
    cleaned_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned_text)  # Remove duplicate words
    cleaned_text = re.sub(r'\b(\d+)\s+(\d+)\b', r'\1\2', cleaned_text)  # Fix split numbers
    return cleaned_text.strip()

if __name__ == '__main__':
    # Define the PDF path and output directory
    pdf_path = '/home/nani/DQ/pdftojson/Allen Dunfee Face Sheet.pdf'  # Update with your PDF file path
    output_dir = 'output'  # Output directory to save the text file and processed images
    lang = 'eng'  # Language for Tesseract OCR

    # Perform OCR and save the extracted text
    text = ocr_process(pdf_path, lang=lang, output_dir=output_dir)

    # Clean and save the extracted text
    if text:
        cleaned_text = clean_extracted_text(text)
        output_text_file = os.path.join(output_dir, "output_text_1.txt")
        try:
            with open(output_text_file, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            print(f"OCR process completed. The cleaned extracted text has been saved to {output_text_file}")
        except Exception as e:
            print(f"Error saving final text file:{e}")