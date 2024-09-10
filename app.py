import os
import json
import re
import cv2
import numpy as np
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

# Set up the Streamlit page
st.set_page_config(layout="wide")
st.title("PDF to Text OCR with Preprocessing")

# PDF file upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

def clean_extracted_text(text):
    """Cleans the extracted text by removing excess whitespace, line breaks, and duplicate words."""
    cleaned_text = re.sub(r'\n+', '\n', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\b(\d+)\s+(\d+)\b', r'\1\2', cleaned_text)
    return cleaned_text.strip()

def preprocess_image(image, output_dir, page_number):
    """Preprocesses the image to improve OCR accuracy, including grayscale conversion, noise removal, and resizing."""
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray)

        # Remove horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        lines = cv2.add(horizontal_lines, vertical_lines)
        cleaned_image = cv2.subtract(inverted_image, lines)
        final_image = cv2.bitwise_not(cleaned_image)

        # Denoising, binarization, and resizing for better OCR results
        denoised = cv2.fastNlMeansDenoising(final_image, h=30)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        resized = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        # Save processed image
        os.makedirs(output_dir, exist_ok=True)
        processed_image_path = os.path.join(output_dir, f'processed_page_{page_number}.png')
        cv2.imwrite(processed_image_path, resized)

    except Exception as e:
        print(f"Error during image preprocessing for page {page_number}: {e}")

    return resized

def ocr_process_page(img, lang='eng', output_dir=None, page_number=None):
    """Processes a single page image for OCR."""
    preprocessed_img = preprocess_image(img, output_dir, page_number)
    text = pytesseract.image_to_string(preprocessed_img, lang=lang, config='--psm 4 --oem 3')
    return text, preprocessed_img

def ocr_process(pdf_path, lang='eng', output_dir=None):
    """Processes the entire PDF for OCR by converting pages to images and running OCR on them."""
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        raise Exception(f"Error during PDF conversion: {e}")

    extracted_text = ""
    processed_images = []

    # Process each page in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ocr_process_page, img, lang, output_dir, i + 1) for i, img in enumerate(images)]
        for future in futures:
            text, processed_img = future.result()
            extracted_text += text + "\n\n"
            processed_images.append(processed_img)

    return extracted_text, processed_images

def parse_text_to_json(cleaned_text):
    """Parses cleaned text into a structured JSON format based on predefined patterns."""
    json_data = {}

    # Define patterns to extract relevant fields
    patterns = {
        "resident_name": re.compile(r"Resident Name\s+(\w+ \w+)"),
        "preferred_name": re.compile(r"Preferred Name\s+(\w+ \w+)"),
        "address": re.compile(r"Previous address\s+([\w\s,]+)"),
        "phone_number": re.compile(r"Previous Phone #\s+([\d-]+)"),
        "admission_date": re.compile(r"Admission Date\s+(\d{2}/\d{2}/\d{4})"),
        "age": re.compile(r"Age\s+(\d+)"),
        "marital_status": re.compile(r"Marital Status\s+(\w+)"),
        "insurance_name": re.compile(r"Insurance Name:\s+([\w\s]+)"),
        "insurance_policy": re.compile(r"Insurance Policy #:\s+(\S+)"),
        "diagnoses": re.compile(r"DIAGNOSIS INFORKA LION\s+(.+?)(?=Date of Discharge|$)", re.DOTALL),
    }

    for field, pattern in patterns.items():
        match = pattern.search(cleaned_text)
        if match:
            json_data[field] = match.group(1).strip()

    # Handle diagnoses section
    diagnoses_section = patterns["diagnoses"].search(cleaned_text)
    if diagnoses_section:
        diagnoses_text = diagnoses_section.group(1)
        json_data["diagnoses"] = []

        # Extract individual diagnosis records
        diagnoses_lines = diagnoses_text.split('\n')
        for line in diagnoses_lines:
            if line.strip():
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    diagnosis = {
                        "code": parts[0],
                        "description": parts[1],
                        "onset_date": parts[2]
                    }
                    json_data["diagnoses"].append(diagnosis)

    return json_data

# Main processing
if uploaded_file is not None:
    output_dir = "output"

    with st.spinner("Processing..."):
        temp_pdf_path = os.path.join("temp.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run the OCR process
        text, processed_images = ocr_process(temp_pdf_path, output_dir=output_dir)
        
        if text:
            st.success("OCR completed!")
            
            # Clean extracted text
            cleaned_text = clean_extracted_text(text)
            json_data = parse_text_to_json(cleaned_text)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Processed Images:")
                for i, processed_img in enumerate(processed_images):
                    st.image(processed_img, caption=f"Processed Page {i+1}", use_column_width=True)
            
            with col2:
                st.subheader("Extracted Text:")
                st.text_area("Cleaned Text", cleaned_text, height=1100)
                
                st.subheader("Extracted JSON:")
                st.json(json_data)

            # Download buttons for text and JSON data
            st.download_button(label="Download Text", data=cleaned_text, file_name="extracted_text.txt", mime="text/plain")
            st.download_button(label="Download JSON", data=json.dumps(json_data, indent=2), file_name="extracted_data.json", mime="application/json")

    # Clean up temporary file
    os.remove(temp_pdf_path)
