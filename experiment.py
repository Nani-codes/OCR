import os
import json
import re
import cv2
import numpy as np
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient

# Set up the Streamlit page
st.set_page_config(layout="wide")
st.title("PDF to Text OCR with Preprocessing and AI Integration")

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

def call_inference_model(text):
    """Calls the Hugging Face model to generate a raw response."""
    try:
        client = InferenceClient(
            model="google/gemma-2-27b-it",  # Make sure this model is available
            token="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",
        )

        # Send the input text to the model
        response = client.chat_completion(
            messages=[{"role": "user", "content": f"Convert the following text to structured JSON:\n\n{text}"}],
            max_tokens=5000,
            stream=False
        )

        # Get the raw text response from the model
        response_text = response['choices'][0]['message']['content']
        return response_text

    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return "Error: Could not retrieve model response."

# In the main Streamlit processing block
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

            # Call Hugging Face model to generate raw response
            with st.spinner("Generating AI response..."):
                raw_response = call_inference_model(cleaned_text)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Processed Images:")
                for i, processed_img in enumerate(processed_images):
                    st.image(processed_img, caption=f"Processed Page {i+1}", use_column_width=True)
            
            with col2:
                st.subheader("Extracted Text:")
                st.text_area("Cleaned Text", cleaned_text, height=1100)
                
                st.subheader("AI Response as Markdown:")
                st.markdown(raw_response)  # Display the AI response as markdown

            # Download buttons for text and raw response as markdown
            st.download_button(label="Download Text", data=cleaned_text, file_name="extracted_text.txt", mime="text/plain")
            st.download_button(label="Download AI Response", data=raw_response, file_name="ai_response.md", mime="text/markdown")

    # Clean up temporary file
    os.remove(temp_pdf_path)