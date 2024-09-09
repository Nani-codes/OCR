import os
import json
import streamlit as st
from ocr_processing import ocr_process, preprocess_image
from text_processing import clean_extracted_text
from json_conversion import parse_text_to_json

st.set_page_config(layout="wide")

st.title("PDF to Text OCR with Preprocessing")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    output_dir = "output"

    with st.spinner("Processing..."):
        temp_pdf_path = os.path.join("temp.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text, processed_images = ocr_process(temp_pdf_path, output_dir=output_dir)
        
        if text:
            st.success("OCR completed!")
            
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
            
            st.download_button(label="Download Text", data=cleaned_text, file_name="extracted_text.txt", mime="text/plain")
            st.download_button(label="Download JSON", data=json.dumps(json_data, indent=2), file_name="extracted_data.json", mime="application/json")

    os.remove(temp_pdf_path)
