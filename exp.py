import fitz  # PyMuPDF for extracting images from PDFs
import pytesseract  # OCR
from PIL import Image
import io
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

# Initialize LayoutLMv3 model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Function to perform OCR and layout detection
def ocr_and_layout_extraction(pdf_path):
    # Open the PDF
    document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        image_list = page.get_images(full=True)

        if image_list:
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # Get image extension
                image = Image.open(io.BytesIO(image_bytes))

                # Ensure the image is in RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Perform OCR
                ocr_text = pytesseract.image_to_string(image)
                
                # Prepare the image for LayoutLMv3 model
                # Add truncation to handle long sequences
                encoding = processor(image, return_tensors="pt", truncation=True, padding=True)
                
                # Get layout predictions from the model
                with torch.no_grad():
                    outputs = model(**encoding)
                layout_predictions = outputs.logits.argmax(-1)

                # Print the OCR text
                print(f"Extracted OCR text from page {page_num + 1}, image {img_index + 1}:")
                print(ocr_text)
                
                # Print the layout information (this is just for debugging; we can store it or process further)
                print(f"Layout information (token predictions): {layout_predictions}")
                print("\n---\n")
        else:
            print(f"No images found on page {page_num + 1}")

    # Close the document when done
    document.close()


# Specify your PDF file path
pdf_path = "/home/nani/DQ/OCR/OCR/Adriann Brierley Face Sheet.pdf"
ocr_and_layout_extraction(pdf_path)
