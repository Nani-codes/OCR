import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from pdf2image import convert_from_path
import json

def perform_trocr_on_image(image_path):
    # Load TrOCR model and processor for printed text
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    # Open and process the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Set the model to evaluation mode
    model.eval()

    # Generate text from the image
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def process_pdf_with_trocr(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    
    # Create a dictionary to store OCR results for each page
    ocr_results = {}

    # Process each page
    for i, page in enumerate(pages):
        # Save page as an image file
        image_path = f"page_{i}.png"
        page.save(image_path, "PNG")

        # Perform OCR on the image
        text = perform_trocr_on_image(image_path)
        
        # Add OCR results to the dictionary
        ocr_results[f"page_{i + 1}"] = text

    # Save the OCR results as JSON
    json_output_path = pdf_path.replace(".pdf", ".json")
    with open(json_output_path, "w") as json_file:
        json.dump(ocr_results, json_file, ensure_ascii=False, indent=4)

    print(f"OCR results saved to {json_output_path}")

# Example usage
pdf_path = "/home/poorna/Desktop/pdf2json/OCR/Allen Dunfee Face Sheet.pdf"
process_pdf_with_trocr(pdf_path)
