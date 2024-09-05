import os
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image
import json
import cv2
import numpy as np

def convert_pdf_to_images(pdf_path, output_dir):
    """Convert PDF to images and save them in the output directory."""
    try:
        images = convert_from_path(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f'page_{i+1}.png')
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"Error during PDF conversion: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess the image for better results."""
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Increase contrast
    contrast = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)

    # Convert back to PIL Image
    processed_image = Image.fromarray(contrast)
    
    return processed_image

def extract_data_with_donut(image_paths, model, processor):
    """Use Donut model to extract structured data from the images."""
    extracted_data = []
    for image_path in image_paths:
        processed_image = preprocess_image(image_path)

        # Preprocess the image and pass it through the model
        pixel_values = processor(processed_image, return_tensors="pt").pixel_values

        # Increase the number of tokens to generate a longer response
        generated_ids = model.generate(pixel_values, max_new_tokens=512)  # Adjust the number of tokens as needed

        # Decode the model's output
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"Extracted Text for {image_path}: {generated_text}")  # Inspect the output
        extracted_data.append(generated_text)

    return extracted_data

def save_extracted_data(output_data, output_dir):
    """Save the extracted data as a JSON file."""
    output_file_path = os.path.join(output_dir, 'extracted_data.json')
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=4)
        print(f"Extracted data saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def main():
    pdf_path = '/home/poorna/Desktop/pdf2json/OCR/Allen Dunfee Face Sheet.pdf'  # Update with your PDF path
    output_dir = 'output'  # Directory to store images and extracted data

    # Step 1: Convert PDF to Images
    print("Converting PDF to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_dir)
    if not image_paths:
        print("Failed to convert PDF to images.")
        return

    # Step 2: Load Donut Model and Processor
    print("Loading Donut model and processor...")
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    # Step 3: Extract Data Using Donut
    print("Extracting data from images using Donut model...")
    extracted_data = extract_data_with_donut(image_paths, model, processor)
    
    if extracted_data:
        # Step 4: Save the Extracted Data
        save_extracted_data(extracted_data, output_dir)
    else:
        print("No data extracted.")

if __name__ == '__main__':
    main()