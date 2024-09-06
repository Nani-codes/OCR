import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast
from PIL import Image
import json
import torch
import os

# Step 1: Preprocess Image (Line Removal)
def preprocess_image(image):
    """Preprocess the image for better OCR results and return the processed image."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Invert the image (make text white and background black)
    inverted_image = cv2.bitwise_not(gray)

    # Use morphology to remove lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Horizontal line kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))   # Vertical line kernel

    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine horizontal and vertical lines
    lines = cv2.add(horizontal_lines, vertical_lines)

    # Subtract lines from the original image
    cleaned_image = cv2.subtract(inverted_image, lines)

    # Re-invert to restore original polarity
    final_image = cv2.bitwise_not(cleaned_image)

    return final_image

# Step 2: Convert PDF Pages to Images
def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to images."""
    images = convert_from_path(pdf_path)
    
    # Save images as temporary files
    image_files = []
    for i, image in enumerate(images):
        image_path = f"page_{i}.png"
        image.save(image_path, 'PNG')
        image_files.append(image_path)
    
    return image_files

# Step 3: Perform OCR using PyTesseract
def perform_ocr_on_image(image_path):
    """Perform OCR on the image and return the result and bounding boxes."""
    # Perform OCR on the preprocessed image
    ocr_result = pytesseract.image_to_string(Image.open(image_path))
    
    # Also get bounding boxes from OCR
    ocr_boxes = pytesseract.image_to_boxes(Image.open(image_path))
    
    ocr_data = []
    for box in ocr_boxes.splitlines():
        b = box.split()
        if len(b) == 6:
            ocr_data.append({
                'text': b[0],
                'bounding_box': [int(b[1]), int(b[2]), int(b[3]), int(b[4])]
            })
    
    return ocr_result, ocr_data

# Step 4: Normalize Bounding Boxes and Process with LayoutLMv3
def layoutlmv3_process_image(image_path, ocr_result, ocr_data):
    """Process the image with LayoutLMv3 and return tokens, bounding boxes, and predictions."""
    # Load pre-trained LayoutLMv3 model and tokenizer
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    
    # Open and process the image
    image = Image.open(image_path)
    
    # Prepare OCR data
    words = [data['text'] for data in ocr_data]
    bounding_boxes = [data['bounding_box'] for data in ocr_data]
    
    # Normalize bounding boxes to the range 0-1000
    normalized_boxes = normalize_bounding_boxes(bounding_boxes, image.size)
    
    # Tokenize and encode inputs
    encoding = tokenizer(words, boxes=normalized_boxes, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Ensure bounding boxes are of the correct type (Long)
    encoding['bbox'] = encoding['bbox'].long()
    
    # Predict tokens with LayoutLMv3
    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1)
    
    # Extract tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
    bounding_boxes = encoding["bbox"].squeeze().tolist()
    
    return tokens, bounding_boxes, predictions

# Normalize bounding box coordinates to the range 0-1000
def normalize_bounding_boxes(bounding_boxes, image_size):
    """Normalize bounding box coordinates to the range 0-1000."""
    width, height = image_size
    normalized_boxes = []
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        normalized_boxes.append([
            x_min / width * 1000,
            y_min / height * 1000,
            x_max / width * 1000,
            y_max / height * 1000
        ])
    return normalized_boxes

# Step 5: Create JSON Output
def create_json_output(tokens, bounding_boxes, predictions):
    """Create a JSON output from tokens, bounding boxes, and predictions."""
    data = {}
    
    # Ensure tokens, bounding boxes, and predictions lists have the same length
    for token, bbox, prediction in zip(tokens, bounding_boxes, predictions.squeeze().tolist()):
        if prediction != 0:  # 0 corresponds to "Other" in most token classification schemes
            data[token] = {
                "text": token,  # Include the text token
                "bounding_box": bbox,
                "label": int(prediction)  # Ensure the label is an integer
            }
    
    return json.dumps(data, indent=4)

# Step 6: Save OCR Results as Text Files
def save_ocr_results_as_text(image_path, ocr_result):
    """Save the OCR results as a text file."""
    # Create an output directory if it doesn't exist
    output_dir = 'ocr_text_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a text file for the OCR result
    file_path = os.path.join(output_dir, f'ocr_{os.path.basename(image_path).replace(".png", ".txt")}')
    with open(file_path, 'w') as f:
        f.write(ocr_result)
    
    print(f"Saved OCR result for {image_path} to {file_path}")

# Step 7: Complete Pipeline to Process PDF and Generate JSON
def process_pdf(pdf_path):
    """Complete pipeline to convert PDF, preprocess images, perform OCR, and generate JSON output."""
    # Step 1: Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path)
    
    for image_path in image_paths:
        # Step 2: Preprocess the image (remove lines)
        preprocessed_image = preprocess_image(Image.open(image_path))
        preprocessed_image_path = f"processed_{os.path.basename(image_path)}"
        cv2.imwrite(preprocessed_image_path, preprocessed_image)

        # Step 3: Perform OCR
        ocr_result, ocr_data = perform_ocr_on_image(preprocessed_image_path)
        print(f"OCR Result for {image_path}:\n", ocr_result)
        
        # Save OCR results as text files
        save_ocr_results_as_text(image_path, ocr_result)
        
        # Step 4: Process with LayoutLMv3 for layout-aware extraction
        tokens, bounding_boxes, predictions = layoutlmv3_process_image(preprocessed_image_path, ocr_result, ocr_data)
        
        # Step 5: Create JSON output
        json_output = create_json_output(tokens, bounding_boxes, predictions)
        
        # Save JSON output
        json_file_path = f"{image_path}.json"
        with open(json_file_path, 'w') as f:
            f.write(json_output)
            
        print(f"JSON output saved: {json_file_path}")

# Provide the path to your PDF file
pdf_path = "/home/poorna/Desktop/pdf2json/OCR/Allen Dunfee Face Sheet.pdf"
process_pdf(pdf_path)
