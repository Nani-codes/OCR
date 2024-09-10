import os
import json
import numpy as np
from transformers import pipeline

def load_text_from_file(file_path):
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def process_text_with_transformer(text):
    """Process the extracted text with a transformer model."""
    # Load a transformer model (e.g., for Named Entity Recognition or Text Classification)
    transformer_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    # Process the extracted text
    results = transformer_pipeline(text)

    # Convert results to JSON format
    json_data = {'entities': results}
    return json_data

def convert_to_serializable(obj):
    """Convert non-serializable data types to serializable ones."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def save_json_to_file(json_data, file_path):
    """Save JSON data to a file."""
    json_data_serializable = convert_to_serializable(json_data)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(json_data_serializable, file, indent=4)
    print(f"JSON data has been saved to {file_path}")

if __name__ == '__main__':
    # Path to the cleaned text file
    text_file_path = 'output/Allen Dunfee Face Sheet.txt'
    json_file_path = 'output/output_data.json'

    # Load the cleaned text
    text = load_text_from_file(text_file_path)
    
    # Process the cleaned text with a transformer model
    json_data = process_text_with_transformer(text)
    
    # Save the JSON output
    save_json_to_file(json_data, json_file_path)
