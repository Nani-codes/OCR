# PDF to Text and JSON Generator

This project processes PDF files, performs Optical Character Recognition (OCR) using Tesseract, and generates clean, structured text. Additionally, it interacts with a local **Ollama** model to convert the extracted text into structured JSON, including null values where applicable.

## Project Overview

The primary purpose of this project is to:
1. **Convert PDF files to text** using image preprocessing and Tesseract OCR.
2. **Clean and format** the extracted text to remove unwanted noise (e.g., excessive spaces, duplicated words).
3. **Generate a structured JSON** file for the cleaned data by sending the processed text as a prompt to a local **Ollama** model.

The project is designed to be flexible, allowing you to preprocess and improve the accuracy of OCR extraction, while also ensuring that the final structured text and JSON representations are clean and reliable.

## Dependencies

This project uses the following Python libraries and tools:

- `os`: For file path handling.
- `cv2 (OpenCV)`: For image preprocessing.
- `numpy`: For image manipulation.
- `pytesseract`: For performing OCR on the processed images.
- `pdf2image`: For converting PDF files to images.
- `subprocess`: For interacting with the local Ollama model.
- `json`: For JSON validation and handling.

Ensure these dependencies are installed before running the project. You can install them using `pip`:

```bash
pip install opencv-python numpy pytesseract pdf2image
