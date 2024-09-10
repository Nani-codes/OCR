PDF to Text OCR with Preprocessing
=====================================

This project is a Streamlit application that processes PDF files using Optical Character Recognition (OCR) to extract text. It includes image preprocessing for enhanced text extraction accuracy, text cleaning, and conversion of the extracted text into JSON format.

Features
--------

### PDF Upload

Upload PDF files and convert them into images.

### Image Preprocessing

Images are preprocessed using techniques such as:

* Denoising
* Thresholding
* Line removal

to improve OCR accuracy.

### Text Extraction

The preprocessed images are passed through Tesseract OCR to extract text.

### Text Cleaning

Clean up the extracted text by removing unwanted characters, extra spaces, and duplicates.

### JSON Conversion

Parsed and structured JSON data is extracted from the text using regular expressions.

### Downloadable Output

Download both the cleaned text and the JSON data for further use.

### Parallel Processing

Uses concurrent processing for faster PDF to text conversion.

Requirements
------------

* Python 3.8+
* Tesseract-OCR
* Required Python packages (listed in `requirements.txt`)

Setup
-----

### Clone the repository

```bash
git clone https://github.com/Nani-codes/OCR.git
cd OCR
```
### Installing Modules

```bash
pip install -r requirements.txt
```
### Installing Tesseract

```bash
sudo apt install tesseract-ocr
```
### Installing Tesseract

```bash
sudo apt install tesseract-ocr
```
### Run Streamlit Application

```bash
streamlit run app.py
```
Project Structure
-----------------

### Files

* `app.py`: Main Streamlit app for PDF upload, OCR, text cleaning, and JSON conversion.
* `ocr_processing.py`: Contains image preprocessing and OCR-related functions.
* `text_processing.py`: Cleans the extracted text by removing extra characters and formatting issues.
* `json_conversion.py`: Converts cleaned text into structured JSON format.
* `requirements.txt`: Contains the list of Python packages required to run the project.



