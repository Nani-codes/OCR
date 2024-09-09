import re

def clean_extracted_text(text):
    cleaned_text = re.sub(r'\n+', '\n', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\b(\d+)\s+(\d+)\b', r'\1\2', cleaned_text)
    return cleaned_text.strip()
