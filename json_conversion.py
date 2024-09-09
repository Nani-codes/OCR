import re
import json

def parse_text_to_json(text):
    json_data = {}

    # Define patterns to extract relevant fields
    patterns = {
        "resident_name": re.compile(r"Resident Name\s+(\w+ \w+)"),
        "preferred_name": re.compile(r"Preferred Name\s+(\w+ \w+)"),
        "address": re.compile(r"Previous address\s+([\w\s,]+)"),
        "phone_number": re.compile(r"Previous Phone #\s+([\d-]+)"),
        "admission_date": re.compile(r"Admission Date\s+(\d{2}/\d{2}/\d{4})"),
        "age": re.compile(r"Age\s+(\d+)"),
        "marital_status": re.compile(r"Marital Status\s+(\w+)"),
        "insurance_name": re.compile(r"Insurance Name:\s+([\w\s]+)"),
        "insurance_policy": re.compile(r"Insurance Policy #:\s+(\S+)"),
        "diagnoses": re.compile(r"DIAGNOSIS INFORKA LION\s+(.+?)(?=Date of Discharge|$)", re.DOTALL),
    }

    for field, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            json_data[field] = match.group(1).strip()

    # Handle the diagnoses section more accurately
    diagnoses_section = patterns["diagnoses"].search(text)
    if diagnoses_section:
        diagnoses_text = diagnoses_section.group(1)
        json_data["diagnoses"] = []
        
        # Extract individual diagnosis records
        diagnoses_lines = diagnoses_text.split('\n')
        for line in diagnoses_lines:
            if line.strip():
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    diagnosis = {
                        "code": parts[0],
                        "description": parts[1],
                        "onset_date": parts[2]
                    }
                    json_data["diagnoses"].append(diagnosis)

    return json_data
