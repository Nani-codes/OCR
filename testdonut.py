import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Path to your custom image file
image_path = "/home/poorna/Desktop/Poornateja_Reddy_Bankdetails.jpg"  # Update this with the path to your image

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")

# Prepare decoder inputs
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

# Preprocess the image into pixel values
pixel_values = processor(image, return_tensors="pt").pixel_values

# Print shape of pixel_values
print("Pixel values shape:", pixel_values.shape)

# Generate the output using the model
outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=512,  # Adjust as needed
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# Decode the output sequence
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove the first task start token

# Convert the sequence into structured JSON
json_output = processor.token2json(sequence)

# Print the JSON output
print("Extracted JSON:", json_output)
