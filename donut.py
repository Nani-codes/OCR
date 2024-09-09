from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image
import torch
import re

# Initialize Donut Processor and Model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Task prompt specific to medical sheet
task_prompt = "Extract medical information"

def process_image_with_donut(image_path, model, processor):
    image = Image.open(image_path)
    
    # Convert image to pixel values
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(f"Processing {image_path}, pixel values shape: {pixel_values.shape}")
    
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

    # Generate output with Donut model
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Post-process and decode
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    return sequence

def process_pdf_with_donut(pdf_path, model, processor):
    images = convert_from_path(pdf_path)
    
    for i, image in enumerate(images):
        image_path = f"page_{i}.png"
        image.save(image_path, 'PNG')

        # Process each image with Donut
        sequence = process_image_with_donut(image_path, model, processor)
        print(f"Extracted sequence from {image_path}:")
        print(sequence)

        # Convert the extracted sequence to JSON
        json_output = processor.token2json(sequence)
        json_path = f"{image_path}.json"
        with open(json_path, 'w') as f:
            f.write(json_output)
        print(f"Saved JSON output to {json_path}")

# Provide the path to your PDF containing medical sheets
pdf_path = "/home/poorna/Desktop/pdf2json/OCR/Allen Dunfee Face Sheet.pdf"
process_pdf_with_donut(pdf_path, model, processor)
