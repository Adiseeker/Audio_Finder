# -*- coding: utf-8 -*-

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import json
from tqdm import tqdm
import os
import time
import ollama
import codecs
import logging

# Configure logging
logging.basicConfig(
    filename="process_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set CUDA_LAUNCH_BLOCKING to 1 for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Read list of file paths from lista.txt
input_file = 'voiceapp//lista.txt'
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        file_paths = f.readlines()
except FileNotFoundError:
    logging.error(f"Input file '{input_file}' not found. Exiting.")
    exit("Input file not found.")

# Create a pipeline for automatic speech recognition
try:
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float32,
        device="cuda:0",  # Use CUDA GPU device 0 (change to 'mps' for Mac devices)
        model_kwargs={
            "attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        },
    )
except Exception as e:
    logging.error(f"Failed to load the Whisper model: {e}")
    exit("Failed to load the Whisper model.")

# Output folder setup
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Summarize each file and save the output
total_files = len(file_paths)
pbar = tqdm(total=total_files, desc="Processing files")

for file_path in file_paths:
    file_path = file_path.strip()  # Remove whitespace and newlines

    # Check if the file exists
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        pbar.update(1)
        continue

    start_time = time.time()

    try:
        # Process the audio file with the pipeline
        outputs = pipe(
            file_path,
            chunk_length_s=30,  # Split the audio into 30-second chunks
            batch_size=24,  # Process 24 chunks in parallel
            return_timestamps=True,  # Include timestamps in the output
        )
    except RuntimeError as e:
        logging.error(f"Error processing file {file_path}: {e}")
        pbar.update(1)
        continue

    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create folder for output
    output_subfolder = os.path.join(output_folder, file_name)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Save the output to a JSON file
    try:
        with open(os.path.join(output_subfolder, 'output.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save JSON output for file {file_path}: {e}")
        pbar.update(1)
        continue

    # Save the transcript to an SRT file
    try:
        with open(os.path.join(output_subfolder, 'transcript.srt'), 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(outputs['chunks']):
                start_time = chunk['timestamp'][0]
                end_time = chunk['timestamp'][1]
                text = chunk['text']
                f.write(f"{i+1}\n")
                f.write(f"{start_time:.3f} --> {end_time:.3f}\n")
                f.write(text + "\n\n")
    except Exception as e:
        logging.error(f"Failed to save SRT file for file {file_path}: {e}")
        pbar.update(1)
        continue

    # Summarize the text using ollama
    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {
                    "role": "assistant",
                    "content": """1. **Thoroughly analyze the article** to identify all key arguments, themes, and conclusions.
                                    2. **Generate a detailed, comprehensive summary** that:
                                    - Fully develops sentences and ideas, avoiding overly concise descriptions.
                                    - Explains context, background, and implications for all discussed points.
                                    - Includes examples or additional insights to ensure the summary is rich and explanatory.

                                    3. Format the summary in **Markdown**, ensuring clarity with:
                                    - Headings (`#`, `##`, `###`) for major sections and subsections.
                                    - Bullet points for listing key arguments or points.
                                    - **Well-developed paragraphs** to elaborate on significant themes and provide detailed context.

                                    PLz DON'T MAKE UP information, use only the information from the article. Summarization should be LONG not short. 
                                    Please summarize the article in correct form of Polish language.
                                """,
                },
                {
                    "role": "user",
                    "content": "Article: \n" + outputs['text'],
                },
            ],
        )

        # Extract the summary from the response
        if isinstance(response, list):
            summary = "".join(msg['content'] for msg in response if 'content' in msg)
        elif isinstance(response, dict):
            summary = response.get('message', {}).get('content', "No content available")
        else:
            logging.warning(f"Unexpected response format for file {file_path}: {response}")
            summary = "Error summarizing file."

        # Save the summary to a Markdown file
        summary_file_path = os.path.join(output_subfolder, 'summary.md')
        with codecs.open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    except Exception as e:
        logging.error(f"Failed to summarize text for file {file_path}: {e}")
        continue

    # Update the progress bar
    elapsed_time = time.time() - start_time
    pbar.update(1)
    pbar.set_postfix({'file': file_name, 'time': f"{elapsed_time:.2f}s"})

pbar.close()
logging.info("Processing completed.")
