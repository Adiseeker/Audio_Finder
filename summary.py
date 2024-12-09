from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
from tqdm import tqdm
import os
import time

# Load pre-trained T5 model and tokenizer
model_name = "t5-3b"  # You can also use "t5-small", "t5-large", etc.
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
token_limit = 1024

# Function to calculate the number of chunks needed based on the token limit
def calculate_chunks(text, tokenizer, token_limit):
    # Tokenize the entire input text
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]

    # Calculate the total number of tokens
    total_tokens = tokens.size(0)

    # Calculate the number of chunks
    num_chunks = math.ceil(total_tokens / token_limit)

    # Split tokens into chunks
    token_chunks = [tokens[i * token_limit: (i + 1) * token_limit] for i in range(num_chunks)]

    # Convert token chunks back into text chunks for summarization
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

    return text_chunks

# Function to summarize a chunk of text
def summarize_chunk(chunk, model, tokenizer):
    input_text = "summarize: " + chunk
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=150,  # Max length of the summary
        min_length=50,  # Min length of the summary
        num_beams=2,  # Beam search for generating diverse summaries
        length_penalty=2.0,
        early_stopping=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to summarize long text using dynamic chunking
def summarize_long_text(text, model, tokenizer, token_limit):
    text_chunks = calculate_chunks(text, tokenizer, token_limit)
    summaries = []
    for chunk in text_chunks:
        summary = summarize_chunk(chunk, model, tokenizer)
        summaries.append(summary)
    final_summary = " ".join(summaries)
    return final_summary

# Read list of file paths from lista.txt
with open('lista.txt', 'r') as f:
    file_paths = f.readlines()

# Summarize each file and save the summary
total_files = len(file_paths)
pbar = tqdm(total=total_files, desc="Summarizing files")

for file_path in file_paths:
    file_path = file_path.strip()
    # Replace .mp3 with .txt
    file_path = file_path.replace('.mp3', '.txt')
    
    # Check if file exists
    if os.path.exists(file_path):
        start_time = time.time()
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        final_summary = summarize_long_text(text, model, tokenizer, token_limit)

        # Save summary to new file
        summary_file_path = file_path + '-summary.txt'
        with open(summary_file_path, 'w') as f:
            f.write(final_summary)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Summarized {file_path} in {elapsed_time:.2f} seconds")

        pbar.update(1)
    else:
        print(f"File not found: {file_path}")

pbar.close()