import ollama
import os
import time
from tqdm import tqdm

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
            
            # Summarize the text using ollama
            response = ollama.chat(
                model='llama3.2:3b',
                messages=[
                    {
                        "role": "assistant",
                        "content": "Summary below text",
                        'role': 'user',
                        'content': text,
                    },
                ]
            )
            
            # Handle the response
            if isinstance(response, list):
                # Iterate if it's a list of messages (common in streaming formats)
                for message in response:
                    if 'content' in message:
                        summary = message['content']
            elif isinstance(response, dict):
                # Access message content if it's in a single response dictionary
                summary = response.get('message', {}).get('content', "No content available")
            else:
                print("Unexpected response format:", response)
                summary = "Error summarizing file"
            
            # Save the summary to a file
            summary_file_path = file_path.replace('.txt', '_summary.txt')
            with open(summary_file_path, 'w') as f:
                f.write(summary)
            
            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'file': file_path, 'time': f"{time.time() - start_time:.2f} seconds"})