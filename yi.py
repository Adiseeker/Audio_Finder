import ollama
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load a pre-trained model for semantic similarity (sentence-transformers/all-distilroberta-v1)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")

# Function to compute cosine similarity between two text embeddings
def compute_similarity(text1, text2):
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        emb1 = model(**inputs1).last_hidden_state.mean(dim=1)
        emb2 = model(**inputs2).last_hidden_state.mean(dim=1)

    cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    return cosine_sim.item()

# Read list of file paths from lista.txt
with open('D:\\Ai\\Audio-Classifier\\voiceapp\\lista.txt', 'r') as f:
    file_paths = f.readlines()

# Summarize each file and save the summary
total_files = len(file_paths)
pbar = tqdm(total=total_files, desc="Summarizing files")

# Number of times to run the model for each file
iterations = 10  # You can change the number of attempts per file

for file_path in file_paths:
    file_path = file_path.strip()
    # Replace .mp3 with .txt
    file_path = file_path.replace('.mp3', '.txt')
    
    # Check if file exists
    if os.path.exists(file_path):
        start_time = time.time()
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

            # List to store summaries for this file
            summaries = []
            similarity_scores = []

            # Generating multiple summaries for each file
            for i in range(1, iterations + 1):
                # Summarize the text using ollama
                response = ollama.chat(
                    model='llama3.2:3b',
                    messages=[{
                        "role": "assistant",
                        "content": """
                            ---
                            **"Analyze the provided text using the following framework:**  
                            1. **Key Themes**: Identify and explain the main themes or topics discussed in the text.  
                            2. **Impacts**: Assess the broader impacts, highlighting economic, technological, political, and social dimensions.  
                            3. **Examples and Evidence**: Draw connections to real-world examples or supporting evidence that underline the key points.  
                            4. **Opportunities and Risks**: Explore potential opportunities and risks suggested by the text.  
                            5. **Conclusion**: Summarize the implications and suggest future considerations or actions that align with the insights presented in the text."
                            ---
                        """,
                        'role': 'user',
                        'content': text,
                    }]
                )

                # Handle the response
                if isinstance(response, list):
                    for message in response:
                        if 'content' in message:
                            summary = message['content']
                elif isinstance(response, dict):
                    summary = response.get('message', {}).get('content', "No content available")
                else:
                    print("Unexpected response format:", response)
                    summary = "Error summarizing file"

                # Save the summary to the list
                summaries.append(summary)

                # Compute the similarity between the original text and the summary
                similarity = compute_similarity(text, summary)
                similarity_scores.append(similarity)

                # Save each summary to a file with a number suffix
                summary_file_path = file_path.replace('.txt', f'_summary_{i}.txt')
                with open(summary_file_path, 'w') as f:
                    f.write(summary)

            # Select the best summary based on the highest similarity score
            best_summary_index = similarity_scores.index(max(similarity_scores))
            best_summary = summaries[best_summary_index]

            # Save the best summary to a file
            best_summary_file_path = file_path.replace('.txt', '_best_summary.txt')
            with open(best_summary_file_path, 'w') as f:
                f.write(best_summary)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'file': file_path, 'best_iteration': best_summary_index + 1, 'time': f"{time.time() - start_time:.2f} seconds"})
