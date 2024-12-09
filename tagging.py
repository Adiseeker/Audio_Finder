import os
import time
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import ollama

# Load a pre-trained model for semantic similarity
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")

# Function to compute embeddings
def compute_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Function to compute cosine similarity between two embeddings
def compute_cosine_similarity(embedding1, embedding2):
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# Function to split text into chunks of approximately 100 words
def split_into_chunks(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Read list of file paths from lista.txt
with open('D:\\Ai\\Audio-Classifier\\voiceapp\\lista-1.txt', 'r') as f:
    file_paths = f.readlines()

# Summarize each file and save the summary
total_files = len(file_paths)
pbar = tqdm(total=total_files, desc="Tagging files")

# Number of iterations for each file
iterations = 5  # Number of times to run the model for each file

for file_path in file_paths:
    file_path = file_path.strip()
    # Replace .mp3 with .txt
    file_path = file_path.replace('.mp3', '.txt')
    
    # Check if file exists
    if os.path.exists(file_path):
        start_time = time.time()
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # Split the text into chunks of 100 words
        chunks = split_into_chunks(original_text, chunk_size=100)

        all_tags_per_iteration = []  # To store tags per iteration
        similarity_scores_per_iteration = []  # To store similarity scores per iteration

        for iteration in range(1, iterations + 1):
            iteration_tags = []  # Tags for this iteration
            iteration_similarities = []  # Similarities for this iteration

            # For each chunk, generate tags
            for chunk_index, chunk in enumerate(chunks, start=1):
                # Generate tags for each chunk using Ollama
                response = ollama.chat(
                    model='mistral:7b',
                    messages=[{
                        "role": "user",
                        "content": f"""
                        You are a professional tagger. Your task is to analyze a given text and return 3 highly relevant tags to the main topics and themes of the text. 

                        Guidelines:
                        1. Only provide the tags, nothing else.
                        2. Each tag must be a single word, not a phrase.
                        3. Separate the tags with commas, without spaces or additional formatting.

                        Example Input:
                        "Artificial intelligence is transforming industries like healthcare, finance, and transportation."

                        Example Output:
                        ai,technology,automation

                        Now, generate tags for the following text:
                        "{chunk}"
                        """
                    }]
                )

                # Extract and save the tags from the response
                if isinstance(response, list):
                    for message in response:
                        if 'content' in message:
                            tags = message['content'].split(',')
                elif isinstance(response, dict):
                    tags = response.get('message', {}).get('content', "No content available").split(',')
                else:
                    print("Unexpected response format:", response)
                    tags = ["error", "generating", "tags"]

                # Compute cosine similarity for each tag with the current chunk
                chunk_embedding = compute_embeddings([chunk])  # Get embedding for the current chunk
                tag_similarities = []
                for tag in tags:
                    tag_embedding = compute_embeddings([tag])  # Get embedding for the current tag
                    similarity = compute_cosine_similarity(tag_embedding, chunk_embedding)
                    tag_similarities.append((tag, similarity))

                # Store the tags and their similarities
                iteration_tags.extend(tags)
                iteration_similarities.extend(tag_similarities)

            # Sort tags based on cosine similarity, descending order
            sorted_tags = sorted(iteration_similarities, key=lambda x: x[1], reverse=True)

            # Store the sorted tags for this iteration
            all_tags_per_iteration.append(sorted_tags)

            # Save the sorted tags to a CSV file
            csv_output_path = file_path.replace('.txt', f'_tags_{iteration}.csv')
            with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['tag_name', 'cosine_similarity'])  # Write header

                # Write sorted tags and their corresponding cosine similarities
                for tag, similarity in sorted_tags:
                    csv_writer.writerow([tag, f"{similarity:.4f}"])

        # Now, evaluate which iteration produced the best set of tags
        avg_similarities_per_iteration = []
        for iteration, iteration_tags in enumerate(all_tags_per_iteration):
            avg_similarity = sum([similarity for _, similarity in iteration_tags]) / len(iteration_tags)
            avg_similarities_per_iteration.append((iteration + 1, avg_similarity))

        # Choose the best iteration based on highest average similarity
        best_iteration = max(avg_similarities_per_iteration, key=lambda x: x[1])[0]

        # Save the best set of tags to a new CSV
        best_iteration_tags = all_tags_per_iteration[best_iteration - 1]
        best_csv_output_path = file_path.replace('.txt', '_best_tags.csv')
        with open(best_csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['tag_name', 'cosine_similarity'])  # Write header

            # Write the best tags and their similarities
            for tag, similarity in best_iteration_tags:
                csv_writer.writerow([tag, f"{similarity:.4f}"])

        # Update the progress bar
        pbar.update(1)
        pbar.set_postfix({'file': file_path, 'best_iteration': best_iteration, 'time': f"{time.time() - start_time:.2f} seconds"})
