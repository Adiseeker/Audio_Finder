import os
import csv
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import ollama
from difflib import SequenceMatcher
import pandas as pd

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

# Step 1: Combine all tags into combine_tags.csv
def combine_tags_from_files(lista_path, output_path):
    combined_tags = set()
    with open(lista_path, 'r') as f:
        file_paths = [line.strip() for line in f.readlines()]

    total_files = len(file_paths)
    pbar = tqdm(total=total_files, desc="Combining tags from files")

    for file_path in file_paths:
        file_path = file_path.replace('.mp3', '_best_tags.csv')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if 'tag_name' in row:
                        combined_tags.add(row['tag_name'].strip())
        pbar.update(1)
    pbar.close()

    # Write to combine_tags.csv
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tag_name'])
        for tag in sorted(combined_tags):  # Sort for consistency
            writer.writerow([tag])

# Step 2: Clean tags
def clean_tags(input_path, output_path):
    def is_similar(word1, word2, threshold=0.8):
        return SequenceMatcher(None, word1, word2).ratio() > threshold

    cleaned_tags = []
    with open(input_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        tags = [row['tag_name'].strip() for row in reader if 'tag_name' in row]

    # Remove duplicates and similar tags
    for tag in tags:
        if tag not in cleaned_tags and len(tag) <= 50 and re.match("^[a-zA-Z0-9_-]+$", tag):
            if all(not is_similar(tag, existing_tag) for existing_tag in cleaned_tags):
                cleaned_tags.append(tag)

    # Write cleaned tags to a new file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tag_name'])
        for tag in sorted(cleaned_tags):  # Sort for consistency
            writer.writerow([tag])

# Step 3: Generate relevant tags using cleaned tags for each file
def generate_tags_with_llm(lista_path, cleaned_tags_path):
    # Load predefined tags
    with open(cleaned_tags_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        predefined_tags = [row['tag_name'] for row in reader if 'tag_name' in row]

    # Limit the number of tags to 50
    max_tags = 50
    predefined_tags = predefined_tags[:max_tags]

    # Read list of file paths from lista.txt
    with open(lista_path, 'r') as f:
        file_paths = [line.strip().replace('.mp3', '.txt') for line in f.readlines()]

    total_files = len(file_paths)
    pbar = tqdm(total=total_files, desc="Generating tags with LLM")

    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()

            # Split the text into chunks
            chunks = split_into_chunks(original_text, chunk_size=100)

            all_tags_with_similarity = []

            for chunk in chunks:
                # Use LLM to generate tags from the predefined tag list
                response = ollama.chat(
                    model='mistral:7b',
                    messages=[{
                        "role": "user",
                        "content": f"""
                        You are a professional tagger. Your task is to analyze the given text and select exactly 3 relevant tags from the provided list of tags.

                        Instructions:
                        1. Do not summarize the text.
                        2. Only select 3 tags from the provided list.
                        3. Do not provide any extra text or explanations, just return the 3 tags.
                        4. Separate the tags with commas (without spaces or other punctuation).

                        Select 3 tags from the following list:
                        {', '.join(predefined_tags)}

                        Text:
                        "{chunk}"
                        """
                    }]
                )

                # Parse the response to extract the tags
                tags = []
                if isinstance(response, list):
                    for message in response:
                        if 'content' in message:
                            tags = message['content'].strip().split(',')
                            break
                elif isinstance(response, dict):
                    tags = response.get('message', {}).get('content', "No content available").strip().split(',')
                else:
                    tags = ["error", "generating", "tags"]

                # Compute cosine similarity for each tag
                chunk_embedding = compute_embeddings([chunk])  # Get embedding for the current chunk
                for tag in tags:
                    tag_embedding = compute_embeddings([tag])  # Get embedding for the current tag
                    similarity = compute_cosine_similarity(tag_embedding, chunk_embedding)
                    all_tags_with_similarity.append((tag, similarity))

            # Convert to DataFrame for easier duplicate removal and sorting
            df = pd.DataFrame(all_tags_with_similarity, columns=['tag_name', 'cosine_similarity'])

            # Remove duplicates based on 'tag_name', keeping the one with the highest cosine similarity
            df = df.loc[df.groupby('tag_name')['cosine_similarity'].idxmax()]

            # Save tags with cosine similarity to final_tags.csv for this file
            output_csv_path = file_path.replace('.txt', '_tag_final.csv')
            df.to_csv(output_csv_path, index=False, columns=['tag_name', 'cosine_similarity'], encoding='utf-8')

        pbar.update(1)

    pbar.close()

# Main execution flow
if __name__ == "__main__":
    lista_path = 'D:\\Ai\\Audio-Classifier\\voiceapp\\lista-1.txt'
    combined_tags_path = 'D:\\Ai\\Audio-Classifier\\voiceapp\\combine_tags.csv'
    cleaned_tags_path = 'D:\\Ai\\Audio-Classifier\\voiceapp\\cleaned_tags.csv'

    # Step 1: Combine tags
    combine_tags_from_files(lista_path, combined_tags_path)

    # Step 2: Clean tags
    clean_tags(combined_tags_path, cleaned_tags_path)

    # Step 3: Generate relevant tags using cleaned tags for each file
    generate_tags_with_llm(lista_path, cleaned_tags_path)
