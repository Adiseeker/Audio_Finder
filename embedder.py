import os
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download NLTK tokenizer data (only if not downloaded)
nltk.download('punkt', quiet=True)

# Function to read the content of a file
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return ""

# Function to tokenize a document into words
def preprocess_text(text):
    return word_tokenize(text.lower())  # Simple tokenization and lowercasing

# Function to load and replace extensions, then train embeddings
def load_and_train_embeddings(lista_file_path):
    modified_paths = []

    try:
        # Read the current contents of lista.txt
        with open(lista_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Replace .mp3 with .txt in each line and save to the list
        modified_paths = [line.strip().replace('.mp3', '.txt') for line in lines]

    except Exception as e:
        logging.error(f"An error occurred while reading the list file: {e}")
        return None

    # Load documents
    documents = []
    for file_path in modified_paths:
        if os.path.exists(file_path):
            text = read_txt_file(file_path)
            if text:  # Ensure text is not empty
                tokens = preprocess_text(text)
                documents.append(tokens)
        else:
            logging.warning(f"File not found: {file_path}")

    # Train a Word2Vec model if there are documents loaded
    if documents:
        model = Word2Vec(sentences=documents, vector_size=300, window=5, min_count=2, workers=4)
        # Save the model
        model.save("word2vec.model")
        logging.info("Model saved as word2vec.model")
        
        return model
    else:
        logging.error("No documents loaded. Exiting the training process.")
        return None

# Path to the lista.txt file
lista_file_path = 'lista.txt'

# Call the function to load paths and train embeddings
word2vec_model = load_and_train_embeddings(lista_file_path)

# Example: Access the vector for a specific word if the model is trained
if word2vec_model:
    try:
        word_vector = word2vec_model.wv['radio']  # Example: Get vector for the word 'radio'
        logging.info(f"Vector for 'radio': {word_vector}")
    except KeyError:
        logging.warning("The word 'radio' is not in the vocabulary.")
