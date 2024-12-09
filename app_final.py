from io import BytesIO
import streamlit as st
import gensim.downloader
import numpy as np
from dotenv import dotenv_values
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import zipfile
import rarfile
import os



# Load environment variables
env = dotenv_values(".env")

# Constants
model_name = 'word2vec-google-news-300'   #'word2vec.model'
EMBEDDING_DIM = 300
QDRANT_COLLECTION_NAME = "transcripts"

# DB Functions
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],
    )

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Creating collection")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Collection already exists")

# Load Word2Vec model once
@st.cache_resource
def load_word2vec_model(model_name):
    try:
        st.write("Loading Word2Vec model, please wait...")
        word2vec_model = gensim.downloader.load(model_name)
        #word2vec_model=gensim.models.Word2Vec.load(model_name)
        st.success("Word2Vec model loaded successfully!")
        return word2vec_model
    except Exception as e:
        st.error(f"Error loading Word2Vec model: {e}")
        return None

# Get embeddings for text
def get_embeddings(text, word2vec_model):
    words = text.split()
    embeddings = []
    
    for word in words:
        if word in word2vec_model:
            embeddings.append(word2vec_model[word])
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        st.warning("No valid words found for embedding.")
        return None


def add_note_to_db(note_title, note_text, word2vec_model):
    qdrant_client = get_qdrant_client()
    
    # Check if the note title already exists
    existing_notes = list_notes_from_db(note_title, word2vec_model)
    if existing_notes:
        st.warning(f"Note with title '{note_title}' already exists in the database. Skipping this note.")
        return

    point_id = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME).count + 1
    embedding = get_embeddings(note_text, word2vec_model)

    if embedding is not None:
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"title": note_title, "text": note_text},
            )]
        )

def list_notes_from_db(query, word2vec_model, search_type="semantic"):
    qdrant_client = get_qdrant_client()

    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
        return [{"title": note.payload.get("title", ""), "text": note.payload["text"], "score": None} for note in notes]

    if search_type == "full_text":
        notes = []
        qdrant_client = get_qdrant_client()
        points = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=1000)[0]
        for point in points:
            payload = point.payload
            text = payload["text"]
            if query.lower() in text.lower():
                notes.append({"title": payload["title"], "text": text, "score": None})
        return notes
    elif search_type == "semantic":
        # Semantic search using embeddings
        embedding = get_embeddings(query, word2vec_model)
        if embedding is not None:
            notes = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=embedding,
                limit=10,
            )
            return [{"title": note.payload.get("title", ""), "text": note.payload["text"], "score": note.score} for note in notes]
    
    

# Function to extract text from zip or rar files
def extract_text_from_files(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/plain":
            texts.append((uploaded_file.name, uploaded_file.getvalue().decode('utf-8')))
        elif uploaded_file.type == "application/zip":
            with zipfile.ZipFile(uploaded_file) as zip_file:
                for file_info in zip_file.infolist():
                    if file_info.filename.endswith('.txt'):
                        with zip_file.open(file_info) as file:
                            texts.append((file_info.filename, file.read().decode('utf-8')))
        elif uploaded_file.type == "application/x-rar":
            with rarfile.RarFile(uploaded_file) as rar_file:
                for file_info in rar_file.infolist():
                    if file_info.filename.endswith('.txt'):
                        with rar_file.open(file_info) as file:
                            texts.append((file_info.filename, file.read().decode('utf-8')))
    return texts

# Main application
st.set_page_config(page_title="Wyszukiwarka Audycji", layout="centered")

# Check for Qdrant URL and API Key
if not st.session_state.get("qdrant_url") or not st.session_state.get("qdrant_api_key"):
    env_qdrant_url = env.get("QDRANT_URL")
    env_qdrant_api_key = env.get("QDRANT_API_KEY")

    if env_qdrant_url and env_qdrant_api_key:
        st.session_state["qdrant_url"] = env_qdrant_url
        st.session_state["qdrant_api_key"] = env_qdrant_api_key
    else:
        st.session_state["qdrant_url"] = st.text_input("Enter Qdrant URL")
        st.session_state["qdrant_api_key"] = st.text_input("Enter Qdrant API Key", type="password")

    if st.session_state["qdrant_url"] and st.session_state["qdrant_api_key"]:
        st.rerun()

if not st.session_state.get("qdrant_url") or not st.session_state.get("qdrant_api_key"):
    st.stop()

# Session state initialization
if "note_texts" not in st.session_state:
    st.session_state["note_texts"] = []

st.title("Wyszukiwarka Audycji")
assure_db_collection_exists()

# Load the Word2Vec model once
word2vec_model = load_word2vec_model(model_name)

# Create tabs for adding and searching notes
add_tab, search_tab = st.tabs(["Dodaj transkrypt", "Wyszukaj transkrypty"])

with add_tab:
    uploaded_files = st.file_uploader("Upload your text files (txt, zip, rar)", type=["txt", "zip", "rar"], accept_multiple_files=True)

    if uploaded_files:
        # Extract text from uploaded files
        st.session_state["note_texts"] = extract_text_from_files(uploaded_files)

        # Display notes and save all notes to the database
        if st.session_state["note_texts"]:
            for uploaded_file_name, note_text in st.session_state["note_texts"]:
                file_title = os.path.splitext(uploaded_file_name)[0]  # Get filename without extension
                st.text_area(f"Uploaded File: {file_title}", note_text, height=200)

            if st.button("Save All Notes"):
                # Create a progress bar
                progress_bar = st.progress(0)
                total_notes = len(st.session_state["note_texts"])

                for idx, (uploaded_file_name, note_text) in enumerate(st.session_state["note_texts"]):
                    file_title = os.path.splitext(uploaded_file_name)[0]  # Get filename without extension
                    add_note_to_db(file_title, note_text, word2vec_model)

                    # Update the progress bar
                    progress_percentage = (idx + 1) / total_notes
                    progress_bar.progress(progress_percentage)

                st.success("All notes have been saved to the database.")

with search_tab:
    query = st.text_input("Wyszukaj audycje", key="search_query")

    # Add a toggle to choose between full-text and semantic search
    search_type = st.toggle("Semantic search", value=True)

    # Convert toggle value to search mode
    search_mode = "semantic" if search_type else "full_text"

    if st.button("Szukaj"):
        if query is not None and query != "":
            for note in list_notes_from_db(query, word2vec_model, search_mode):
                with st.container():
                    expander = st.expander(note['title'])
                    with expander:
                        st.write(note['text'])
                    if note['score'] is not None:
                        st.write(f"Score: {note['score']:.4f}")
        else:
            st.error("Please enter a search query")