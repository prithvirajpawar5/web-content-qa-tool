import faiss
import pickle
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
import os

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Replace if not using env variables

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
INDEX_PATH = "data/faiss_index.bin"  # Use .bin for FAISS native format
# INDEX_PATH = "data/faiss_index.pkl"
DATA_PATH = "data/content.pkl"

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)  # Proper FAISS loading
    else:
        return faiss.IndexFlatL2(1536)  # Create a new FAISS index

def save_index(index):
    faiss.write_index(index, INDEX_PATH)  # Proper FAISS saving

# def load_index():
#     try:
#         with open(INDEX_PATH, "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         return faiss.IndexFlatL2(1536)

# def save_index(index):
#     with open(INDEX_PATH, "wb") as f:
#         pickle.dump(index, f)


# def ingest_url(url, OPENAI_API_KEY):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     text = " ".join([p.text for p in soup.find_all("p")])
    
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY).embed_documents([text])
#     index = load_index()
#     embeddings_np = np.array(embeddings).astype('float32')  # Convert to NumPy array
#     index.add(embeddings_np)
#     # index.add(embeddings)
#     save_index(index)
    
#     with open(DATA_PATH, "wb") as f:
#         pickle.dump(text, f)

def ingest_urls(urls, openai_api_key):
    """Ingests content from multiple URLs and updates the FAISS index."""
    content_dict = load_content()
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index = load_index()

    for url in urls:
        text = extract_text_from_url(url)  # You should have a function for this
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_documents([text])
        embeddings_np = np.array(embeddings).astype('float32')  # Convert to NumPy array
        # vector = embeddings.embed(text)  # Convert text to embeddings
        index.add(embeddings_np)
        content_dict[url] = text
        # index.add(vector)

    save_index(index)
    save_content(content_dict)

    # with open(DATA_PATH, "wb") as f:
    #     pickle.dump(text, f)

def load_content():
    """Loads existing content from content.pkl or returns an empty dict."""
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_content(content_dict):
    """Saves the updated content dictionary to content.pkl."""
    with open(DATA_PATH, "wb") as f:
        pickle.dump(content_dict, f)

def extract_text_from_url(url):
    """Fetches and extracts readable text from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unnecessary elements like scripts, styles, and navbars
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        # Extract visible text
        text = " ".join(soup.stripped_strings)
        return text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
