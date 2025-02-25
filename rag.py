import faiss
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
import os
import numpy as np

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Replace if not using env variables

INDEX_PATH = "data/faiss_index.bin"  # Use .bin for FAISS native format

# INDEX_PATH = "data/faiss_index.pkl"
DATA_PATH = "data/content.pkl"

def load_content():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)

def answer_query(query, OPENAI_API_KEY):
    index = faiss.read_index(INDEX_PATH)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY).embed_documents([query])

    embeddings_np = np.array(embeddings).astype('float32')  # Convert to NumPy array
    # index.add(embeddings_np)

    D, I = index.search(embeddings_np, 1)
    content = load_content()
    
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    # llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        #Context: 
        {content}

        #Question:
        {query}

        #Answer:"""
        )
    # response = llm(f"Based on the content: {content}, answer: {query}")
    messages = prompt.invoke({"query": query, "content": content})
    response = llm.invoke(messages)
    return response.content