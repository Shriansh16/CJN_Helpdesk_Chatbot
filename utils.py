from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import streamlit as st
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
load_dotenv()
pkey=os.getenv("PINECONE_API_KEY")


import pickle

def download_embeddings():
    embedding_path = "local_embeddings"

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)

    return embedding
def find_match(input):
    pc = PineconeClient(api_key=pkey)
    model=download_embeddings()
    index_name='cjn-chatbot'
    index=pc.Index(index_name)
    vectorstore = Pinecone(
    index, model.embed_query,"text"
                       )
    result=vectorstore.similarity_search(
    input,  # our search query
    k=6  # return 6 most relevant docs
      )
    return result
