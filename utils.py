from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import streamlit as st
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
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
    k=5  # return 6 most relevant docs
      )
    return result
def query_refiner(conversation, query):
    api_key1 = "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"
    client = Groq(api_key=api_key1)
    response = client.chat.completions.create(
    model="gemma-7b-it",
    messages=[{"role": "system", "content": "You are a specialized question builder. Your task is to make necessary changes to the question provided by the user according to the given conversation log."},
           {"role": "user", "content": f"Given the following user query and conversation log, refine the query to make it most relevant for retrieving an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
    ],
    temperature=0.5,
    max_tokens=256,
    top_p=1,
    stream=False,
    stop=None,
     )
    return response.choices[0].message.content
def get_conversation_string():
    conversation_string = ""
    # Get the last two exchanges
    requests = st.session_state['requests'][-2:]  # Last two requests
    responses = st.session_state['responses'][-2:]  # Last two responses
    
    # Iterate over the two most recent exchanges
    for i in range(len(requests)):
        conversation_string += "Human: " + requests[i] + "\n"
        conversation_string += "Bot: " + responses[i] + "\n"
    
    return conversation_string

def load_pdf(pdf_path):
    loader=DirectoryLoader(pdf_path,glob='*.pdf',loader_cls=PyPDFLoader)
    document=loader.load()
    return document