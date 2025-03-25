import chromadb
import uuid
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from src.helper import load_pdf, text_split, RecursiveCharacterTextSplitter


# Loading the data part

extracted_data = load_pdf('/Users/visakh/Desktop/Gen_AI/medical_chat_bot/Data/')

text_chunks = text_split(extracted_data)

# Load the embedding model 
embeddings = HuggingFaceEmbeddings()

# Create a local db for chroma
persistant_db = 'doc_db'

# Creating sample files for testing purpose 
import random
sampled_chunks = random.sample(text_chunks, max(1, int(len(text_chunks) * 0.05)))

vectordb = Chroma.from_documents(
    documents=sampled_chunks,
    embedding=embeddings,
    persist_directory=persistant_db,
)
