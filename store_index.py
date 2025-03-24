import chromadb
import uuid
from src.helper import load_pdf, text_split, RecursiveCharacterTextSplitter


# Loading the data part

extracted_data = load_pdf('/Users/visakh/Desktop/Gen_AI/medical_chat_bot/Data/')

text_chunks = text_split(extracted_data)

# Creating chromadb client
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="medical_chat_bot")

sample_chunks = text_chunks[:int(len(text_chunks) * 0.01)]
documents = [chunk.page_content for chunk in sample_chunks]
ids = [str(uuid.uuid4()) for _ in documents]

collection.add(
    documents=documents,
    ids=ids
)
