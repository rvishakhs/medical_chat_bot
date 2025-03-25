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
sampled_chunks = random.sample(text_chunks, max(1, int(len(text_chunks) * 0.005)))

vectordb = Chroma.from_documents(
    documents=sampled_chunks,
    embedding=embeddings,
    persist_directory=persistant_db,
)



# # Creating chromadb client
# chroma_client = chromadb.Client()

# collection = chroma_client.create_collection(name="medical_chat_bot")

# sample_chunks = text_chunks[:int(len(text_chunks) * 0.01)]
# documents = [chunk.page_content for chunk in sample_chunks]
# ids = [str(uuid.uuid4()) for _ in documents]

# collection.add(
#     documents=documents,
#     ids=ids
# )

# def get_query_results(question):
#     results = collection.query(
#         query_texts=[question],
#         n_results=5,
#         where={"metadata_field": "is_equal_to_this"},
#         where_document={"$contains":"search_string"}
#     )

#     return results