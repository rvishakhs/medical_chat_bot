from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter


# Extract data from the pdf
def load_pdf(data):
    loader = DirectoryLoader(data, 
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


# Create text chunks 
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk
