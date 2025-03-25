from flask import Flask, render_template, jsonify, request
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#from langchain.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain_groq.chat_models import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.prompt import * 
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import chromadb
import os

os.environ["GROQ_API_KEY"] = "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"

# Initialising flask 
app = Flask(__name__)

# Retriving Chroma db collection and chain them into LLM 

embeddings = HuggingFaceEmbeddings()

persistant_db = 'doc_db'

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=persistant_db,
)

#retriever = vectordb.as_retriever(search_kwargs={'k': 2})

Prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt" : Prompt}

# Initialize Memory

memory = ConversationBufferMemory(memory_key="chat_history",output_key="result")


# LLm section 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    verbose=False
)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs,
    memory=memory,
    )


# Create the Conversational Retrieval Chain with the Custom Prompt
# Conversational Chain
conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    memory=memory,
    combine_docs_chain_kwargs=chain_type_kwargs,
    return_source_documents=True
)

chat_history = []

# App Routing 
@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    result = conv_qa_chain.invoke({"question": msg})
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080, debug=False)
