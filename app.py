from flask import Flask, render_template, jsonify, request
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from src.prompt import * 
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

retriever = vectordb.as_retriever()

Prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt" : Prompt}

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.5,

)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)


# App Routing 
@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    result=qa_chain({"query":input})
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080, debug=True)
