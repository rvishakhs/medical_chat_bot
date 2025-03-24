from flask import Flask, render_template, jsonify, request
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import * 
import chromadb
import os

from store_index import get_query_results

# Initialising flask 
app = Flask(__name__)

# Retriving Chroma db collection and chain them into LLM 
query_reslts = get_query_results()

Prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt" : Prompt}


qa = RetrievalQA.from_chain_type(
    llm="gpt2",
    chain_type="stuff",
    retriever=get_query_results(),
    return_source_documents = True,
    chain_type_kwargs=chain_type_kwargs
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
    result=qa({"query":input})
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080, debug=True)
