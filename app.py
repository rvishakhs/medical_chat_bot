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


# App Routing 
@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
