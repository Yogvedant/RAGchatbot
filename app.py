from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from src.helper import download_huggingface_embeddings
from langchain.vectorstores import Pinecone
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import os
import re

#Initializing flask.
app = Flask(__name__)


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "bns-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = Ollama(model="deepseek-r1:7b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answering_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

#clean response:
def clean_and_format_response(response_text):
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # Remove numbering like "1.**text**:" or "1.**text**"
    cleaned = re.sub(r'\d+\.\*\*(.*?)\*\*:', r'**\1:**', cleaned)
    
    # Remove any remaining numbering like "1." at start of lines
    cleaned = re.sub(r'^\d+\.', '', cleaned, flags=re.MULTILINE)
    
    # Convert **text** to <strong>text</strong>
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cleaned)
    
    # Add line breaks before each bold header
    cleaned = re.sub(r'<strong>(.*?)</strong>:', r'<br><br><strong>\1:</strong>', cleaned)
    
    # Convert line breaks
    cleaned = re.sub(r'\n\n', '<br><br>', cleaned)
    cleaned = re.sub(r'\n', '<br>', cleaned)
    
    # Clean up leading breaks
    cleaned = re.sub(r'^(<br>\s*)+', '', cleaned)
    
    # Clean up any leading <br> tags
    cleaned = re.sub(r'^(<br>\s*)+', '', cleaned)
    
    return cleaned




@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    raw_answer = response["answer"]
    formatted_answer = clean_and_format_response(raw_answer)
    print("Response: ", formatted_answer)
    return str(formatted_answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)