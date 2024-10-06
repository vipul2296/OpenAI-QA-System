from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import PyPDF2
# from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader,PyPDFLoader
import langchain
import os
from config import data_dir, openai_api_key, faiss_indexer_path,admin_key
from functools import wraps
os.makedirs(data_dir,exist_ok=True)
os.makedirs(faiss_indexer_path,exist_ok=True)
app = Flask(__name__)
cors = CORS(app,resources={r"/*":{"origin":"*"}})

import openai
from datetime import datetime
openai.api_key = openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

def ask_gpt3(content,question):

    prompt = f"Please answer the following question based on the information provided below. If the answer is not found in the content, please reply with 'don't know'.\n\n{content}\n\nQuestion: {question}"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.5
    )

    answer = response.choices[0].text.strip()
    return answer

def is_valid_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, 'rb') as pdf_file:
            PyPDF2.PdfReader(pdf_file)
            return True
    except Exception as e:
        print(f"File validation failed due to {e}")
        return False
    
def restrict_access(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the user is authorized
        auth_header = request.headers.get('Authorization')
        print(f"auth_header: {auth_header}")
        if auth_header != admin_key:
            
            return jsonify({'error': 'Unauthorized access'}), 401
        return func(*args, **kwargs)
    return wrapper

@app.route("/getfiles",methods = ['GET'])
def get_files():
    files = os.listdir(data_dir)
    return jsonify({"files":files})

# This endpoint can only be accessed by admin users with a valid admin authorization token
@app.route("/upload",methods=['POST'])
@restrict_access
def upload_file():
    
    t1 = datetime.now()
    uploaded_files = request.files.getlist('file')
    filename = uploaded_files[0].filename
    
    if filename in os.listdir(data_dir):
        return jsonify("Filename already in use. Please try a different filename and upload again")
    filepath = os.path.join(data_dir,filename)
    indexer_path = os.path.join(faiss_indexer_path,filename)
    uploaded_files[0].save(filepath)
    print(f"File saved into local directory!!!")
    if not is_valid_pdf(filepath):
        return jsonify({"status":"File is corrupted. Please try again."})
    
    print("File validation completed!!!")
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    print(docs,embeddings)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(indexer_path)
    t2 = datetime.now()
    print(f"time taken: {t2-t1}")
    return jsonify({"status":"File succesfully uploaded !!!"})

@app.route("/query",methods=['POST'])
def ask_query():
    
    t1 = datetime.now()
    filename = request.form.get('filename')
    query = request.form.get('query')
    
    print(request.form,filename,query)
    path = os.path.join(data_dir,filename)
    embeddings = OpenAIEmbeddings()
    indexer_path = os.path.join(faiss_indexer_path,filename)
    if not os.path.exists(indexer_path):
        return jsonify({"status":"File does not exists. Please provide correct filename"})
    db = FAISS.load_local(indexer_path, embeddings)
    embedding_vector = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(embedding_vector)
    print(f"content: {docs[0].page_content}")
    
    start = datetime.now()
    answer = ask_gpt3(docs[0].page_content,query)
    answer = answer.split("\n")[0]
    answer = answer.replace("Answer:","").strip()
    response = {"query":query,"answer":answer.split("\n")[0]}
    end = datetime.now()
    t2 = datetime.now()
    print(f"time taken: {t2-t1}")
    print(f"time taken by gpt3: {end-start}")
    return jsonify(response)

if __name__=='__main__':
    # app.config['TIMEOUT'] = 180
    app.run(host='0.0.0.0',port=5001,debug=True,use_reloader=True)