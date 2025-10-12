import streamlit as st
import requests
import logging
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
import os
from config import DATA_PATH
from src.processing import generate_data_store
from src.response import query_rag
from streamlit_chat import message

# Load environment variables from a .env file
load_dotenv()
ALLOWED_EXTENSIONS = ['pdf', "docx", "txt"]

st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📄 Chat with Your Documents")

# --- Upload Document ---
st.sidebar.header("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    print(f"uploaded_file.type.lower(): {uploaded_file.type.lower()}")
    if uploaded_file.type.lower() in ALLOWED_EXTENSIONS:
        logging.info("File Extension is correct!")
        # Save the file
        save_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        generate_data_store(save_path)
        logging.info("File Processing Succesfull!!!")
    else:
        exts = ",".join(ALLOWED_EXTENSIONS)
        st.sidebar.warning(f"The Uploaded File Format is not supported. Please make sure the uploaded file has the below format\n {exts}")
    
# --- List Documents ---
st.sidebar.header("📁 Uploaded Documents")
doc_list_response = os.listdir(DATA_PATH)
if doc_list_response:
    for doc in doc_list_response:
        st.sidebar.markdown(f"📄 {doc}")
else:
    st.sidebar.info("No documents uploaded yet.")

# --- Chat Interface ---
# st.header("💬 Ask Questions")
if "messages" not in st.session_state:
    st.session_state.messages = []

# question = st.text_input("Ask a question based on your documents:")
st.header("💬 Ask Questions")

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
question = st.chat_input("Type your question here:")

if question:
    with st.chat_message("user"):
        st.markdown(question)
            
    with st.spinner("Thinking..."):
        try:
            response_details,response_text = query_rag(question)
            print(f"response_text: {response_text}")
            st.session_state.messages.append({"role":"User","content":question})
            st.session_state.messages.append({"role":"assistant","content":response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)
        except Exception as ex:
            # st.error("Failed to get answer from the server.")
            st.error("Something went wrong while generating the response.")
            logging.error(f"[RESPONSE GENERATION ERROR] due to {ex}")