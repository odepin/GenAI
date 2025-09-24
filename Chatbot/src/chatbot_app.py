# Import necessary libraries
import os
import shutil
import hashlib
from glob import glob
from datetime import datetime

import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import find_dotenv, load_dotenv

# Set OpenAI API key
load_dotenv(find_dotenv())
os.environ['OPENAI_API_KEY'] = str(os.getenv("OPENAI_API_KEY"))

# Set up temporary directory
tmp_directory = 'tmp'

# Constants for UI labels
UPLOAD_NEW = "Upload new one"
ALREADY_UPLOADED = "Already Uploaded"

# Available OpenAI models
OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k"
]

# Device options
DEVICE_OPTIONS = ["cpu", "cuda"]

# Function to load documents from a directory
def load_docs(directory: str):
    """
    Load documents from the given directory.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to load models and data once the server starts
@st.cache_resource
def startup_event(last_update: str, selected_model: str, selected_device: str):
    """
    Load all the necessary models and data once the server starts.
    """
    print(f"{last_update} - Model: {selected_model}, Device: {selected_device}")
    directory = 'tmp/'
    documents = load_docs(directory)
    docs = split_docs(documents)

    # Set device for embeddings
    device_name = selected_device if selected_device == "cpu" else "cuda"
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={"device": device_name}
    )
    
    persist_directory = "chroma_db"

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    # Use selected OpenAI model
    temperature = 0.7
    if "gpt-5" in selected_model:
        temperature = 1
    llm = ChatOpenAI(model_name=selected_model, temperature=temperature)
    
    db = Chroma.from_documents(docs, embeddings)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

    return db, chain

# Function to query the model and get an answer
def get_answer(query: str, db, chain):
    """
    Queries the model with a given question and returns the answer.
    """
    matching_docs_score = db.similarity_search_with_score(query)

    matching_docs = [doc for doc, score in matching_docs_score]
    answer = chain.run(input_documents=matching_docs, question=query)

    # Prepare the sources
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]

    return {"answer": answer, "sources": sources}

# Function to start the chatbot
def start_chatbot(selected_model: str, selected_device: str):
    # Create a unique cache key that includes model and device selection
    cache_key = f"{last_db_updated}_{selected_model}_{selected_device}"
    
    db, chain = startup_event(cache_key, selected_model, selected_device)
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = selected_model
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Clear messages if model changed
    if st.session_state.get("current_model") != selected_model:
        st.session_state.messages = []
        st.session_state["current_model"] = selected_model
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = get_answer(st.session_state.messages[-1]["content"], db, chain)
            answer = full_response['answer']
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


# Sidebar UI for selecting knowledge base
st.sidebar.header("Configuration")

content_type = st.sidebar.radio("Which Knowledge base you want to use?",
                                [ALREADY_UPLOADED, UPLOAD_NEW])

# UI logic for uploading a new knowledge base
if content_type == UPLOAD_NEW:
    uploaded_files = st.sidebar.file_uploader("Choose a txt file", accept_multiple_files=True)

    uploaded_file_names = [file.name for file in uploaded_files]

    if uploaded_files is not None and len(uploaded_files):
        if os.path.exists(tmp_directory):
            shutil.rmtree(tmp_directory)

        os.makedirs(tmp_directory)

        if len(uploaded_files):
            last_db_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        for file in uploaded_files:
            with open(f"{tmp_directory}/{file.name}", 'wb') as temp:
                temp.write(file.getvalue())
                temp.seek(0)

curr_dir = [path.split(os.path.sep)[-1] for path in glob(tmp_directory + '/*')]

# UI for displaying current knowledge base
if content_type == ALREADY_UPLOADED:
    st.sidebar.write("Current Knowledge Base")
    if len(curr_dir):
        st.sidebar.write(curr_dir)
    else:
        st.sidebar.write('**No KB Uploaded**')

# Model and device selection dropdowns
st.sidebar.subheader("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select OpenAI Model:",
    options=OPENAI_MODELS,
    index=4,  # Default to gpt-3.5-turbo
    help="Choose the OpenAI model for generating responses"
)

selected_device = st.sidebar.selectbox(
    "Select Device for Embeddings:",
    options=DEVICE_OPTIONS,
    index=0,  # Default to CPU
    help="Choose CPU or GPU (CUDA) for processing embeddings"
)

# Display current selections
st.sidebar.info(f"**Selected Model:** {selected_model}")
st.sidebar.info(f"**Selected Device:** {selected_device}")

# Calculate a hash to determine if the knowledge base has changed
last_db_updated = hashlib.md5(','.join(curr_dir).encode()).hexdigest()

# Start the chatbot if a knowledge base is loaded
if curr_dir and len(curr_dir):
    start_chatbot(selected_model, selected_device)
else:
    st.header('No KB Loaded, use the left menu to start')
    st.write("Please upload documents using the sidebar to begin chatting with your knowledge base.")
    
    # Display current configuration even when no KB is loaded
    st.subheader("Current Configuration")
    st.write(f"**Model:** {selected_model}")
    st.write(f"**Device:** {selected_device}")