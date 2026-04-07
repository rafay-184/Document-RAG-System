import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Grab the secret key from the Hugging Face vault automatically
if "api_key" not in st.session_state: 
    st.session_state.api_key = os.environ.get("GOOGLE_API_KEY")

# --- 1. SET UP THE STREAMLIT UI ---
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with your Data (RAG System)")
st.write("Upload a PDF document and ask questions about its content.")

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Setup")
    # Using a password field so your API key stays hidden on screen
    #api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    st.markdown("---")
    st.subheader("Upload Document")
    pdf_docs = st.file_uploader("Upload your PDF File", type="pdf")

# --- 2. BACKEND LOGIC FUNCTIONS ---

def get_pdf_text(pdf):
    """Extracts text from the uploaded PDF."""
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the massive text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Converts chunks into vectors and stores them in FAISS database."""
    # Tell Google we are passing the API key directly
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """Sets up the prompt and the AI model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the uploaded document." Do not provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- 3. MAIN APP EXECUTION ---

if pdf_docs and api_key:
    # If the user has uploaded a file and provided a key, process it!
    with st.spinner("Processing PDF... Please wait."):
        # Step A: Read Text
        raw_text = get_pdf_text(pdf_docs)
        # Step B: Chunk Text
        text_chunks = get_text_chunks(raw_text)
        # Step C: Create Vector Database
        vector_store = get_vector_store(text_chunks, api_key)
        
    st.success("PDF Processed Successfully! You can now chat with it.")
    
    st.markdown("---")
    
    # Step D: Chat Interface
    user_question = st.text_input("Ask a question about the document:")
    
    if user_question:
        with st.spinner("Searching document for the answer..."):
            # 1. Search the database for the most relevant chunks
            docs = vector_store.similarity_search(user_question)
            
            # 2. Load the AI Brain
            chain = get_conversational_chain()
            
            # 3. Pass the relevant chunks and the question to the AI
            response = chain(
                {"input_documents": docs, "question": user_question}, 
                return_only_outputs=True
            )
            
            # 4. Display the result
            st.write("### AI Response:")
            st.info(response["output_text"])

elif not api_key:
    st.warning("👈 Please enter your Gemini API Key in the sidebar to start.")
elif not pdf_docs:
    st.info("👈 Please upload a PDF document in the sidebar to start.")