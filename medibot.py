
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
dotenv_path = ".env"
load_dotenv(dotenv_path)

# Page configuration
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🩺",
    layout="centered"
)

# Dark theme CSS
st.markdown(
    """
    <style>
    body, .stApp { background-color: #121212; color: #E0E0E0; }
    .stTextInput>div>div>input { background-color: #1E1E1E !important; color: #E0E0E0 !important; border: none; }
    .stButton>button { background-color: #BB86FC !important; color: #121212 !important; border: none; }
    .chat-box { background-color: #1E1E1E; color: #E0E0E0; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .chat-user { background-color: #2A2A2A; color: #E0E0E0; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🩺 Medical AI Chatbot")

# Build QA chain factory
def build_qa_chain():
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore_db_faiss", embedder, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use only the context to answer. If you don’t know, say you don't know.\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    )
    llm = ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Initialize or prime chain on first load
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_qa_chain()

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    css_class = "chat-user" if msg["role"] == "user" else "chat-box"
    speaker = "You:" if msg["role"] == "user" else "MediBot:"
    st.markdown(f"<div class='{css_class}'><strong>{speaker}</strong> {msg['content']}</div>", unsafe_allow_html=True)

# Input at bottom for new messages
def submit():
    user_msg = st.session_state.user_input.strip()
    if user_msg:
        # Append user
        st.session_state.history.append({"role": "user", "content": user_msg})
        st.session_state.user_input = ""
        # Generate bot reply
        with st.spinner("MediBot is thinking…"):
            out = st.session_state.qa_chain.invoke({"query": user_msg})
            st.session_state.history.append({"role": "assistant", "content": out["result"]})

st.text_input("", key="user_input", placeholder="Type your message…", on_change=submit)
