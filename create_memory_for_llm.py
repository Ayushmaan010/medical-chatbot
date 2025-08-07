# create_memory_for_llm.py

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR    = "data/"
OUTPUT_DIR  = "vectorstore_db_faiss"

def load_pdfs(path: str):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

def main():
    print("➔ Loading PDFs from", DATA_DIR)
    docs = load_pdfs(DATA_DIR)
    print(f"   • Found {len(docs)} PDF documents.")

    print("➔ Splitting into chunks…")
    chunks = split_into_chunks(docs)
    print(f"   • Created {len(chunks)} text chunks.")

    print("➔ Creating embeddings…")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("➔ Building FAISS index…")
    db = FAISS.from_documents(chunks, embedder)
    db.save_local(OUTPUT_DIR)
    print(f"✅ FAISS index saved to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
