# connect_memory_with_llm.py

import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 1. Load your GROQ key
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

# 2. Initialize embeddings & FAISS
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    "vectorstore_db_faiss",
    embed_model,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. Define your prompt
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use only the context to answer. If you don’t know, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )
)

# 4. Set up the GROQ LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.0,
    groq_api_key=GROQ_KEY
)

# 5. Build the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def main():
    print("👋 Welcome to your Medical QA Bot (type ‘exit’ to quit)\n")
    while True:
        query = input("❓ Ask a question: ")
        if query.strip().lower() == "exit":
            break

        result = qa.invoke({"query": query})
        answer = result["result"]
        docs   = result["source_documents"]

        # 6. Print the answer
        print("\n🤖 Answer:\n", answer)

        # 7. Deduplicate and print sources
        print("\n📄 Source(s):")
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source")
            if src not in seen:
                print(" -", src)
                seen.add(src)
        print()  # blank line before next prompt

if __name__ == "__main__":
    main()
