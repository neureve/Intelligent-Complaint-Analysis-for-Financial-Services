import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load data
with open('data/cleaned_complaints.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Parameters
chunk_size = 500
chunk_overlap = 50
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Chunking strategy
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Create LangChain-compatible documents with metadata
documents = []
for entry in data:
    text = entry['complaint_text']
    metadata = {
        'complaint_id': entry['complaint_id'],
        'product': entry['product'],
        'category': entry['category']
    }
    chunks = splitter.create_documents([text], metadatas=[metadata]*1)
    documents.extend(chunks)

# Embedding
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Indexing with FAISS
vectorstore = FAISS.from_documents(documents, embedding)

# Save vector store
os.makedirs("vector_store", exist_ok=True)
vectorstore.save_local("vector_store")

print(f"Saved {len(documents)} chunks to vector_store/")
