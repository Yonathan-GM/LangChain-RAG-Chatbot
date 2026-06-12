# ingestion.py
# RAG Document Ingestion Pipeline
# Loads PDFs, creates embeddings, and stores in Pinecone vector database

import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Get index name from environment variables
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Create index if it doesn't exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # 768 dimensions required for Google embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize embeddings with Google's Gemini model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# Connect vector store to Pinecone index
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load PDF documents from the 'documents' directory
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Larger chunks = fewer API calls
    chunk_overlap=200,    # Small overlap to maintain context
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(raw_documents)

# Generate unique IDs for each document chunk
uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

# Add documents to Pinecone vector database
vector_store.add_documents(documents=documents, ids=uuids)

print(f"✅ Successfully added {len(documents)} document chunks to Pinecone!")