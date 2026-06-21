# chatbot_rag.py
# Advanced RAG Chatbot Interface with Professional Features
# Streamlit app for intelligent document Q&A with advanced capabilities

import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
import hashlib

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Pinecone as PineconeVectorStoreLegacy

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-document {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: 600;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "chat_history": [],
        "document_count": 0,
        "total_chunks": 0,
        "last_query_time": None,
        "query_count": 0,
        "feedback_data": {},
        "current_llm": "llama-3.1-8b-instant",
        "temperature": 0.3,
        "top_k": 3,
        "score_threshold": 0.5,
        "show_sources": True,
        "chat_sessions": {},
        "current_session": "default",
        "rag_enabled": True,
        "vector_store_initialized": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize system message
    if not st.session_state.messages:
        st.session_state.messages.append(
            SystemMessage("You are an advanced assistant for question-answering tasks. Provide accurate, concise, and helpful responses based on the provided context.")
        )

# Initialize vector store with error handling
@st.cache_resource
def initialize_vector_store():
    """Initialize Pinecone vector store with caching"""
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        index = pc.Index(index_name)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        st.session_state.vector_store_initialized = True
        return vector_store, embeddings, index, pc
    except Exception as e:
        st.error(f"Failed to initialize vector store: {str(e)}")
        return None, None, None, None

# Document upload and processing
def process_uploaded_file(uploaded_file):
    """Process uploaded documents and add to vector store"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load document based on type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_path)
        elif uploaded_file.name.endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(temp_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return False
        
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
            chunk.metadata["upload_date"] = datetime.now().isoformat()
        
        # Add to vector store
        if st.session_state.vector_store_initialized:
            vector_store, _, _, _ = initialize_vector_store()
            vector_store.add_documents(chunks)
            
            # Update session stats
            st.session_state.document_count += 1
            st.session_state.total_chunks += len(chunks)
            
            # Clean up
            os.remove(temp_path)
            return True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

# Retrieval function with advanced options
def retrieve_documents(query: str, top_k: int = 3, score_threshold: float = 0.5):
    """Retrieve relevant documents with advanced filtering"""
    try:
        if not st.session_state.vector_store_initialized:
            return []
        
        vector_store, _, _, _ = initialize_vector_store()
        
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": top_k,
                "score_threshold": score_threshold
            }
        )
        
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return []

# Generate response with context
def generate_response(query: str, context: str, llm_model: str, temperature: float):
    """Generate response using LLM with context"""
    try:
        llm = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name=llm_model,
            temperature=temperature,
        )
        
        system_prompt = """You are an advanced assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question accurately.
        If you don't know the answer, just say that you don't know.
        Provide a comprehensive but concise answer.
        Include relevant details and examples when applicable.
        
        Context: {context}"""
        
        system_prompt_fmt = system_prompt.format(context=context)
        
        # Prepare messages
        messages = [
            SystemMessage(system_prompt_fmt),
            HumanMessage(query)
        ]
        
        # Generate response
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar for controls and settings
def render_sidebar():
    """Render the sidebar with controls and settings"""
    with st.sidebar:
        # Use a working image service
        # st.image("https://dummyimage.com/300x100/1f77b4/ffffff&text=DocQnA", use_container_width=True)
        st.markdown("""
        <div style="background: #1f77b4; 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;
                    font-family: Arial, sans-serif;">
            <span style="font-size: 40px;">📚</span>
            <h2 style="color: white; margin: 0;">AI Knowledge Assistant</h2>
            <p style="color: #e0e0e0; font-size: 14px; margin: 0;">v2.0</p>
        </div>
""", unsafe_allow_html=True)
        st.markdown("## 📊 Dashboard")
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Documents", st.session_state.document_count)
        with col2:
            st.metric("📝 Chunks", st.session_state.total_chunks)
        
        st.metric("❓ Queries", st.session_state.query_count)
        
        # Document Upload
        st.markdown("## 📤 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx', 'doc'],
            help="Upload PDF, TXT, or DOCX files for Q&A"
        )
        
        if uploaded_file:
            if st.button("📥 Process Document", use_container_width=True):
                with st.spinner("Processing document..."):
                    if process_uploaded_file(uploaded_file):
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")
        
        st.divider()
        
        # Settings
        st.markdown("## ⚙️ Settings")
        
        # LLM Model Selection
        llm_options = [
            "llama-3.1-8b-instant",
            "llama-3.2-3b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        st.session_state.current_llm = st.selectbox(
            "🤖 LLM Model",
            options=llm_options,
            index=llm_options.index(st.session_state.current_llm) if st.session_state.current_llm in llm_options else 0
        )
        
        # Temperature
        st.session_state.temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values = more creative responses"
        )
        
        # Retrieval settings
        st.markdown("### 🔍 Retrieval Settings")
        
        st.session_state.top_k = st.slider(
            "Number of documents (k)",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k
        )
        
        st.session_state.score_threshold = st.slider(
            "Relevance threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.score_threshold,
            step=0.05
        )
        
        # Toggle options
        st.session_state.show_sources = st.toggle(
            "Show sources",
            value=st.session_state.show_sources
        )
        
        st.session_state.rag_enabled = st.toggle(
            "RAG enabled",
            value=st.session_state.rag_enabled
        )
        
        st.divider()
        
        # Actions
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [SystemMessage("You are an advanced assistant for question-answering tasks.")]
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("📊 Export Chat", use_container_width=True):
            export_chat()
        
        # Status
        st.divider()
        st.markdown("### 📡 Status")
        status = "✅ Active" if st.session_state.vector_store_initialized else "⚠️ Disconnected"
        st.info(f"Vector Store: {status}")
        st.caption(f"Session: {st.session_state.current_session}")

# Export chat history
def export_chat():
    """Export chat history as a text file"""
    if not st.session_state.messages:
        st.warning("No chat history to export")
        return
    
    export_content = f"""Document Q&A Chat Export
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Messages: {len(st.session_state.messages)}
{'='*50}

"""
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            export_content += f"User: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            export_content += f"Assistant: {msg.content}\n\n"
    
    st.download_button(
        label="📥 Download Chat Export",
        data=export_content,
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Display sources
def display_sources(docs: List[Any]):
    """Display source documents with metadata"""
    if not docs:
        return
    
    st.markdown("### 📚 Sources")
    for i, doc in enumerate(docs):
        with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}", expanded=False):
            st.markdown(f"**Content:**\n{doc.page_content[:500]}...")
            st.markdown(f"**Relevance Score:** {doc.metadata.get('score', 'N/A')}")
            st.markdown(f"**Metadata:** {doc.metadata}")

# Main chat interface
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Initialize vector store
    vector_store, _, _, _ = initialize_vector_store()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<p class="main-header">📚 AI Knowledge Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Ask questions about your documents with advanced RAG capabilities</p>', unsafe_allow_html=True)
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
                        
                        # Show sources if enabled and available
                        if st.session_state.show_sources and hasattr(message, 'sources'):
                            display_sources(message.sources)
        
        # Chat input
        prompt = st.chat_input("Ask a question about your documents...", key="chat_input")
        
        if prompt:
            # Add user message
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.messages.append(HumanMessage(prompt))
            
            # Show thinking indicator
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    # Update query count
                    st.session_state.query_count += 1
                    st.session_state.last_query_time = datetime.now()
                    
                    # Retrieve documents
                    docs = []
                    context = ""
                    
                    if st.session_state.rag_enabled and st.session_state.vector_store_initialized:
                        docs = retrieve_documents(
                            prompt,
                            top_k=st.session_state.top_k,
                            score_threshold=st.session_state.score_threshold
                        )
                        
                        if docs:
                            # Prepare context from retrieved documents
                            context = "\n\n".join(d.page_content for d in docs)
                    
                    # Generate response
                    if context or not st.session_state.rag_enabled:
                        response = generate_response(
                            prompt,
                            context,
                            st.session_state.current_llm,
                            st.session_state.temperature
                        )
                    else:
                        response = "I couldn't find relevant information in the documents. Please try rephrasing your question or upload more documents."
                    
                    # Display response
                    st.markdown(response)
                    
                    # Store response with sources
                    ai_message = AIMessage(response)
                    if docs and st.session_state.show_sources:
                        ai_message.sources = docs
                    st.session_state.messages.append(ai_message)
                    
                    # Display sources
                    if docs and st.session_state.show_sources:
                        display_sources(docs)
        
        # Information footer
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"💬 Messages: {len(st.session_state.messages)}")
        with col2:
            st.caption(f"❓ Queries: {st.session_state.query_count}")
        with col3:
            if st.session_state.last_query_time:
                st.caption(f"🕐 Last query: {st.session_state.last_query_time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()