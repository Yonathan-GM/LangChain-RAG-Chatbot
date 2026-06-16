# chatbot_rag.py
# RAG Chatbot Interface
# Streamlit app for asking questions about your documents using RAG

import streamlit as st
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("📚 Document Q&A Chatbot")
st.caption("Ask questions about your uploaded documents")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize embeddings model (Google Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# Connect to Pinecone vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage("You are an assistant for question-answering tasks.")
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input bar
prompt = st.chat_input("Ask a question about your documents...")

# Process user prompt
if prompt:

    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
    )

    # Retrieve relevant documents from Pinecone
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # Build system prompt with retrieved context
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}"""

    system_prompt_fmt = system_prompt.format(context=docs_text)

    # Debug output (visible in terminal)
    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # Add system prompt to message history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # Generate response from LLM
    result = llm.invoke(st.session_state.messages).content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))