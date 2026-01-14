import os
import  streamlit as st
import chromadb

from langchain_community.vectorstores import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="NCERT RAG Tutor",
    page_icon="📘",
    layout="wide"
)

st.title("📘 NCERT Buddy")
st.caption("English & Science | Class 5 | Powered by Hugging Face + Chroma")

# -------------------------------
# Load Embeddings (MUST match ingestion)
# -------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -------------------------------
# Load Chroma Vector Store
# -------------------------------
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_data",
        collection_name="education",
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()

# -------------------------------
# Create Subject-wise Retrievers
# -------------------------------
english_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4, "filter": {"subject": "english"}}
)

science_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4, "filter": {"subject": "science"}}
)

n_retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)
# -------------------------------
# Load Hugging Face LLM
# -------------------------------
@st.cache_resource
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        task="conversational",
        temperature=0
    )
    return ChatHuggingFace(llm=endpoint)

llm = load_llm()

# -------------------------------
# Prompt Template
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """You are a helpful teacher for 5th grade English and Science.

Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer in simple, clear language:
"""
)

# -------------------------------
# Helper: format retrieved docs
# -------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------
# Build RAG chain dynamically
# -------------------------------
def get_rag_chain(subject: str):
    if subject == "english":
        retriever = english_retriever
    elif subject=="science":
        retriever = science_retriever
    else:
        retriever=n_retriever

    return (
        {
            "context": itemgetter("question")|retriever | format_docs,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# -------------------------------
# Streamlit UI
# -------------------------------
subject = st.selectbox(
    "📚 Select Subject",
    options=["All","english", "science"]
)

question = st.text_input(
    "❓ Ask a question from the textbook",
    placeholder="e.g. What is the poem Papa’s Spectacles about?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking like a teacher 🤔"):
            rag_chain = get_rag_chain(subject)
            answer = rag_chain.invoke({"question": question})

        st.subheader("✅ Answer")
        st.write(answer)