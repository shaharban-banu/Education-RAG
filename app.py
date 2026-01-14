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


st.markdown("""
<style>
/* Keep black background (Streamlit default dark) */
.stApp {
    background-color: black;
}

/* Main title */
h1 {
    font-size: 44px !important;
    font-weight: 800;
    color: #ffffff;
}

/* Section headings */
h3 {
    font-size: 28px !important;
    font-weight: 700;
    color: #f1f1f1;
}

/* Labels (Selectbox, Input labels) */
label {
    font-size: 22px !important;
    font-weight: 700;
    color: #eaeaea;
}
            

/* ---- Control overall content width ---- */
.block-container {
    padding-left: 2rem;
}

/* ---- Make Selectbox width smaller ---- */
.stSelectbox > div {
    width: 50% !important;
}

/* ---- Make TextInput (Ask Question) SAME width ---- */
.stTextInput > div {
    width: 50% !important;
}

/* Improve text readability */
label {
    font-size: 22px !important;
    font-weight: 700;
}

.stTextInput input {
    font-size: 18px !important;
}

            
            /* Button styling */
.stButton > button {
    background-color: #22c55e;
    color: black;
    font-size: 18px;
    font-weight: 700;
    padding: 10px 26px;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #16a34a;
}

/* Answer card */
.answer-box {
    background: #111827;
    border-left: 6px solid #22c55e;
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 18px;
    color: #f9fafb;
    line-height: 1.6;
}
            
</style>
""", unsafe_allow_html=True)




st.title("📘 NCERT Buddy")
st.markdown(
    "<h3>Your friendly study helper for Class 5 🌟</h3>",
    unsafe_allow_html=True
)

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
# subject = st.selectbox(
#     "📚 Select Subject",
#     options=["All","english", "science"]
# )
subject = st.selectbox(
    "📚 Choose your subject",
    options=["All","English 📖", "Science 🔬"]
)

# Normalize subject value
subject = subject.lower().split()[0]



question = st.text_input(
    "❓ Ask a question from the textbook",
    placeholder="e.g. What is the poem Papa’s Spectacles about?"
)

if st.button("🎈 Ask NCERT Buddy"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking like a teacher 🤔"):
            rag_chain = get_rag_chain(subject)
            answer = rag_chain.invoke({"question": question})

        # st.subheader("✅ Answer")
        # st.write(answer)
        st.markdown("### ✅ Answer ✨")

        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
    "<hr><center>🌈 Keep learning, keep asking questions! 🌈</center>",
    unsafe_allow_html=True
    )
