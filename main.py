import os
import pdfplumber
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Assistant Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------- STYLING ----------
st.markdown("""
<style>
/* Header Styling */
h1 {
    color: #4B6CB7;
    font-size: 2.5rem;
    text-align: center;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    padding: 1rem;
}

/* Input Box Styling */
div.stTextInput > label {
    font-weight: bold;
    color: #4B6CB7;
}

/* Button Styling */
button[kind="primary"] {
    background-color: #4B6CB7;
    color: white;
    font-weight: bold;
}

/* Chat Bubble Style */
div[data-testid="stMarkdownContainer"] p {
    font-size: 1rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.header("🤖 Assistant Chatbot")
st.subheader("Upload your document and ask questions!")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 📄 Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")
    st.markdown("---")
    st.markdown("💡 **Tip:** Ask questions about the uploaded PDF. The assistant will answer only based on the document.")

# ---------- PDF PROCESSING ----------
if file is not None:
    # Extract text
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # Store in FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User input
    user_question = st.text_input("💬 Type your question here")

    # Prepare retriever
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # LLM & Prompt
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. Only use information from the provided context - do not use outside knowledge.\n"
         "4. Summarize long information, ideally in bullets where needed.\n"
         "5. If the information is not in the context, politely say: "
         "\"I'm sorry, but I don't have enough information to answer that based on the provided context.\"\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Display response
    if user_question:
        response = chain.invoke(user_question)
        st.markdown(f"**Assistant:** {response}")
