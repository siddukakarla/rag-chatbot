import os
import pdfplumber
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- Streamlit Config ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.header("📄 RAG PDF Chatbot")


# ---------- Load OpenAI Key from Streamlit Secrets ----------

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ---------- Sidebar ----------
with st.sidebar:
    st.title("Upload Document")
    file = st.file_uploader("Upload a PDF file", type="pdf")


# ---------- Process PDF ----------
if file is not None:

    # Extract text safely
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        st.error("No readable text found in PDF")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    st.success(f"Document processed | Chunks created: {len(chunks)}")

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=800
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions from a PDF document.\n\n"
         "Rules:\n"
         "1. Use ONLY the provided context\n"
         "2. If answer not in context, say politely 'Not found in document'\n"
         "3. Give clear and structured answers\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User Question
    user_question = st.text_input("Ask a question from the PDF")

    if user_question:
        with st.spinner("Thinking..."):
            response = chain.invoke(user_question)
        st.markdown("### Answer")
        st.write(response)
