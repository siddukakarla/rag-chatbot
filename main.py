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
}

/* Fix chat input to bottom and style the send icon area */
.stChatInput {
    padding-bottom: 20px;
}

/* Chat bubble adjustments */
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.header("🤖 Assistant Chatbot")

# ---------- INITIALIZE SESSION STATE ----------
# This keeps the chat history visible when the app reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 📄 Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("💡 **Tip:** Ask questions about the uploaded PDF.")

# ---------- PDF PROCESSING & CHAL LOGIC ----------
if file is not None:
    # We use a spinner so the user knows the PDF is being processed
    with st.spinner("Processing PDF..."):
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

        # Prepare retriever
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
             "2. Only use information from the provided context.\n"
             "3. If the information is not in the context, say you don't know.\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    # ---------- DISPLAY CHAT HISTORY ----------
    # This renders all previous messages in the UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------- CHAT INPUT AREA ----------
    # st.chat_input automatically anchors to the bottom and has a built-in enter/send icon
    if user_question := st.chat_input("Type your question here..."):
        
        # 1. Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # 2. Add user message to session history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # 3. Generate response from the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(user_question)
                st.markdown(response)
        
        # 4. Add assistant response to session history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👋 Please upload a PDF file in the sidebar to begin.")
