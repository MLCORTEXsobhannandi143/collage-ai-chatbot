# Import required libraries
import streamlit as st
import os
import tempfile
import pandas as pd
from datetime import date
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set page config for better look
st.set_page_config(page_title="College AI Assistant", page_icon="🌟", layout="wide")

# Custom CSS for beautiful frontend
st.markdown("""
<style>
    .main {background-color: #f0f2f6; padding: 20px;}
    .title {color: #1e3d59; text-align: center; font-size: 40px; font-weight: bold; margin-bottom: 10px;}
    .subtitle {text-align: center; color: #ff6f61; font-size: 20px; margin-bottom: 30px;}
    .sidebar .sidebar-content {background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .uploadedFile {color: #4caf50; font-weight: bold;}
    .stButton > button {background-color: #4caf50; color: white; border-radius: 5px; width: 100%;}
    .stDataFrame {background-color: #ffffff; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">🌟 My College AI Assistant 🌟</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG Chatbot + Attendance Tracking</p>', unsafe_allow_html=True)

# Tabs for navigation
tab1, tab2 = st.tabs(["📚 Chat with Documents", "📅 Attendance Tracker"])

with tab1:
    st.header("Chat with College Documents")

    llm = OllamaLLM(model="llama3.1")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    DB_PATH = "chroma_db"

    if "vectorstore" not in st.session_state:
        if os.path.exists(DB_PATH):
            st.session_state.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        else:
            st.session_state.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    vectorstore = st.session_state.vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    template = """Answer only from the given context. If you don't know, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        vectorstore.add_documents(chunks)
        st.success(f"✅ {uploaded_file.name} uploaded!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response = chain.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("Attendance Tracker")

    ATTENDANCE_FILE = "attendance.csv"

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Date", "Name", "Roll", "Status"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    name = st.text_input("Your Name")
    roll = st.text_input("Roll Number")

    if st.button("Mark Present"):
        if name and roll:
            today = date.today().strftime("%Y-%m-%d")
            new_entry = pd.DataFrame([{"Date": today, "Name": name, "Roll": roll, "Status": "Present"}])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            st.success(f"✅ Marked Present for {name} ({roll}) on {today}")
        else:
            st.error("Enter name and roll")

    if st.button("View My Attendance"):
        if name and roll:
            my_att = df[(df["Name"] == name) & (df["Roll"] == roll)]
            if not my_att.empty:
                st.dataframe(my_att)
                present_count = len(my_att[my_att["Status"] == "Present"])
                total = len(my_att)
                percentage = (present_count / total) * 100 if total > 0 else 0
                st.metric("Attendance Percentage", f"{percentage:.2f}%")
            else:
                st.info("No record found")
        else:
            st.error("Enter name and roll")

    if st.button("View All Attendance"):
        st.dataframe(df)