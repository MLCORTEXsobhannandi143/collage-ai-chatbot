import streamlit as st
import os
import tempfile
import pandas as pd
from datetime import date, datetime, time
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from hashlib import sha256

# ==================== COLLEGE CONFIGURATION ====================
COLLEGE_CODE = "BWU2025"
COLLEGE_NAME = "BRAINWARE UNIVERSITY"

DATA_FOLDER = f"data_{COLLEGE_CODE}"
os.makedirs(DATA_FOLDER, exist_ok=True)

ATTENDANCE_FILE = os.path.join(DATA_FOLDER, "attendance.csv")
RESULTS_FILE = os.path.join(DATA_FOLDER, "results.csv")
LIBRARY_BORROW_FILE = os.path.join(DATA_FOLDER, "library_borrow.csv")
LIBRARY_UPLOAD_DIR = os.path.join(DATA_FOLDER, "library_uploads")
LIBRARY_PREV_QUESTIONS_DIR = os.path.join(LIBRARY_UPLOAD_DIR, "previous_questions")
LIBRARY_STORIES_DIR = os.path.join(LIBRARY_UPLOAD_DIR, "stories")
os.makedirs(LIBRARY_PREV_QUESTIONS_DIR, exist_ok=True)
os.makedirs(LIBRARY_STORIES_DIR, exist_ok=True)
STUDENT_PASS_FILE = os.path.join(DATA_FOLDER, "student_passwords.csv")

# Page config
st.set_page_config(page_title=f"{COLLEGE_NAME} AI Assistant", page_icon="🌟", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .title {color: #1e3d59; text-align: center; font-size: 40px; font-weight: bold;}
    .subtitle {text-align: center; color: #ff6f61; font-size: 20px;}
    .stButton > button {background-color: #4caf50; color: white; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<h1 class="title">🌟 {COLLEGE_NAME} AI Assistant 🌟</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG Chatbot + Attendance + Library </p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📚 Chat with Documents", "📅 Attendance", "📖 Library"])

# ==================== Chat with Documents Tab ====================
with tab1:
    st.header("Chat with College Documents")

    llm = OllamaLLM(model="llama3.1")
    embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L12-v2",
       model_kwargs={"device": "cpu"},
       encode_kwargs={"normalize_embeddings": True})

    DB_PATH = os.path.join(DATA_FOLDER, "chroma_db")
    os.makedirs(DB_PATH, exist_ok=True)

    if "vectorstore" not in st.session_state:
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
        st.success(f"✅ {uploaded_file.name} uploaded and indexed!")

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

    
# ==================== Attendance Tracker Tab ====================
with tab2:
    st.header("📅 Attendance Tracker")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Date", "Department", "Subject", "ClassType", "Roll", "Status"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    HOLIDAYS = ["2026-01-01", "2026-01-26", "2026-03-25", "2026-04-14", "2026-05-01", "2026-08-15", "2026-10-02", "2026-12-25"]

    today_str = date.today().strftime("%Y-%m-%d")

    department_prefix = {
        "CSE": "BWU/CSE/25/",
        "EEE": "BWU/EEE/25/",
        "Civil": "BWU/CE/25/",
        "BBA": "BWU/BBA/25/",
    }

    department_config = {
        "CSE": {
            "subjects": {
                "CSE101 - Programming": {"password": "cse101", "class_start": "10:00", "class_end": "11:00"},
                "CSE102 - Data Structure": {"password": "cse102", "class_start": "11:00", "class_end": "12:00"},

            }
        },
        "EEE": {
            "subjects": {
                "EEE101 - Basic Electrical": {"password": "eee101", "class_start": "09:00", "class_end": "10:00"},
            }
        },
    }

    st.subheader("Teacher Section")
    selected_dept = st.selectbox("Select Department", list(department_config.keys()))
    selected_subject = st.selectbox("Select Subject", list(department_config[selected_dept]["subjects"].keys()))
    subject_password = st.text_input(f"{selected_subject} Password", type="password")

    correct_password = department_config[selected_dept]["subjects"][selected_subject]["password"]
    teacher_access = (subject_password == correct_password)

    if teacher_access:
        st.success(f"✅ Access granted for {selected_subject}")

        is_extra = st.checkbox("Extra Class?")

        if not is_extra:
            if today_str in HOLIDAYS:
                st.error("🚫 Holiday")
                st.stop()

            start_str = department_config[selected_dept]["subjects"][selected_subject]["class_start"]
            end_str = department_config[selected_dept]["subjects"][selected_subject]["class_end"]
            current_time = datetime.now().time()
            start_time = time.fromisoformat(start_str)
            end_time = time.fromisoformat(end_str)
            if not (start_time <= current_time <= end_time):
                st.error(f"🚫 Outside class time")
                st.stop()

        st.write("### Mark Present")
        roll_last = st.text_input("Roll Last Digits (e.g. 290)")

        if st.button("Mark Present"):
            if roll_last.strip():
                full_roll = department_prefix[selected_dept] + roll_last.strip()
                class_type = "Extra" if is_extra else "Regular"
                new_entry = pd.DataFrame([{
                    "Date": today_str,
                    "Department": selected_dept,
                    "Subject": selected_subject,
                    "ClassType": class_type,
                    "Roll": full_roll,
                    "Status": "Present"
                }])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(ATTENDANCE_FILE, index=False)
                st.success(f"✅ {full_roll} marked Present")
            else:
                st.error("Enter roll")

        st.write("### Summary")
        sub_df = df[(df["Department"] == selected_dept) & (df["Subject"] == selected_subject)]
        if not sub_df.empty:
            summary = sub_df.groupby(["ClassType", "Roll"]).agg(Total=('Status', 'count'), Present=('Status', 'sum')).reset_index()
            summary["Percentage"] = (summary["Present"] / summary["Total"] * 100).round(2)
            st.dataframe(summary)
            if st.button("Full Record"):
                st.dataframe(sub_df)
        else:
            st.info("No record")

    else:
        if subject_password != "":
            st.error("😕 Wrong password")

    st.subheader("Student Section")
    view_dept = st.selectbox("Your Department", list(department_prefix.keys()), key="att_dept")
    view_roll_last = st.text_input("Your Roll Last Digits", key="att_roll")

    if st.button("View My Attendance"):
        if view_roll_last.strip():
            full_roll = department_prefix[view_dept] + view_roll_last.strip()
            my_att = df[df["Roll"] == full_roll]
            if not my_att.empty:
                summary = my_att.groupby(["Department", "Subject", "ClassType"]).agg(Total=('Status', 'count'), Present=('Status', 'sum')).reset_index()
                summary["Percentage"] = (summary["Present"] / summary["Total"] * 100).round(2)
                st.dataframe(summary)
                if st.button("Full Record"):
                    st.dataframe(my_att)
            else:
                st.info("No record")
        else:
            st.error("Enter roll")

# ==================== Library Tab ====================
with tab3:
    st.header("📖 College Library")

    MAX_BORROW_DAYS = 15
    FINE_PER_DAY = 5

    # return book and update file
    def load_borrow_df():
        if os.path.exists(LIBRARY_BORROW_FILE):
           return pd.read_csv(LIBRARY_BORROW_FILE)
        else:
            df = pd.DataFrame(columns=["Date", "Roll", "BookName", "ReturnDate"])
            df.to_csv(LIBRARY_BORROW_FILE, index=False)
            return df
    borrow_df = load_borrow_df()
    # Offline Library - Issue & Return (Teacher only)
    st.subheader("Offline Library - Issue & Return Book (Teacher Only)")
    lib_pass = st.text_input("Teacher Password", type="password", key="offline_lib_pass")

    if lib_pass == "teacher2026":
        st.success("✅ Teacher access granted")

        # Issue Book
        st.write("#### Issue Book")
        issue_dept = st.selectbox("Student Department", list(department_prefix.keys()), key="offline_issue_dept")
        issue_roll_last = st.text_input("Student Roll Last Digits", key="offline_issue_roll")
        book_name = st.text_input("Book Name", key="offline_book_name")

        if st.button("Issue Book", key="offline_issue_button"):
            if issue_roll_last.strip() and book_name:
                full_roll = department_prefix[issue_dept] + issue_roll_last.strip()
                return_date = date.today() + pd.Timedelta(days=MAX_BORROW_DAYS)
                new_borrow = pd.DataFrame([{
                    "Date": today_str,
                    "Roll": full_roll,
                    "BookName": book_name,
                    "ReturnDate": return_date.strftime("%Y-%m-%d")
                }])
                borrow_df = pd.concat([borrow_df, new_borrow], ignore_index=True)
                borrow_df.to_csv(LIBRARY_BORROW_FILE, index=False)
                st.success(f"✅ '{book_name}' issued to {full_roll}")
            else:
                st.error("Enter roll and book name")

        # Return Book
        st.write("#### Return Book")
        return_dept = st.selectbox("Student Department for Return", list(department_prefix.keys()), key="offline_return_dept")
        return_roll_last = st.text_input("Student Roll Last Digits for Return", key="offline_return_roll")

        if st.button("Show Borrowed Books", key="offline_show_return"):
            if return_roll_last.strip():
                full_roll = department_prefix[return_dept] + return_roll_last.strip()
                student_books = borrow_df[borrow_df["Roll"] == full_roll]
                if not student_books.empty:
                    book_to_return = st.selectbox("Select Book to Return", student_books["BookName"].tolist(), key="offline_select_return")
                if st.button("Confirm Return", key="confirm_return_button"):
                        selected_row = student_books[student_books["BookName"] == book_to_return].iloc[0]
                        issue_date = pd.to_datetime(selected_row["Date"])
                        days_used = (date.today() - issue_date).days

                        if days_used > MAX_BORROW_DAYS:
                            days_overdue = days_used - MAX_BORROW_DAYS
                            fine = days_overdue * FINE_PER_DAY
                            st.warning(f"⚠️ Fine: ৳{fine} (Overdue by {days_overdue} days)")
                        else:
                            st.success("✅ No fine – returned within time limit!")

                        # uses for delete book
                        borrow_df = borrow_df[borrow_df["BookName"] != book_to_return]
                        borrow_df = borrow_df.reset_index(drop=True)
                        borrow_df.to_csv(LIBRARY_BORROW_FILE, index=False)
                        
                        borrow_df = load_borrow_df()

                        st.success(f"✅ '{book_to_return}' returned and removed from list")
                        st.rerun()  
                else:
                    st.info("No borrowed books")
            else:
                st.error("Enter roll")

        # All Issued Books
        st.write("### All Issued Books")
        if not borrow_df.empty:
            today_pd = pd.to_datetime(date.today())
            borrow_df["ReturnDate"] = pd.to_datetime(borrow_df["ReturnDate"])
            borrow_df["DaysOverdue"] = (today_pd - borrow_df["ReturnDate"]).dt.days
            borrow_df["DaysOverdue"] = borrow_df["DaysOverdue"].clip(lower=0)
            borrow_df["Fine"] = borrow_df["DaysOverdue"] * FINE_PER_DAY
            st.dataframe(borrow_df[["Date", "Roll", "BookName", "ReturnDate", "DaysOverdue", "Fine"]])
        else:
            st.info("No books issued")

    else:
        if lib_pass != "":
            st.error("Wrong password")

    # Student - My Borrowed Books
    st.subheader("My Borrowed Books (Offline)")
    stu_lib_dept = st.selectbox("Your Department", list(department_prefix.keys()), key="stu_offline_dept")
    stu_lib_roll_last = st.text_input("Your Roll Last Digits", key="stu_offline_roll")

    if st.button("View My Borrowed Books", key="stu_offline_view"):
        if stu_lib_roll_last.strip():
            full_roll = department_prefix[stu_lib_dept] + stu_lib_roll_last.strip()
            my_books = borrow_df[borrow_df["Roll"] == full_roll].copy()
            if not my_books.empty:
                today_pd = pd.to_datetime(date.today())
                my_books["ReturnDate"] = pd.to_datetime(my_books["ReturnDate"])
                my_books["DaysLeft"] = (my_books["ReturnDate"] - today_pd).dt.days
                my_books["Fine"] = my_books["DaysLeft"].apply(lambda x: max(-x * FINE_PER_DAY, 0) if x < 0 else 0)
                st.dataframe(my_books[["BookName", "Date", "ReturnDate", "DaysLeft", "Fine"]])
            else:
                st.info("No borrowed books")
        else:
            st.error("Enter roll")

    # Online Resources - Upload & View
    st.subheader("Online Study Resources (Upload & View)")

    # Teacher - Upload Study Notes (Subject password required)
    st.subheader("Teacher - Upload Study Notes (Password Required)")
    note_pass = st.text_input("Enter Subject Password to Upload Notes", type="password", key="note_upload_password")

    note_dept = st.selectbox("Department for Notes Upload", list(department_config.keys()), key="note_upload_dept")
    note_subject = st.selectbox("Subject for Notes Upload", list(department_config[note_dept]["subjects"].keys()), key="note_upload_subject")

    # chaking password of subject
    correct_pass = department_config[note_dept]["subjects"][note_subject]["password"]

    if note_pass == correct_pass:
        st.success(f"✅ Access granted – you can upload notes for {note_subject}")

        note_upload = st.file_uploader("Upload Study Notes (PDF/TXT)", type=["pdf", "txt"], key="secure_note_upload")

        if note_upload:
            dept_dir = os.path.join(LIBRARY_UPLOAD_DIR, note_dept, "notes", note_subject)
            os.makedirs(dept_dir, exist_ok=True)
            file_path = os.path.join(dept_dir, note_upload.name)
            with open(file_path, "wb") as f:
                f.write(note_upload.getbuffer())
            st.success(f"✅ Notes uploaded successfully for {note_subject}")

    else:
        if note_pass != "":
            st.error("😕 Wrong password – only the subject teacher can upload notes")
        st.info("Enter the correct subject password to upload notes")
    # View Study Notes
    st.write("#### View Study Notes")
    view_note_dept = st.selectbox("Your Department", list(department_config.keys()), key="view_note_dept")
    view_note_subject = st.selectbox("Subject", list(department_config[view_note_dept]["subjects"].keys()), key="view_note_subject")
    note_search = st.text_input("Search notes by name", key="note_search")

    note_dir = os.path.join(LIBRARY_UPLOAD_DIR, view_note_dept, "notes", view_note_subject)
    if os.path.exists(note_dir):
        note_files = os.listdir(note_dir)
        if note_search:
            note_files = [f for f in note_files if note_search.lower() in f.lower()]
        if note_files:
            st.write(f"### {view_note_subject} Notes ({len(note_files)} found)")
            for file in note_files:
                file_path = os.path.join(note_dir, file)
                with open(file_path, "rb") as f:
                    st.download_button(f"📄 {file}", f, file_name=file)
        else:
            st.info("No matching notes")
    else:
        st.info("No notes uploaded")

    # Previous Year Questions
    st.write("#### Previous Year Questions")
    prev_dept = st.selectbox("Department for Previous Questions", list(department_config.keys()), key="prev_dept_view")
    prev_search = st.text_input("Search previous questions by name", key="prev_search_key")

    prev_dir = os.path.join(LIBRARY_PREV_QUESTIONS_DIR, prev_dept)
    if os.path.exists(prev_dir):
        prev_files = os.listdir(prev_dir)
        if prev_search:
            prev_files = [f for f in prev_files if prev_search.lower() in f.lower()]
        if prev_files:
            st.write(f"### {prev_dept} Previous Questions ({len(prev_files)} found)")
            for file in prev_files:
                file_path = os.path.join(prev_dir, file)
                with open(file_path, "rb") as f:
                    st.download_button(f"📄 {file}", f, file_name=file)
        else:
            st.info("No matching questions")
    else:
        st.info("No questions uploaded")

    prev_upload = st.file_uploader("Upload Previous Question Paper", type=["pdf", "txt"], key="prev_upload_key")
    if prev_upload:
        dept_dir = os.path.join(LIBRARY_PREV_QUESTIONS_DIR, prev_dept)
        os.makedirs(dept_dir, exist_ok=True)
        file_path = os.path.join(dept_dir, prev_upload.name)
        with open(file_path, "wb") as f:
            f.write(prev_upload.getbuffer())
        st.success("Uploaded")

    # Stories
    st.write("#### Stories & Novels")
    story_search = st.text_input("Search stories by name", key="story_search_key")

    if os.path.exists(LIBRARY_STORIES_DIR):
        story_files = os.listdir(LIBRARY_STORIES_DIR)
        if story_search:
            story_files = [f for f in story_files if story_search.lower() in f.lower()]
        if story_files:
            st.write(f"### Stories ({len(story_files)} found)")
            for file in story_files:
                file_path = os.path.join(LIBRARY_STORIES_DIR, file)
                with open(file_path, "rb") as f:
                    st.download_button(f"📖 {file}", f, file_name=file)
        else:
            st.info("No matching stories")
    else:
        st.info("No stories uploaded")

    story_upload = st.file_uploader("Upload Story/Novel", type=["pdf", "txt"], key="story_upload_key")
    if story_upload:
        file_path = os.path.join(LIBRARY_STORIES_DIR, story_upload.name)
        with open(file_path, "wb") as f:
            f.write(story_upload.getbuffer())
        st.success("Uploaded to Stories")

    # Recommended Books from CSV (Teacher uploads CSV)
    st.subheader("Recommended Books for Subject (From CSV)")

    # Teacher - Upload CSV for recommended books
    rec_csv_pass = st.text_input("Teacher Password for Upload Recommended Books CSV", type="password", key="rec_csv_pass")

    if rec_csv_pass == "teacher2026":  # changeable
        st.success("✅ Access granted for CSV upload")
        rec_csv_upload = st.file_uploader("Upload recommended_books.csv (Department, Subject, BookName)", type=["csv"], key="rec_csv_upload")

        if rec_csv_upload:
            rec_df = pd.read_csv(rec_csv_upload)
            rec_df.to_csv(os.path.join(DATA_FOLDER, "recommended_books.csv"), index=False)
            st.success("✅ Recommended books CSV uploaded and updated!")

    # Student - View Recommended Books
    rec_view_dept = st.selectbox("Select Department for Recommended Books", list(department_config.keys()), key="rec_view_dept")
    rec_view_subject = st.selectbox("Select Subject", list(department_config[rec_view_dept]["subjects"].keys()), key="rec_view_subject")

    rec_csv_file = os.path.join(DATA_FOLDER, "recommended_books.csv")
    if os.path.exists(rec_csv_file):
        rec_df = pd.read_csv(rec_csv_file)
        filtered_books = rec_df[(rec_df["Department"] == rec_view_dept) & (rec_df["Subject"] == rec_view_subject)]["BookName"].tolist()
        if filtered_books:
            st.write(f"### Recommended Books for {rec_view_subject}")
            for book in filtered_books:
                st.write(f"📚 {book}")
        else:
            st.info("No recommended books for this subject yet")
    else:
        st.info("No recommended books CSV uploaded yet")

    if st.button("Show Recommended Books", key="show_rec_csv"):
        pass  
