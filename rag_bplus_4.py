import tempfile
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datetime import datetime

# Streamlit 페이지 설정
st.set_page_config(
    page_title="벼락치기 도우미",
    page_icon="⏳",
)

# Streamlit 앱 설정
def main():
    st.title("⏳ 대학생 벼락치기 도우미")

    # 상태 초기화
    if "uploaded_text" not in st.session_state:
        st.session_state.uploaded_text = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "roadmap" not in st.session_state:
        st.session_state.roadmap = None
    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    # 사이드바 UI 설정
    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")

        create_summary = st.checkbox("핵심 요약 생성", value=True)
        create_roadmap = st.checkbox("공부 로드맵 생성", value=True)
        create_quiz = st.checkbox("예상 문제 생성", value=True)

    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return
        if not uploaded_files:
            st.warning("강의 자료를 업로드해주세요!")
            return
        if not exam_date:
            st.warning("시험 날짜를 선택해주세요!")
            return

        today = datetime.now().date()
        days_left = (exam_date - today).days
        if days_left <= 0:
            st.warning("시험 날짜는 오늘 이후여야 합니다!")
            return

        st.session_state.uploaded_text = extract_text_from_files(uploaded_files)
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        vectorstore = create_vectorstore(text_chunks, openai_api_key)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

        if create_summary:
            st.session_state.summary = summarize_text(text_chunks, llm)

        if create_roadmap and st.session_state.summary:
            st.session_state.roadmap = create_study_roadmap(st.session_state.summary, llm, days_left)

        if create_quiz and st.session_state.summary:
            st.session_state.quiz = generate_quiz_questions(st.session_state.summary, llm)

    if st.session_state.summary:
        st.subheader("📌 핵심 요약")
        st.markdown(st.session_state.summary)

    if st.session_state.roadmap:
        st.subheader("📋 공부 로드맵")
        st.markdown(st.session_state.roadmap)

    if st.session_state.quiz:
        st.subheader("❓ 예상 문제")
        st.markdown(st.session_state.quiz)


# 파일에서 텍스트 추출
def extract_text_from_files(files):
    doc_list = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(file.read())
            temp_file_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_file_path)
        else:
            st.warning(f"지원하지 않는 파일 형식입니다: {file.name}")
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)
        os.remove(temp_file_path)
    return doc_list

# 텍스트 청크로 분할
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500
    )
    return text_splitter.split_documents(text)

# 벡터 저장소 생성 (OpenAI 임베딩 사용)
def create_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(text_chunks, embeddings)

if __name__ == "__main__":
    main()
