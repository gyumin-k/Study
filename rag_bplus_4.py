import tempfile
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datetime import datetime


# Streamlit 앱 설정
def main():
    st.set_page_config(
        page_title="벼락치기 도우미",
        page_icon="⏳",
    )
    st.title("⏳ 대학생 벼락치기 도우미")

    if "uploaded_text" not in st.session_state:
        st.session_state.uploaded_text = None

    if "roadmap" not in st.session_state:
        st.session_state.roadmap = None

    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")

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

        # 시험까지 남은 기간 계산
        days_left = (exam_date - datetime.now().date()).days
        if days_left <= 0:
            st.warning("시험 날짜는 오늘보다 이후여야 합니다!")
            return

        # 업로드한 파일 텍스트 추출
        st.session_state.uploaded_text = extract_text_from_files(uploaded_files)

        # 벡터 저장소 및 요약 생성
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        vectorstore = create_vectorstore(text_chunks)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

        # 핵심 요약 생성
        st.session_state.summary = summarize_text(text_chunks, llm)

        # 공부 로드맵 생성
        st.session_state.roadmap = create_study_roadmap(st.session_state.summary, llm, days_left)

        # 예상 문제 생성
        st.session_state.quiz = generate_quiz_questions(st.session_state.summary, llm)

    if st.session_state.uploaded_text:
        st.subheader("📌 핵심 요약")
        st.markdown(st.session_state.summary)

        st.subheader("📋 공부 로드맵")
        st.markdown(st.session_state.roadmap)

        st.subheader("❓ 예상 문제")
        st.markdown(st.session_state.quiz)


# 파일에서 텍스트 추출
def extract_text_from_files(files):
    doc_list = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
            tmp_file.write(file.read())
            temp_file_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_file_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)

        # 임시 파일 삭제
        os.remove(temp_file_path)
    return doc_list


# 텍스트 청크로 분할
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(text)


# 벡터 저장소 생성
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return FAISS.from_documents(text_chunks, embeddings)


# 텍스트 요약
def summarize_text(text_chunks, llm):
    summaries = []
    for chunk in text_chunks:
        text = chunk.page_content
        messages = [
            SystemMessage(content="You are a helpful assistant that summarizes text."),
            HumanMessage(content=f"Summarize the following text:\n\n{text}")
        ]
        response = llm(messages)
        summaries.append(response.content)
    return "\n".join(summaries)


# 공부 로드맵 생성
def create_study_roadmap(summary, llm, days_left):
    messages = [
        SystemMessage(content="You are a helpful assistant that creates study plans."),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 {days_left}일 동안 효과적으로 공부할 계획을 만들어주세요:
        {summary}
        """)
    ]
    response = llm(messages)
    return response.content


# 예상 문제 생성
def generate_quiz_questions(summary, llm):
    messages = [
        SystemMessage(content="You are a helpful assistant that generates quiz questions."),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 시험에 나올 수 있는 5개의 예상 문제를 만들어주세요:
        {summary}
        """)
    ]
    response = llm(messages)
    return response.content


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

