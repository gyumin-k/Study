import tempfile
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, Document
from datetime import datetime

# Streamlit 페이지 설정
st.set_page_config(
    page_title="벼락치기 도우미",
    page_icon="⏳",
)

# 파일에서 텍스트 추출
def extract_text_from_file(file):
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
        st.warning(f"지원하지 않는 파일 형식입니다: {file.name}")
        return ""

    documents = loader.load_and_split()
    full_text = " ".join([doc.page_content for doc in documents])

    os.remove(temp_file_path)
    return full_text

# 텍스트 청크로 분할
def split_text_into_chunks(uploaded_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    documents = [Document(page_content=text) for text in uploaded_text.values()]
    return text_splitter.split_documents(documents)

# 텍스트 요약
def summarize_text(text_chunks, llm, max_summary_length=2000):
    def process_chunk(chunk):
        text = chunk.page_content
        messages = [
            SystemMessage(content="당신은 유능한 한국어 요약 도우미입니다."),
            HumanMessage(content=f"다음 텍스트를 한국어로 요약해주세요:\n\n{text}")
        ]
        response = llm(messages)
        return response.content

    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(process_chunk, text_chunks))
    
    combined_summary = "\n".join(summaries)
    return combined_summary[:max_summary_length] + "..." if len(combined_summary) > max_summary_length else combined_summary

# 공부 로드맵 생성
def create_study_roadmap(summary, llm, days_left, max_summary_length=2000):
    if len(summary) > max_summary_length:
        summary = summary[:max_summary_length] + "..."
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 유능한 공부 로드맵 작성 도우미입니다."),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 {days_left}일 동안 효과적으로 공부할 수 있는 계획을 작성해주세요:
        {summary}
        """)
    ]
    response = llm(messages)
    return response.content

# 예상 문제 생성
def generate_quiz_questions(text, llm, use_exam_format=False):
    system_message = "당신은 한국 대학생을 위한 예상 문제를 작성하는 도우미입니다."
    if use_exam_format:
        system_message += " 기출문제 형식을 반영하여 작성해주세요."
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 예상 문제를 작성해주세요:
        {text}
        - 각 문제에는 중요도를 '높음', '중간', '낮음'으로 표시해주세요.
        """)
    ]
    response = llm(messages)
    return response.content

# Streamlit 앱 설정
def main():
    st.title("⏳ 대학생 벼락치기 도우미")

    if "lecture_text" not in st.session_state:
        st.session_state.lecture_text = {}

    if "exam_text" not in st.session_state:
        st.session_state.exam_text = {}

    if "summary" not in st.session_state:
        st.session_state.summary = {}

    if "roadmap" not in st.session_state:
        st.session_state.roadmap = None

    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    with st.sidebar:
        lecture_files = st.file_uploader("📄 강의자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True, key="lecture")
        exam_files = st.file_uploader("📄 기출문제 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True, key="exam")
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")

    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return

        if not lecture_files:
            st.warning("강의자료를 업로드해주세요!")
            return

        # 시험까지 남은 기간 계산
        days_left = (exam_date - datetime.now().date()).days
        if days_left <= 0:
            st.warning("시험 날짜는 오늘 이후여야 합니다!")
            return

        # 강의자료와 기출문제 텍스트 추출
        for file in lecture_files:
            st.session_state.lecture_text[file.name] = extract_text_from_file(file)
        for file in exam_files:
            st.session_state.exam_text[file.name] = extract_text_from_file(file)

        # 강의자료 및 기출문제 텍스트 분리
        lecture_chunks = split_text_into_chunks(st.session_state.lecture_text)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

        # 요약 생성
        st.session_state.summary = summarize_text(lecture_chunks, llm)

        # 공부 로드맵 생성
        st.session_state.roadmap = create_study_roadmap(
            st.session_state.summary, llm, days_left
        )

        # 예상 문제 생성
        if st.session_state.exam_text:  # 기출문제가 있는 경우
            exam_chunks = split_text_into_chunks(st.session_state.exam_text)
            all_text = "\n".join([chunk.page_content for chunk in lecture_chunks + exam_chunks])
            st.session_state.quiz = generate_quiz_questions(all_text, llm, use_exam_format=True)
        else:  # 강의자료만 있는 경우
            all_text = "\n".join([chunk.page_content for chunk in lecture_chunks])
            st.session_state.quiz = generate_quiz_questions(all_text, llm, use_exam_format=False)

    # 결과 출력
    if st.session_state.summary:
        st.subheader("📌 핵심 요약")
        st.markdown(st.session_state.summary)

    if st.session_state.roadmap:
        st.subheader("📋 공부 로드맵")
        st.markdown(st.session_state.roadmap)

    if st.session_state.quiz:
        st.subheader("❓ 예상 문제")
        st.markdown(st.session_state.quiz)

if __name__ == "__main__":
    main()
