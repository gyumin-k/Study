import tempfile
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime

# Streamlit 페이지 설정
st.set_page_config(page_title="벼락치기 도우미", page_icon="⏳")

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

# 텍스트 요약
def summarize_text(text, llm, max_summary_length=2000):
    chunks = [text[i:i + 3000] for i in range(0, len(text), 3000)]  # 텍스트를 나눔
    summaries = []
    for chunk in chunks:
        messages = [
            SystemMessage(content="당신은 유능한 한국어 요약 도우미입니다."),
            HumanMessage(content=f"다음 텍스트를 요약해주세요:\n\n{chunk}")
        ]
        response = llm(messages)
        summaries.append(response.content[:max_summary_length])

    return "\n".join(summaries)

# 기출문제 형식 추출
def extract_exam_format(text, llm):
    messages = [
        SystemMessage(content="당신은 문제지 형식을 분석하는 도우미입니다."),
        HumanMessage(content=f"""
        다음 텍스트에서 문제지의 형식을 분석해주세요:
        {text}
        형식적인 구조(객관식, 주관식, 보기 형식 등)에 집중해주세요.
        """)
    ]
    response = llm(messages)
    return response.content

# 예상 문제 생성
def generate_quiz_questions(summary, exam_format, llm, max_chunk_size=2000):
    chunks = [summary[i:i + max_chunk_size] for i in range(0, len(summary), max_chunk_size)]
    quiz_results = []

    for chunk in chunks:
        messages = [
            SystemMessage(content="당신은 한국 대학생을 위한 예상 문제를 작성하는 도우미입니다."),
            HumanMessage(content=f"""
            다음 요약된 텍스트를 기반으로 예상 문제를 작성해주세요:
            {chunk}
            문제의 형식은 다음에 맞춰주세요:
            {exam_format}
            """)
        ]
        response = llm(messages)
        quiz_results.append(response.content)

    return "\n\n".join(quiz_results)

# Streamlit 앱
def main():
    st.title("⏳ 대학생 벼락치기 도우미")

    if "lecture_text" not in st.session_state:
        st.session_state.lecture_text = {}

    if "exam_format" not in st.session_state:
        st.session_state.exam_format = None

    if "summary" not in st.session_state:
        st.session_state.summary = None

    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    # Sidebar 설정
    with st.sidebar:
        st.header("📂 파일 업로드")
        lecture_files = st.file_uploader("강의자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        exam_files = st.file_uploader("기출문제 업로드 (형식만 사용)", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        process_button = st.button("🚀 예상 문제 생성")

    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return

        if not lecture_files:
            st.warning("강의자료를 업로드해주세요!")
            return

        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

        # 강의자료 텍스트 추출 및 요약
        for file in lecture_files:
            st.session_state.lecture_text[file.name] = extract_text_from_file(file)

        lecture_text = "\n".join(st.session_state.lecture_text.values())
        st.session_state.summary = summarize_text(lecture_text, llm)

        # 기출문제 형식 추출
        if exam_files:
            exam_text = "\n".join([extract_text_from_file(file) for file in exam_files])
            st.session_state.exam_format = extract_exam_format(exam_text, llm)
        else:
            st.session_state.exam_format = "객관식과 주관식 문제로 구성된 기본 문제 형식"

        # 예상 문제 생성
        st.session_state.quiz = generate_quiz_questions(
            st.session_state.summary,
            st.session_state.exam_format,
            llm
        )

    # 메인 화면 출력
    st.subheader("📌 강의자료 요약")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("강의자료를 업로드하고 요약 결과를 확인하세요.")

    st.subheader("📋 기출문제 형식")
    if st.session_state.exam_format:
        st.markdown(st.session_state.exam_format)
    else:
        st.info("기출문제를 업로드하면 형식을 분석하여 표시합니다.")

    st.subheader("❓ 예상 문제")
    if st.session_state.quiz:
        st.markdown(st.session_state.quiz)
    else:
        st.info("예상 문제를 생성하려면 강의자료와 기출문제를 업로드하세요.")

if __name__ == "__main__":
    main()
