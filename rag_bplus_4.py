import tempfile
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rouge_score import rouge_scorer
from langdetect import detect_langs

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

    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        process_button = st.button("🚀 벼락치기 시작하기")

        create_summary = st.checkbox("핵심 요약 생성", value=True)
        create_roadmap = st.checkbox("공부 로드맵 생성", value=True)
        create_quiz = st.checkbox("예상 문제 생성", value=True)
        show_metrics = st.checkbox("요약 성능 평가 표시", value=True)

    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return
        if not uploaded_files:
            st.warning("강의 자료를 업로드해주세요!")
            return

        st.session_state.uploaded_text = extract_text_from_files(uploaded_files)
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

        if create_summary:
            st.session_state.summary = summarize_text(text_chunks, llm)

        if create_roadmap and st.session_state.summary:
            st.session_state.roadmap = create_study_roadmap(st.session_state.summary, llm, 7)

        if create_quiz and st.session_state.summary:
            st.session_state.quiz = generate_quiz_questions(st.session_state.summary, llm)

        if show_metrics and st.session_state.summary:
            st.subheader("📊 요약 성능 평가")
            metrics = evaluate_summary(st.session_state.uploaded_text, st.session_state.summary)
            for metric, score in metrics.items():
                st.write(f"**{metric}:** {score:.2f}")

    if st.session_state.summary:
        st.subheader("📌 핵심 요약")
        st.markdown(st.session_state.summary)

    if st.session_state.roadmap:
        st.subheader("📋 공부 로드맵")
        st.markdown(st.session_state.roadmap)

    if st.session_state.quiz:
        st.subheader("❓ 예상 문제")
        st.markdown(st.session_state.quiz)

# 텍스트 추출 함수
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

        # 임시 파일 삭제
        os.remove(temp_file_path)
    return doc_list

# 병렬 처리된 요약문 후처리 함수
def refine_summary(summaries, llm):
    combined_summary = "\n".join(summaries)
    messages = [
        SystemMessage(content="당신은 요약문을 정리하고 간결하게 만드는 도우미입니다."),
        HumanMessage(content=f"다음 요약문들을 읽고 하나의 일관된 요약문으로 정리해주세요:\n\n{combined_summary}")
    ]
    response = llm(messages)
    return response.content

# 텍스트 청크로 분할
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return text_splitter.split_documents(text)

if __name__ == "__main__":
    main()
