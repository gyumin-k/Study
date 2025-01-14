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

    # 사이드바 UI 설정
    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        process_button = st.button("🚀 벼락치기 시작하기")

        # 체크박스 설정
        create_summary = st.checkbox("핵심 요약 생성", value=True)
        create_roadmap = st.checkbox("공부 로드맵 생성", value=True)
        create_quiz = st.checkbox("예상 문제 생성", value=True)
        show_metrics = st.checkbox("요약 성능 평가 표시", value=True)

    # 기능별 처리
    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return
        if not uploaded_files:
            st.warning("강의 자료를 업로드해주세요!")
            return

        # 파일에서 텍스트 추출
        st.session_state.uploaded_text = extract_text_from_files(uploaded_files)
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

        # 핵심 요약 생성
        if create_summary:
            st.session_state.summary = summarize_text(text_chunks, llm)

        # 공부 로드맵 생성
        if create_roadmap and st.session_state.summary:
            st.session_state.roadmap = create_study_roadmap(st.session_state.summary, llm, days_left=7)

        # 예상 문제 생성
        if create_quiz and st.session_state.summary:
            st.session_state.quiz = generate_quiz_questions(st.session_state.summary, llm)

        # 요약 성능 평가
        if show_metrics and st.session_state.summary:
            st.subheader("📊 요약 성능 평가")
            metrics = evaluate_summary(st.session_state.uploaded_text, st.session_state.summary)
            for metric, score in metrics.items():
                st.write(f"**{metric}:** {score:.2f}")

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
        os.remove(temp_file_path)
    return doc_list

# 텍스트 청크로 분할 함수
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return text_splitter.split_documents(text)

# 요약 생성 함수
def summarize_text(text_chunks, llm, max_summary_length=2000):
    def process_chunk(chunk):
        text = chunk.page_content
        detected_languages = detect_langs(text)
        if any(lang.lang == "ko" and lang.prob > 0.5 for lang in detected_languages):
            system_prompt = "당신은 유능한 한국어 요약 도우미입니다."
            human_prompt = f"""
            다음 텍스트를 한국어로 요약해주세요:
            1. 핵심 주제만 간결히 포함
            2. 중요 키워드를 강조
            3. 불필요한 정보 및 중복 제거
            텍스트:\n\n{text}
            """
        else:
            system_prompt = "You are a skilled English summarization assistant."
            human_prompt = f"""
            Please summarize the following text:
            1. Include only the main points.
            2. Highlight important keywords.
            3. Remove unnecessary details and redundancy.
            Text:\n\n{text}
            """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = llm(messages)
        return response.content

    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(process_chunk, text_chunks))

    # 후처리로 요약문 정리
    return refine_summary(summaries, llm)

# 병렬 처리된 요약문 후처리 함수
def refine_summary(summaries, llm):
    combined_summary = "\n".join(summaries)
    messages = [
        SystemMessage(content="당신은 요약문을 정리하고 간결하게 만드는 도우미입니다."),
        HumanMessage(content=f"다음 요약문들을 읽고 하나의 일관된 요약문으로 정리해주세요:\n\n{combined_summary}")
    ]
    response = llm(messages)
    return response.content

if __name__ == "__main__":
    main()
