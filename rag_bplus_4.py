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

        # 체크박스 설정
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

        # 시험까지 남은 기간 계산
        today = datetime.now().date()
        days_left = (exam_date - today).days
        if days_left <= 0:
            st.warning("시험 날짜는 오늘 이후여야 합니다!")
            return

        # 파일에서 텍스트 추출
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
        chunk_size=1500,  # 청크 크기 조정
        chunk_overlap=500  # 중복 설정
    )
    return text_splitter.split_documents(text)

# 벡터 저장소 생성 (OpenAI 임베딩 사용)
def create_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(text_chunks, embeddings)

# 텍스트 요약 (프롬프트 개선 및 병렬 처리)
def summarize_text(text_chunks, llm, max_summary_length=2000):
    def process_chunk(chunk):
        text = chunk.page_content
        messages = [
            SystemMessage(content="당신은 한국 대학생의 시험 준비를 돕는 요약 전문가입니다."),
            HumanMessage(content=f"""
            다음 텍스트를 학교 시험 준비를 위해 요약해주세요:
            1. 핵심 개념, 정의, 주요 이론 및 차이점을 간결하게 포함하세요.
            2. 불필요한 배경 정보는 생략하고 시험에 나올 가능성이 높은 내용을 강조하세요.
            3. 간단한 문장으로 요약하고, 이해하기 쉽도록 정리하세요.

            텍스트:
            {text}
            """)
        ]
        response = llm(messages)
        return response.content

    # 병렬 처리로 청크별 요약 생성
    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(process_chunk, text_chunks))

    # 요약 결과 결합
    combined_summary = "\n".join(summaries)
    return combined_summary[:max_summary_length] + "..." if len(combined_summary) > max_summary_length else combined_summary

# 공부 로드맵 생성
def create_study_roadmap(summary, llm, days_left):
    messages = [
        SystemMessage(content="당신은 대학생이 시험 준비를 잘 할 수 있도록 도움을 주는 로드맵 전문가입니다."),
        HumanMessage(content=f"""
        다음 요약문을 바탕으로 {days_left}일 동안 효과적으로 공부할 수 있는 로드맵을 작성해주세요:
        1. 남은 일수를 고려하여 매일 공부할 주제와 분량을 제안해주세요.
        2. 중요한 개념, 이론, 차이점을 학습하는 데 우선순위를 두세요.
        3. 복습과 예상 문제 풀이를 포함한 효율적인 학습 계획을 제시하세요.

        요약문:
        {summary}
        """)
    ]
    response = llm(messages)
    return response.content

# 예상 문제 생성
def generate_quiz_questions(summary, llm):
    messages = [
        SystemMessage(content="당신은 예상 문제를 작성하는 전문가입니다."),
        HumanMessage(content=f"""
        다음 요약문을 바탕으로 학교 시험에 나올 가능성이 높은 10개 이상의 예상 문제를 작성해주세요:
        1. 시험에서 중요도가 높은 개념과 이론에 기반한 문제를 작성하세요.
        2. 문제마다 중요도를 '높음', '중간', '낮음'으로 표시하세요.
        3. 명확하고 구체적인 질문을 작성하세요.

        요약문:
        {summary}
        """)
    ]
    response = llm(messages)
    return response.content

if __name__ == "__main__":
    main()
