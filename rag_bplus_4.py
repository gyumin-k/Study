import tempfile
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datetime import datetime
from rouge_score import rouge_scorer  # ROUGE 평가 라이브러리

# Streamlit 페이지 설정 (가장 첫 번째 명령어)
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
        # 파일 업로드 및 OpenAI API 키 입력
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")

        # 각 기능 활성화 여부 선택
        create_summary = st.checkbox("핵심 요약 생성", value=True)
        create_roadmap = st.checkbox("공부 로드맵 생성", value=True)
        create_quiz = st.checkbox("예상 문제 생성", value=True)
        show_metrics = st.checkbox("요약 성능 평가 표시", value=True)

    if process_button:
        # 필수 입력 확인
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

        # 파일에서 텍스트 추출
        st.session_state.uploaded_text = extract_text_from_files(uploaded_files)

        # 텍스트 청크 분할 및 LLM 초기화
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")  # GPT-4 유지

        # 기능별 처리
        if create_summary:
            st.session_state.summary = summarize_text(text_chunks, llm)

        if create_roadmap and st.session_state.summary:
            st.session_state.roadmap = create_study_roadmap(st.session_state.summary, llm, days_left)

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
            st.warning(f"지원하지 않는 파일 형식입니다: {file.name}")
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)

        # 임시 파일 삭제
        os.remove(temp_file_path)
    return doc_list


# 텍스트 청크로 분할
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(text)


# 텍스트 요약
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
            Please summarize the following text in English:
            1. Include only the main points
            2. Highlight important keywords
            3. Remove unnecessary details and redundancy
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
    
    # 병렬 처리된 요약문 결합 및 후처리
    combined_summary = "\n".join(summaries)
    messages = [
        SystemMessage(content="당신은 요약문을 정리하는 도우미입니다."),
        HumanMessage(content=f"다음 요약문을 간결하고 일관되게 정리해주세요:\n\n{combined_summary}")
    ]
    response = llm(messages)
    return response.content[:max_summary_length] + "..." if len(response.content) > max_summary_length else response.content



# 공부 로드맵 생성
def create_study_roadmap(summary, llm, days_left):
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 유능한 공부 로드맵 작성 도우미입니다."),
        HumanMessage(content=f"다음 텍스트를 기반으로 {days_left}일 동안 효과적으로 공부할 계획을 작성해주세요:\n\n{summary}")
    ]
    response = llm(messages)
    return response.content


# 예상 문제 생성
def generate_quiz_questions(summary, llm):
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 예상 문제 작성 도우미입니다."),
        HumanMessage(content=f"""
        다음 텍스트를 기반으로 중요도를 표시한 10개 이상의 예상 문제를 작성해주세요:
        {summary}
        - 예상 문제는 명확하고 구체적으로 작성해주세요.
        - 각 문제에는 중요도를 '높음', '중간', '낮음'으로 표시해주세요.
        """)
    ]
    response = llm(messages)
    return response.content


# 요약 성능 평가
def evaluate_summary(original_text, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    original_text_combined = "\n".join([doc.page_content for doc in original_text])
    scores = scorer.score(original_text_combined, generated_summary)
    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }


if __name__ == "__main__":
    main()
