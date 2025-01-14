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

    if "uploaded_text" not in st.session_state:
        st.session_state.uploaded_text = None

    if "summary" not in st.session_state:
        st.session_state.summary = None

    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")
        show_metrics = st.checkbox("요약 성능 평가 표시", value=True)

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
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")  # GPT-4 유지

        # 텍스트 요약 생성
        st.session_state.summary = summarize_text(text_chunks, llm)

        # 자동 평가 지표 표시
        if show_metrics and st.session_state.summary:
            st.subheader("📊 요약 성능 평가")
            metrics = evaluate_summary(st.session_state.uploaded_text, st.session_state.summary)
            for metric, score in metrics.items():
                st.write(f"**{metric}:** {score:.2f}")

    if st.session_state.summary:
        st.subheader("📌 핵심 요약")
        st.markdown(st.session_state.summary)


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
        chunk_size=3000,  # 크기 증가로 처리량 감소
        chunk_overlap=200
    )
    return text_splitter.split_documents(text)


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

    with ThreadPoolExecutor(max_workers=4) as executor:  # 병렬 처리
        summaries = list(executor.map(process_chunk, text_chunks))
    
    combined_summary = "\n".join(summaries)
    return combined_summary[:max_summary_length] + "..." if len(combined_summary) > max_summary_length else combined_summary


# 자동 평가 지표 계산
def evaluate_summary(original_text, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 원본 텍스트 전체를 하나로 결합
    original_text_combined = "\n".join([doc.page_content for doc in original_text])
    
    # ROUGE 점수 계산
    scores = scorer.score(original_text_combined, generated_summary)
    metrics = {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }
    return metrics


if __name__ == "__main__":
    main()
