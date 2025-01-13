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
            st.warning("시험 날짜는 오늘보다 이후여야

