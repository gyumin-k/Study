def summarize_text(text_chunks, llm, max_summary_length=2000):
    def process_chunk(chunk):
        text = chunk.page_content
        detected_languages = detect_langs(text)
        if any(lang.lang == "ko" and lang.prob > 0.5 for lang in detected_languages):
            system_prompt = "당신은 유능한 한국어 요약 도우미입니다."
            human_prompt = f"""
            다음 텍스트를 한국어로 요약해주세요:
            1. 핵심 주제를 3문장 이내로 요약.
            2. 중요한 키워드 강조.
            3. 중복 및 불필요한 정보 제거.
            텍스트:\n\n{text}
            """
        else:
            system_prompt = "You are an expert in summarizing English text."
            human_prompt = f"""
            Please summarize the following text:
            1. Summarize the main points in less than 3 sentences.
            2. Highlight important keywords.
            3. Remove redundant and irrelevant information.
            Text:\n\n{text}
            """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = llm(messages)
        return response.content

    # 병렬 처리
    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(process_chunk, text_chunks))
    
    # 후처리
    return refine_summary(summaries, llm)
