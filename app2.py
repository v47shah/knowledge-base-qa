import streamlit as st
import openai
import os
from sentence_transformers import SentenceTransformer
from dla_utils2 import (
    extract_text_from_pdf,
    extract_text_from_markdown,
    preprocess_text_with_sources,
    retrieve_relevant_chunks,
    build_prompt
)
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("🏢 Enterprise Knowledge Base Search (PDF + Markdown)")
openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Upload multiple files
uploaded_files = st.file_uploader("📤 Upload PDF or Markdown files", type=["pdf", "md"], accept_multiple_files=True)

if uploaded_files:
    all_chunks, all_embeddings, all_sources = [], [], []

    with st.spinner("📚 Processing uploaded files..."):
        for f in uploaded_files:
            ext = os.path.splitext(f.name)[1].lower()
            if ext == ".pdf":
                text = extract_text_from_pdf(f)
            elif ext == ".md":
                text = extract_text_from_markdown(f)
            else:
                st.warning(f"Unsupported file type: {f.name}")
                continue

            chunks, embeddings, sources = preprocess_text_with_sources(f, text, model)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            all_sources.extend(sources)

    st.success("✅ All files processed. Ready for search!")

    query = st.text_input("❓ Ask a question about the uploaded documents")
    if query and openai.api_key:
        top_chunks, top_sources, top_scores = retrieve_relevant_chunks(
            query, model, np.array(all_embeddings), all_chunks, all_sources
        )

        similarity_threshold = 0.6
        if all(score < similarity_threshold for score in top_scores):
            st.markdown("### 🧠 Answer:")
            st.write("Not enough information in the documents to answer that question.")
        else:
            prompt = build_prompt(query, top_chunks)
            with st.spinner("💬 Generating answer..."):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown("### 🧠 Answer:")
                st.write(response['choices'][0]['message']['content'])

                st.markdown("#### 📁 Sources:")
                for chunk, source, score in zip(top_chunks, top_sources, top_scores):
                    st.markdown(f"- **{source}** (similarity: {score:.2f})")
