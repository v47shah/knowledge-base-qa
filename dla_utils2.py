# dla_utils2.py

from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

# Extract text from Markdown
def extract_text_from_markdown(file):
    return file.read().decode("utf-8")

# Group similar sentences into semantic chunks
def group_semantic_chunks(sentences, embeddings, threshold=0.7, max_chunk_size=5):
    chunks, cur_chunk, cur_emb = [], [sentences[0]], [embeddings[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [np.mean(cur_emb, axis=0)])[0][0]
        if sim > threshold and len(cur_chunk) < max_chunk_size:
            cur_chunk.append(sentences[i])
            cur_emb.append(embeddings[i])
        else:
            chunks.append(" ".join(cur_chunk))
            cur_chunk, cur_emb = [sentences[i]], [embeddings[i]]
    if cur_chunk:
        chunks.append(" ".join(cur_chunk))
    return chunks

# Sentence → embedding → chunk → chunk embeddings
def preprocess_text_with_sources(file, raw_text, model):
    sentences = sent_tokenize(raw_text)
    embeddings = model.encode(sentences, convert_to_numpy=True)
    chunks = group_semantic_chunks(sentences, embeddings)
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
    sources = [file.name] * len(chunks)
    return chunks, chunk_embeddings, sources

# Given a query, retrieve most relevant chunks and similarity scores
def retrieve_relevant_chunks(query, model, chunk_embeddings, chunks, sources, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    top_sources = [sources[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    return top_chunks, top_sources, top_scores

# Create LLM-ready prompt
def build_prompt(query, chunks):
    context = "\n\n".join(chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
