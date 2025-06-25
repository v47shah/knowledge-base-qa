# knowledge-base-qa
 A Streamlit app that enables question-answering over PDF and Markdown documents using semantic chunking, vector search, and GPT-based responses. Ideal for building enterprise-ready document search and internal knowledge assistants. It also prevents hallucinations using a threshold cosine similarity check.

Built with:
- 🔍 `sentence-transformers` for semantic search
- 💬 OpenAI GPT (via `gpt-3.5-turbo`) for answers
- 🖥️ Streamlit for a friendly UI

---

## Features

- Upload **PDF** or **Markdown** files
- Preprocess content into semantically meaningful chunks
- Embed chunks using `all-MiniLM-L6-v2`
- Retrieve the top relevant sections using cosine similarity
- Query using natural language and receive GPT-generated answers
- No SSO or database required — works locally with file uploads

---

## Use Cases

- Company handbooks or HR policies
- Investment or legal guides
- Internal product documentation
- AI-powered chatbot on static documents

---

## 📂 Project Structure

```bash
├── app.py               # Streamlit frontend
├── dla_utils.py         # Backend logic (text extraction, chunking, etc.)
├── requirements.txt     # Dependency list
└── README.md            # This file

🛠 Installation
1) Clone the repo:
git clone https://github.com/v47shah/enterprise-doc-qa.git
cd knowledge-base-qa

2) Install dependencies:
pip install -r requirements.txt

3) Run the app:
streamlit run app2.py

4) Enter your OpenAI API key when prompted.

**How to Evaluate**
# Upload one or more documents (.pdf or .md)

# Wait for them to be processed and embedded.

# Type in your question (e.g., "What are the rules for investment in Australia?")

# Read the extracted context and GPT-generated answer.

# Observe the fallback behavior when similarity is too low. (Prevents hallucinations)

