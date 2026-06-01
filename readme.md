# RAG Resume Builder

An AI-powered resume builder using Retrieval-Augmented Generation.

Enter your profile once. Paste any job description. Get a tailored CV instantly.

## Tech stack
- Embeddings: `sentence-transformers` (free, local)
- LLM: Groq + LLaMA 3.3 70B (free API)
- Vector DB: ChromaDB
- UI: Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your `GROQ_API_KEY` in a `.env` file or Streamlit Cloud secrets.