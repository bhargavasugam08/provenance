# PDF RAG with Groq + LangSmith

Chat with any PDF. Answers come only from the document, with page citations.
Scanned (image-only) pages are read automatically with OCR.

- LLM: Groq (`openai/gpt-oss-120b`, change with `GROQ_MODEL`)
- Embeddings: local `all-MiniLM-L6-v2` (Groq has no embedding API)
- Vector store: FAISS, cached by PDF content hash + settings
- LangSmith: traces for ingest and every question

No PDF is bundled. Upload one in the app, or drop files into `data/`.

## 1. Set up keys
```bash
cp .env.example .env      # Windows: copy .env.example .env
# edit .env and add GROQ_API_KEY (and LANGSMITH_API_KEY)
```

## 2a. Run with a venv (Python 3.10-3.12)
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Terminal chat instead: `python cli.py data/yourfile.pdf`

## 2b. Run with Docker
```bash
docker compose up --build
```
or without compose:
```bash
docker build -t pdf-rag .
docker run -p 8501:8501 --env-file .env -v "$(pwd)/data:/app/data" -v "$(pwd)/.indices:/app/.indices" pdf-rag
```
Open http://localhost:8501 . Keys are passed at run time and are never baked into the image.

## Notes
- First run downloads the embedding model (already baked into the Docker image).
- OCR works well on printed scans. Handwriting and diagrams are read poorly.
- Streamlit Community Cloud: push without `.env`, add keys under Settings -> Secrets.

## Layout
```
app.py              Streamlit UI
cli.py              terminal chat (optional)
rag/config.py       settings and key checks
rag/engine.py       ingest, cache, chain, LangSmith tracing
rag/ocr.py          OCR fallback for scanned pages
Dockerfile, docker-compose.yml, .dockerignore
data/               put PDFs here (empty by default)
```
