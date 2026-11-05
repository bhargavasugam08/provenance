# Provenance

**Ask questions about your PDFs and see where the answers came from.** Provenance is a document question-answering app built with retrieval-augmented generation (RAG). It extracts text from PDFs, uses OCR for scanned pages, retrieves relevant passages, and answers using the document context with page citations.

## Features

- **PDF chat:** upload a PDF in the Streamlit app or select a PDF saved in `data/`.
- **Page-grounded answers:** responses are generated from retrieved document passages and include page references. The app also shows short source excerpts.
- **Scanned-PDF support:** pages with little or no extractable text are rendered with PyMuPDF and processed with RapidOCR and OpenCV.
- **Local semantic search:** document passages are embedded with a Sentence Transformers model and indexed in FAISS.
- **Index reuse:** indexes are cached using the PDF content hash and indexing settings, so an unchanged document can be reused.
- **Optional tracing:** LangSmith tracing is enabled when `LANGSMITH_API_KEY` is configured.
- **Two interfaces:** use the Streamlit web app or the command-line chat client.

## How it works

1. `rag.engine.load_pdf` extracts a page at a time with LangChain's `PyPDFLoader`.
2. Pages with fewer than 20 extracted characters are sent to `rag.ocr.ocr_pages`, which renders them with PyMuPDF and runs RapidOCR. OCR results are added to the corresponding pages.
3. `RecursiveCharacterTextSplitter` divides the text into overlapping chunks. Sentence Transformers creates local embeddings, and FAISS stores them in a content- and settings-based cache under `.indices/`.
4. A question retrieves the most relevant chunks. The Groq chat model answers from those passages, and the app displays page numbers and source excerpts.

The embedding model runs locally; question answering uses the Groq API. If LangSmith tracing is enabled, traces are sent to the configured LangSmith project. Do not use this app with documents you are not permitted to process or send to these services.

## Requirements

- Python 3.10–3.12 for a local environment; the Docker image uses Python 3.11.
- A Groq API key. The default model is `openai/gpt-oss-120b`; availability depends on your Groq account.
- Internet access for the Groq API and, on first use, downloading the embedding model and dependencies.
- Docker and Docker Compose for the containerized option.

## Quick start: local Python

Clone the repository and move into the project directory, then create and activate a virtual environment:

```bash
git clone https://github.com/bhargavasugam08/provenance.git
cd provenance
python -m venv .venv
```

Activate it and install dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file from the example and add your Groq key:

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`. Start the web app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (normally <http://localhost:8501>), then upload a PDF in the sidebar or choose one already in `data/`.

## Configuration

Set values in `.env` or the deployment environment. `GROQ_API_KEY` is required. The remaining settings are optional.

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required API key for Groq chat completions. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model used to generate answers. |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face/Sentence Transformers model used for local embeddings. |
| `CHUNK_SIZE` | `1000` | Maximum chunk size used when splitting extracted text. |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent text chunks. |
| `TOP_K` | `4` | Number of matching chunks retrieved for a question. |
| `LANGSMITH_API_KEY` | unset | Optional key that enables LangSmith tracing. |
| `LANGSMITH_TRACING` | set by the app | Tracing is enabled when a LangSmith key is present and disabled otherwise. |
| `LANGSMITH_PROJECT` | `pdf-rag-groq` | LangSmith project name when tracing is enabled. |

Never commit `.env` or expose API keys in source code. For hosted deployments, configure secrets through the platform's environment/secrets settings.

## Command-line interface

The CLI accepts a PDF path, prepares or reuses its index, and starts an interactive question-and-answer session:

```bash
python cli.py data/yourfile.pdf
```

Type `exit` or `quit` to end the session. Answers include the pages used as sources.

## Run with Docker Compose

Create `.env` as described above, then build and start the service:

```bash
docker compose up --build
```

Visit <http://localhost:8501>. Compose mounts `data/` and `.indices/` into the container so PDFs and generated indexes persist across container restarts. Stop the service with `Ctrl+C`; use `docker compose down` to remove the running container.

To build and run the image directly instead:

```bash
docker build -t provenance .
docker run --rm -p 8501:8501 --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.indices:/app/.indices" \
  provenance
```

The Dockerfile installs CPU-only PyTorch and pre-downloads the default embedding model during image build. Rebuild the image after changing dependencies in `requirements.txt`.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub without committing `.env`.
2. Create an app from the repository with `app.py` as the entry point.
3. Add `GROQ_API_KEY` in the app's **Settings → Secrets**. Add `LANGSMITH_API_KEY` only if you want tracing.
4. Ensure the deployment installs packages from `requirements.txt`. That file includes `pymupdf`, `rapidocr-onnxruntime`, and `opencv-python-headless` for scanned-PDF OCR.

For any platform that builds from a Dockerfile, rebuild/redeploy the image after dependency changes so the updated packages are installed in the runtime.

## Troubleshooting

### Scanned PDF reports `Missing module: cv2`

`cv2` is provided by the `opencv-python-headless` package, which is listed in `requirements.txt`. Redeploy or rebuild the runtime so it installs the updated requirements. For Docker, use `docker compose up --build`. You can check the imports in the active environment with:

```bash
python -c "import cv2, pymupdf; from rapidocr_onnxruntime import RapidOCR; print('OCR dependencies imported successfully')"
```

If the error remains, inspect deployment logs for the complete `OCR initialization failed` traceback and confirm the service is running the newly built environment.

### `GROQ_API_KEY` is missing

Set `GROQ_API_KEY` in your local `.env` or in the deployment platform's secret settings, then restart or redeploy the app.

### Groq model unavailable or rate limited

Check that the configured `GROQ_MODEL` is available to your API key. If you receive a rate-limit error, wait and retry or review your Groq account limits.

### First run is slow

The app downloads the embedding model on first use in a local Python environment. Docker images pre-download the default model at build time. OCR and indexing a large PDF can also take several minutes.

### OCR produces little or no text

OCR is best suited to clear, printed scans. Handwriting, low-resolution pages, unusual layouts, and diagrams may not be recognized accurately. Password-protected or blank PDFs may not produce readable text.

## Project structure

```text
.
├── app.py                 Streamlit web interface
├── cli.py                 Interactive command-line client
├── rag/
│   ├── config.py          Environment settings, directories, and key checks
│   ├── engine.py          PDF ingestion, chunking, embeddings, FAISS, retrieval, and answers
│   └── ocr.py             OCR fallback for scanned pages
├── data/                  Local PDF storage (created/used by the app)
├── .indices/              Cached FAISS indexes (created as needed)
├── requirements.txt       Python dependencies
├── Dockerfile             Container build and Streamlit startup
└── docker-compose.yml     Container service and persistent mounts
```

## Limitations

- Answer quality depends on PDF text quality, OCR quality, chunking, retrieval, and the selected Groq model. Verify important claims against the cited pages.
- The application provides document-grounded answers, but citations are retrieved source passages rather than a guarantee that every generated statement is correct.
- Local FAISS indexes and PDFs are stored on disk; deployments with ephemeral storage may lose them when restarted unless persistent storage is configured.
- The default prompt directs the model to answer only from retrieved document context. It may say it does not know when relevant information is not retrieved or is absent from the PDF.
