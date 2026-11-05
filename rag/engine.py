"""PDF RAG engine: Groq LLM + local HF embeddings + FAISS + LangSmith tracing."""
import os
import json
import uuid
import hashlib
from pathlib import Path

from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from . import config as cfg
from .ocr import ocr_pages


# ---------------- embeddings (loaded once) ----------------
_EMB_CACHE = {}

def get_embeddings(model_name: str = cfg.EMBED_MODEL):
    if model_name not in _EMB_CACHE:
        _EMB_CACHE[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMB_CACHE[model_name]


# ---------------- traced ingest steps ----------------
MIN_PAGE_CHARS = 20


@traceable(name="load_pdf", tags=["ingest"])
def load_pdf(path: str):
    docs = PyPDFLoader(path).load()          # one Document per page
    empty = [i for i, d in enumerate(docs) if len(d.page_content.strip()) < MIN_PAGE_CHARS]
    if empty:                                 # scanned pages -> OCR fallback
        for i, text in ocr_pages(path, empty).items():
            docs[i].page_content = text
            docs[i].metadata["ocr"] = True
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError(
            "No readable text was found in this PDF, even after OCR. "
            "It may be blank, password-protected, or contain only handwriting/diagrams."
        )
    return docs

@traceable(name="split_documents", tags=["ingest"])
def split_documents(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

@traceable(name="embed_and_index", tags=["ingest"])
def build_vectorstore(splits, embed_model_name):
    return FAISS.from_documents(splits, get_embeddings(embed_model_name))


# ---------------- cache key (content hash, ignores mtime) ----------------
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name) -> str:
    meta = {
        "sha256": file_sha256(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v3",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


# ---------------- traced load / build ----------------
@traceable(name="load_index", tags=["index", "cache_hit"])
def load_index_run(index_dir: Path, embed_model_name: str):
    return FAISS.load_local(
        str(index_dir), get_embeddings(embed_model_name),
        allow_dangerous_deserialization=True,  # safe: we only load indexes we built ourselves
    )

@traceable(name="build_index", tags=["index", "cache_miss"])
def build_index_run(pdf_path, index_dir: Path, chunk_size, chunk_overlap, embed_model_name):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap)
    if not splits:
        raise ValueError("The PDF produced no text chunks to index.")
    vs = build_vectorstore(splits, embed_model_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf": os.path.basename(pdf_path),
        "pages": len(docs),
        "ocr_pages": sum(1 for d in docs if d.metadata.get("ocr")),
        "chunks": len(splits),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

@traceable(name="setup_index", tags=["setup"])
def load_or_build_index(pdf_path, chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP,
                        embed_model_name=cfg.EMBED_MODEL, force_rebuild=False):
    key = index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = cfg.INDEX_ROOT / key
    if index_dir.exists() and not force_rebuild:
        return load_index_run(index_dir, embed_model_name)
    return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)


# ---------------- chain ----------------
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a PDF using ONLY the provided context.\n"
     "Rules:\n"
     "1. If the context answers the question, answer it and cite page numbers like (p. 12).\n"
     "2. If the context does not cover the topic asked about but does cover related topics, "
     "say clearly that the exact topic is not in the document, then briefly list the related "
     "topics it does cover (with page numbers) and offer to explain those. "
     "Do not use outside knowledge to explain the missing topic.\n"
     "3. If nothing in the context is related at all, say you don't know and suggest checking "
     "that the right PDF is selected."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(f"[p. {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs)

def build_chain(vectorstore, k: int = cfg.TOP_K):
    llm = ChatGroq(model=cfg.GROQ_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    answer_chain = (
        (lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | PROMPT | llm | StrOutputParser()
    )
    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(answer=answer_chain)
    )


def ask(chain, question: str, session_id: str = "cli", pdf_name: str = ""):
    """Run one query. Returns (result_dict, run_id)."""
    run_id = uuid.uuid4()
    result = chain.invoke(
        question,
        config={
            "run_id": run_id,
            "run_name": "pdf_rag_query",
            "tags": ["qa", "groq"],
            "metadata": {
                "session_id": session_id,
                "pdf": pdf_name,
                "llm": cfg.GROQ_MODEL,
                "embeddings": cfg.EMBED_MODEL,
                "k": cfg.TOP_K,
            },
        },
    )
    return result, run_id


def sources_summary(docs):
    out = []
    for d in docs:
        page = d.metadata.get("page", 0) + 1
        snippet = " ".join(d.page_content.split())[:200]
        out.append({"page": page, "snippet": snippet})
    return out


def flush_traces():
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass
