import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INDEX_ROOT = ROOT / ".indices"
DATA_DIR = ROOT / "data"
INDEX_ROOT.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# LangSmith: tracing turns on automatically when a key exists
LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))
if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "pdf-rag-groq")
else:
    os.environ["LANGSMITH_TRACING"] = "false"


def missing_keys():
    """Return an error message if required keys are missing, else None."""
    if not os.getenv("GROQ_API_KEY"):
        return ("GROQ_API_KEY is missing. Create a .env file (copy .env.example) "
                "or pass it with docker --env-file .env.")
    return None


def check_keys():
    msg = missing_keys()
    if msg:
        raise SystemExit(msg)
