"""Streamlit UI:  streamlit run app.py"""
import html
import os
import uuid
from pathlib import Path

import streamlit as st
from rag import config as cfg
from rag.engine import (load_or_build_index, build_chain, ask, sources_summary,
                        file_sha256)

st.set_page_config(page_title="Provenance", page_icon="📄", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

:root {
  --bg: #131B2A;
  --panel: #1B2536;
  --line: #2C3A52;
  --ink: #E6EAF2;
  --muted: #93A1B8;
  --mark: #F2D45C;
  --mark-ink: #241E05;
}

/* type: sans for the interface, serif for reading */
.stApp, .stApp p, .stApp label, .stApp button, .stApp textarea, .stApp li {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

/* quiet the Streamlit chrome */
footer, #MainMenu, [data-testid="stDecoration"], .stDeployButton { display: none !important; }

div.block-container { max-width: 800px !important; padding-top: 2.6rem; padding-bottom: 7rem; }

/* header */
.app-title {
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 500;
  font-size: 2.7rem;
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--ink);
  margin: 0 0 .5rem;
}
.app-sub { color: var(--muted); font-size: 1rem; margin: 0 0 1.6rem; max-width: 34rem; line-height: 1.55; }
.app-sub b { color: var(--ink); font-weight: 500; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-size: 1.05rem; font-weight: 600; margin-bottom: .4rem; }
[data-testid="stFileUploaderDropzone"] { background: var(--bg); border: 1px dashed var(--line); border-radius: 8px; }

/* buttons: outlined, highlighter on hover */
.stButton > button {
  background: transparent;
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 8px;
  font-weight: 500;
  transition: border-color .15s, color .15s;
}
.stButton > button:hover { border-color: var(--mark); color: var(--mark); background: transparent; }
.stButton > button:focus-visible { outline: 2px solid var(--mark); outline-offset: 2px; }

/* conversation */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none; }
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; gap: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .75rem 1rem;
  margin: 1.4rem 0 .5rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1.14rem;
  line-height: 1.7;
}

/* page citations: the one highlighted element */
.cites { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; margin: .5rem 0 .2rem; }
.cites-label { color: var(--muted); font-size: .85rem; margin-right: .3rem; }
.chip {
  display: inline-block;
  background: var(--mark);
  color: var(--mark-ink);
  font-size: .78rem;
  font-weight: 600;
  padding: .1rem .6rem;
  border-radius: 999px;
}
.excerpt { border-left: 2px solid var(--line); padding: .2rem .9rem; margin: .6rem 0; color: var(--muted); font-size: .9rem; line-height: 1.55; }
.excerpt .chip { margin-right: .4rem; }
[data-testid="stExpander"] { border: none; background: transparent; }
[data-testid="stExpander"] summary { color: var(--muted); padding-left: 0; }

/* input */
[data-testid="stChatInput"] { border-radius: 10px; }

.hint { color: var(--muted); font-size: .9rem; margin: 1.2rem 0 .5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

err = cfg.missing_keys()
if err:
    st.error(err)
    st.stop()


@st.cache_resource(show_spinner=False)
def get_chain(pdf_path: str, sha: str):
    return build_chain(load_or_build_index(pdf_path))


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "model_not_found" in low or "does not exist" in low:
        return (f"The Groq model `{cfg.GROQ_MODEL}` is not available for your key. "
                "Set GROQ_MODEL in .env to another model, e.g. `openai/gpt-oss-20b`.")
    if "rate_limit" in low or "429" in low:
        return "Groq rate limit reached. Wait a minute and try again."
    if "invalid_api_key" in low or "401" in low:
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    return f"Something went wrong: {msg}"


def render_sources(sources):
    pages = sorted({s["page"] for s in sources})
    chips = "".join(f'<span class="chip">p. {p}</span>' for p in pages)
    st.markdown(f'<div class="cites"><span class="cites-label">Found on</span>{chips}</div>',
                unsafe_allow_html=True)
    with st.expander("See the passages"):
        for s in sources:
            st.markdown(
                f'<div class="excerpt"><span class="chip">p. {s["page"]}</span>'
                f'{html.escape(s["snippet"])}…</div>',
                unsafe_allow_html=True,
            )


st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("current_pdf", None)
st.session_state.setdefault("ready", set())

with st.sidebar:
    st.markdown("### Your document")
    up = st.file_uploader("Upload a PDF", type="pdf")
    existing = sorted(p.name for p in cfg.DATA_DIR.glob("*.pdf"))
    choice = st.selectbox("Or open a saved one", ["—"] + existing)

    pdf_path = None
    if up is not None:
        pdf_path = cfg.DATA_DIR / Path(up.name).name
        buf = up.getbuffer()
        if not pdf_path.exists() or pdf_path.stat().st_size != len(buf):
            pdf_path.write_bytes(buf)
    elif choice != "—":
        pdf_path = cfg.DATA_DIR / choice

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<h1 class="app-title">Provenance</h1>', unsafe_allow_html=True)

if pdf_path is None:
    st.markdown(
        '<p class="app-sub">Ask questions about a PDF and see where every answer came from. '
        'Answers use only your document and point to the pages they were found on.</p>'
        '<p class="app-sub">Upload a PDF in the sidebar, or open one you saved earlier.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

pdf_path = str(pdf_path)
sha = file_sha256(pdf_path)
key = f"{pdf_path}:{sha}"

st.markdown(f'<p class="app-sub">Reading <b>{html.escape(os.path.basename(pdf_path))}</b></p>',
            unsafe_allow_html=True)

try:
    if key in st.session_state.ready:
        chain = get_chain(pdf_path, sha)
    else:
        with st.status("Preparing your PDF… the first time can take a few minutes "
                       "(scanned pages need OCR).", expanded=False) as status:
            chain = get_chain(pdf_path, sha)
            status.update(label="PDF ready", state="complete")
        st.session_state.ready.add(key)
except ValueError as e:                      # readable problems (no text, OCR missing)
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not process this PDF: {e}")
    st.stop()

# new document -> fresh conversation
if st.session_state.current_pdf != pdf_path:
    st.session_state.current_pdf = pdf_path
    st.session_state.messages = []

q = st.chat_input("Ask a question about your document")
if not q:
    q = st.session_state.pop("pending", None)

# starter questions on an empty conversation
if not st.session_state.messages and not q:
    st.markdown('<p class="hint">Not sure where to start?</p>', unsafe_allow_html=True)
    for text in ["What is this document about?",
                 "List the main topics it covers",
                 "Summarize it in five points"]:
        if st.button(text, key=f"starter-{text}"):
            st.session_state.pending = text
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m["sources"])

if q:
    with st.chat_message("user"):
        st.markdown(q)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Reading the document…"):
                result, _ = ask(chain, q, session_id=st.session_state.session_id,
                                pdf_name=os.path.basename(pdf_path))
            st.markdown(result["answer"])
    except Exception as e:
        st.error(friendly_error(e))
        st.stop()
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources_summary(result["docs"]),
    })
    st.rerun()