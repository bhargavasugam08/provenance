"""CLI chat:  python cli.py data/yourfile.pdf"""
import sys
import uuid
from rag import config as cfg
from rag.engine import load_or_build_index, build_chain, ask, sources_summary, flush_traces


def main():
    cfg.check_keys()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python cli.py path/to/file.pdf")
    pdf = sys.argv[1]

    print("Preparing index (first run embeds the PDF, later runs load from cache)...")
    try:
        vs = load_or_build_index(pdf)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")
    chain = build_chain(vs)
    session = uuid.uuid4().hex[:8]

    print("Ask questions. Type 'exit' to quit.\n")
    try:
        while True:
            q = input("Q: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            try:
                result, _ = ask(chain, q, session_id=session, pdf_name=pdf)
            except Exception as e:
                print(f"\nError: {e}\n")
                continue
            print("\nA:", result["answer"])
            pages = sorted({s["page"] for s in sources_summary(result["docs"])})
            print(f"   sources: pages {pages}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        flush_traces()


if __name__ == "__main__":
    main()
