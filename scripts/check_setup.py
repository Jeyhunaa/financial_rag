"""Offline integrity/preflight checks; no network or Gemini calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT / "data"

def main():
    questions = DATA / "financebench_open_source.jsonl"
    if questions.exists():
        n = sum(bool(line.strip()) for line in questions.open(encoding="utf-8"))
        print(f"FinanceBench questions: {n}")
    else:
        print("FinanceBench questions missing; run python prepare.py")
    manifest = ART / "manifest.json"
    if not manifest.exists():
        print("FAISS index not prepared; run python prepare.py")
        return 1
    m = json.loads(manifest.read_text(encoding="utf-8"))
    chunks = ART / "chunks.json"
    index = ART / "index.faiss"
    if not chunks.is_file() or not index.is_file():
        print("Incomplete index artifacts; rerun python prepare.py")
        return 1
    count = len(json.loads(chunks.read_text(encoding="utf-8")))
    print(f"Indexed chunks: {count}; embedding model: {m.get('embedding_model')}")
    if count != m.get("chunks"):
        print("ERROR: manifest and chunks disagree")
        return 1
    print("Artifact preflight: PASS (metadata checks; not a full FAISS load)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
