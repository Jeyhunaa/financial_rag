"""Retrieval-only demo. No Gemini key or API request needed."""
import argparse
from rag import FinancialRAG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", default="dense", choices=["dense", "hybrid", "hybrid_balanced", "page_hybrid"])
    args = parser.parse_args()
    rag = FinancialRAG().load()
    for rank, item in enumerate(rag.retrieve(args.question, mode=args.mode), 1):
        print(f"{rank}. {item['doc']} · PDF p.{item['page'] + 1} [zero-index {item['page']}]")
        print(item['text'][:240].replace("\n", " "), "\n")

if __name__ == "__main__":
    main()
