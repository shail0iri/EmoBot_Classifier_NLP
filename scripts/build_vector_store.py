import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from rag.vector_store import EmotionVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Build or rebuild the EmoBot RAG Chroma vector store."
    )
    parser.add_argument(
        "--knowledge-base",
        default=str(PROJECT_ROOT / "knowledge_base"),
        help="Folder containing .txt knowledge-base files.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(PROJECT_ROOT / "data" / "vector_store"),
        help="Folder where ChromaDB should persist embeddings.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before rebuilding it.",
    )
    args = parser.parse_args()

    kb_path = Path(args.knowledge_base)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge-base folder not found: {kb_path}")

    store = EmotionVectorStore(persist_directory=args.persist_dir)
    if args.reset:
        store.reset_collection()

    before = store.collection.count()
    store.build_from_text_files(str(kb_path))
    after = store.collection.count()

    print(f"Vector store documents before: {before}")
    print(f"Vector store documents after: {after}")
    print(f"Knowledge base used: {kb_path}")


if __name__ == "__main__":
    main()
