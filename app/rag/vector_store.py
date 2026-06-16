import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class EmotionVectorStore:
    COLLECTION_NAME = "emotion_support_kb"

    def __init__(self, persist_directory="./data/vector_store"):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded")

        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector DB ready. Documents: {self.collection.count()}")

    def reset_collection(self):
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector DB collection reset")

    def add_documents(self, texts, metadatas=None, ids=None):
        if not texts:
            return

        print(f"Embedding {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids,
        )
        print(f"Added {len(texts)} documents")

    def search(self, query, n_results=3):
        query_embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
        )
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        return documents, distances, metadatas

    def build_from_text_files(self, directory_path):
        all_texts = []
        all_metadatas = []

        for file_path in sorted(Path(directory_path).glob("*.txt")):
            print(f"Reading: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._chunk_text(content, chunk_size=160)
            for chunk_index, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                all_texts.append(chunk)
                all_metadatas.append(
                    {"source": file_path.name, "chunk": chunk_index}
                )

        if all_texts:
            self.add_documents(all_texts, all_metadatas)
            print(f"Built vector store: {len(all_texts)} chunks")
        else:
            print("No text files found")

    def _chunk_text(self, text, chunk_size=160):
        words = text.split()
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
