import faiss
import numpy as np
import pickle
import os


class FAISSStore:

    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []
        self.embeddings = []

    def add(self, embeddings, texts, metadata):

        embeddings = np.array(embeddings).astype("float32")

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {self.index.d}"
            )

        self.index.add(embeddings)

        self.embeddings.extend(embeddings)  # ADD THIS

        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_vector, k=5):

        # ensure correct shape
        query_vector = np.array(query_vector).astype("float32").reshape(1, -1)

        # FAISS search
        distances, indices = self.index.search(query_vector, k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):

            if idx == -1:
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(dist)
            })

        return results

    def save(self, path):

        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/meta.pkl", "wb") as f:
            pickle.dump((self.texts, self.metadata, self.embeddings), f)

    def load(self, path):

        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/meta.pkl", "rb") as f:
            self.texts, self.metadata, self.embeddings = pickle.load(f)

        self.embeddings = np.array(self.embeddings).astype("float32")