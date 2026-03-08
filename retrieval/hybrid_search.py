from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:

    def __init__(self, texts, embeddings, embed_fn):

        self.texts = texts
        self.embeddings = embeddings
        self.embed_fn = embed_fn

        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=5, alpha=0.6):

        # vector similarity
        query_vec = self.embed_fn(query)
        vector_scores = np.dot(self.embeddings, query_vec)

        # bm25 keyword scoring
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)

        # normalize scores
        vector_scores = vector_scores / (np.max(vector_scores) + 1e-9)
        bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-9)

        # hybrid fusion
        hybrid_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

        ranked = np.argsort(hybrid_scores)[::-1][:k]

        results = []

        for i in ranked:
            results.append({
                "text": self.texts[i],
                "hybrid_score": float(hybrid_scores[i]),
                "vector_score": float(vector_scores[i]),
                "bm25_score": float(bm25_scores[i]),
                "index": i
            })

        return results