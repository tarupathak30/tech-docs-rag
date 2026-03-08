from vector_db.faiss_store import FAISSStore
from app.rag_pipeline import retrieve, rag_pipeline
import os


INDEX_PATH = "data/faiss_index"


def main():

    # ----------------------------
    # Load Vector Store
    # ----------------------------
    store = FAISSStore()

    if not os.path.exists(INDEX_PATH):
        print("FAISS index not found. Run ingestion first.")
        return

    store.load(INDEX_PATH)
    print("FAISS index loaded\n")


    # ----------------------------
    # Test Queries
    # ----------------------------
    queries = [
        "Why does FastAPI return 422 error?",
        "How does dependency injection work in FastAPI?",
        "How to run FastAPI with Uvicorn?",
        "What is request validation in FastAPI?"
    ]


    for q in queries:

        print("\n==============================")
        print("QUERY:", q)
        print("==============================\n")

        # ----------------------------
        # Retrieval
        # ----------------------------
        docs = retrieve(q)

        print("Top Retrieved Chunks:\n")

        for i, d in enumerate(docs):

            print(f"Result {i+1}")
            print("Score:", round(d["score"], 4))
            print("Source:", d["metadata"].get("source"))
            print("Text:", d["text"][:300])
            print()


        # ----------------------------
        # RAG Answer
        # ----------------------------
        result = rag_pipeline(q)

        print("\nGenerated Answer:\n")
        print(result["answer"])
        print("\n\n")


if __name__ == "__main__":
    main()