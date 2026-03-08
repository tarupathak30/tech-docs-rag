from langchain_groq import ChatGroq
from processing.embedder import embed_texts
from vector_db.faiss_store import FAISSStore
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import rerank
from utils.web_search import fetch_web_docs
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_PATH = "data/faiss_index"

store = FAISSStore()

if os.path.exists(INDEX_PATH):
    store.load(INDEX_PATH)
else:
    raise ValueError("FAISS index not found. Run ingestion first.")


# initialize hybrid search ONCE
hybrid = HybridSearch(
    texts=store.texts,
    embeddings=store.embeddings,
    embed_fn=lambda q: embed_texts([q])[0]
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


def retrieve(query, k=5):

    # hybrid retrieval
    results = hybrid.search(query, k=20)

    docs = []

    for r in results:
        docs.append({
            "text": r["text"],
            "metadata": store.metadata[r["index"]],
            "score": r["hybrid_score"]
        })

    # rerank
    docs = rerank(query, docs, top_k=k)

    return docs


def build_prompt(query, docs):

    if not docs:
        return None

    context = "\n\n".join(
        f"[Source {i+1}] {d['text']}" for i, d in enumerate(docs)
    )

    return f"""
You are a technical documentation assistant.

Use ONLY the provided context.

If the answer is not present say:
"I don't know based on the documentation."

Context:
{context}

Question:
{query}
"""



def generate_answer(prompt):

    if prompt is None:
        return "No relevant documentation found."

    response = llm.invoke([
        {"role": "user", "content": prompt}
    ])

    return response.content



def rag_pipeline(query, threshold=0.35):

    docs = retrieve(query)

    scores = [d["score"] for d in docs]
    best_score = max(scores) if scores else 0

    print(f"Best hybrid score: {best_score}")

    if best_score < threshold:

        print("Low confidence → triggering web search")

        docs = fetch_web_docs(query, store)

        store.save("data/faiss_index")

    # remove duplicate sources
    unique = {}

    for d in docs:
        src = d["metadata"]["source"]

        if src not in unique:
            unique[src] = d

    docs = list(unique.values())

    prompt = build_prompt(query, docs)

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": docs,
        "retrieval_score": best_score
    }
    
    
    
if __name__ == "__main__":

    # query = "What is request validation in FastAPI?"

    result = rag_pipeline(query)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for s in result["sources"]:
        print("-", s["metadata"]["source"])