from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.rag_pipeline import rag_pipeline

app = FastAPI(
    title="Tech Docs RAG API",
    description="Retrieval-Augmented QA for technical documentation",
    version="1.0"
)


class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    text: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieval_score: float


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):

    result = rag_pipeline(req.query)

    sources = [
        {
            "text": s["text"],
            "source": s["metadata"]["source"]
        }
        for s in result["sources"]
    ]

    return {
        "answer": result["answer"],
        "sources": sources,
        "retrieval_score": result["retrieval_score"]
    }