from fastapi import FastAPI
from app.rag_pipeline import retrieve, build_prompt

app = FastAPI()

@app.post("/ask")
def ask(question: str):

    docs = retrieve(question)

    prompt = build_prompt(question, docs)

    return {
        "question": question,
        "context": docs,
        "prompt_for_llm": prompt
    }