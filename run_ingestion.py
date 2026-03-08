from ingestion.scraper import scrape_page
from ingestion.cleaner import clean_text
from processing.chunker import chunk_text
from processing.embedder import embed_texts
from vector_db.faiss_store import FAISSStore

urls = [
    "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    "https://huggingface.co/docs/transformers/index"
]

store = FAISSStore()

for url in urls:

    page = scrape_page(url)

    clean = clean_text(page["text"])

    chunks = chunk_text(clean)

    embeddings = embed_texts(chunks)

    metadata = [{"source": url}] * len(chunks)

    store.add(embeddings, chunks, metadata)

store.save("data/faiss_index")