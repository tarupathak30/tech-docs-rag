from ddgs import DDGS
from ingestion.scraper import scrape_page
from ingestion.cleaner import clean_text
from processing.chunker import chunk_text
from processing.embedder import embed_texts


def fetch_web_docs(query, store, max_results=3):

    docs = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)

        for r in results:

            url = r["href"]

            page = scrape_page(url)

            if not page:
                continue

            text = clean_text(page["text"])

            chunks = chunk_text(text)

            embeddings = embed_texts(chunks)

            metadata = [{"source": url}] * len(chunks)

            store.add(embeddings, chunks, metadata)

            for chunk in chunks[:2]:
                docs.append({
                    "text": chunk,
                    "metadata": {"source": url}
                })

    print("FAISS index size:", len(store.texts))
    store.save("data/faiss_index")
    print("FAISS index size:", store.index.ntotal)
    return docs