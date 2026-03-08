from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv() 
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )