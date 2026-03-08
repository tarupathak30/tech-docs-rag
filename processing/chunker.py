import re

def chunk_text(text, chunk_size=350, overlap=80):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()

        if current_len + len(words) > chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words.copy()
            current_len = len(current_chunk)

        current_chunk.extend(words)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks