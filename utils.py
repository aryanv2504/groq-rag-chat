import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def split_text(text, chunk_size=500, overlap=200):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return model, embeddings, chunks


def create_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

