from pathlib import Path
import json

import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Project paths: run this script from the project root
PDF_PATH = Path("documents/RAG_Release1_Pipeline_Python.pdf")
VECTOR_DIR = Path("vector_db")
CHUNKS_PATH = VECTOR_DIR / "chunks.json"
INDEX_PATH = VECTOR_DIR / "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"


# 1. Read PDF
if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

reader = PdfReader(str(PDF_PATH))
print("Number of pages:", len(reader.pages))

text_parts = []

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()

    if page_text and page_text.strip():
        text_parts.append(page_text)

text = "\n\n".join(text_parts)

if not text.strip():
    raise ValueError(
        "No extractable text found. "
        "The PDF may require OCR."
    )


# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)

chunks = splitter.split_text(text)

if not chunks:
    raise ValueError("No chunks were generated.")

print("Number of chunks:", len(chunks))


# 3. Generate normalized embeddings
model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    convert_to_numpy=True
)

embeddings = np.asarray(embeddings, dtype=np.float32)

print("Embedding shape:", embeddings.shape)


# 4. Create FAISS index
dimension = embeddings.shape[1]

# Inner product on normalized vectors = cosine similarity
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


# 5. Save index and corresponding chunks
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(INDEX_PATH))

with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)


print("Indexing completed successfully.")
print("Vectors stored:", index.ntotal)
print("Index saved to:", INDEX_PATH)
print("Chunks saved to:", CHUNKS_PATH)
