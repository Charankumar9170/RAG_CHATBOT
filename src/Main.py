import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm import generate_answer


# Configuration
INDEX_PATH = "vector_db/index.faiss"
CHUNKS_PATH = "vector_db/chunks.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Starting value only; calibrate using your own PDF.
THRESHOLD = 0.40


# 1. Load the saved FAISS index
index = faiss.read_index(INDEX_PATH)


# 2. Load the corresponding chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# 3. Validate index and chunks
if index.ntotal != len(chunks):
    raise ValueError(
        f"Index contains {index.ntotal} vectors, "
        f"but chunks.json contains {len(chunks)} chunks. "
        "Run Indexing.py again."
    )

if index.ntotal == 0:
    raise ValueError(
        "The vector database is empty. Run Indexing.py first."
    )


# 4. Load the same embedding model used for indexing
model = SentenceTransformer(MODEL_NAME)


# 5. Accept questions continuously
while True:

    query = input("\nEnter your query (or 'exit' to quit): ").strip()

    if query.lower() in {"exit", "quit"}:
        break

    if not query:
        print("Please enter a question.")
        continue

    # 6. Generate normalized query embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        [query_embedding],
        dtype=np.float32
    )

    # 7. Search for the top matching chunks
    k = min(TOP_K, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    # 8. Select relevant chunks
    relevant_chunks = []

    for score, chunk_index in zip(scores[0], indices[0]):

        # FAISS uses -1 when no result is available
        if chunk_index < 0:
            continue

        if score >= THRESHOLD:
            relevant_chunks.append(chunks[chunk_index])

    # 9. Stop if no chunk passes the threshold
    if not relevant_chunks:
        print("\nAnswer: Could not find")
        continue

    # # Optional: inspect retrieval results while testing
    # print("\nRetrieved chunks:", len(relevant_chunks))
    # for i, chunk in enumerate(relevant_chunks, start=1):
    #     print(f"\n--- Chunk {i} ---")
    #     print(chunk)

    # 10. Send the query and retrieved chunks to the LLM
    try:
        answer = generate_answer(query, relevant_chunks)
        print("\nAnswer:")
        print(answer)

    except Exception as error:
        print("\nLLM request failed:", error)
