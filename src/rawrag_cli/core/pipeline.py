from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import requests

from .chunker import chunk_text
from .config import DB_PATH, OLLAMA_URL, chunk_overlap, chunk_size
from .database import count_chunks, get_chunks, initialize_database, insert_chunk
from .embeddings import get_embedding
from .vector import VectorMath


class IndexNotFoundError(RuntimeError):
    """Raised when a query is attempted before an index has been built."""


class IndexConsistencyError(RuntimeError):
    """Raised when the SQLite and vector indexes do not match."""


class NoRelevantChunksError(RuntimeError):
    """Raised when a query does not retrieve any indexed chunks."""


def index_directory(directory: str | Path = ".") -> int:
    """Rebuild the local index from Python files and return the file count."""
    filepaths = list(Path(directory).rglob("*.py"))
    if not filepaths:
        raise FileNotFoundError("No .py files found in the specified directory.")

    for index_file in (DB_PATH, "rawrag_vectors.npy", "vector_ids.json"):
        if os.path.exists(index_file):
            os.remove(index_file)

    initialize_database()
    vector_store = VectorMath()

    for path in filepaths:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for chunk in chunks:
            vector = get_embedding(chunk)
            chunk_id = insert_chunk(str(path), chunk)
            vector_store.add_vector(vector, chunk_id)

    if vector_store.matrix is None:
        raise FileNotFoundError("No non-empty Python code found in the specified directory.")

    vector_store.save_index()
    return len(filepaths)


def query_codebase(question: str, model: str = "llama3") -> tuple[str, list[str]]:
    """Retrieve relevant code chunks and ask the configured Ollama model."""
    if not all(os.path.exists(path) for path in (DB_PATH, "rawrag_vectors.npy", "vector_ids.json")):
        raise IndexNotFoundError("No vector index found. Run `rawrag index` to build it.")

    try:
        vector_store = VectorMath()
        chunk_count = count_chunks()
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise IndexConsistencyError(str(error)) from error

    if vector_store.matrix is None:
        raise IndexNotFoundError("No vector index found. Run `rawrag index` to build it.")

    if chunk_count != vector_store.matrix.shape[0]:
        raise IndexConsistencyError("The SQLite and vector indexes are out of sync. Run `rawrag index` again.")

    chunk_ids = vector_store.search(get_embedding(question), top_k=3)
    if not chunk_ids:
        raise NoRelevantChunksError("No relevant chunks found for the query. Run `rawrag index` first.")

    chunks = get_chunks(chunk_ids)
    if not chunks:
        raise NoRelevantChunksError("No matching chunks found in the database. Run `rawrag index` first.")

    context_string = "\n\n".join(chunks)
    prompt = f"""You are a helpful coding assistant answering questions about a local codebase.
Use ONLY the provided codebase context below to answer the user's question. If the answer is not in the context, say you don't know.

Codebase Context:
{context_string}

Question: {question}
Answer:"""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    answer = response.json().get("response", "No response received.")
    return answer, chunks
