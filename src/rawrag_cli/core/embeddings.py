from sentence_transformers import SentenceTransformer
import numpy as np
import threading

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a given text using the SentenceTransformer model.
    """
    embeddings = _get_model().encode(text)
    return np.array(embeddings, dtype=np.float32)