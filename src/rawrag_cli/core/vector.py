import numpy as np
import os
import json

class VectorMath:
    def __init__(self, index_path="rawrag_vectors.npy", id_path="vector_ids.json"):
        self.index_path = index_path
        self.id_path = id_path
        self.vector_ids = {}
        # Load existing matrix and IDs from disk if they exist, otherwise start fresh
        if os.path.exists(self.index_path):
            self.matrix = np.atleast_2d(np.load(self.index_path))
            self.load_ids()
            self.validate_index()
        else:
            self.matrix = None

    def load_ids(self):
        """Load vector IDs from a JSON file."""
        if os.path.exists(self.id_path):
            with open(self.id_path, 'r') as f:
                self.vector_ids = json.load(f)
        else:
            raise RuntimeError(
                f"{self.index_path} exists without {self.id_path}; delete both index files before re-indexing."
            )

    def validate_index(self):
        """Ensure the vector matrix and ID mapping are synchronized."""
        if self.matrix is None:
            return
        if len(self.vector_ids) != self.matrix.shape[0]:
            raise RuntimeError(
                "Vector index is out of sync: "
                f"{len(self.vector_ids)} vector IDs but matrix has {self.matrix.shape[0]} rows."
            )

    def add_vector(self, vector, vector_id):
        row_index = 0 if self.matrix is None else self.matrix.shape[0]
        if self.matrix is None:
            self.matrix = np.atleast_2d(vector)
        else:
            self.matrix = np.vstack([self.matrix, np.atleast_2d(vector)])
        self.vector_ids[str(row_index)] = vector_id
        return self.matrix.shape

    def search(self, query_vector, top_k=3):
        if self.matrix is None:
            return []
        self.validate_index()
        
        q = np.array(query_vector)
        dot = np.dot(self.matrix, q)
        norm = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(q)
        
        # Avoid division by zero
        similarity = np.divide(dot, norm, out=np.zeros_like(dot), where=norm!=0)
        
        # Get top_k indices
        top_indices = np.argsort(similarity)[::-1][:top_k]
        return [self.vector_ids[str(idx)] for idx in top_indices.tolist()]

    def save_index(self):
        """Saves the NumPy matrix and vector IDs to disk."""
        if self.matrix is not None:
            np.save(self.index_path, self.matrix)
            with open(self.id_path, 'w') as f:
                json.dump(self.vector_ids, f)