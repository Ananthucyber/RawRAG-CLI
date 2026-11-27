import numpy as np
import os

class VectorMath:
    def __init__(self, index_path="rawrag_vectors.npy"):
        self.index_path = index_path

        """
        Initialize the vector index

        Load existing matrix from disk if it exists, otherwise start fresh

        Args:
            index_path (str): Path to the NumPy file used to store the vector index. Defaults to "rawrag_vectors.npy".
        """
        if os.path.exists(self.index_path):
            self.matrix = np.load(self.index_path)
        else:
            self.matrix = None

    def add_vector(self, vector):
        """
        Initialize the vector intex.

        If the index is empty initializes the matrix with the provided vector. Otherwise appends the vector as a new row.

        Args:
             vector (np.ndarray): The vector to add to the index.
        
        return:
            Shape of the updated matrix

        """
        if self.matrix is None:
            self.matrix = np.atleast_2d(vector)
        else:
            self.matrix = np.vstack([self.matrix, np.atleast_2d(vector)])
        return self.matrix.shape

    def search(self, query_vector, top_k=3):
        """
        search for the most similar vector

        Args:
            query_vector (np.ndarray): The vector used to search the index.
            top_k (int): Maximum number of matching vector indices to return. Defaults is 3.

         Returns:
            list[int]: Indices of the most similar vectors. Returns an empty list if the index has no vectors.
        """
        if self.matrix is None:
            return []
        
        q = np.array(query_vector)
        dot = np.dot(self.matrix, q)
        norm = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(q)
        
        # Avoid division by zero
        similarity = np.divide(dot, norm, out=np.zeros_like(dot), where=norm!=0)
        
        # Get top_k indices
        top_indices = np.argsort(similarity)[::-1][:top_k]
        return top_indices.tolist()

    def save_index(self):
        """Saves the NumPy matrix to disk."""
        if self.matrix is not None:
            np.save(self.index_path, self.matrix)