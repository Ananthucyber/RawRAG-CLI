import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from rawrag_cli.core import pipeline


class PipelineTests(unittest.TestCase):
    def test_index_directory_indexes_python_files_through_shared_pipeline(self):
        with TemporaryDirectory() as directory:
            sample_path = Path(directory, "sample.py")
            sample_path.write_text("print('hello')", encoding="utf-8")
            for ignored_dir in (".venv", "venv", "__pycache__"):
                ignored_path = Path(directory, ignored_dir, "ignored.py")
                ignored_path.parent.mkdir(parents=True)
                ignored_path.write_text("print('ignored')", encoding="utf-8")

            vector_store = Mock()
            on_file = Mock()

            with (
                patch.object(pipeline.os.path, "exists", return_value=False),
                patch.object(pipeline, "initialize_database") as initialize_database,
                patch.object(pipeline, "VectorMath", return_value=vector_store),
                patch.object(pipeline, "chunk_text", return_value=["code chunk"]),
                patch.object(pipeline, "get_embedding", return_value=np.zeros(384)),
                patch.object(pipeline, "insert_chunk", return_value=7),
            ):
                file_count = pipeline.index_directory(directory, on_file=on_file)

        self.assertEqual(file_count, 1)
        on_file.assert_called_once_with(sample_path)
        initialize_database.assert_called_once_with()
        vector_store.add_vector.assert_called_once()
        vector_store.save_index.assert_called_once_with()

    def test_index_directory_rejects_files_that_produce_no_chunks(self):
        with TemporaryDirectory() as directory:
            Path(directory, "empty.py").write_text("", encoding="utf-8")
            vector_store = Mock()
            vector_store.matrix = None

            with (
                patch.object(pipeline.os.path, "exists", return_value=False),
                patch.object(pipeline, "initialize_database"),
                patch.object(pipeline, "VectorMath", return_value=vector_store),
                patch.object(pipeline, "chunk_text", return_value=[]),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "No non-empty Python code"):
                    pipeline.index_directory(directory)

        vector_store.save_index.assert_not_called()

    def test_query_uses_selected_model_and_returns_retrieved_chunks(self):
        vector_store = Mock()
        vector_store.matrix = np.zeros((1, 384))
        vector_store.search.return_value = [7]
        response = Mock()
        response.json.return_value = {"response": "Found it."}

        with (
            patch.object(pipeline.os.path, "exists", return_value=True),
            patch.object(pipeline, "VectorMath", return_value=vector_store),
            patch.object(pipeline, "count_chunks", return_value=1),
            patch.object(pipeline, "get_embedding", return_value=np.zeros(384)),
            patch.object(pipeline, "get_chunks", return_value=["File: src/example.py\nCode:\nexample"]),
            patch.object(pipeline.requests, "post", return_value=response) as post,
        ):
            answer, chunks = pipeline.query_codebase("where is it?", model="qwen2.5-coder")

        self.assertEqual(answer, "Found it.")
        self.assertEqual(chunks, ["File: src/example.py\nCode:\nexample"])
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen2.5-coder")

    def test_query_without_vector_files_reports_missing_index(self):
        with (
            patch.object(pipeline.os.path, "exists", return_value=False),
            patch.object(pipeline, "VectorMath") as vector_math,
        ):
            with self.assertRaises(pipeline.IndexNotFoundError):
                pipeline.query_codebase("question")

        vector_math.assert_not_called()


if __name__ == "__main__":
    unittest.main()
