import unittest
import tempfile
import os
import json
from batch_rag.batch_processor import BatchProcessor
from unittest.mock import patch
import faiss


class TestBatchProcessor(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test output
        self.test_dir = tempfile.TemporaryDirectory()
        self.processor = BatchProcessor(output_dir=self.test_dir.name)

        # Fake article data to inject for index building
        self.mock_articles = [
            {
                "title": "Test Article",
                "date": "2023-01-01",
                "url": "https://example.com/article",
                "image_url": None,
                "body": "This is a test body of the article.",
                "image_caption": "Test image caption"
            }
        ]

        with open(self.processor.article_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_articles, f)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_build_index_creates_index_file(self):
        """Test that build_index creates a FAISS index and metadata."""
        self.processor.build_index()

        self.assertTrue(os.path.exists(self.processor.index_path), "FAISS index file not created.")
        self.assertTrue(os.path.exists(self.processor.metadata_path), "Metadata file not created.")

        index = faiss.read_index(self.processor.index_path)
        self.assertEqual(index.ntotal, 1, "Index should contain one entry.")

    def test_metadata_file_contains_expected_fields(self):
        """Ensure metadata JSON contains the correct fields."""
        self.processor.build_index()

        with open(self.processor.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertIn("title", metadata[0])
        self.assertIn("body", metadata[0])
        self.assertIn("image_caption", metadata[0])

    @patch("batch_rag.batch_processor.BatchProcessor.extract_articles")
    @patch("batch_rag.batch_processor.BatchProcessor.build_index")
    def test_run_all_executes_both(self, mock_index, mock_extract):
        """Make sure run_all() calls extract_articles and build_index."""
        self.processor.run_all()
        mock_extract.assert_called_once()
        mock_index.assert_called_once()

