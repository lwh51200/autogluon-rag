import unittest
from unittest.mock import MagicMock, patch

import torch

from agrag.modules.retriever.rerankers.reranker import Reranker


class TestReranker(unittest.TestCase):
    @patch("agrag.modules.retriever.rerankers.reranker.AutoModel.from_pretrained")
    @patch("agrag.modules.retriever.rerankers.reranker.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

        self.reranker = Reranker(
            model_name="some-model",
            batch_size=2,
            hf_model_params={},
            hf_tokenizer_init_params={},
            hf_tokenizer_params={"padding": True, "max_length": 512, "return_tensors": "pt"},
            hf_forward_params={},
        )

    def test_rerank(self):
        query = "Some query"
        text_chunks = ["test chunk 1", "test chunk 2"]

        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        self.mock_model.return_value.logits = torch.tensor([[0.1], [0.2], [0.3]])

        with patch.object(self.reranker, "model", return_value=[torch.tensor([[0.1], [0.2], [0.3]])]):
            sorted_text_chunks = self.reranker.rerank(query, text_chunks)

        self.assertEqual(sorted_text_chunks, ["test chunk 2", "test chunk 1"])

    def test_rerank_top_k(self):
        query = "Some query"
        text_chunks = ["test chunk 1", "test chunk 2"]

        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        self.reranker.top_k = 1

        self.mock_model.return_value.logits = torch.tensor([[0.1], [0.2], [0.3]])

        with patch.object(self.reranker, "model", return_value=[torch.tensor([[0.1], [0.2], [0.3]])]):
            sorted_text_chunks = self.reranker.rerank(query, text_chunks)

        self.assertEqual(sorted_text_chunks, ["test chunk 2"])

    def test_rerank_return_scores(self):
        query = "Some query"
        text_chunks = ["test chunk 1", "test chunk 2"]

        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        with patch.object(self.reranker, "model", return_value=[torch.tensor([[0.1], [0.2]])]):
            scored = self.reranker.rerank(query, text_chunks, return_scores=True)

        # Returns (text, score) tuples sorted by descending score.
        self.assertEqual([t for t, _ in scored], ["test chunk 2", "test chunk 1"])
        self.assertAlmostEqual(scored[0][1], 0.2, places=5)
        self.assertAlmostEqual(scored[1][1], 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
