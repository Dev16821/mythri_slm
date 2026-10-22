"""
Tokenizer wrapper for MYTHRI SLM.

Uses DistilBERT's WordPiece tokenizer — no need to train a tokenizer
from scratch for Experiment 0. The model's embedding layer is trained
from scratch, but the tokenizer is pre-trained.
"""

from typing import List, Dict, Optional
from transformers import AutoTokenizer


class MythriTokenizer:
    """Wraps HuggingFace tokenizer with MYTHRI-specific convenience methods."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def encode(
        self,
        text: str,
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = True,
    ) -> Dict:
        """
        Encode a single text into model-ready tensors.
        
        Returns dict with:
            input_ids: (seq_len,)
            attention_mask: (seq_len,)
        """
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding="max_length" if padding else False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = True,
    ) -> Dict:
        """Encode a batch of texts."""
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding="max_length" if padding else False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)