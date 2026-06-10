"""Transformer-based classifier for text detection."""

from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TransformerClassifier:
    """Transformer-based classifier using HuggingFace models."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.eval()

    def predict(self, texts: List[str]) -> List[int]:
        """Predict class labels for texts."""
        probabilities = self.predict_proba(texts)
        return [int(max(enumerate(probs), key=lambda x: x[1])[0]) for probs in probabilities]

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """Predict class probabilities for texts."""
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.tolist()

    @classmethod
    def load_model(
        cls,
        path: str,
        num_labels: int = 2,
        max_length: int = 512,
    ) -> "TransformerClassifier":
        """Load a trained transformer classifier from disk."""
        instance = cls.__new__(cls)
        instance.model_name = path
        instance.num_labels = num_labels
        instance.max_length = max_length
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.eval()
        return instance
