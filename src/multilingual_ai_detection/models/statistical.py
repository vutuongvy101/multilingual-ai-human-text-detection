"""Statistical (TF-IDF + sklearn) classifier for text detection."""

from typing import List, Tuple, Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class StatisticalClassifier:
    """TF-IDF based classifier with sklearn models."""

    def __init__(
        self,
        classifier: str = "logistic",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.classifier = classifier
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline: Pipeline = None

    def _build_pipeline(self) -> Pipeline:
        if self.classifier == "logistic":
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        elif self.classifier == "naive_bayes":
            clf = MultinomialNB()
        else:
            raise ValueError(
                f"Unknown classifier: {self.classifier}. "
                "Use 'logistic' or 'naive_bayes'."
            )

        return Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                    ),
                ),
                ("classifier", clf),
            ]
        )

    def fit(self, texts: List[str], labels: List[int]) -> "StatisticalClassifier":
        """Train the classifier on text data."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: List[str]) -> List[int]:
        """Predict class labels for texts."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """Predict class probabilities for texts."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.pipeline.predict_proba(texts).tolist()

    @classmethod
    def save_model(cls, model: "StatisticalClassifier", path: str) -> None:
        """Save a trained classifier to disk."""
        joblib.dump(model, path)

    @classmethod
    def load_model(cls, path: str) -> "StatisticalClassifier":
        """Load a trained classifier from disk."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected StatisticalClassifier, got {type(model)}")
        return model
