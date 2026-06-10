"""Text processing utilities."""

import re
from typing import Dict, List, Tuple

import numpy as np


def clean_text(text: str) -> str:
    """Clean and normalize text for processing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    return text.strip()


def extract_text_features(text: str) -> Dict[str, float]:
    """Extract basic text features for analysis.

    Args:
        text: Input text

    Returns:
        Dictionary of features
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "n_words": 0,
            "n_chars": 0,
            "lexical_diversity": 0.0,
            "avg_word_length": 0.0,
            "punctuation_ratio": 0.0,
        }

    # Basic tokenization (simple word split)
    words = re.findall(r'\b\w+\b', text.lower())
    n_words = len(words)
    n_chars = len(text)

    # Lexical diversity
    unique_words = set(words)
    lexical_diversity = len(unique_words) / n_words if n_words > 0 else 0.0

    # Average word length
    avg_word_length = sum(len(word) for word in words) / n_words if n_words > 0 else 0.0

    # Punctuation ratio
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    punctuation_ratio = punctuation_count / n_chars if n_chars > 0 else 0.0

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "punctuation_ratio": punctuation_ratio,
    }