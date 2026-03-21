"""Tests for multilingual AI detection package."""

import pytest
from multilingual_ai_detection.utils import clean_text, extract_text_features


class TestUtils:
    """Test utility functions."""

    def test_clean_text(self):
        """Test text cleaning function."""
        # Basic cleaning
        assert clean_text("  hello   world  ") == "hello world"

        # URL removal
        assert clean_text("Check this http://example.com") == "Check this"

        # Email removal
        assert clean_text("Contact me at test@example.com") == "Contact me at"

        # Empty input
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_extract_text_features(self):
        """Test text feature extraction."""
        text = "Hello world! This is a test."
        features = extract_text_features(text)

        assert "n_words" in features
        assert "n_chars" in features
        assert "lexical_diversity" in features
        assert "avg_word_length" in features
        assert "punctuation_ratio" in features

        assert features["n_words"] == 6  # Hello, world, This, is, a, test
        assert features["n_chars"] == len(text)

        # Empty text
        empty_features = extract_text_features("")
        assert empty_features["n_words"] == 0
        assert empty_features["lexical_diversity"] == 0.0


class TestDataLoading:
    """Test data loading functions."""

    def test_stable_prompt_id(self):
        """Test stable prompt ID generation."""
        from multilingual_ai_detection.data import stable_prompt_id

        # Same inputs should give same ID
        id1 = stable_prompt_id("en", "What is AI?")
        id2 = stable_prompt_id("en", "What is AI?")
        assert id1 == id2

        # Different inputs should give different IDs
        id3 = stable_prompt_id("en", "What is ML?")
        assert id1 != id3

        # Different languages should give different IDs
        id4 = stable_prompt_id("zh", "What is AI?")
        assert id1 != id4


if __name__ == "__main__":
    pytest.main([__file__])