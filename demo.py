#!/usr/bin/env python3
"""Demo script showing how to use the multilingual AI detection package."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multilingual_ai_detection.data import load_multilingual_dataset
from multilingual_ai_detection.models import StatisticalClassifier, TransformerClassifier
from multilingual_ai_detection.evaluation import evaluate_model


def demo_data_loading():
    """Demo data loading functionality."""
    print("🔍 Loading multilingual dataset...")
    dataset = load_multilingual_dataset("data", seed=42)

    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples")
    print(f"Test: {len(dataset['test'])} samples")

    # Show sample
    sample = dataset["train"][0]
    print(f"\nSample text: {sample['text'][:100]}...")
    print(f"Label: {'Human' if sample['label'] == 0 else 'AI'}")
    print(f"Language: {sample['lang']}")


def demo_statistical_model():
    """Demo statistical model training and inference."""
    print("\n🤖 Training statistical model...")

    # Load data
    dataset = load_multilingual_dataset("data", seed=42)
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Train model
    model = StatisticalClassifier(classifier="logistic", max_features=5000)
    model.fit(train_texts, train_labels)

    # Make predictions
    predictions = model.predict(test_texts[:5])  # First 5 samples
    probabilities = model.predict_proba(test_texts[:5])

    print("Sample predictions:")
    for i, (text, pred, prob) in enumerate(zip(test_texts[:5], predictions, probabilities)):
        print(f"  {i+1}. {'Human' if pred == 0 else 'AI'} "
              ".3f"
              f"  Text: {text[:50]}...")


def demo_transformer_model():
    """Demo transformer model (if available)."""
    print("\n🚀 Loading transformer model...")

    model_path = "models/transformer"
    if Path(model_path).exists():
        try:
            model = TransformerClassifier.load_model(model_path)

            # Test inference
            test_texts = [
                "This is a human-written message about artificial intelligence.",
                "The weather today is sunny and warm, perfect for outdoor activities."
            ]

            predictions = model.predict(test_texts)
            print("Transformer predictions:")
            for text, pred in zip(test_texts, predictions):
                print(f"  {'Human' if pred == 0 else 'AI'}: {text[:50]}...")

        except Exception as e:
            print(f"Could not load transformer model: {e}")
    else:
        print("Transformer model not found. Train it first with: python scripts/train_transformer.py")


def main():
    """Run the demo."""
    print("🎯 Multilingual AI-Human Text Detection Demo")
    print("=" * 50)

    try:
        demo_data_loading()
        demo_statistical_model()
        demo_transformer_model()

        print("\n✅ Demo completed successfully!")
        print("\nTo train models:")
        print("  python scripts/train_statistical.py")
        print("  python scripts/train_transformer.py")
        print("\nTo run inference:")
        print("  python scripts/infer.py --text 'Your text here'")
        print("\nTo start web demo:")
        print("  streamlit run scripts/web_demo.py")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have the required data files in the 'data/' directory.")


if __name__ == "__main__":
    main()