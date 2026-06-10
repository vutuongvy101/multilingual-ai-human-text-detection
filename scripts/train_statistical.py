#!/usr/bin/env python3
"""Train statistical model for AI-human text detection."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multilingual_ai_detection.data import load_multilingual_dataset
from multilingual_ai_detection.training import train_statistical_model


def main():
    parser = argparse.ArgumentParser(description="Train statistical model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing dataset files"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logistic", "naive_bayes"],
        default="logistic",
        help="Type of classifier to use"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum number of TF-IDF features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/statistical",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    dataset = load_multilingual_dataset(args.data_dir, seed=args.seed)

    # Prepare training data
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["text"]
    val_labels = dataset["validation"]["label"]

    print(f"Training on {len(train_texts)} samples")
    print(f"Validating on {len(val_texts)} samples")

    # Train model
    print(f"Training {args.classifier} classifier...")
    model = train_statistical_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        classifier=args.classifier,
        max_features=args.max_features,
        output_dir=args.output_dir
    )

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()