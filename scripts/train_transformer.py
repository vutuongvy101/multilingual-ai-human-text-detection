#!/usr/bin/env python3
"""Train transformer model for AI-human text detection."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multilingual_ai_detection.data import load_multilingual_dataset
from multilingual_ai_detection.training import train_transformer_model


def main():
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing dataset files"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/transformer",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
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

    print(f"Training on {len(dataset['train'])} samples")
    print(f"Validating on {len(dataset['validation'])} samples")

    # Train model
    print(f"Training {args.model_name}...")
    model = train_transformer_model(
        train_dataset=dataset["train"],
        val_dataset=dataset["validation"],
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()