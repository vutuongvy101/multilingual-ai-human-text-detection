#!/usr/bin/env python3
"""Run inference with trained model."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multilingual_ai_detection.models import load_model


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["statistical", "transformer"],
        required=True,
        help="Type of model"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="File containing texts to classify (one per line)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for predictions"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type)

    # Get texts to classify
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Must provide either --text or --text-file")
        return

    print(f"Classifying {len(texts)} text(s)...")

    # Make predictions
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    # Prepare results
    class_names = ["Human", "AI"]
    results = []

    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        result = {
            "index": i,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": class_names[pred],
            "confidence": max(prob) if isinstance(prob, list) else prob,
        }
        if isinstance(prob, list):
            result["probabilities"] = {class_names[j]: p for j, p in enumerate(prob)}
        results.append(result)

    # Print results
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(".4f")
        if "probabilities" in result:
            print("Probabilities:")
            for cls, prob in result["probabilities"].items():
                print(".4f")

    # Save to file if requested
    if args.output_file:
        import json
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()