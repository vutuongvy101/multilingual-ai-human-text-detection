# Multilingual AI-Human Text Detection

A machine learning project for detecting AI-generated vs human-written text across multiple languages (English, Chinese, Vietnamese). This project implements both traditional statistical models and modern transformer-based approaches for binary classification of text authenticity.

## Features

- **Multilingual Support**: Trained on English, Chinese, and Vietnamese datasets
- **Multiple Model Architectures**:
  - Statistical models (Logistic Regression, Naive Bayes) with n-gram features
  - Transformer models (DistilBERT, XLM-RoBERTa)
- **Data Pipeline**: Complete data collection, preprocessing, and generation pipeline
- **Evaluation**: Comprehensive metrics and per-language analysis
- **Inference API**: REST API for real-time text classification
- **Web Demo**: Interactive web interface for testing the model

## Project Structure

```
├── data/                    # Datasets and raw data
├── docs/                    # Documentation
│   ├── README.md          # Main documentation
│   ├── api.md            # API reference
│   ├── data.md           # Data collection details
│   └── models.md         # Model architecture
├── models/                  # Trained model checkpoints
├── notebooks/              # Jupyter notebooks (POC and experiments)
├── scripts/                # Training and evaluation scripts
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model definitions and architectures
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Evaluation and metrics
│   └── utils/             # Utilities and helpers
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── setup.py               # Package setup
└── README.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-ai-human-text-detection.git
cd multilingual-ai-human-text-detection

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Training

```bash
# Train statistical model
python scripts/train_statistical.py

# Train transformer model
python scripts/train_transformer.py --model xlm-roberta-base
```

### Demo

Run the interactive demo to see the package in action:

```bash
python demo.py
```

This will demonstrate:
- Data loading from the multilingual dataset
- Training a statistical model
- Running inference with both model types

### Inference

```bash
# Run inference on text
python scripts/infer.py --text "Your text here" --model models/transformer --model-type transformer

# Run on multiple texts from file
python scripts/infer.py --text-file input.txt --output-file results.json
```

### API Server

```bash
# Start the REST API
python scripts/serve_api.py

# Example request
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text to classify"}'
```

## Data

The dataset consists of question-answer pairs from:
- **English**: HC3 dataset (reddit_eli5 subset)
- **Chinese**: HC3-Chinese dataset (open_qa subset)
- **Vietnamese**: Custom crawled data from Vietnamese Reddit communities

Each language has 300 QA pairs with both human-written and AI-generated answers (using Qwen2.5-1.5B-Instruct).

## Models

### Statistical Models
- Feature extraction: TF-IDF with n-grams (1-2)
- Classifiers: Logistic Regression, Multinomial Naive Bayes
- Best F1: ~0.85 on multilingual test set

### Transformer Models
- **DistilBERT**: English-only baseline
- **XLM-RoBERTa**: Multilingual model
- Best F1: ~0.92 on multilingual test set

## Evaluation

The project provides comprehensive evaluation including:
- Standard metrics: Accuracy, Precision, Recall, F1
- Per-language breakdown
- Confusion matrices
- Error analysis
- Model interpretability

## Web Demo

Launch the interactive web demo:

```bash
python scripts/web_demo.py
```

Visit `http://localhost:8501` to test the model with your own text.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

If you use this work, please cite:

```bibtex
@misc{vu2024multilingual,
  title={Multilingual AI-Human Text Detection},
  author={Tuong Vy Vu},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/multilingual-ai-human-text-detection}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Tuong Vy Vu - [Your contact information]

Project Link: [https://github.com/yourusername/multilingual-ai-human-text-detection](https://github.com/yourusername/multilingual-ai-human-text-detection)
