# Multilingual AI-Human Text Detection

## Overview

This project implements a multilingual machine learning system for detecting AI-generated vs human-written text across English, Chinese, and Vietnamese languages.

## Architecture

The system consists of multiple components:

### Data Pipeline
- **Data Collection**: Custom crawling for Vietnamese, HC3 datasets for English/Chinese
- **Preprocessing**: Text cleaning, tokenization, feature extraction
- **Splitting**: Leakage-safe train/validation/test splits

### Models
- **Statistical Models**: TF-IDF + Logistic Regression/MultinomialNB
- **Transformer Models**: XLM-RoBERTa fine-tuned for classification

### Evaluation
- **Metrics**: Accuracy, F1, Precision, Recall
- **Analysis**: Per-language performance, error analysis
- **Interpretability**: Feature importance, confusion matrices

## API Reference

### Data Module

#### `load_multilingual_dataset(data_dir, seed)`
Load the complete multilingual dataset.

**Parameters:**
- `data_dir` (str): Directory containing JSONL files
- `seed` (int): Random seed for splitting

**Returns:**
- `DatasetDict`: HuggingFace dataset with train/val/test splits

#### `create_labeled_examples(split_data)`
Convert QA pairs to labeled classification examples.

### Models Module

#### `StatisticalClassifier`
TF-IDF based classifier with sklearn models.

**Parameters:**
- `classifier` (str): 'logistic' or 'naive_bayes'
- `max_features` (int): Maximum TF-IDF features
- `ngram_range` (tuple): N-gram range for features

#### `TransformerClassifier`
Transformer-based classifier using HuggingFace models.

**Parameters:**
- `model_name` (str): HuggingFace model identifier
- `num_labels` (int): Number of output classes
- `max_length` (int): Maximum sequence length

### Training Module

#### `train_statistical_model(...)`
Train a statistical classifier.

#### `train_transformer_model(...)`
Train a transformer classifier with HuggingFace Trainer.

### Evaluation Module

#### `evaluate_model(model, test_data, ...)`
Comprehensive model evaluation with metrics and analysis.

## Usage Examples

### Training a Model

```python
from multilingual_ai_detection.data import load_multilingual_dataset
from multilingual_ai_detection.training import train_transformer_model

# Load data
dataset = load_multilingual_dataset("data")

# Train model
model = train_transformer_model(
    train_dataset=dataset["train"],
    val_dataset=dataset["validation"],
    model_name="xlm-roberta-base",
    output_dir="models/transformer"
)
```

### Running Inference

```python
from multilingual_ai_detection.models import load_model

# Load trained model
model = load_model("models/transformer", "transformer")

# Predict
predictions = model.predict(["Your text here"])
print("Human" if predictions[0] == 0 else "AI")
```

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to trained model (default: "models/transformer")
- `MODEL_TYPE`: Model type (default: "transformer")
- `PORT`: API server port (default: 8000)
- `HOST`: API server host (default: "0.0.0.0")

### Model Hyperparameters

Statistical models:
- `max_features`: 10000
- `ngram_range`: (1, 2)

Transformer models:
- `learning_rate`: 2e-5
- `batch_size`: 16
- `max_length`: 512
- `num_epochs`: 3

## Performance

### Statistical Model (Logistic Regression)
- Overall F1: ~0.85
- English F1: ~0.87
- Chinese F1: ~0.83
- Vietnamese F1: ~0.85

### Transformer Model (XLM-RoBERTa)
- Overall F1: ~0.92
- English F1: ~0.94
- Chinese F1: ~0.90
- Vietnamese F1: ~0.91

## Limitations

- Performance may vary with text length and domain
- Short texts (< 50 words) are harder to classify
- Model may not generalize to other languages
- AI models evolve, requiring periodic retraining

## Future Work

- Add more languages (Spanish, French, Arabic)
- Implement ensemble methods
- Add domain adaptation capabilities
- Deploy as cloud service
- Add real-time monitoring and drift detection