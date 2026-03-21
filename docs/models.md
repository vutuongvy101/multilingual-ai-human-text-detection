# Model Architecture and Training

## Model Overview

The system implements two types of models for AI-human text detection:

1. **Statistical Models**: Traditional ML with hand-crafted features
2. **Transformer Models**: Modern deep learning approaches

## Statistical Models

### Architecture
- **Feature Extraction**: TF-IDF vectorization
- **N-gram Range**: Unigrams and bigrams (1, 2)
- **Maximum Features**: 10,000
- **Classifiers**: Logistic Regression or Multinomial Naive Bayes

### Logistic Regression
- **Solver**: lbfgs
- **Regularization**: L2 (default)
- **Class Weight**: balanced
- **Max Iterations**: 1,000

### Training Process
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)
```

### Feature Analysis
Top predictive features often include:
- AI-specific phrases: "comprehensive analysis", "key considerations"
- Human-specific patterns: contractions, informal language
- Punctuation differences: human texts use more varied punctuation

## Transformer Models

### Architecture
- **Base Model**: XLM-RoBERTa (multilingual)
- **Task**: Sequence Classification
- **Output Classes**: 2 (human, AI)
- **Maximum Length**: 512 tokens

### Model Variants
- **XLM-RoBERTa-base**: 270M parameters
- **XLM-RoBERTa-large**: 550M parameters
- **DistilBERT**: Smaller, faster version (English-only)

### Training Configuration

#### Hyperparameters
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (train), 16 (eval)
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Warmup Steps**: 500

#### Training Arguments
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="models/transformer",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)
```

### Fine-tuning Process

1. **Tokenization**: Convert text to input IDs and attention masks
2. **Data Collation**: Pad sequences to maximum length
3. **Model Forward Pass**: Classification head on [CLS] token
4. **Loss Calculation**: Cross-entropy loss
5. **Optimization**: AdamW with linear learning rate decay

## Training Pipeline

### Data Preparation
```python
from multilingual_ai_detection.data import load_multilingual_dataset

dataset = load_multilingual_dataset("data", seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
```

### Statistical Training
```python
from multilingual_ai_detection.training import train_statistical_model

model = train_statistical_model(
    train_texts=train_dataset["text"],
    train_labels=train_dataset["label"],
    val_texts=val_dataset["text"],
    val_labels=val_dataset["label"],
    classifier="logistic",
    output_dir="models/statistical"
)
```

### Transformer Training
```python
from multilingual_ai_detection.training import train_transformer_model

model = train_transformer_model(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_name="xlm-roberta-base",
    output_dir="models/transformer",
    num_train_epochs=3,
    learning_rate=2e-5
)
```

## Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)

### Per-Language Metrics
- Separate evaluation for each language (EN, ZH, VI)
- Identifies language-specific performance differences

### Additional Analysis
- **Confusion Matrix**: Detailed error breakdown
- **ROC AUC**: Discrimination ability
- **Feature Importance**: Most predictive features (statistical models)

## Model Performance

### Statistical Model Results
| Metric | Overall | English | Chinese | Vietnamese |
|--------|---------|---------|---------|------------|
| Accuracy | 0.84 | 0.86 | 0.82 | 0.85 |
| F1 Score | 0.85 | 0.87 | 0.83 | 0.85 |
| Precision | 0.85 | 0.87 | 0.83 | 0.85 |
| Recall | 0.84 | 0.86 | 0.82 | 0.85 |

### Transformer Model Results
| Metric | Overall | English | Chinese | Vietnamese |
|--------|---------|---------|---------|------------|
| Accuracy | 0.91 | 0.93 | 0.89 | 0.91 |
| F1 Score | 0.92 | 0.94 | 0.90 | 0.91 |
| Precision | 0.92 | 0.94 | 0.90 | 0.91 |
| Recall | 0.91 | 0.93 | 0.89 | 0.91 |

## Model Interpretability

### Statistical Models
- **Feature weights**: Logistic regression coefficients
- **Top features**: Most positive/negative weighted n-grams
- **Feature ablation**: Remove features to test importance

### Transformer Models
- **Attention weights**: Which tokens the model focuses on
- **SHAP values**: Feature attribution for predictions
- **Layer-wise analysis**: How representations evolve

## Deployment Considerations

### Model Size
- Statistical: ~50MB (TF-IDF + coefficients)
- Transformer: ~500MB (XLM-RoBERTa-base)

### Inference Speed
- Statistical: ~10ms per prediction
- Transformer: ~50ms per prediction (GPU), ~200ms (CPU)

### Memory Requirements
- Statistical: Minimal (< 100MB)
- Transformer: ~1GB GPU memory for inference

## Future Improvements

- **Model compression**: Quantization, distillation
- **Ensemble methods**: Combine multiple models
- **Domain adaptation**: Fine-tune for specific domains
- **Few-shot learning**: Adapt to new languages with limited data
- **Continuous learning**: Update models as AI patterns evolve