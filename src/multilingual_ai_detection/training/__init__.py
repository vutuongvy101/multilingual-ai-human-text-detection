"""Training utilities for different model types."""

from .common import compute_metrics, tokenize_function
from .statistical import train_statistical_model
from .transformer import train_transformer_model

__all__ = [
    "compute_metrics",
    "tokenize_function",
    "train_statistical_model",
    "train_transformer_model",
]
    """Train a statistical classifier.

    Args:
        train_texts: Training text data
        train_labels: Training labels
        val_texts: Validation text data (optional)
        val_labels: Validation labels (optional)
        classifier: Type of classifier ('logistic' or 'naive_bayes')
        max_features: Maximum features for TF-IDF
        output_dir: Directory to save the model
        **kwargs: Additional arguments for StatisticalClassifier

    Returns:
        Trained StatisticalClassifier
    """
    # Initialize model
    model = StatisticalClassifier(
        classifier=classifier,
        max_features=max_features,
        **kwargs
    )

    # Fit the model
    model.fit(train_texts, train_labels)

    # Evaluate on validation set if provided
    if val_texts is not None and val_labels is not None:
        val_predictions = model.predict(val_texts)
        val_metrics = compute_metrics((model.predict_proba(val_texts), val_labels))
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / f"{classifier}_model.joblib"
    model.save_model(model, str(model_path))

    return model


def tokenize_function(tokenizer, max_length: int = 512):
    """Create a tokenize function for transformers."""
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Will be handled by data collator
            max_length=max_length,
        )
    return tokenize


def train_transformer_model(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    model_name: str = "xlm-roberta-base",
    output_dir: str = "models/transformer",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    **kwargs
) -> TransformerClassifier:
    """Train a transformer-based classifier.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        model_name: HuggingFace model name
        output_dir: Output directory for model and checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        load_best_model_at_end: Whether to load best model at end
        metric_for_best_model: Metric to use for best model selection
        greater_is_better: Whether higher metric values are better
        **kwargs: Additional training arguments

    Returns:
        Trained TransformerClassifier
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification
    )

    # Tokenize datasets
    tokenize_fn = tokenize_function(tokenizer, max_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    if val_dataset is not None:
        val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        evaluation_strategy="steps" if val_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=2,
        **kwargs
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)

    # Create and return the classifier wrapper
    classifier = TransformerClassifier(
        model_name=output_dir,  # Load from saved path
        num_labels=2,
        max_length=max_length
    )

    return classifier


def train_model(
    model_type: str,
    train_data: Union[Dataset, tuple],
    val_data: Optional[Union[Dataset, tuple]] = None,
    **kwargs
) -> Union[StatisticalClassifier, TransformerClassifier]:
    """Unified training function for different model types.

    Args:
        model_type: Type of model ('statistical' or 'transformer')
        train_data: Training data (Dataset for transformer, tuple of (texts, labels) for statistical)
        val_data: Validation data (optional)
        **kwargs: Model-specific training arguments

    Returns:
        Trained model
    """
    if model_type == "statistical":
        if not isinstance(train_data, tuple) or len(train_data) != 2:
            raise ValueError("For statistical models, train_data must be a tuple of (texts, labels)")

        train_texts, train_labels = train_data
        val_texts, val_labels = val_data if val_data else (None, None)

        return train_statistical_model(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            **kwargs
        )

    elif model_type == "transformer":
        if not isinstance(train_data, Dataset):
            raise ValueError("For transformer models, train_data must be a Dataset")

        return train_transformer_model(
            train_dataset=train_data,
            val_dataset=val_data,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")