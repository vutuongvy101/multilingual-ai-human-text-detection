"""Transformer model training utilities."""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..models import TransformerClassifier
from .common import compute_metrics


def tokenize_function(tokenizer, max_length: int = 512):
    """Create tokenization function for transformers.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenization function
    """
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    return tokenize


def _columns_to_remove(dataset: Dataset) -> list:
    """Return dataset columns to drop after tokenization, keeping labels."""
    return [col for col in dataset.column_names if col != "label"]


def train_transformer_model(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    model_name: str = "xlm-roberta-base",
    num_labels: int = 2,
    max_length: int = 512,
    output_dir: str = "models/transformer",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    logging_steps: int = 100,
    save_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    dataloader_pin_memory: bool = False,
    eval_strategy: str = "steps",
    save_strategy: str = "steps",
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
        num_labels: Number of output labels
        max_length: Maximum sequence length
        output_dir: Output directory for model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps
        logging_steps: Logging frequency
        save_steps: Saving frequency
        gradient_accumulation_steps: Number of steps to accumulate gradients
        dataloader_pin_memory: Whether to pin memory in dataloaders
        eval_strategy: Evaluation strategy
        save_strategy: Saving strategy
        load_best_model_at_end: Whether to load best model
        metric_for_best_model: Metric for best model
        greater_is_better: Whether higher metric is better
        **kwargs: Additional training arguments

    Returns:
        Trained TransformerClassifier
    """
    # Initialize model and tokenizer
    model = TransformerClassifier(
        model_name=model_name,
        num_labels=num_labels,
        max_length=max_length
    )

    # Tokenize datasets
    tokenized_train = train_dataset.map(
        tokenize_function(model.tokenizer, max_length),
        batched=True,
        remove_columns=_columns_to_remove(train_dataset),
    )

    tokenized_val = None
    if val_dataset is not None:
        tokenized_val = val_dataset.map(
            tokenize_function(model.tokenizer, max_length),
            batched=True,
            remove_columns=_columns_to_remove(val_dataset),
        )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_pin_memory=dataloader_pin_memory,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        **kwargs
    )

    # Initialize trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # From common.py
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(output_dir)
    model.tokenizer.save_pretrained(output_dir)

    # Load the best model if available
    if load_best_model_at_end and val_dataset is not None:
        best_model_path = trainer.state.best_model_checkpoint
        if best_model_path:
            model.model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
            model.tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    return model