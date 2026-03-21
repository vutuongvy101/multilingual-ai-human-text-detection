"""Utility functions and helpers."""

from .text_processing import clean_text, extract_text_features
from .data_utils import (
    analyze_dataset_statistics,
    split_dataset_by_language,
    balance_dataset,
    create_train_val_test_split,
    save_dataset_splits,
)

__all__ = [
    "clean_text",
    "extract_text_features",
    "analyze_dataset_statistics",
    "split_dataset_by_language",
    "balance_dataset",
    "create_train_val_test_split",
    "save_dataset_splits",
]

    # Basic tokenization (simple word split)
    words = re.findall(r'\b\w+\b', text.lower())
    n_words = len(words)
    n_chars = len(text)

    # Lexical diversity
    unique_words = set(words)
    lexical_diversity = len(unique_words) / n_words if n_words > 0 else 0.0

    # Average word length
    avg_word_length = sum(len(word) for word in words) / n_words if n_words > 0 else 0.0

    # Punctuation ratio
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    punctuation_ratio = punctuation_count / n_chars if n_chars > 0 else 0.0

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "punctuation_ratio": punctuation_ratio,
    }


def analyze_dataset_statistics(dataset: pd.DataFrame) -> pd.DataFrame:
    """Analyze basic statistics of the dataset.

    Args:
        dataset: Dataset with 'text' and 'label' columns

    Returns:
        DataFrame with statistics by class
    """
    stats = []

    for label in dataset["label"].unique():
        subset = dataset[dataset["label"] == label]
        features = subset["text"].apply(extract_text_features)

        # Aggregate features
        agg_stats = {}
        for feature_name in features.iloc[0].keys():
            values = [f[feature_name] for f in features]
            agg_stats[f"{feature_name}_mean"] = np.mean(values)
            agg_stats[f"{feature_name}_std"] = np.std(values)
            agg_stats[f"{feature_name}_min"] = np.min(values)
            agg_stats[f"{feature_name}_max"] = np.max(values)

        agg_stats["count"] = len(subset)
        agg_stats["label"] = label

        stats.append(agg_stats)

    return pd.DataFrame(stats)


def split_dataset_by_language(
    dataset: pd.DataFrame,
    languages: List[str]
) -> Dict[str, pd.DataFrame]:
    """Split dataset by language.

    Args:
        dataset: Dataset with 'lang' column
        languages: List of languages to split by

    Returns:
        Dictionary mapping language codes to subsets
    """
    splits = {}
    for lang in languages:
        splits[lang] = dataset[dataset["lang"] == lang].copy()
    return splits


def balance_dataset(
    dataset: pd.DataFrame,
    label_column: str = "label",
    method: str = "undersample"
) -> pd.DataFrame:
    """Balance dataset by class labels.

    Args:
        dataset: Input dataset
        label_column: Name of label column
        method: Balancing method ('undersample' or 'oversample')

    Returns:
        Balanced dataset
    """
    label_counts = dataset[label_column].value_counts()
    min_count = label_counts.min()

    balanced_dfs = []
    for label in label_counts.index:
        subset = dataset[dataset[label_column] == label]
        if method == "undersample":
            # Random undersampling
            subset = subset.sample(n=min_count, random_state=42)
        elif method == "oversample":
            # Random oversampling
            subset = subset.sample(n=min_count, replace=True, random_state=42)
        balanced_dfs.append(subset)

    return pd.concat(balanced_dfs, ignore_index=True)


def create_train_val_test_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_column: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test split.

    Args:
        dataset: Input dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify_column: Column to stratify by
        random_state: Random state

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    if stratify_column and stratify_column in dataset.columns:
        # Stratified split
        from sklearn.model_selection import train_test_split

        # First split: train + (val+test)
        train_df, temp_df = train_test_split(
            dataset,
            train_size=train_ratio,
            stratify=dataset[stratify_column],
            random_state=random_state
        )

        # Second split: val and test
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adj,
            stratify=temp_df[stratify_column],
            random_state=random_state
        )
    else:
        # Random split
        n_total = len(dataset)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Shuffle
        dataset_shuffled = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_df = dataset_shuffled[:n_train]
        val_df = dataset_shuffled[n_train:n_train + n_val]
        test_df = dataset_shuffled[n_train + n_val:]

    return train_df, val_df, test_df


def save_dataset_splits(
    splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    output_dir: str,
    prefix: str = "dataset"
):
    """Save dataset splits to CSV files.

    Args:
        splits: Tuple of (train_df, val_df, test_df)
        output_dir: Output directory
        prefix: Prefix for filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = splits

    train_df.to_csv(f"{output_dir}/{prefix}_train.csv", index=False)
    val_df.to_csv(f"{output_dir}/{prefix}_val.csv", index=False)
    test_df.to_csv(f"{output_dir}/{prefix}_test.csv", index=False)

    print(f"Saved splits to {output_dir}:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")