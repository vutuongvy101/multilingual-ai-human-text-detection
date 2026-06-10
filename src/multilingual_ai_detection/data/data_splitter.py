"""Data splitting utilities."""

import random
from typing import List, Tuple

from datasets import Dataset, DatasetDict
from .data_loader import stable_prompt_id


def split_by_ratio(
    rows: List[dict],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split data by ratio at the prompt level.

    Args:
        rows: List of data records
        seed: Random seed
        train_ratio: Training split ratio
        val_ratio: Validation split ratio

    Returns:
        Tuple of (train_rows, val_rows, test_rows)
    """
    rows = rows.copy()
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]
    return train_rows, val_rows, test_rows


def create_labeled_examples(split_data: List[dict]) -> Dataset:
    """Flatten QA pairs into labeled examples for classification.

    Args:
        split_data: List of QA records

    Returns:
        HuggingFace Dataset with text, label, lang, gen, prompt_id columns
    """
    texts, labels, langs, gens, prompt_ids = [], [], [], [], []
    for row in split_data:
        lang = row.get("lang", "en")
        q = row.get("question", "")
        pid = stable_prompt_id(lang, q)
        human = row.get("human_answers") or []
        ai = row.get("ai_answers") or []
        if not human or not ai:
            continue
        h = human[0] if isinstance(human[0], str) else str(human[0])
        a = ai[0] if isinstance(ai[0], str) else str(ai[0])
        texts.extend([h, a])
        labels.extend([0, 1])  # 0=human, 1=machine
        langs.extend([lang, lang])
        gens.extend(["human", "oss"])  # oss = open-source model
        prompt_ids.extend([pid, pid])
    return Dataset.from_dict({
        "text": texts,
        "label": labels,
        "lang": langs,
        "gen": gens,
        "prompt_id": prompt_ids,
    })


def load_multilingual_dataset(
    data_dir: str = "data",
    seed: int = 42
) -> DatasetDict:
    """Load the complete multilingual dataset.

    Args:
        data_dir: Directory containing the JSONL files
        seed: Random seed for splitting

    Returns:
        DatasetDict with train, validation, test splits
    """
    from pathlib import Path
    from .data_loader import load_qa_jsonl

    data_dir = Path(data_dir)

    # Load all three languages
    en_rows = load_qa_jsonl(data_dir / "en_qa.jsonl", "en")
    zh_rows = load_qa_jsonl(data_dir / "zh_qa.jsonl", "zh")
    vi_rows = load_qa_jsonl(data_dir / "vi_qa.jsonl", "vi")

    print(f"EN: {len(en_rows)} QA pairs")
    print(f"ZH: {len(zh_rows)} QA pairs")
    print(f"VI: {len(vi_rows)} QA pairs")
    print(f"Total prompts: {len(en_rows) + len(zh_rows) + len(vi_rows)}")

    # Split each language separately
    en_train, en_val, en_test = split_by_ratio(en_rows, seed=seed)
    zh_train, zh_val, zh_test = split_by_ratio(zh_rows, seed=seed)
    vi_train, vi_val, vi_test = split_by_ratio(vi_rows, seed=seed)

    raw_train = en_train + zh_train + vi_train
    raw_val = en_val + zh_val + vi_val
    raw_test = en_test + zh_test + vi_test

    random.Random(seed).shuffle(raw_train)
    random.Random(seed).shuffle(raw_val)
    random.Random(seed).shuffle(raw_test)

    # Create labeled datasets
    train_data = create_labeled_examples(raw_train)
    val_data = create_labeled_examples(raw_val)
    test_data = create_labeled_examples(raw_test)

    return DatasetDict({
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    })