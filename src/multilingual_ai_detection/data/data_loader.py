"""Data loading utilities."""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def stable_prompt_id(lang: str, question: str) -> str:
    """Create a stable ID for a prompt/question to ensure leakage-safe splitting.

    Args:
        lang: Language code (en/zh/vi)
        question: The question text

    Returns:
        SHA1 hash of the language-question pair
    """
    s = f"{lang}||{question}".strip().encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]


def load_qa_jsonl(path: Union[str, Path], lang: str) -> List[Dict]:
    """Load QA JSONL file from en_qa.jsonl, zh_qa.jsonl, or vi_qa.jsonl.

    Args:
        path: Path to the JSONL file
        lang: Language code

    Returns:
        List of QA records
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            human = obj.get("human_answers") or []
            ai = obj.get("ai_answers") or []
            if not q or not human or not ai:
                continue
            h = human[0] if isinstance(human[0], str) else str(human[0])
            a = ai[0] if isinstance(ai[0], str) else str(ai[0])
            rows.append({
                "question": q,
                "human_answers": [h],
                "ai_answers": [a],
                "index": obj.get("index") or obj.get("post_id"),
                "source": obj.get("source", f"{lang}_built"),
                "lang": lang,
            })
    return rows


def load_hc3_dataset(
    languages: Optional[List[str]] = None
) -> Dict[str, Dataset]:
    """Load HC3 datasets for specified languages.

    Args:
        languages: List of languages to load (en, zh). If None, loads all.

    Returns:
        Dictionary mapping language codes to datasets
    """
    if languages is None:
        languages = ["en", "zh"]

    datasets = {}
    for lang in languages:
        if lang == "en":
            # Load HC3 English
            dataset = load_dataset("Hello-SimpleAI/HC3", "all")
            # Use reddit_eli5 subset
            datasets["en"] = dataset["train"].filter(
                lambda x: x["source"] == "reddit_eli5"
            )
        elif lang == "zh":
            # Load HC3 Chinese
            dataset = load_dataset("Hello-SimpleAI/HC3-Chinese", "open_qa")
            datasets["zh"] = dataset["train"]

    return datasets