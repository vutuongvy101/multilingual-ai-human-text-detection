# Data Collection and Preparation

## Dataset Overview

The multilingual AI-human text detection dataset consists of question-answer pairs from three languages:

- **English**: 300 QA pairs from HC3 (reddit_eli5 subset)
- **Chinese**: 300 QA pairs from HC3-Chinese (open_qa subset)
- **Vietnamese**: 300 QA pairs from custom Reddit crawling

Each language has balanced human-written and AI-generated answers.

## Data Sources

### English (HC3)
- **Source**: [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- **Subset**: reddit_eli5 (informal, community-driven Q&A)
- **Rationale**: Natural human writing patterns, diverse topics

### Chinese (HC3-Chinese)
- **Source**: [HC3-Chinese dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)
- **Subset**: open_qa (general question-answering)
- **Rationale**: Closest equivalent to English HC3 for Chinese

### Vietnamese (Custom)
- **Source**: Reddit communities (r/vozforums, r/TroChuyenLinhTinh, r/VietNamNation, r/VietTalk)
- **Method**: Custom web crawling with quality filtering
- **Rationale**: No existing Vietnamese HC3 equivalent

## Data Collection Process

### Vietnamese Crawling Strategy

1. **Community Selection**
   - Target: Active Vietnamese Q&A communities
   - Criteria: High engagement, informal discussion

2. **Post Filtering**
   - Question detection using Vietnamese question patterns
   - Minimum score threshold (20+ upvotes)
   - Age limit (10 years max)

3. **Answer Selection**
   - Highest-scored top-level comment
   - Minimum length (30+ words)
   - Exclude deleted/removed content

4. **Quality Control**
   - Remove duplicates
   - Filter spam/low-quality content
   - Manual review of samples

### AI Generation

- **Model**: Qwen2.5-1.5B-Instruct
- **Prompt Template**: "Answer this question in {language} in 100-300 words: {question}"
- **Consistency**: Same model and template across all languages

## Data Format

### Raw JSONL Format
```json
{
  "question": "What is artificial intelligence?",
  "human_answers": ["AI is..."],
  "ai_answers": ["Artificial intelligence refers to..."],
  "index": "post_id_or_index",
  "source": "en_reddit_eli5",
  "lang": "en"
}
```

### Processed Dataset Format
```json
{
  "text": "AI is a field of computer science...",
  "label": 0,  // 0=human, 1=AI
  "lang": "en",
  "gen": "human",  // "human" or "oss" (open-source model)
  "prompt_id": "sha1_hash"
}
```

## Data Splitting

### Strategy
- **Prompt-level splitting**: Prevents leakage between train/val/test
- **Language-balanced**: Equal distribution across languages
- **Ratios**: 70% train, 15% validation, 15% test

### Leakage Prevention
- Same question never appears in multiple splits
- Stable prompt IDs ensure reproducible splits
- Cross-validation safe

## Quality Metrics

### Human Text Criteria
- Natural language patterns
- Contextual responses
- Variable length and style
- Community-validated (upvotes)

### AI Text Criteria
- Consistent generation quality
- Language-appropriate responses
- Controlled length (100-300 words)
- Same model across languages

## Dataset Statistics

| Language | Samples | Human | AI | Avg Length | Lexical Diversity |
|----------|---------|-------|----|------------|-------------------|
| English  | 600     | 300   | 300 | 148 words  | 0.78             |
| Chinese  | 600     | 300   | 300 | 142 words  | 0.76             |
| Vietnamese| 600    | 300   | 300 | 135 words  | 0.80             |

## Usage

```python
from multilingual_ai_detection.data import load_multilingual_dataset

# Load processed dataset
dataset = load_multilingual_dataset("data", seed=42)

print(f"Train: {len(dataset['train'])}")
print(f"Validation: {len(dataset['validation'])}")
print(f"Test: {len(dataset['test'])}")
```

## Future Improvements

- Expand to more languages
- Add domain-specific datasets
- Include more AI generators (GPT, Claude, etc.)
- Implement data augmentation
- Add quality scoring and filtering