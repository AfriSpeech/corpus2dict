# corpus2nouns

> **High-throughput engine for extracting, cleaning, and reducing noun phrases from massive text corpora and instruction fine-tuning datasets.**

`corpus2nouns` is designed to process multi-million row datasets (Parquet, JSONL, Hugging Face Hub datasets) at scale, stripping out code, equations, and structured syntax, extracting linguistically valid noun chunks with spaCy, and optionally reducing them to pure, deduplicated nominal concepts.

---

## Key Capabilities

- **Deep Structured Text & Code Cleaning**:
  - Drops fenced code blocks (` ```...``` `), inline backticks (`` `user_id` ``), LaTeX formulas (`$...$`, `$$...$$`), HTML/XML tags, file paths, URLs, and CLI command lines.
  - Automatically filters snake_case (`api_key`), camelCase (`userId`), code syntax operators (`{`, `}`, `[`, `]`, `()`, `=>`), file extensions (`.py`, `.ts`, `.json`), and reserved programming keywords (`function`, `return`, `def`, `class`, etc.).
- **Boundary Article & Stopword Stripping**:
  - Removes definite articles (`the`), indefinite articles (`a`, `an`), demonstratives, and pronouns from chunk boundaries (*"the renewable energy"* &rarr; *"renewable energy"*).
- **Adjective & Determiner Dropping (Pure Noun Reduction)**:
  - Strips modifying adjectives and determiners (*"clean drinking water"* &rarr; *"drinking water"*, *"crucial role"* &rarr; *"role"*), deduplicating variants and summing their frequencies.
- **High-Throughput Parallel Pipeline**:
  - Multi-process parallel worker architecture built for multicore servers and cloud instances.
  - Granular chunk caching (`.json.gz`) enabling instant checkpoint resumption if interrupted.
- **Hugging Face Hub Integration**:
  - Automatically compiles multi-subset Parquet files with YAML frontmatter dataset cards for multi-config loading via `datasets`.

---

## Installation

```bash
pip install corpus2nouns
python -m spacy download en_core_web_sm
```

---

## Quickstart

### 1. Python API

```python
from corpus2nouns import strip_structured_text, extract_from_text, reduce_to_pure_nouns

raw_text = """
We deployed the new artificial intelligence model. 
Please update `api_key` in `config.py` and restart the backend server.
"""

# 1. Clean structured code and syntax
cleaned = strip_structured_text(raw_text)

# 2. Extract noun phrases (articles and stopwords stripped)
phrases = extract_from_text(cleaned)
print(phrases)
# Output: ['artificial intelligence model', 'backend server']

# 3. Reduce adjectives to pure nouns / compound nouns
pairs = [(p, 10) for p in phrases]
pure_nouns = reduce_to_pure_nouns(pairs)
print(pure_nouns)
# Output: Counter({'intelligence model': 10, 'backend server': 10})
```

### 2. Command Line Interface (CLI)

#### Extract from a Dataset Directory

```bash
# Extract noun phrases across all subset directories
corpus2nouns extract /path/to/dataset_dir \
    --output-dir ./extracted_nouns \
    --workers 16
```

#### Extract & Reduce to Pure Nouns Directly

```bash
corpus2nouns extract /path/to/dataset_dir \
    --output-dir ./pure_nouns \
    --workers 16 \
    --reduce
```

#### Publish directly to Hugging Face

```bash
corpus2nouns publish ./pure_nouns \
    --repo-id AfriSpeech/SFT-base-en-nouns
```

---

## License

Apache License 2.0. Developed by [AfriSpeech](https://github.com/AfriSpeech).
