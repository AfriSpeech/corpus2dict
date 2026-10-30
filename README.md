# corpus2dict

> **High-throughput engine for extracting, cleaning, reducing, and translating noun lexicons and Gloss-MT dictionaries from massive text corpora and instruction fine-tuning datasets.**

`corpus2dict` is designed to process multi-million row datasets (Parquet, JSONL, Hugging Face Hub datasets) at scale, stripping out code, equations, and structured syntax, extracting linguistically valid noun chunks with spaCy, reducing them to pure nominal concepts, and translating them with LLMs (e.g. Gemini) to produce offline bilingual dictionaries compatible with [**`AfriSpeech/afrigloss-mt`**](https://github.com/AfriSpeech/afrigloss-mt).

---

## Key Capabilities

- **Deep Structured Text & Code Cleaning**:
  - Drops fenced code blocks (` ```...``` `), inline backticks (`` `user_id` ``), LaTeX formulas (`$...$`, `$$...$$`), HTML/XML tags, file paths, URLs, and CLI command lines.
  - Automatically filters snake_case (`api_key`), camelCase (`userId`), code syntax operators (`{`, `}`, `[`, `]`, `()`, `=>`), file extensions (`.py`, `.ts`, `.json`), and reserved programming keywords (`function`, `return`, `def`, `class`, etc.).
- **Boundary Article & Stopword Stripping**:
  - Removes definite articles (`the`), indefinite articles (`a`, `an`), demonstratives, and pronouns from chunk boundaries (*"the renewable energy"* &rarr; *"renewable energy"*).
- **Adjective & Determiner Dropping (Pure Noun Reduction)**:
  - Strips modifying adjectives and determiners (*"clean drinking water"* &rarr; *"drinking water"*, *"crucial role"* &rarr; *"role"*), deduplicating variants and summing their frequencies.
- **Data-Driven Casing Classification (Proper vs Common Nouns)**:
  - Analyzes empirical orthography outside sentence boundaries to accurately tag common nouns vs proper entities and all-caps acronyms (`FDA`, `HIV`, `NASA`).
- **Gloss-MT Ready Schema**:
  - Exports standard columns (`source_phrase`, `target_phrase`, `translated`, `confidence`, `noun_type`, `frequency`) ready for drop-in use with `GlossaryDictionary.load_tsv()`.
- **High-Throughput Parallel Pipeline**:
  - Multi-process worker architecture and async batch translation designed for cloud GPU/CPU servers.
  - Granular chunk caching (`.json.gz`, SQLite) enabling instant checkpoint resumption if interrupted.
- **Hugging Face Hub Integration**:
  - Automatically compiles multi-subset Parquet files with YAML frontmatter dataset cards for multi-config loading via `datasets`.

---

## Installation

```bash
pip install corpus2dict
python -m spacy download en_core_web_sm
```

---

## Quickstart

### 1. Python API

```python
from corpus2dict import strip_structured_text, extract_from_text, reduce_to_pure_nouns

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

#### Extract & Reduce from a Dataset Directory

```bash
# Extract and reduce to pure nouns with multi-worker parallelism
corpus2dict extract /path/to/dataset_dir \
    --output-dir ./extracted_dict \
    --workers 16 \
    --reduce
```

#### Publish directly to Hugging Face

```bash
corpus2dict publish ./extracted_dict \
    --repo-id AfriSpeech/SFT-base-en-dict
```

---

## Using Dictionaries with Afri-Gloss-MT

```python
from afrigloss.dictionary import GlossaryDictionary
from datasets import load_dataset

# Load from Hugging Face
ds = load_dataset("AfriSpeech/SFT-base-en-dict", split="train", streaming=True)

# Build offline glossary
glossary = GlossaryDictionary()
for row in ds:
    glossary.add(row["source_phrase"], row["target_phrase"], row["confidence"])

# Query
print(glossary.lookup("renewable energy"))
```

---

## License

Apache License 2.0. Developed by [AfriSpeech](https://github.com/AfriSpeech).
