"""Publisher module for creating multi-subset dataset cards and uploading to Hugging Face Hub."""

import os
import yaml
import pyarrow.parquet as pq
from huggingface_hub import HfApi

def generate_hf_readme(data_dir: str, repo_id: str) -> str:
    """Generate a comprehensive README.md with YAML metadata for all subsets."""
    subsets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "all"])
    
    configs = [
        {
            "config_name": "default",
            "data_files": [{"split": "train", "path": "data/all/train-*"}]
        },
        {
            "config_name": "all",
            "data_files": [{"split": "train", "path": "data/all/train-*"}]
        }
    ]
    for s in subsets:
        configs.append({
            "config_name": s,
            "data_files": [{"split": "train", "path": f"data/{s}/train-*"}]
        })
        
    frontmatter = {
        "configs": configs,
        "language": ["en"],
        "license": "apache-2.0",
        "task_categories": ["feature-extraction", "token-classification"],
        "tags": [
            "noun-phrases",
            "lexicon",
            "nlp",
            "spacy",
            "corpus2nouns",
            "vocabulary",
            "frequencies"
        ]
    }
    
    yaml_text = yaml.dump(frontmatter, sort_keys=False)
    
    # Calculate stats
    stats_rows = []
    all_dir = os.path.join(data_dir, "all")
    if os.path.exists(all_dir):
        all_files = [os.path.join(all_dir, f) for f in os.listdir(all_dir) if f.endswith(".parquet")]
        total_rows = sum(pq.read_metadata(f).num_rows for f in all_files)
        total_size = sum(os.path.getsize(f) for f in all_files) / (1024**2)
        stats_rows.append(("all (default deduped)", total_rows, total_size))
        
    for s in subsets:
        s_dir = os.path.join(data_dir, s)
        s_files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith(".parquet")]
        r = sum(pq.read_metadata(f).num_rows for f in s_files)
        sz = sum(os.path.getsize(f) for f in s_files) / (1024**2)
        stats_rows.append((s, r, sz))
        
    table_md = "| Subset Config | Unique Nouns | File Size |\n| :--- | :--- | :--- |\n"
    for name, r, sz in stats_rows:
        table_md += f"| `{name}` | {r:,} | {sz:.2f} MB |\n"
        
    body = f"""# {repo_id}

Extracted and deduplicated English nouns and nominal compounds produced by [**`corpus2nouns`**](https://github.com/AfriSpeech/corpus2nouns).

---

## Dataset Subsets & Statistics

{table_md}

---

## Schema

### Global Deduplicated View (`default` / `all`)
```json
{{
  "phrase": "space station",
  "frequency": 24510,
  "subset_count": 8,
  "subsets": ["industry_aerospace", "infinity_domain_commonsense"]
}}
```

### Subset Level
```json
{{
  "phrase": "space station",
  "frequency": 8943
}}
```

---

## How to Load

```python
from datasets import load_dataset, get_dataset_config_names

# 1. Load global deduped across all subsets
ds = load_dataset("{repo_id}")
# or streaming
ds = load_dataset("{repo_id}", split="train", streaming=True)

# 2. Load specific subset
ds_sub = load_dataset("{repo_id}", "industry_aerospace")

# 3. Check configs
configs = get_dataset_config_names("{repo_id}")
```
"""
    return "---\n" + yaml_text + "---\n\n" + body

def publish_dataset(output_dir: str, repo_id: str, token: str | None = None):
    """Upload dataset directory and README to Hugging Face Hub."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    
    print(f"Ensuring repository exists: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    data_dir = os.path.join(output_dir, "data") if os.path.exists(os.path.join(output_dir, "data")) else output_dir
    readme_content = generate_hf_readme(data_dir, repo_id)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print(f"Uploading files to Hugging Face Hub ({repo_id})...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish dataset via corpus2nouns"
    )
    print(f"Dataset live at: https://huggingface.co/datasets/{repo_id}")
