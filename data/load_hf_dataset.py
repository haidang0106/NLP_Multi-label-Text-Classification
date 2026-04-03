"""
Download hierarchical_multi_label_dataset from HuggingFace
and save to data/raw_data.json (compatible with existing pipeline).
"""

import json
from pathlib import Path

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_FILE = DATA_DIR / "raw_data.json"


def download_and_save(repo_id: str = "dat7505/hierarchical_multi_label_dataset",
                     split: str = "train",
                     output_path: Path = RAW_FILE) -> None:
    """Load dataset from HuggingFace and save as JSON array."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {repo_id} (split={split})")
    ds = load_dataset(repo_id, split=split)
    print(f"Loaded {len(ds):,} rows")

    records = []
    for row in ds:
        records.append({
            "url":       row.get("url", ""),
            "title":     row.get("title", ""),
            "content":   row.get("content", ""),
            "label_l1":  row.get("label_l1", ""),
            "label_l2":  row.get("label_l2", ""),
            "label_l3":  row.get("label_l3", ""),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records):,} articles to {output_path}")


if __name__ == "__main__":
    download_and_save()
