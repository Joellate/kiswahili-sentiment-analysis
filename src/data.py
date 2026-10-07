"""Loading the AfriSenti Kiswahili (swa) Twitter sentiment dataset.

Source: Muhammad et al. (2023), AfriSenti: A Twitter Sentiment Analysis
Benchmark for African Languages. Hugging Face: shmuhammad/AfriSenti-twitter-sentiment

The official train / validation / test splits are kept unchanged so results
are comparable with the published benchmark.
"""
from pathlib import Path

import pandas as pd

HF_PARQUET = (
    "https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment/"
    "resolve/refs%2Fconvert%2Fparquet/swa/{split}/0000.parquet"
)
SPLITS = ("train", "validation", "test")

# Label ids as defined in the dataset's ClassLabel feature.
LABELS = ["positive", "neutral", "negative"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_split(split: str, cache: bool = True) -> pd.DataFrame:
    """Return one split as a DataFrame with columns `text`, `label`, `label_name`.

    The first call downloads from Hugging Face and caches a CSV in data/.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = DATA_DIR / f"swa_{split}.csv"
    if cache and path.exists():
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.read_parquet(HF_PARQUET.format(split=split))
        df = df.rename(columns={"tweet": "text"})
        if cache:
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(path, index=False)
    df["label_name"] = df["label"].map(ID2LABEL)
    return df


def load_all(cache: bool = True) -> dict:
    """Return {"train": df, "validation": df, "test": df}."""
    return {split: load_split(split, cache) for split in SPLITS}
