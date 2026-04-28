from functools import lru_cache
import os

import pandas as pd

from shared.column_map import COLUMN_MAP


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    path = os.getenv("DATASET_PATH", "./data/Reto_data_20251023_122206.parquet")
    df = pd.read_parquet(path)
    rename = {v: k for k, v in COLUMN_MAP.items() if v is not None and v in df.columns and v != k}
    df = df.rename(columns=rename)
    return df


def get_post(post_id: str) -> dict | None:
    if not post_id:
        return None
    df = load_dataset()
    row = df[df["post_id"].astype(str) == str(post_id)]
    return row.iloc[0].to_dict() if not row.empty else None