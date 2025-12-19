"""
Step 1: 打球結果から長打スコアVを付与する。
- 入力: data/processed/bip.csv
- 出力: data/processed/bip_with_v_and_y.csv (yは後工程で付与)
"""

from __future__ import annotations

import pandas as pd

from utils import event_to_bases, long_hit_score


IN_PATH = "data/processed/bip.csv"
OUT_PATH = "data/processed/bip_with_v_and_y.csv"


def add_long_hit_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bases"] = df["events"].map(event_to_bases).fillna(0).astype(int)
    df["V"] = df["bases"].map(long_hit_score)
    return df


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df_scored = add_long_hit_score(df)
    df_scored.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df_scored)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()

