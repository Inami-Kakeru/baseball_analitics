"""
Layer 2用特徴量セット確定: Layer 2に渡す最終特徴量セットを確定する。
- 入力: data/output/feature_classification.csv
- 出力: data/output/candidate_features.csv（Layer2に渡す候補）
"""

from __future__ import annotations

import pandas as pd


CLASSIFICATION_PATH = "data/output/feature_classification.csv"
OUT_PATH = "data/output/candidate_features.csv"


def main() -> None:
    classified = pd.read_csv(CLASSIFICATION_PATH)
    # A/B/C を候補（Dは除外）
    candidates = classified[classified["category"].isin(["A", "B", "C"])].copy()
    # A→B→C順で優先
    order = {"A": 0, "B": 1, "C": 2}
    candidates["priority"] = candidates["category"].map(order).fillna(99).astype(int)
    candidates = candidates.sort_values(["priority", "feature"]).drop(columns=["priority"])
    candidates.to_csv(OUT_PATH, index=False)
    print(f"exported {len(candidates)} candidate features -> {OUT_PATH}")


if __name__ == "__main__":
    main()

