"""
単変量スクリーニング（Layer1.5）:
- y1: P(y1=1)
- 不利条件窓: P(y1=1 | is_adverse=True)  ※ y3を作らない
- y2: E(V | y1=1)

設計:
- 各変数を10分位に分割
- 分位ごとに確率/期待値を算出
- 条件別（pitch/plate_z/count/選手）で再計算
"""

from __future__ import annotations

import pandas as pd
import numpy as np


IN_PATH = "data/intermediate/contact_quality.csv"
OUT_PATH = "data/output/screening_results.csv"

STRATA_COLS = ["pitch_group", "plate_z_group", "count_group", "player_id"]


def _qcut10(s: pd.Series) -> pd.Series:
    # duplicates='drop' でユニーク値不足に耐える（分位数が10未満になる場合あり）
    return pd.qcut(s, q=10, labels=False, duplicates="drop") + 1


def _segment_iter(df: pd.DataFrame):
    yield ("overall", df)
    for col in STRATA_COLS:
        if col in df.columns:
            for v in sorted(df[col].dropna().unique()):
                yield (f"{col}={v}", df[df[col] == v])


def _screen_one(
    df: pd.DataFrame,
    feature: str,
    target: str,
    value_col: str,
) -> pd.DataFrame:
    d = df[[feature, value_col]].dropna().copy()
    if len(d) < 30:
        return pd.DataFrame()

    try:
        d["decile"] = _qcut10(d[feature])
    except Exception:
        return pd.DataFrame()

    g = d.groupby("decile", dropna=True)[value_col].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "metric", "count": "n"})
    g.insert(0, "feature", feature)
    g.insert(0, "target", target)
    return g


def run_screening(df: pd.DataFrame) -> pd.DataFrame:
    # 必須列
    for c in ("y1", "y2", "is_adverse"):
        if c not in df.columns:
            raise KeyError(f"'{c}' がありません。00_prepare_layer1_5.py → 01 → 02 の順で実行してください。")

    # 数値特徴量のみ（y/y1/y2/Vやビン系は除外）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"y", "y1", "y2", "V", "ev_bin", "la_bin"}
    features = [
        c
        for c in numeric_cols
        if c not in exclude and not c.startswith("ev_bin") and not c.startswith("la_bin")
    ]

    out = []
    for seg_name, seg_df in _segment_iter(df):
        # y1: 全体（P(y1=1)）
        for f in features:
            tmp = _screen_one(seg_df, f, target="y1", value_col="y1")
            if len(tmp):
                tmp.insert(0, "segment", seg_name)
                out.append(tmp)

        # y1 under adverse (y3相当): 目的変数はy1のまま、観測窓を切る
        adv = seg_df[seg_df["is_adverse"] == True]  # noqa: E712
        for f in features:
            tmp = _screen_one(adv, f, target="y1|adverse", value_col="y1")
            if len(tmp):
                tmp.insert(0, "segment", seg_name)
                out.append(tmp)

        # y2: y1==1内のE[V]
        hvz = seg_df[seg_df["y1"] == 1]
        for f in features:
            tmp = _screen_one(hvz, f, target="y2=E[V|y1=1]", value_col="y2")
            if len(tmp):
                tmp.insert(0, "segment", seg_name)
                out.append(tmp)

    if not out:
        return pd.DataFrame(columns=["segment", "target", "feature", "decile", "metric", "n"])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    df = pd.read_csv(IN_PATH)
    results = run_screening(df)
    results.to_csv(OUT_PATH, index=False)
    print(f"saved screening results -> {OUT_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()

