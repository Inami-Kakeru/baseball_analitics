"""
第1層の共通ユーティリティ。
"""

from __future__ import annotations

import pandas as pd


EVENT_BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}


def event_to_bases(event: str) -> int:
    """打球結果文字列を塁打数に変換。未知は0。"""
    if pd.isna(event):
        return 0
    return EVENT_BASES.get(str(event).lower(), 0)


def long_hit_score(bases: int) -> int:
    """V = max(bases - 1, 0)"""
    return max(bases - 1, 0)


def make_ev_la_bins(
    df: pd.DataFrame, ev_bin: float = 2.0, la_bin: float = 2.0
) -> pd.DataFrame:
    """EV/LAをビニングしてビンインデックスを付与。"""
    df = df.copy()
    df["ev_bin"] = (df["launch_speed"] // ev_bin * ev_bin).astype(float)
    df["la_bin"] = (df["launch_angle"] // la_bin * la_bin).astype(float)
    return df


def assign_high_value_label(
    df: pd.DataFrame,
    value_map: pd.DataFrame,
    top_quantile: float = 0.15,
) -> pd.DataFrame:
    """
    value_mapの上位q%セルを高価値帯域とみなし、該当打球にy=1を付与。
    """
    df_bins = make_ev_la_bins(df)
    threshold = value_map["E_V"].quantile(1 - top_quantile)
    high_cells = value_map.loc[value_map["E_V"] >= threshold, ["ev_bin", "la_bin"]]
    high_set = set(map(tuple, high_cells.values))

    def is_high(row: pd.Series) -> int:
        return int((row["ev_bin"], row["la_bin"]) in high_set)

    df_bins["y"] = df_bins.apply(is_high, axis=1)
    return df_bins

