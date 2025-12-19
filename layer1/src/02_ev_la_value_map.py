"""
Step 2: EV×LAグリッドでE[V]を集計し、価値マップを生成する（選手別対応）。
- 入力: data/processed/bip_with_v_and_y.csv
- 出力: data/processed/value_map.csv（EV×LAビンごとの平均Vと件数、player_id付き）
"""

from __future__ import annotations

import pandas as pd

from utils import make_ev_la_bins


IN_PATH = "data/processed/bip_with_v_and_y.csv"
OUT_PATH = "data/processed/value_map.csv"
DEFAULT_BIN_SIZE = {"ev_bin": 2.0, "la_bin": 2.0}
MIN_COUNT = 2


def build_value_map(
    df: pd.DataFrame,
    ev_bin: float = 2.0,
    la_bin: float = 2.0,
    min_count: int = MIN_COUNT,
) -> pd.DataFrame:
    """選手別に価値マップを生成"""
    df_binned = make_ev_la_bins(df, ev_bin=ev_bin, la_bin=la_bin)
    
    # player_idがある場合は選手別に集計、ない場合は全体で集計
    group_cols = ["ev_bin", "la_bin"]
    if "player_id" in df_binned.columns:
        group_cols = ["player_id"] + group_cols
    
    grouped = (
        df_binned.groupby(group_cols, dropna=True)["V"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "E_V", "count": "n"})
    )
    # サンプルが少ないセルは無効化（E_VをNaNに）
    grouped.loc[grouped["n"] < min_count, "E_V"] = pd.NA
    return grouped


def main() -> None:
    df = pd.read_csv(IN_PATH)
    value_map = build_value_map(df, **DEFAULT_BIN_SIZE, min_count=MIN_COUNT)
    value_map.to_csv(OUT_PATH, index=False)
    if "player_id" in value_map.columns:
        n_players = value_map["player_id"].nunique()
        print(f"saved value map ({n_players} players) -> {OUT_PATH}")
    else:
        print(f"saved value map -> {OUT_PATH}")


if __name__ == "__main__":
    main()


