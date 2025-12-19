from __future__ import annotations

"""
共通前処理:
- contact_quality.csv を読み込み
- two_strike, in_band, out_band を追加
- primary_kpi を選択して列を作成（xSLG優先、なければV）
- phase_all_base_table.csv を出力

注意:
- 欠損は「分析ごとに必要列だけdrop」する方針なので、この段階では全体dropしない。
"""

from pathlib import Path

import pandas as pd

from ._utils_phase import Band, choose_primary_kpi_col, ensure_columns, read_band_from_single_source


CONTACT_PATH = Path("data/intermediate/contact_quality.csv")
BAND_SOURCE = Path("data/output/ohtani_attack_angle_band_rate.csv")  # 単一ソース
OUT_PATH = Path("data/output/phase_all_base_table.csv")


def main() -> None:
    df = pd.read_csv(CONTACT_PATH)
    ensure_columns(df, ["player_id", "attack_angle", "bat_speed", "strikes", "y1", "launch_speed"])

    band: Band = read_band_from_single_source(BAND_SOURCE)

    df["two_strike"] = (df["strikes"] == 2)
    df["in_band"] = (df["attack_angle"] >= band.lower) & (df["attack_angle"] <= band.upper)
    df["out_band"] = ~df["in_band"]

    kpi_col, kpi_label = choose_primary_kpi_col(df)
    df["primary_kpi"] = df[kpi_col]
    df["primary_kpi_source_col"] = kpi_col
    df["primary_kpi_label"] = kpi_label

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")
    print(f"primary KPI: {kpi_label} (from column '{kpi_col}')")
    print(f"band: [{band.lower}, {band.upper}] (single source: {BAND_SOURCE})")


if __name__ == "__main__":
    main()


