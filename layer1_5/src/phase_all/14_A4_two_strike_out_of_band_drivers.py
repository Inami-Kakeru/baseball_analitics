from __future__ import annotations

"""
A4: two_strikeで帯域外に出る“崩れ要因”を2〜3変数で特定（仮説探索）

目的変数:
- out_band（帯域外フラグ）

条件:
- two_strike == True のみ

候補変数（存在するものだけ、優先順で最大3つ）:
1) swing_length
2) swing_path_tilt
3) intercept_ball_minus_batter_pos_y_inches
4) intercept_ball_minus_batter_pos_x_inches

方法:
- 各featureを benchmark_group overall 分布で10分位境界固定
- decile × P(out_band=1) を group別（ohtani / benchmark_group）に算出

出力:
- data/output/A4_two_strike_out_of_band_drivers.csv
  columns: feature, group, decile, n, p_out_band
- 図: data/output/plots/A4_two_strike_out_of_band_drivers.png

解釈制約:
- 因果は禁止（帯域外になりやすい特徴が見える、まで）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._utils_phase import assign_bins_by_edges, ensure_columns, q_edges_from_series


BASE_TABLE = Path("data/output/phase_all_base_table.csv")
OUT_CSV = Path("data/output/A4_two_strike_out_of_band_drivers.csv")
OUT_PNG = Path("data/output/plots/A4_two_strike_out_of_band_drivers.png")

OHTANI_ID = "ohtani"

FEATURE_PRIORITY = [
    "swing_length",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_y_inches",
    "intercept_ball_minus_batter_pos_x_inches",
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("decile", dropna=True)
        .agg(n=("out_band", "count"), p_out_band=("out_band", "mean"))
        .reset_index()
        .sort_values("decile")
    )
    g["decile"] = g["decile"].astype(int)
    return g


def main() -> None:
    df = pd.read_csv(BASE_TABLE)
    ensure_columns(df, ["player_id", "two_strike", "out_band"])

    # two_strikeのみ
    df2 = df[df["two_strike"] == True].copy()  # noqa: E712

    available = [f for f in FEATURE_PRIORITY if f in df.columns]
    selected = available[:3]
    if len(selected) == 0:
        raise KeyError("A4候補変数が1つも見つかりません（swing/intercept系の列を確認）。")

    rows = []
    # decile境界は benchmark_group overall（全行）で固定
    for feat in selected:
        edges = q_edges_from_series(df[feat].dropna(), q=10)
        df2["decile"] = assign_bins_by_edges(df2[feat], edges)

        for gname, gdf in {
            "ohtani": df2[df2["player_id"] == OHTANI_ID],
            "benchmark_group": df2,
        }.items():
            s = gdf.dropna(subset=["decile", "out_band"]).copy()
            agg = summarize(s)
            agg.insert(0, "group", gname)
            agg.insert(0, "feature", feat)
            rows.append(agg)

    out = pd.concat(rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # 図: featureごとにsubplot（ohtani vs benchmark_group）
    import matplotlib.pyplot as plt

    feats = selected
    fig, axes = plt.subplots(len(feats), 1, figsize=(8, 3 * len(feats)), sharex=True)
    if len(feats) == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats):
        sub_o = out[(out["feature"] == feat) & (out["group"] == "ohtani")].sort_values("decile")
        sub_b = out[(out["feature"] == feat) & (out["group"] == "benchmark_group")].sort_values("decile")
        ax.plot(sub_o["decile"], sub_o["p_out_band"], marker="o", label="ohtani")
        ax.plot(sub_b["decile"], sub_b["p_out_band"], marker="o", label="benchmark_group")
        ax.set_title(f"{feat} | two_strike: P(out_band=1)")
        ax.set_ylabel("P(out_band=1)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("decile (fixed on benchmark_group overall)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


