"""
B-1: attack_angle × bat_speed｜2次元ヒートマップ

目的:
- attack_angle と bat_speed の組み合わせで P(y1=1) を可視化する。
- 対象: ohtani / league
- 回帰・最適ゾーン断定は禁止。構造の可視化のみ。

仕様:
- attack_angle を 2deg ビン
- bat_speed を 2mph ビン
- 各セルで P(y1=1), count を算出
- count < MIN_COUNT は NaN

出力:
- CSV（任意だが出す）:
  - data/output/attack_angle_bat_speed_heatmap_ohtani.csv
  - data/output/attack_angle_bat_speed_heatmap_league.csv
- 図:
  - data/output/plots/attack_angle_bat_speed_heatmap_ohtani.png
  - data/output/plots/attack_angle_bat_speed_heatmap_league.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")

MIN_COUNT = 30
ANGLE_BIN = 2.0
BAT_BIN = 2.0

OHTANI_ID = "ohtani"

OUT_DIR = Path("data/output")
PLOT_DIR = Path("data/output/plots")


def bin_floor(x: pd.Series, step: float) -> pd.Series:
    return (np.floor(x.astype(float) / step) * step).astype(float)


def build_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["attack_angle", "bat_speed", "y1"]).copy()
    d["y1"] = d["y1"].astype(int)
    d["attack_angle_bin"] = bin_floor(d["attack_angle"], ANGLE_BIN)
    d["bat_speed_bin"] = bin_floor(d["bat_speed"], BAT_BIN)

    g = (
        d.groupby(["attack_angle_bin", "bat_speed_bin"])["y1"]
        .agg(py1="mean", n="count")
        .reset_index()
    )
    g.loc[g["n"] < MIN_COUNT, "py1"] = np.nan
    return g


def save_plot(hm: pd.DataFrame, title: str, out_png: Path) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    pivot = hm.pivot(index="attack_angle_bin", columns="bat_speed_bin", values="py1")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot.sort_index(ascending=False), cmap="magma", vmin=0, vmax=np.nanmax(hm["py1"]))
    plt.xlabel("bat_speed bin (mph)")
    plt.ylabel("attack_angle bin (deg)")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(IN_PATH)
    required = ["player_id", "attack_angle", "bat_speed", "y1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    league = df.copy()
    oht = df[df["player_id"] == OHTANI_ID].copy()

    for name, sdf in [("league", league), ("ohtani", oht)]:
        hm = build_heatmap(sdf)
        out_csv = OUT_DIR / f"attack_angle_bat_speed_heatmap_{name}.csv"
        out_png = PLOT_DIR / f"attack_angle_bat_speed_heatmap_{name}.png"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        hm.to_csv(out_csv, index=False)
        save_plot(hm, title=f"attack_angle × bat_speed: P(y1=1) ({name})", out_png=out_png)
        print(f"saved csv -> {out_csv}")
        print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    main()


