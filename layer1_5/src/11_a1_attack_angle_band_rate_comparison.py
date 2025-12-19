"""
A-1: 大谷 vs 他選手｜attack_angle 帯域“逸脱率”比較

目的:
- 固定された attack_angle 帯域（lower_bound/upper_bound）を用い、
  各グループの帯域内率を overall / two_strike で比較する。
- 再学習・再最適化・回帰は禁止。

入力:
- data/intermediate/contact_quality.csv
- data/output/ohtani_attack_angle_band_rate.csv  (lower_bound, upper_bound)

出力:
- data/output/attack_angle_band_rate_comparison.csv
- data/output/plots/attack_angle_band_rate_comparison.png（任意：棒グラフ）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CONTACT_PATH = Path("data/intermediate/contact_quality.csv")
BAND_PATH = Path("data/output/ohtani_attack_angle_band_rate.csv")

OUT_CSV = Path("data/output/attack_angle_band_rate_comparison.csv")
OUT_PNG = Path("data/output/plots/attack_angle_band_rate_comparison.png")

OHTANI_ID = "ohtani"
TOP_EV_PCT = 0.10


def load_band() -> tuple[float, float]:
    b = pd.read_csv(BAND_PATH)
    lower = float(b["lower_bound"].iloc[0])
    upper = float(b["upper_bound"].iloc[0])
    return lower, upper


def in_band_rate(df: pd.DataFrame, lower: float, upper: float) -> float:
    if len(df) == 0:
        return float("nan")
    m = (df["attack_angle"] >= lower) & (df["attack_angle"] <= upper)
    return float(m.mean())


def power_hitter_ids(df: pd.DataFrame, top_pct: float = TOP_EV_PCT) -> set[str]:
    # player_id別 平均launch_speedで上位10%をパワーヒッター群とする（定義は固定・単純）
    g = df.groupby("player_id", dropna=True)["launch_speed"].mean().sort_values(ascending=False)
    k = max(1, int(np.ceil(len(g) * top_pct)))
    return set(g.head(k).index.astype(str).tolist())


def main() -> None:
    df = pd.read_csv(CONTACT_PATH)
    required = ["player_id", "attack_angle", "strikes", "launch_speed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["player_id", "attack_angle", "strikes", "launch_speed"]).copy()
    df["strikes"] = df["strikes"].astype(int)

    lower, upper = load_band()

    pow_ids = power_hitter_ids(df, TOP_EV_PCT)

    groups: dict[str, pd.DataFrame] = {
        "ohtani": df[df["player_id"] == OHTANI_ID],
        "league": df,  # 全体平均
        "power_hitters_top10_ev": df[df["player_id"].isin(pow_ids)],
    }

    rows = []
    for gname, gdf in groups.items():
        for cond, sdf in [
            ("overall", gdf),
            ("two_strike", gdf[gdf["strikes"] == 2]),
        ]:
            rows.append(
                {
                    "player_group": gname,
                    "condition": cond,
                    "n": int(len(sdf)),
                    "in_band_rate": in_band_rate(sdf, lower, upper),
                }
            )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved csv -> {OUT_CSV}")

    # 図（任意）
    import matplotlib.pyplot as plt

    pivot = out.pivot(index="player_group", columns="condition", values="in_band_rate")
    pivot = pivot.reindex(["ohtani", "league", "power_hitters_top10_ev"])
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("in_band_rate")
    ax.set_title("attack_angle band in-rate comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


