"""
A-3: 個人内分散｜attack_angle のブレ比較

目的:
- attack_angle の分布のブレを、選手内分散として比較する。
- 対象: ohtani / league / power_hitters_top10_ev
- 条件: overall / two_strike（strikes==2）
- 因果主張・回帰は禁止。分布要約のみ。

入力:
- data/intermediate/contact_quality.csv

出力:
- data/output/attack_angle_variance_comparison.csv
- data/output/plots/attack_angle_variance_comparison.png（任意：箱ひげ図）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")
OUT_CSV = Path("data/output/attack_angle_variance_comparison.csv")
OUT_PNG = Path("data/output/plots/attack_angle_variance_comparison.png")

OHTANI_ID = "ohtani"
TOP_EV_PCT = 0.10


def power_hitter_ids(df: pd.DataFrame, top_pct: float = TOP_EV_PCT) -> set[str]:
    g = df.groupby("player_id", dropna=True)["launch_speed"].mean().sort_values(ascending=False)
    k = max(1, int(np.ceil(len(g) * top_pct)))
    return set(g.head(k).index.astype(str).tolist())


def summarize(vals: pd.Series) -> dict:
    vals = vals.dropna().astype(float)
    if len(vals) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "iqr": np.nan}
    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "iqr": float(q3 - q1),
    }


def main() -> None:
    df = pd.read_csv(IN_PATH)
    required = ["player_id", "attack_angle", "strikes", "launch_speed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["player_id", "attack_angle", "strikes", "launch_speed"]).copy()
    df["strikes"] = df["strikes"].astype(int)

    pow_ids = power_hitter_ids(df, TOP_EV_PCT)
    groups: dict[str, pd.DataFrame] = {
        "ohtani": df[df["player_id"] == OHTANI_ID],
        "league": df,
        "power_hitters_top10_ev": df[df["player_id"].isin(pow_ids)],
    }

    rows = []
    for gname, gdf in groups.items():
        for cond, sdf in [
            ("overall", gdf),
            ("two_strike", gdf[gdf["strikes"] == 2]),
        ]:
            s = summarize(sdf["attack_angle"])
            rows.append(
                {
                    "player_group": gname,
                    "condition": cond,
                    **s,
                }
            )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved csv -> {OUT_CSV}")

    # 図（任意）：箱ひげ（group×condition）
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    order = ["ohtani", "league", "power_hitters_top10_ev"]
    plot_rows = []
    for gname in order:
        for cond in ["overall", "two_strike"]:
            sdf = groups[gname]
            if cond == "two_strike":
                sdf = sdf[sdf["strikes"] == 2]
            vals = sdf["attack_angle"].dropna().astype(float).to_numpy()
            plot_rows.append(((gname, cond), vals))

    labels = [f"{g}\\n{c}" for (g, c), _ in plot_rows]
    data = [v for _, v in plot_rows]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("attack_angle")
    ax.set_title("attack_angle distribution (boxplot)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


