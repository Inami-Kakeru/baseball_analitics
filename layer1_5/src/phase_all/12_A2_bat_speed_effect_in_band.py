from __future__ import annotations

"""
A2: 帯域内に限定して、bat_speed の“出力レバー”を定量化

仕様:
- サブセット: in_band==True
- bat_speed を benchmark_group overall 分布で10分位境界固定（他へ適用、切り直し禁止）
- 集計: decileごとの mean_primary_kpi, P(y1=1), n
- group: ohtani / benchmark_group
- condition: overall / two_strike

出力:
- data/output/A2_bat_speed_effect_in_band.csv
- data/output/plots/A2_bat_speed_effect_in_band.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._utils_phase import assign_bins_by_edges, ensure_columns, q_edges_from_series


BASE_TABLE = Path("data/output/phase_all_base_table.csv")
OUT_CSV = Path("data/output/A2_bat_speed_effect_in_band.csv")
OUT_PNG = Path("data/output/plots/A2_bat_speed_effect_in_band.png")

OHTANI_ID = "ohtani"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("decile", dropna=True)
        .agg(n=("primary_kpi", "count"), mean_primary_kpi=("primary_kpi", "mean"), py1=("y1", "mean"))
        .reset_index()
        .sort_values("decile")
    )
    g["decile"] = g["decile"].astype(int)
    return g


def main() -> None:
    df = pd.read_csv(BASE_TABLE)
    ensure_columns(df, ["player_id", "two_strike", "in_band", "bat_speed", "primary_kpi", "y1"])

    # サブセット: in_band
    df_in = df[df["in_band"] == True].copy()  # noqa: E712

    # decile境界は benchmark_group overall で固定
    bench_over = df_in.copy()
    bench_over = bench_over.dropna(subset=["bat_speed"])
    edges = q_edges_from_series(bench_over["bat_speed"], q=10)
    df_in["decile"] = assign_bins_by_edges(df_in["bat_speed"], edges)

    groups: dict[str, pd.DataFrame] = {
        "ohtani": df_in[df_in["player_id"] == OHTANI_ID],
        "benchmark_group": df_in,
    }

    rows = []
    for gname, gdf in groups.items():
        for cond, sdf in [
            ("overall", gdf),
            ("two_strike", gdf[gdf["two_strike"] == True]),  # noqa: E712
        ]:
            s = sdf.dropna(subset=["decile", "primary_kpi", "y1"]).copy()
            agg = summarize(s)
            agg.insert(0, "condition", cond)
            agg.insert(0, "group", gname)
            rows.append(agg)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # 図: 2×2（group×condition）で線を描く（回帰なし）
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=False)
    for i, gname in enumerate(["ohtani", "benchmark_group"]):
        for j, cond in enumerate(["overall", "two_strike"]):
            ax = axes[i, j]
            sub = out[(out["group"] == gname) & (out["condition"] == cond)].sort_values("decile")
            ax.plot(sub["decile"], sub["mean_primary_kpi"], marker="o", label="mean_primary_kpi")
            ax2 = ax.twinx()
            ax2.plot(sub["decile"], sub["py1"], marker="o", color="tab:orange", label="py1")
            ax.set_title(f"{gname} | {cond}")
            ax.set_xlabel("bat_speed decile (fixed on benchmark_group overall)")
            ax.set_ylabel("mean_primary_kpi")
            ax2.set_ylabel("P(y1=1)")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


