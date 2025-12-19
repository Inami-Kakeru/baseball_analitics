from __future__ import annotations

"""
A3: 帯域内に限定して、contact_efficiency（launch_speed / bat_speed）の特徴を翻訳

仕様:
- サブセット: in_band==True
- contact_efficiency を計算（bat_speed==0は除外）
- decile境界は benchmark_group overall 分布で固定
- 集計: decileごとの mean_primary_kpi, P(y1=1), n
- group: ohtani / benchmark_group
- condition: overall / two_strike

出力:
- data/output/A3_contact_efficiency_effect_in_band.csv
- data/output/plots/A3_contact_efficiency_effect_in_band.png

重要な注意（この分析の出力/解釈）:
- 因果は主張しない
- 「帯域内で当たり負けしにくい特徴として相関が見える」まで
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._utils_phase import assign_bins_by_edges, ensure_columns, q_edges_from_series, safe_div


BASE_TABLE = Path("data/output/phase_all_base_table.csv")
OUT_CSV = Path("data/output/A3_contact_efficiency_effect_in_band.csv")
OUT_PNG = Path("data/output/plots/A3_contact_efficiency_effect_in_band.png")

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
    ensure_columns(df, ["player_id", "two_strike", "in_band", "launch_speed", "bat_speed", "primary_kpi", "y1"])

    df_in = df[df["in_band"] == True].copy()  # noqa: E712
    df_in["contact_efficiency"] = safe_div(df_in["launch_speed"], df_in["bat_speed"])

    bench_over = df_in.dropna(subset=["contact_efficiency"]).copy()
    edges = q_edges_from_series(bench_over["contact_efficiency"], q=10)
    df_in["decile"] = assign_bins_by_edges(df_in["contact_efficiency"], edges)

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
            ax.set_xlabel("contact_efficiency decile (fixed on benchmark_group overall)")
            ax.set_ylabel("mean_primary_kpi")
            ax2.set_ylabel("P(y1=1)")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")

    print(
        "\n[NOTE] contact_efficiencyは特徴（相関）としてのみ扱う。"
        "『上げれば良くなる』等の因果主張はしない。"
    )


if __name__ == "__main__":
    main()


