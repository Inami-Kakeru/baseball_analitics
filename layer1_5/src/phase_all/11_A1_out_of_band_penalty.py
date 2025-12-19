from __future__ import annotations

"""
A1: 帯域外ペナルティの定量化（最重要）

group×conditionで in_band/out_band を比較して以下を出力:
- mean_primary_kpi_in, mean_primary_kpi_out, diff, ratio
- py1_in, py1_out, diff_py1
- n_in, n_out, n_total

出力:
- data/output/A1_out_of_band_penalty.csv
- data/output/plots/A1_out_of_band_penalty.png（任意）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._utils_phase import ensure_columns


BASE_TABLE = Path("data/output/phase_all_base_table.csv")
OUT_CSV = Path("data/output/A1_out_of_band_penalty.csv")
OUT_PNG = Path("data/output/plots/A1_out_of_band_penalty.png")

OHTANI_ID = "ohtani"


def safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return float("nan")
    return a / b


def summarize_group(df: pd.DataFrame) -> dict:
    d_in = df[df["in_band"] == True]  # noqa: E712
    d_out = df[df["out_band"] == True]  # noqa: E712
    mean_in = float(d_in["primary_kpi"].mean())
    mean_out = float(d_out["primary_kpi"].mean())
    py1_in = float(d_in["y1"].mean()) if len(d_in) else float("nan")
    py1_out = float(d_out["y1"].mean()) if len(d_out) else float("nan")
    return {
        "n_total": int(len(df)),
        "n_in": int(len(d_in)),
        "n_out": int(len(d_out)),
        "mean_primary_kpi_in": mean_in,
        "mean_primary_kpi_out": mean_out,
        "diff_primary_kpi": mean_in - mean_out,
        "ratio_primary_kpi": safe_ratio(mean_in, mean_out),
        "py1_in": py1_in,
        "py1_out": py1_out,
        "diff_py1": py1_in - py1_out,
    }


def main() -> None:
    df = pd.read_csv(BASE_TABLE)
    ensure_columns(df, ["player_id", "two_strike", "in_band", "out_band", "primary_kpi", "y1"])

    groups: dict[str, pd.DataFrame] = {
        "ohtani": df[df["player_id"] == OHTANI_ID],
        "benchmark_group": df,
    }

    rows = []
    for gname, gdf in groups.items():
        for cond, sdf in [
            ("overall", gdf),
            ("two_strike", gdf[gdf["two_strike"] == True]),  # noqa: E712
        ]:
            # 必要列の欠損のみdrop（全体drop禁止の方針に従う）
            s = sdf.dropna(subset=["primary_kpi", "y1", "in_band", "out_band"]).copy()
            rec = {"group": gname, "condition": cond}
            rec.update(summarize_group(s))
            rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # 図（任意）: diff_primary_kpi と diff_py1 を並べる（回帰なし）
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, col, title in [
        (axes[0], "diff_primary_kpi", "Primary KPI: in_band - out_band"),
        (axes[1], "diff_py1", "P(y1=1): in_band - out_band"),
    ]:
        pivot = out.pivot(index="group", columns="condition", values=col)
        pivot = pivot.reindex(["ohtani", "benchmark_group"])
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


