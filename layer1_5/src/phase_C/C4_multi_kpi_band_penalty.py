from __future__ import annotations

"""
C-4: KPIをxSLG以外でも確認（納得の幅）

仕様:
- overall / two_strike 両方
- group: ohtani / benchmark_group
- KPI候補（存在するものだけ）:
  - estimated_slg_using_speedangle（必須）
  - estimated_woba_using_speedangle
  - barrel（0/1）
  - hard_hit（0/1）
- in/out の平均（または率）と差分を出す
- 図は不要（CSVのみ）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from layer1_5.src.phase_C.utils import add_band_flags, choose_xslg_col, load_band_bounds, load_contact_quality


OUT_CSV = Path("data/output/C4_multi_kpi_band_penalty.csv")


def is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    if len(vals) == 0:
        return False
    return set(vals).issubset({0, 1, 0.0, 1.0, True, False})


def summarize(df: pd.DataFrame, kpi: str) -> dict:
    d = df[df["attack_angle_available"] == True].copy()  # noqa: E712
    d_in = d[d["in_band"] == True]  # noqa: E712
    d_out = d[d["out_band"] == True]  # noqa: E712
    mean_in = float(d_in[kpi].mean())
    mean_out = float(d_out[kpi].mean())
    return {
        "n_total": int(len(df)),
        "n_attack_angle_available": int(len(d)),
        "n_in": int(len(d_in)),
        "n_out": int(len(d_out)),
        "mean_in": mean_in,
        "mean_out": mean_out,
        "diff": mean_in - mean_out,
    }


def main() -> None:
    df = add_band_flags(load_contact_quality(), load_band_bounds())
    # xSLG必須
    _ = choose_xslg_col(df)

    candidates = [
        "estimated_slg_using_speedangle",
        "estimated_woba_using_speedangle",
        "barrel",
        "hard_hit",
    ]
    kpis = [c for c in candidates if c in df.columns]
    if "estimated_slg_using_speedangle" not in kpis:
        raise KeyError("C4: estimated_slg_using_speedangle は必須です。")

    rows = []
    for cond_name, dcond in [
        ("overall", df),
        ("two_strike", df[df["two_strike"] == True]),  # noqa: E712
    ]:
        for group_name, gdf in {
            "ohtani": dcond[dcond["player_id"] == "ohtani"],
            "benchmark_group": dcond,
        }.items():
            for kpi in kpis:
                s = gdf.dropna(subset=[kpi, "y1"]).copy()
                rec = {"condition": cond_name, "group": group_name, "kpi": kpi}
                rec.update(summarize(s, kpi))
                rec["kpi_type"] = "rate" if is_binary(s[kpi]) else "mean"
                rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()


