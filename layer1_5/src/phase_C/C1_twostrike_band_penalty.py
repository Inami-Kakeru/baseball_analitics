from __future__ import annotations

"""
C-1: two_strike内だけで帯域外ペナルティが成立するか（主張の芯）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from layer1_5.src.phase_C.utils import (
    add_band_flags,
    choose_xslg_col,
    load_band_bounds,
    load_contact_quality,
)


OUT_CSV = Path("data/output/C1_twostrike_band_penalty.csv")
OUT_PNG = Path("data/output/plots/phase_C/C1_twostrike_band_penalty.png")


def safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return float("nan")
    return a / b


def summarize(df: pd.DataFrame, xslg_col: str) -> dict:
    # attack_angle非欠損のみが分母
    d = df[df["attack_angle_available"] == True].copy()  # noqa: E712
    n_total = int(len(df))
    n_avail = int(len(d))
    d_in = d[d["in_band"] == True]  # noqa: E712
    d_out = d[d["out_band"] == True]  # noqa: E712
    mean_in = float(d_in[xslg_col].mean())
    mean_out = float(d_out[xslg_col].mean())
    py1_in = float(d_in["y1"].mean()) if len(d_in) else float("nan")
    py1_out = float(d_out["y1"].mean()) if len(d_out) else float("nan")
    return {
        "n_total": n_total,
        "n_attack_angle_available": n_avail,
        "n_in": int(len(d_in)),
        "n_out": int(len(d_out)),
        "mean_xslg_in": mean_in,
        "mean_xslg_out": mean_out,
        "diff_xslg": mean_in - mean_out,
        "ratio_xslg": safe_ratio(mean_in, mean_out),
        "py1_in": py1_in,
        "py1_out": py1_out,
        "diff_py1": py1_in - py1_out,
    }


def main() -> None:
    df = load_contact_quality()
    band = load_band_bounds()
    df = add_band_flags(df, band)
    xslg_col = choose_xslg_col(df)

    # two_strikeのみ
    df2 = df[df["two_strike"] == True].copy()  # noqa: E712

    groups = {
        "ohtani": df2[df2["player_id"] == "ohtani"],
        "benchmark_group": df2,
    }

    rows = []
    for gname, gdf in groups.items():
        s = gdf.dropna(subset=[xslg_col, "y1"]).copy()
        rec = {"group": gname}
        rec.update(summarize(s, xslg_col))
        rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # 図: diff_xslg と diff_py1
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(out["group"], out["diff_xslg"])
    axes[0].set_title("diff_xslg (in - out) | two_strike")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(out["group"], out["diff_py1"])
    axes[1].set_title("diff_py1 (in - out) | two_strike")
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


