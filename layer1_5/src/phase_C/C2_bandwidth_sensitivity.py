from __future__ import annotations

"""
C-2: 帯域幅の感度分析（±2/±4/±6度）で結論が崩れないか
"""

from pathlib import Path

import numpy as np
import pandas as pd

from layer1_5.src.phase_C.utils import (
    Band,
    add_band_flags,
    choose_xslg_col,
    load_band_bounds,
    load_contact_quality,
)


OUT_CSV = Path("data/output/C2_bandwidth_sensitivity.csv")
OUT_PNG = Path("data/output/plots/phase_C/C2_bandwidth_sensitivity.png")


def safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return float("nan")
    return a / b


def summarize(df: pd.DataFrame, xslg_col: str) -> dict:
    d = df[df["attack_angle_available"] == True].copy()  # noqa: E712
    d_in = d[d["in_band"] == True]  # noqa: E712
    d_out = d[d["out_band"] == True]  # noqa: E712
    mean_in = float(d_in[xslg_col].mean())
    mean_out = float(d_out[xslg_col].mean())
    py1_in = float(d_in["y1"].mean()) if len(d_in) else float("nan")
    py1_out = float(d_out["y1"].mean()) if len(d_out) else float("nan")
    return {
        "n_total": int(len(df)),
        "n_attack_angle_available": int(len(d)),
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


def band_variants(base: Band) -> list[tuple[str, Band]]:
    out: list[tuple[str, Band]] = [("base", base)]
    for d in [2, 4, 6]:
        out.append((f"narrow{d}", Band(base.lower + d, base.upper - d)))
        out.append((f"wide{d}", Band(base.lower - d, base.upper + d)))
    return out


def main() -> None:
    df = load_contact_quality()
    base_band = load_band_bounds()
    xslg_col = choose_xslg_col(df)

    variants = band_variants(base_band)
    rows = []
    for band_name, band in variants:
        if band.lower >= band.upper:
            print(f"[SKIP] band '{band_name}' invalid (lower>=upper): {band}")
            continue

        d0 = add_band_flags(df, band)
        for cond_name, dcond in [
            ("overall", d0),
            ("two_strike", d0[d0["two_strike"] == True]),  # noqa: E712
        ]:
            for group_name, gdf in {
                "ohtani": dcond[dcond["player_id"] == "ohtani"],
                "benchmark_group": dcond,
            }.items():
                s = gdf.dropna(subset=[xslg_col, "y1"]).copy()
                rec = {"band": band_name, "condition": cond_name, "group": group_name}
                rec.update(summarize(s, xslg_col))
                rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # 図: diff_xslg を band×group で可視化（overall/two_strikeはsubplot）
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cond in zip(axes, ["overall", "two_strike"]):
        sub = out[out["condition"] == cond].copy()
        # pivot: band x group
        piv = sub.pivot_table(index="band", columns="group", values="diff_xslg", aggfunc="mean")
        piv = piv.reindex(sorted(piv.index, key=lambda x: (0 if x == "base" else 1, x)))
        piv.plot(kind="bar", ax=ax)
        ax.set_title(f"diff_xslg by band | {cond}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel("band setting")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


