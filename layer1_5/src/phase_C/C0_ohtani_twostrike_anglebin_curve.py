from __future__ import annotations

"""
実験2:
帯域内 vs 帯域外の“差”を、角度ビンで連続表示（C1の説得力を上げる）

出力:
- CSV: data/output/C0_ohtani_twostrike_anglebin_curve.csv
- 図 : data/output/plots/phase_C/C0_ohtani_twostrike_anglebin_curve.png

仕様:
- 大谷 two_strike のみ
- attack_angle を等幅ビン（デフォルト 2deg 刻み）
- 各ビンの mean xSLG と P(y1=1) を連続で表示
- 固定帯域 [lower, upper] を背景色でハイライト
- n < MIN_N のビンは “薄色/点線” で表示し、ルールをタイトルに明記
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONTACT_PATH = Path("data/intermediate/contact_quality.csv")
BAND_PATH = Path("data/output/ohtani_attack_angle_band_rate.csv")

OUT_CSV = Path("data/output/C0_ohtani_twostrike_anglebin_curve.csv")
OUT_PNG = Path("data/output/plots/phase_C/C0_ohtani_twostrike_anglebin_curve.png")

BIN_DEG = 2.0
MIN_N = 30


def _must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 必須列が不足: {missing}")


def _load_band() -> tuple[float, float]:
    b = pd.read_csv(BAND_PATH)
    _must_have(b, ["lower_bound", "upper_bound"], "ohtani_attack_angle_band_rate.csv")
    lower = float(b["lower_bound"].iloc[0])
    upper = float(b["upper_bound"].iloc[0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"帯域が不正: lower={lower}, upper={upper}")
    return lower, upper


def main() -> None:
    df = pd.read_csv(CONTACT_PATH)
    _must_have(
        df,
        ["player_id", "strikes", "attack_angle", "y1", "estimated_slg_using_speedangle"],
        "contact_quality.csv",
    )

    lower, upper = _load_band()

    d0 = df[(df["player_id"] == "ohtani") & (df["strikes"] == 2)].copy()
    n_total = int(len(d0))
    d = d0.dropna(subset=["attack_angle", "y1", "estimated_slg_using_speedangle"]).copy()
    n_avail = int(len(d))
    if n_avail == 0:
        raise ValueError("two_strikeで attack_angle/y1/xSLG が揃う行が0です")

    aa = d["attack_angle"].to_numpy(dtype=float)
    lo = float(np.floor(np.nanmin(aa) / BIN_DEG) * BIN_DEG)
    hi = float(np.ceil(np.nanmax(aa) / BIN_DEG) * BIN_DEG)
    edges = np.arange(lo, hi + BIN_DEG, BIN_DEG)
    if len(edges) < 3:
        raise ValueError("attack_angleの範囲が狭すぎてビニングできません")

    d["angle_bin"] = pd.cut(d["attack_angle"], bins=edges, include_lowest=True)
    g = (
        d.groupby("angle_bin", dropna=True)
        .agg(
            n=("y1", "count"),
            py1=("y1", "mean"),
            mean_xslg=("estimated_slg_using_speedangle", "mean"),
            aa_mean=("attack_angle", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("aa_mean")
    )
    g["keep"] = g["n"] >= MIN_N

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT_CSV, index=False)

    fig, ax1 = plt.subplots(figsize=(11.5, 5.2))
    ax2 = ax1.twinx()

    # 固定帯域ハイライト
    ax1.axvspan(lower, upper, alpha=0.2)
    ax1.axvline(lower, linestyle="--", linewidth=1.2)
    ax1.axvline(upper, linestyle="--", linewidth=1.2)

    kept = g[g["keep"] == True]
    dropped = g[g["keep"] == False]

    # kept: 実線
    ax1.plot(kept["aa_mean"], kept["mean_xslg"], marker="o", label="mean xSLG (kept)")
    ax2.plot(kept["aa_mean"], kept["py1"], marker="s", label="P(y1=1) (kept)")

    # dropped: 薄色点線（どこが検証薄いかを視覚化）
    if len(dropped):
        ax1.plot(
            dropped["aa_mean"],
            dropped["mean_xslg"],
            linestyle="--",
            alpha=0.25,
            marker="o",
            label="mean xSLG (n<MIN_N)",
        )
        ax2.plot(
            dropped["aa_mean"],
            dropped["py1"],
            linestyle="--",
            alpha=0.25,
            marker="s",
            label="P(y1=1) (n<MIN_N)",
        )

    miss = 1.0 - (n_avail / n_total) if n_total else float("nan")
    title = (
        f"Ohtani two_strike | n_total={n_total}, n_avail={n_avail}, missing_rate={miss:.3f} "
        f"| bin={BIN_DEG}deg | MIN_N={MIN_N}"
    )
    ax1.set_title(title)
    ax1.set_xlabel("attack_angle (deg) [equal-width bins]")
    ax1.set_ylabel("mean estimated_slg_using_speedangle")
    ax2.set_ylabel("P(y1=1)")
    ax1.grid(True, alpha=0.3)

    # 凡例統合
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()

    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()


