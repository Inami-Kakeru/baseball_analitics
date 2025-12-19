from __future__ import annotations

"""
C-3: two_strikeで out_band になりやすい「ドライバー」を単変量で抽出

仕様:
- two_strikeのみ
- 目的: out_band（attack_angle非欠損かつ帯域外）
- 特徴量候補: 数値列のみ
- 除外: y1, attack_angle, launch_angle, KPI列（estimated_*等）, in/out系, ID系
- 単変量: qcut 10分位（duplicates='drop'）
- 評価: Spearman rho（符号と絶対値のみ、p値不要）

出力:
- data/output/C3_twostrike_outband_drivers.csv
- 図（top5）: data/output/plots/phase_C/C3_outband_driver_top5.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

from layer1_5.src.phase_C.utils import (
    add_band_flags,
    load_band_bounds,
    load_contact_quality,
    spearman_corr_no_scipy,
)


OUT_CSV = Path("data/output/C3_twostrike_outband_drivers.csv")
OUT_PNG = Path("data/output/plots/phase_C/C3_outband_driver_top5.png")

TOP_K = 20
TOP_PLOT = 5


def is_kpi_col(name: str) -> bool:
    n = str(name).lower()
    return n.startswith("estimated_") or ("xslg" in n) or ("xwoba" in n) or ("woba" in n)


def excluded(name: str) -> bool:
    n = str(name)
    if n in {"y1", "attack_angle", "launch_angle", "out_band", "in_band", "two_strike", "attack_angle_available"}:
        return True
    if n in {"player_id", "batter", "pitcher", "game_pk"}:
        return True
    if is_kpi_col(n):
        return True
    return False


def compute_feature(df: pd.DataFrame, feat: str) -> tuple[float, int, float]:
    """
    (rho, direction, effect_delta)
    effect_delta: top decile p(out_band) - bottom decile p(out_band)
    """
    d = df[[feat, "out_band"]].dropna().copy()
    if len(d) < 200:
        return float("nan"), 0, float("nan")
    try:
        d["decile"] = pd.qcut(d[feat], q=10, labels=False, duplicates="drop") + 1
    except Exception:
        return float("nan"), 0, float("nan")
    agg = d.groupby("decile")["out_band"].agg(["mean", "count"]).reset_index().sort_values("decile")
    if len(agg) < 2:
        return float("nan"), 0, float("nan")
    rho = spearman_corr_no_scipy(agg["decile"], agg["mean"])
    direction = int(np.sign(rho)) if not np.isnan(rho) else 0
    # delta between max and min decile index
    lo = float(agg["mean"].iloc[0])
    hi = float(agg["mean"].iloc[-1])
    delta = float(hi - lo)
    return float(rho), direction, delta


def main() -> None:
    df = add_band_flags(load_contact_quality(), load_band_bounds())
    df2 = df[df["two_strike"] == True].copy()  # noqa: E712
    # out_bandは attack_angle非欠損が前提（定義済み）

    # numeric cols
    numeric = df2.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if not excluded(c)]
    if not feats:
        raise ValueError("候補featureが空です（除外条件が強すぎる可能性）。")

    summary_rows = []
    for f in feats:
        rho, direction, delta = compute_feature(df2, f)
        if np.isnan(rho):
            continue
        summary_rows.append(
            {
                "row_type": "summary",
                "feature": f,
                "spearman_rho": rho,
                "direction": "+" if direction > 0 else "-" if direction < 0 else "0",
                "effect_delta": delta,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("summaryが空になりました（qcut不可/欠損が多い可能性）。")

    # rank: abs(rho) then abs(delta)
    summary["abs_rho"] = summary["spearman_rho"].abs()
    summary["abs_delta"] = summary["effect_delta"].abs()
    summary = summary.sort_values(["abs_rho", "abs_delta"], ascending=False).head(TOP_K).drop(columns=["abs_rho", "abs_delta"])

    # decile tables for top20
    decile_rows = []
    for f in summary["feature"].tolist():
        d = df2[[f, "out_band"]].dropna().copy()
        try:
            d["decile"] = pd.qcut(d[f], q=10, labels=False, duplicates="drop") + 1
        except Exception:
            continue
        agg = d.groupby("decile")["out_band"].agg(p_out_band="mean", n="count").reset_index().sort_values("decile")
        if len(agg) == 0:
            continue
        agg.insert(0, "feature", f)
        agg.insert(0, "row_type", "decile")
        decile_rows.append(agg)

    deciles = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()

    out = pd.concat([summary.drop(columns=[]), deciles], ignore_index=True, sort=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # plot top5
    top5 = summary.head(TOP_PLOT)["feature"].tolist()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    for f in top5:
        sub = deciles[deciles["feature"] == f].sort_values("decile")
        plt.plot(sub["decile"], sub["p_out_band"], marker="o", label=f)
    plt.xlabel("decile")
    plt.ylabel("P(out_band=1)")
    plt.title("two_strike out_band drivers (top5)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


