from __future__ import annotations

"""
C-6: 大谷の“改善プラン直結”診断セット

出力:
- data/output/C6_ohtani_action_diagnosis_summary.csv
- data/output/C6_ohtani_twostrike_outband_drivers.csv
- data/output/plots/phase_C/C6_ohtani_summary.png

仕様:
- player_id=='ohtani' のみ
- overall / two_strike それぞれで in_band_rate と in/out ペナルティ（xSLG, y1）
- two_strikeだけで out_band drivers（数値列をC3と同様に単変量抽出、top5）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from layer1_5.src.phase_C.utils import (
    add_band_flags,
    choose_xslg_col,
    load_band_bounds,
    load_contact_quality,
    spearman_corr_no_scipy,
)


OUT_SUM = Path("data/output/C6_ohtani_action_diagnosis_summary.csv")
OUT_DRV = Path("data/output/C6_ohtani_twostrike_outband_drivers.csv")
OUT_PNG = Path("data/output/plots/phase_C/C6_ohtani_summary.png")


def safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return float("nan")
    return a / b


def summarize_penalty(df: pd.DataFrame, xslg_col: str) -> dict:
    d = df[df["attack_angle_available"] == True].copy()  # noqa: E712
    n_total = int(len(df))
    n_avail = int(len(d))
    in_rate = float(d["in_band"].mean()) if len(d) else float("nan")
    d_in = d[d["in_band"] == True]  # noqa: E712
    d_out = d[d["out_band"] == True]  # noqa: E712
    mean_in = float(d_in[xslg_col].mean())
    mean_out = float(d_out[xslg_col].mean())
    py1_in = float(d_in["y1"].mean()) if len(d_in) else float("nan")
    py1_out = float(d_out["y1"].mean()) if len(d_out) else float("nan")
    return {
        "n_total": n_total,
        "n_attack_angle_available": n_avail,
        "in_band_rate": in_rate,
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


def excluded(name: str) -> bool:
    n = str(name)
    if n in {"y1", "attack_angle", "launch_angle", "out_band", "in_band", "two_strike", "attack_angle_available"}:
        return True
    if n in {"player_id", "batter", "pitcher", "game_pk"}:
        return True
    if n.startswith("estimated_"):
        return True
    if "xslg" in n.lower() or "woba" in n.lower():
        return True
    return False


def top_outband_drivers(df: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if not excluded(c)]
    rows = []
    for f in feats:
        d = df[[f, "out_band"]].dropna().copy()
        if len(d) < 200:
            continue
        try:
            d["decile"] = pd.qcut(d[f], q=10, labels=False, duplicates="drop") + 1
        except Exception:
            continue
        agg = d.groupby("decile")["out_band"].agg(p_out_band="mean", n="count").reset_index().sort_values("decile")
        if len(agg) < 2:
            continue
        rho = spearman_corr_no_scipy(agg["decile"], agg["p_out_band"])
        delta = float(agg["p_out_band"].iloc[-1] - agg["p_out_band"].iloc[0])
        rows.append({"feature": f, "spearman_rho": rho, "direction": "+" if rho > 0 else "-" if rho < 0 else "0", "effect_delta": delta})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["abs_rho"] = res["spearman_rho"].abs()
    res["abs_delta"] = res["effect_delta"].abs()
    res = res.sort_values(["abs_rho", "abs_delta"], ascending=False).head(topk).drop(columns=["abs_rho", "abs_delta"])
    return res


def main() -> None:
    df = add_band_flags(load_contact_quality(), load_band_bounds())
    xslg_col = choose_xslg_col(df)

    oht = df[df["player_id"] == "ohtani"].copy()
    if len(oht) == 0:
        raise ValueError("ohtaniの行が0です（player_id表記を確認）。")

    rows = []
    for cond, sdf in [
        ("overall", oht),
        ("two_strike", oht[oht["two_strike"] == True]),  # noqa: E712
    ]:
        s = sdf.dropna(subset=[xslg_col, "y1"]).copy()
        rec = {"condition": cond}
        rec.update(summarize_penalty(s, xslg_col))
        rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    OUT_DRV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SUM, index=False)
    print(f"saved -> {OUT_SUM}")

    # drivers（two_strikeのみ）
    two = oht[oht["two_strike"] == True].copy()  # noqa: E712
    drivers = top_outband_drivers(two, topk=5)
    if drivers.empty:
        # 落ちた理由をログ
        print("[LOG] C6 drivers: empty (maybe n too small or qcut failed for all features).")
        drivers.to_csv(OUT_DRV, index=False)
    else:
        drivers.to_csv(OUT_DRV, index=False)
    print(f"saved -> {OUT_DRV}")

    # 図（1枚で見える）
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    # bar: in_band_rate (two rows)
    ax.bar(out["condition"], out["in_band_rate"], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("in_band_rate (attack_angle available denom)")
    ax.set_title("Ohtani: in_band_rate (overall vs two_strike)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


