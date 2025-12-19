from __future__ import annotations

"""
C-5: 球種・コース別は「補助資料」に隔離（本筋に混ぜない）

仕様:
- two_strikeのみ
- 目的: P(y1=1) の形状がグループで変わるか（最適値推定は禁止）
- decile境界: overall分布（全体）で attack_angle を qcut して境界固定
  その境界を two_strike に適用（グループ内で切り直し禁止）
- 分割:
  - pitch_group があれば使用（FF/Breaking/Offspeed/Other）
  - plate_z_group があれば使用（Low/Mid/High）
  - どちらも無ければスキップしてログ

出力:
- data/output/C5_shape_by_group.csv
- data/output/plots/phase_C/C5_shape_by_group.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

from layer1_5.src.phase_C.utils import assign_bins_by_edges, load_band_bounds, load_contact_quality, q_edges


OUT_CSV = Path("data/output/C5_shape_by_group.csv")
OUT_PNG = Path("data/output/plots/phase_C/C5_shape_by_group.png")


def summarize(df: pd.DataFrame, group_dim: str, group_val: str) -> pd.DataFrame:
    g = (
        df.groupby("decile")["y1"]
        .agg(py1="mean", n="count")
        .reset_index()
        .sort_values("decile")
    )
    g.insert(0, "group_value", group_val)
    g.insert(0, "group_dim", group_dim)
    g["decile"] = g["decile"].astype(int)
    return g


def main() -> None:
    df = load_contact_quality()
    # bandはここでは使わないが、前提として固定済み（C系）
    _ = load_band_bounds()

    if "pitch_group" not in df.columns and "plate_z_group" not in df.columns:
        print("[SKIP] pitch_group/plate_z_group が無いので C5 をスキップ")
        return

    df = df.dropna(subset=["attack_angle", "y1", "strikes"]).copy()
    df["y1"] = df["y1"].astype(int)
    df["strikes"] = df["strikes"].astype(int)

    # overall分布でdecile境界固定
    edges = q_edges(df["attack_angle"], q=10)
    df["decile"] = assign_bins_by_edges(df["attack_angle"], edges)

    # two_strikeのみ
    df2 = df[df["strikes"] == 2].copy()

    rows = []
    # pitch_group
    if "pitch_group" in df2.columns:
        for v in sorted(df2["pitch_group"].dropna().unique()):
            sub = df2[df2["pitch_group"] == v].dropna(subset=["decile"])
            if len(sub) == 0:
                continue
            rows.append(summarize(sub, "pitch_group", str(v)))
    # plate_z_group
    if "plate_z_group" in df2.columns:
        for v in ["Low", "Mid", "High"]:
            sub = df2[df2["plate_z_group"] == v].dropna(subset=["decile"])
            if len(sub) == 0:
                continue
            rows.append(summarize(sub, "plate_z_group", str(v)))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")

    # 図: 線が多い場合は pitch_group のみを描画
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plotted = False
    if "pitch_group" in out["group_dim"].unique():
        sub = out[out["group_dim"] == "pitch_group"]
        for v in sub["group_value"].unique():
            s = sub[sub["group_value"] == v].sort_values("decile")
            plt.plot(s["decile"], s["py1"], marker="o", label=f"pitch:{v}")
            plotted = True
    elif "plate_z_group" in out["group_dim"].unique():
        sub = out[out["group_dim"] == "plate_z_group"]
        for v in sub["group_value"].unique():
            s = sub[sub["group_value"] == v].sort_values("decile")
            plt.plot(s["decile"], s["py1"], marker="o", label=f"z:{v}")
            plotted = True

    if plotted:
        plt.xlabel("attack_angle decile (fixed on overall distribution)")
        plt.ylabel("P(y1=1)")
        plt.title("two_strike: shape by group (defense-only)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        plt.close()
        print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


