"""
B-2: plate_z別に切らなかった理由の検証（防御用）

目的:
- plate_z_group（Low/Mid/High）別に見たとき、
  attack_angle に対する P(y1=1) の形状がどれだけズレるかを確認する（補足のみ）。
- 本分析への組み込みは禁止（あくまで補助資料）。

仕様:
- attack_angle 10分位は overall（全選手）分布で固定（切り直さない）
- plate_z_group × decile で P(y1=1), n を算出

出力:
- CSV: data/output/attack_angle_platez_shape.csv
- 図（任意）: data/output/plots/attack_angle_platez_shape.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")
OUT_CSV = Path("data/output/attack_angle_platez_shape.csv")
OUT_PNG = Path("data/output/plots/attack_angle_platez_shape.png")


def make_quantile_edges(x: pd.Series, q: int = 10) -> np.ndarray:
    probs = np.linspace(0, 1, q + 1)
    edges = x.quantile(probs, interpolation="linear").to_numpy(dtype=float)
    edges = edges[~np.isnan(edges)]
    edges = np.unique(edges)
    return edges


def assign_bins(x: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) < 2:
        return pd.Series([np.nan] * len(x), index=x.index)
    b = pd.cut(x, bins=edges, include_lowest=True, right=True, labels=False)
    return b.astype("float") + 1


def main() -> None:
    df = pd.read_csv(IN_PATH)
    required = ["attack_angle", "y1", "plate_z_group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["attack_angle", "y1", "plate_z_group"]).copy()
    df["y1"] = df["y1"].astype(int)

    # overall分布でdecile境界固定
    edges = make_quantile_edges(df["attack_angle"], q=10)
    n_bins = len(edges) - 1
    if n_bins < 10:
        print(f"[WARN] attack_angle のユニーク不足により、分位ビン数が {n_bins} になりました（10未満）。")
    df["attack_angle_decile"] = assign_bins(df["attack_angle"], edges)

    out = (
        df.groupby(["plate_z_group", "attack_angle_decile"])["y1"]
        .agg(n="count", py1="mean")
        .reset_index()
        .rename(columns={"attack_angle_decile": "decile"})
        .sort_values(["plate_z_group", "decile"])
    )
    out["decile"] = out["decile"].astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved csv -> {OUT_CSV}")

    # 図（任意）：plate_z_groupごとに折れ線
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    for grp in ["Low", "Mid", "High"]:
        sub = out[out["plate_z_group"] == grp]
        if len(sub) == 0:
            continue
        plt.plot(sub["decile"], sub["py1"], marker="o", label=grp)
    plt.xlabel("attack_angle decile (based on overall distribution)")
    plt.ylabel("P(y1=1)")
    plt.title("attack_angle shape by plate_z_group (defense-only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


