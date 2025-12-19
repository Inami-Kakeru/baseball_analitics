"""
attack_angle × y1 形状チェック（2ストライク限定・単変量）

目的:
- Layer1.5における不利条件を「2ストライク（strikes==2）」のみに修正して再検証する。
- 目的変数は常に y1（HVZ所属）で固定。y2/y3は作らない。
- 回帰・多変量・最適化・p値は一切しない（分位×確率のみ）。

入力（固定）:
- data/intermediate/contact_quality.csv
  必須列: attack_angle, y1, strikes

手順（固定）:
- overall（全BIP）の attack_angle 分布で分位境界を一度だけ作成（原則10分位）
- その境界を two_strike（strikes==2）にも適用（切り直し禁止）
- 集計: P(y1=1 | decile) を overall と two_strike で出す

出力（固定）:
- data/output/attack_angle_decile_py1_twostrike.csv
- data/output/plots/attack_angle_decile_vs_py1_twostrike.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")
OUT_CSV = Path("data/output/attack_angle_decile_py1_twostrike.csv")
OUT_PNG = Path("data/output/plots/attack_angle_decile_vs_py1_twostrike.png")


def make_quantile_edges(x: pd.Series, q: int = 10) -> np.ndarray:
    """
    overall分布に基づく分位境界を作る。
    ユニーク不足で境界が潰れる場合は、ビン数が10未満になる（警告して継続）。
    """
    probs = np.linspace(0, 1, q + 1)
    edges = x.quantile(probs, interpolation="linear").to_numpy(dtype=float)
    edges = edges[~np.isnan(edges)]
    edges = np.unique(edges)
    return edges


def assign_bins(x: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) < 2:
        return pd.Series([np.nan] * len(x), index=x.index)
    b = pd.cut(x, bins=edges, include_lowest=True, right=True, labels=False)
    return b.astype("float") + 1  # NaN保持のためfloat


def summarize(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    g = (
        df.groupby(bin_col, dropna=True)["y1"]
        .agg(n="count", py1="mean")
        .reset_index()
        .rename(columns={bin_col: "decile"})
        .sort_values("decile")
    )
    g["decile"] = g["decile"].astype(int)
    return g


def main() -> None:
    df = pd.read_csv(IN_PATH)
    required = ["attack_angle", "y1", "strikes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["attack_angle", "y1", "strikes"]).copy()
    df["y1"] = df["y1"].astype(int)

    # Step B: overall分布で分位境界を固定（10分位が原則）
    edges = make_quantile_edges(df["attack_angle"], q=10)
    n_bins = max(0, len(edges) - 1)
    if n_bins < 10:
        print(f"[WARN] attack_angle のユニーク不足により、分位ビン数が {n_bins} になりました（10未満）。")

    df["attack_angle_decile"] = assign_bins(df["attack_angle"], edges)

    # Step C: overall と two_strike（strikes==2）で P(y1=1 | decile)
    overall = summarize(df, "attack_angle_decile").rename(
        columns={"n": "n_overall", "py1": "py1_overall"}
    )
    two = df[df["strikes"] == 2].copy()
    two_strike = summarize(two, "attack_angle_decile").rename(
        columns={"n": "n_two_strike", "py1": "py1_two_strike"}
    )

    out = overall.merge(two_strike, on="decile", how="left").fillna(
        {"n_two_strike": 0, "py1_two_strike": np.nan}
    )
    out["n_two_strike"] = out["n_two_strike"].astype(int)

    # 小nは警告のみ
    small_overall = out[out["n_overall"] < 30]
    small_two = out[out["n_two_strike"] < 30]
    if len(small_overall):
        print("[WARN] overall で n<30 の分位があります:", small_overall["decile"].tolist())
    if len(small_two):
        print("[WARN] two_strike で n<30 の分位があります:", small_two["decile"].tolist())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved csv -> {OUT_CSV}")

    # Plot: overall + two_strike の2本線（回帰・平滑化なし）
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(out["decile"], out["py1_overall"], marker="o", label="overall")
    plt.plot(out["decile"], out["py1_two_strike"], marker="o", label="two_strike (strikes=2)")
    plt.xlabel("attack_angle decile (based on overall distribution)")
    plt.ylabel("P(y1=1)")
    plt.title("attack_angle decile vs P(y1=1) (two strikes window)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")

    # 形状コメント（自動判定しない）
    print("\n[Shape notes (non-assertive)]")
    print("- overall shape: inspect decile curve (monotonic vs peak/U-shape).")
    print("- two_strike shape: inspect if shape is similar to overall.")
    print("- Do NOT state an 'optimal degree'.\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()


