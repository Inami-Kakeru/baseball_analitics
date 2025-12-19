"""
attack_angle の形状チェック（単変量・分位×確率）

目的:
- attack_angle が y1（HVZ所属）に与える影響が単調か、山型/U字かを確認する。
- 因果主張・回帰・多変量・最適化は禁止。

仕様（固定）:
- 分位（decile）は overall（全BIP）の attack_angle 分布で切る
- adverse は is_adverse=True の観測窓（y3は作らない）
- 出力:
  - layer1_5/data/output/attack_angle_decile_py1.csv
  - layer1_5/data/output/plots/attack_angle_decile_vs_py1.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")
OUT_CSV = Path("data/output/attack_angle_decile_py1.csv")
OUT_PNG = Path("data/output/plots/attack_angle_decile_vs_py1.png")


def make_quantile_edges(x: pd.Series, q: int = 10) -> np.ndarray:
    """
    overall分布に基づく分位境界を作る。
    ユニーク不足で境界が潰れる場合は、ビン数が10未満になる。
    """
    probs = np.linspace(0, 1, q + 1)
    edges = x.quantile(probs, interpolation="linear").to_numpy(dtype=float)
    # NaN除去・単調化
    edges = edges[~np.isnan(edges)]
    edges = np.unique(edges)
    return edges


def assign_bins(x: pd.Series, edges: np.ndarray) -> pd.Series:
    # edgesが2未満ならビニング不能
    if len(edges) < 2:
        return pd.Series([np.nan] * len(x), index=x.index)
    # pd.cutで1..kのビン番号を付与
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
    required = ["attack_angle", "y1", "is_adverse", "player_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["attack_angle", "y1"]).copy()
    df["y1"] = df["y1"].astype(int)

    # Step1: overall分布で分位境界を作成
    edges = make_quantile_edges(df["attack_angle"], q=10)
    n_bins = max(0, len(edges) - 1)
    if n_bins < 10:
        print(f"[WARN] attack_angle のユニーク不足により、分位ビン数が {n_bins} になりました（10未満）。")

    df["attack_angle_decile"] = assign_bins(df["attack_angle"], edges)

    # Step2: overall/adverseで P(y1=1 | bin) を計算
    overall = summarize(df, "attack_angle_decile").rename(
        columns={"n": "n_overall", "py1": "py1_overall"}
    )
    adverse_df = df[df["is_adverse"] == True].copy()  # noqa: E712
    adverse = summarize(adverse_df, "attack_angle_decile").rename(
        columns={"n": "n_adverse", "py1": "py1_adverse"}
    )

    out = overall.merge(adverse, on="decile", how="left").fillna(
        {"n_adverse": 0, "py1_adverse": np.nan}
    )
    out["n_adverse"] = out["n_adverse"].astype(int)

    # Step1補足: 極端に小さいnは警告のみ
    small_overall = out[out["n_overall"] < 30]
    small_adverse = out[out["n_adverse"] < 30]
    if len(small_overall):
        print("[WARN] overall で n<30 の分位があります:", small_overall["decile"].tolist())
    if len(small_adverse):
        print("[WARN] adverse で n<30 の分位があります:", small_adverse["decile"].tolist())

    # 出力
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved csv -> {OUT_CSV}")

    # Step3: 可視化（overall + adverse）
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(out["decile"], out["py1_overall"], marker="o", label="overall")
    plt.plot(out["decile"], out["py1_adverse"], marker="o", label="adverse")
    plt.xlabel("attack_angle decile (based on overall distribution)")
    plt.ylabel("P(y1=1)")
    plt.title("attack_angle decile vs P(y1=1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")

    # Step4: 形状コメント（自動判定はしない、観察のためのメモ）
    print("\n[Summary / Non-assertive notes]")
    print("- overall shape: please inspect decile curve (look for peak/U-shape/non-monotonicity).")
    print("- adverse shape: please inspect if the curve shape is preserved vs overall.")
    print("- Do NOT conclude an 'optimal degree'; this is shape-check only.\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()


