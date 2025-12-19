"""
A-2: two_strike限定｜attack_angle × P(y1) 山形状比較

目的:
- attack_angle に対する P(y1=1) の形状を overall と two_strike で比較する。
- 分位境界は overall（全選手）の attack_angle 分布で固定し、切り直さない。
- 対象: ohtani / league（全体）
- 回帰・最適点推定・多変量は禁止。

入力:
- data/intermediate/contact_quality.csv

出力:
- data/output/attack_angle_decile_py1_shape_comparison.csv
- data/output/plots/attack_angle_decile_py1_shape_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")
OUT_CSV = Path("data/output/attack_angle_decile_py1_shape_comparison.csv")
OUT_PNG = Path("data/output/plots/attack_angle_decile_py1_shape_comparison.png")

OHTANI_ID = "ohtani"


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


def build_one_group(df: pd.DataFrame, player_group: str) -> pd.DataFrame:
    overall = summarize(df, "attack_angle_decile").rename(
        columns={"n": "n_overall", "py1": "py1_overall"}
    )
    two = summarize(df[df["strikes"] == 2], "attack_angle_decile").rename(
        columns={"n": "n_two_strike", "py1": "py1_two_strike"}
    )
    out = overall.merge(two, on="decile", how="left").fillna(
        {"n_two_strike": 0, "py1_two_strike": np.nan}
    )
    out["n_two_strike"] = out["n_two_strike"].astype(int)
    out.insert(0, "player_group", player_group)
    # 仕様上 columns: player_group, decile, n, py1_overall, py1_two_strike なので nはoverallのnを採用
    out["n"] = out["n_overall"].astype(int)
    return out[["player_group", "decile", "n", "py1_overall", "py1_two_strike"]]


def main() -> None:
    df = pd.read_csv(IN_PATH)
    required = ["player_id", "attack_angle", "y1", "strikes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["attack_angle", "y1", "strikes", "player_id"]).copy()
    df["y1"] = df["y1"].astype(int)
    df["strikes"] = df["strikes"].astype(int)

    # overall（全選手）分布で境界固定
    edges = make_quantile_edges(df["attack_angle"], q=10)
    n_bins = len(edges) - 1
    if n_bins < 10:
        print(f"[WARN] attack_angle のユニーク不足により、分位ビン数が {n_bins} になりました（10未満）。")

    df["attack_angle_decile"] = assign_bins(df["attack_angle"], edges)

    oht = df[df["player_id"] == OHTANI_ID].copy()
    league = df.copy()

    out = pd.concat(
        [
            build_one_group(oht, "ohtani"),
            build_one_group(league, "league"),
        ],
        ignore_index=True,
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved csv -> {OUT_CSV}")

    # 図: playerごとに overall/two_strike 2本線
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pg in zip(axes, ["ohtani", "league"]):
        sub = out[out["player_group"] == pg].sort_values("decile")
        ax.plot(sub["decile"], sub["py1_overall"], marker="o", label="overall")
        ax.plot(sub["decile"], sub["py1_two_strike"], marker="o", label="two_strike")
        ax.set_title(pg)
        ax.set_xlabel("attack_angle decile")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("P(y1=1)")
    fig.suptitle("attack_angle decile vs P(y1=1): overall vs two_strike")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()


