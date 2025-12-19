"""
Phase1（最小追加裏付け）: 大谷翔平に限定した attack_angle の帯域内率

目的（固定）:
- Layer1.5 / Layer2-lite で確認された attack_angle の「山の頂点付近帯域」への集中が、
  大谷（player_id='ohtani'）に限定しても、2ストライク条件でも崩れないかを最小限で確認する。

前提（固定）:
- 目的変数は常に y1（HVZ所属）で、ここでは y1 自体は使わず「帯域内率」だけを出す。
- y3は作らない。不利条件は strikes==2 のみ。
- 回帰・最適点推定・多変量・GA2Mは禁止。
- 帯域は再学習しない（全体で決めた帯域をそのまま使う）。

帯域定義（固定）:
- 前回のLayer2-lite（decile vs P(y1=1)）で「山の頂点付近」として扱った
  attack_angle の分位帯を固定し、その分位境界から [lower_bound, upper_bound] を復元する。
  ここでは **decile 4〜9** を「中央帯域（山の頂点付近）」として固定する。

入力:
- layer1_5/data/intermediate/contact_quality.csv

出力:
- layer1_5/data/output/ohtani_attack_angle_band_rate.csv
  columns: condition, n, in_band_rate, lower_bound, upper_bound
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/intermediate/contact_quality.csv")
OUT_PATH = Path("data/output/ohtani_attack_angle_band_rate.csv")

PLAYER_ID = "ohtani"

# 固定: 山の頂点付近帯域（分位で固定）
BAND_DECILE_LO = 4
BAND_DECILE_HI = 9


def make_quantile_edges(x: pd.Series, q: int = 10) -> np.ndarray:
    probs = np.linspace(0, 1, q + 1)
    edges = x.quantile(probs, interpolation="linear").to_numpy(dtype=float)
    edges = edges[~np.isnan(edges)]
    edges = np.unique(edges)
    return edges


def infer_band_bounds_from_overall(df_all: pd.DataFrame) -> tuple[float, float]:
    """
    overall全体分布の10分位境界から、固定decile帯域の角度境界を復元する。
    """
    edges = make_quantile_edges(df_all["attack_angle"], q=10)
    n_bins = len(edges) - 1
    if n_bins < 10:
        raise ValueError(
            f"attack_angle のユニーク不足により10分位境界が復元できません（bins={n_bins}）。"
        )
    # decile k の下端は edges[k-1], 上端は edges[k]
    lower = float(edges[BAND_DECILE_LO - 1])
    upper = float(edges[BAND_DECILE_HI])
    return lower, upper


def band_rate(df: pd.DataFrame, lower: float, upper: float) -> float:
    if len(df) == 0:
        return float("nan")
    in_band = (df["attack_angle"] >= lower) & (df["attack_angle"] <= upper)
    return float(in_band.mean())


def main() -> None:
    df = pd.read_csv(IN_PATH)
    required = ["player_id", "attack_angle", "strikes", "y1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"必須列が不足: {missing}")

    df = df.dropna(subset=["player_id", "attack_angle", "strikes"]).copy()
    df["strikes"] = df["strikes"].astype(int)

    # 固定帯域を overall 全体分布から復元
    lower, upper = infer_band_bounds_from_overall(df)

    oht = df[df["player_id"] == PLAYER_ID].copy()
    two = oht[oht["strikes"] == 2].copy()

    rows = [
        {
            "condition": "overall",
            "n": int(len(oht)),
            "in_band_rate": band_rate(oht, lower, upper),
            "lower_bound": lower,
            "upper_bound": upper,
        },
        {
            "condition": "two_strike",
            "n": int(len(two)),
            "in_band_rate": band_rate(two, lower, upper),
            "lower_bound": lower,
            "upper_bound": upper,
        },
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")

    # 最終コメント（事実のみ・1段落）
    print(
        "\n"
        "大谷翔平に限定しても、attack_angle の固定帯域（全体で定めた中央帯域）への集中は、"
        "overall / two_strike の両条件で大きく崩れなかったかどうかを、このCSVの in_band_rate で確認できる。"
    )


if __name__ == "__main__":
    main()


