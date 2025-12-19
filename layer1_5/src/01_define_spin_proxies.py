"""
回転代理変数の定義: 回転数"そのもの"ではなく、物理的代理変数を構築する。
- 入力: data/intermediate/with_strata.csv
- 出力: data/intermediate/spin_proxy_features.csv
"""

from __future__ import annotations

import pandas as pd


IN_PATH = "data/intermediate/with_strata.csv"
OUT_PATH = "data/intermediate/spin_proxy_features.csv"


def define_spin_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    回転の代理変数を構築。
    spin_rateは結論に使わず、物理的代理変数を構築する（必須4種）。
    """
    df = df.copy()

    # 必須: 回転の代理変数（列名固定）
    if "launch_speed" in df.columns and "bat_speed" in df.columns:
        df["contact_efficiency"] = df["launch_speed"] / (df["bat_speed"] + 1e-6)
    else:
        df["contact_efficiency"] = pd.NA

    if "hit_distance_sc" in df.columns and "launch_speed" in df.columns:
        df["carry_efficiency"] = df["hit_distance_sc"] / (df["launch_speed"] + 1e-6)
    else:
        df["carry_efficiency"] = pd.NA

    if "hyper_speed" in df.columns and "launch_speed" in df.columns:
        df["adjusted_ev_gap"] = df["hyper_speed"] - df["launch_speed"]
    else:
        df["adjusted_ev_gap"] = pd.NA

    if "attack_angle" in df.columns and "bat_speed" in df.columns:
        df["upper_velocity"] = df["attack_angle"] * df["bat_speed"]
    else:
        df["upper_velocity"] = pd.NA

    # 補助（確認用）：spin系の例（結論には使わない）
    if "release_spin_rate" in df.columns and "release_speed" in df.columns:
        df["spin_proxy_1"] = df["release_spin_rate"] / (df["release_speed"] + 1e-6)
    if "spin_axis" in df.columns and "launch_angle" in df.columns:
        df["spin_proxy_2"] = (df["spin_axis"] - df["launch_angle"]).abs()

    return df


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df_proxies = define_spin_proxies(df)
    df_proxies.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df_proxies)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()

