"""
接触質指標の定義: EV減衰・伸びを示す接触質指標を構築する。
- 入力: data/intermediate/spin_proxy_features.csv
- 出力: data/intermediate/contact_quality.csv
"""

from __future__ import annotations

import pandas as pd


IN_PATH = "data/intermediate/spin_proxy_features.csv"
OUT_PATH = "data/intermediate/contact_quality.csv"


def define_contact_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    接触質指標を構築。
    例: 理論EVと実測EVの差分、距離/EV比など
    """
    df = df.copy()
    
    # 例: 接触効率（距離/EV比）
    if "hit_distance_sc" in df.columns and "launch_speed" in df.columns:
        df["contact_quality_1"] = df["hit_distance_sc"] / (df["launch_speed"] + 1e-6)
    
    # 例: EV減衰率（理論値との差分）
    # 簡易版: 角度から期待される距離と実測距離の差分
    if "launch_angle" in df.columns and "hit_distance_sc" in df.columns:
        # 簡易モデル（実際はより複雑な計算が必要）
        df["contact_quality_2"] = df["hit_distance_sc"] - (df["launch_angle"] * 10)
    
    return df


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df_contact = define_contact_quality(df)
    df_contact.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df_contact)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()

