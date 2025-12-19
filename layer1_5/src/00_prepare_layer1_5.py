"""
Layer 1.5準備: Layer 1出力を読み込み、Layer 1.5用に前処理する。
- 入力: data/input/bip_with_v_and_y.csv (Layer 1成果物)
- 出力: data/intermediate/with_strata.csv (層別ラベル + y1/y2 + is_adverse)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


IN_PATH = "data/input/bip_with_v_and_y.csv"
OUT_PATH = "data/intermediate/with_strata.csv"


def load_layer1_output(path: str = IN_PATH) -> pd.DataFrame:
    """Layer 1の成果物を読み込む"""
    return pd.read_csv(path)


def _pitch_group(pitch_type: object) -> str:
    """簡易球種グループ（必要に応じて調整）。"""
    if pd.isna(pitch_type):
        return "Unknown"
    pt = str(pitch_type).upper()

    fastball = {"FF", "SI", "FT", "FC"}
    breaking = {"SL", "CU", "KC", "KN", "SV"}
    offspeed = {"CH", "FS", "FO", "EP", "SC"}

    if pt in fastball:
        return "FF"
    if pt in breaking:
        return "Breaking"
    if pt in offspeed:
        return "Offspeed"
    return "Other"


def _plate_z_group(row: pd.Series) -> str | float:
    """
    sz基準で plate_z を Low/Mid/High に分類。
    plate_z, sz_top, sz_bot が無い場合は NaN。
    """
    for c in ("plate_z", "sz_top", "sz_bot"):
        if c not in row or pd.isna(row[c]):
            return np.nan

    denom = (row["sz_top"] - row["sz_bot"])
    if pd.isna(denom) or denom <= 0:
        return np.nan

    z_norm = (row["plate_z"] - row["sz_bot"]) / denom
    if z_norm < 1 / 3:
        return "Low"
    if z_norm < 2 / 3:
        return "Mid"
    return "High"


def add_strata_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    層別ラベルを付与し、Layer1.5用の観測窓を固定する。
    - y1: Layer1のy（成功状態）をコピー（不変）
    - y2: V | y1==1（成功状態内の量的評価）
    - is_adverse: 不利条件フラグ（y3を“目的変数”として作らない）
    - pitch_group / plate_z_group / count_group: 層別用
    """
    df = df.copy()

    if "y" not in df.columns:
        raise KeyError("Layer1出力に 'y' が見つかりません（y1の元）。")
    if "V" not in df.columns:
        raise KeyError("Layer1出力に 'V' が見つかりません（y2の元）。")

    df["y1"] = df["y"].astype(int)
    df["y2"] = np.where(df["y1"] == 1, df["V"], np.nan)

    # 層別変数
    if "pitch_type" in df.columns:
        df["pitch_group"] = df["pitch_type"].map(_pitch_group)
    else:
        df["pitch_group"] = "Unknown"

    if all(c in df.columns for c in ("plate_z", "sz_top", "sz_bot")):
        df["plate_z_group"] = df.apply(_plate_z_group, axis=1)
    else:
        df["plate_z_group"] = np.nan

    if "strikes" in df.columns:
        df["count_group"] = np.where(
            df["strikes"] == 2, "two_strike", "non_two_strike"
        )
    else:
        df["count_group"] = np.nan

    # 不利条件フラグ（観測窓）
    df["is_adverse"] = (
        (df["plate_z_group"] == "Low")
        | (df["pitch_group"] == "Breaking")
        | (df["count_group"] == "two_strike")
    )

    return df


def main() -> None:
    df = load_layer1_output()
    df_strata = add_strata_labels(df)
    df_strata.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df_strata)} rows -> {OUT_PATH}")
    print(f"is_adverse rate: {df_strata['is_adverse'].mean():.3f}")


if __name__ == "__main__":
    main()

