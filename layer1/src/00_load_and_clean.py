"""
Step 0: Statcast生データのロードとBIP抽出、EV/LAの基本前処理を行う。
- 入力: data/raw/statcast_*.csv（複数選手対応）
- 出力: data/processed/bip.csv (EV/LAが存在する打球のみ、player_idカラム付き)
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd


RAW_DIR = "data/raw"
RAW_PATTERN = "statcast_*.csv"
OUT_PATH = "data/processed/bip.csv"
# 最低限の異常値フィルタ
MIN_EV = 0.0
MIN_LA = -90.0
MAX_LA = 90.0


def extract_player_id(filename: str) -> str:
    """statcast_<player>.csv から player_id を抽出"""
    match = re.search(r"statcast_(.+)\.csv", os.path.basename(filename))
    return match.group(1) if match else "unknown"


def load_all_statcast(raw_dir: str = RAW_DIR, pattern: str = RAW_PATTERN) -> pd.DataFrame:
    """複数のStatcastファイルを読み込み、player_idを付与して結合"""
    all_files = glob.glob(os.path.join(raw_dir, pattern))
    if not all_files:
        raise FileNotFoundError(f"No files found matching {pattern} in {raw_dir}")
    
    dfs = []
    for filepath in all_files:
        player_id = extract_player_id(filepath)
        df = pd.read_csv(filepath)
        df["player_id"] = player_id
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def filter_bip(df: pd.DataFrame) -> pd.DataFrame:
    """EV/LAが欠損でないインプレー打球のみを抽出し、異常値を除外。"""
    required = df.dropna(subset=["launch_speed", "launch_angle", "events"])
    # インプレー判定: eventsが存在するものを採用
    bip = required.copy()
    # 異常値除外
    bip = bip[bip["launch_speed"] > MIN_EV]
    bip = bip[(bip["launch_angle"] >= MIN_LA) & (bip["launch_angle"] <= MAX_LA)]
    return bip


def main() -> None:
    df = load_all_statcast()
    bip = filter_bip(df)
    bip.to_csv(OUT_PATH, index=False)
    print(f"saved {len(bip)} rows ({bip['player_id'].nunique()} players) -> {OUT_PATH}")
    print(f"Players: {', '.join(sorted(bip['player_id'].unique()))}")


if __name__ == "__main__":
    main()

